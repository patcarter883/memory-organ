#!/usr/bin/env python3
"""E0 — the persistence probe (SLIDING_WINDOW.md, first rung; issue #19 Track 4).

Does the product-key bank survive NOT being reset? Every published carry number writes M facts into a
FRESH episodic bank and reads them back within the episode. E0 keeps ONE bank alive across E episodes
of test-time writes (frozen addressing geometry — no gradients anywhere) and probes every fact after
every episode, so recall is measured as a function of WRITE-AGE (episodes since the fact was written)
and of total accumulated writes.

Conditions (each fact probed identically — query "<cargo>:" -> argmax over the FULL vocab == name):
  persistent : one bank, episode e's facts delta-written on top of episodes 0..e-1's. THE MEASUREMENT.
  fresh      : each episode's facts written into their own fresh bank (the published setting — ceiling).
  batch      : ALL E*M facts written into one fresh bank in a single write() call. Separates "too many
               facts" from "facts written over time": if batch holds but persistent decays with age,
               the interference is temporal, not capacity.
  empty      : probe against a zero bank (floor; the read of an empty store is exactly 0).

Facts are unique (cargo -> name) pairs drawn from the same single-token pools the store was trained
on, so writes are in-distribution test-time binds of NOVEL pairings — the same property held-out carry
already measures, minus the reset. Unique keys cap total facts at the cargo-pool size (~200 under the
Qwen tokenizer): with the default N=1024-slot store that is a fill fraction of ~0.2. To probe the
fill-fraction ladder toward and past 1.0, bind a SMALLER store (e.g. --n-sub 8 -> N=64) and rerun —
shrink N rather than invent off-distribution keys.

Run (GPU box; needs a saved single-token dict pk-store checkpoint, e.g. REPRODUCING.md §3's m8.pt):
  python -m cam.persist_probe --load-ckpt ckpt/m8.pt --episodes 16 --M 8 --seed 20260625
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# flat package: sibling imports resolve relatively when imported as cam.X (`python -m cam.X`,
# `import cam.X`) and fall back to a path-hacked absolute import when run as a file (`python cam/X.py`).
try:
    from .m2_adapter import MODEL, DEV
    from .recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder
    from .recall_mag import load_ckpt
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import MODEL, DEV                                      # noqa: E402
    from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
    from recall_mag import load_ckpt                                       # noqa: E402


# ---- core (adapter-only, model-free: CPU-testable) ----------------------------------------------
def draw_facts(rng, key_pool, val_pool, n):
    """n unique (key_tid, val_tid) facts: keys sampled WITHOUT replacement (a key must map to one
    value for the probe to be unambiguous); values sampled with replacement (shared values are fine —
    the store's job is key->value, not value uniqueness). -> LongTensor [n, 2]."""
    assert n <= len(key_pool), \
        f"{n} facts need {n} unique keys but the pool has {len(key_pool)} — lower --episodes/--M"
    keys = rng.choice(len(key_pool), size=n, replace=False)
    vals = rng.integers(0, len(val_pool), size=n)
    return torch.tensor([[key_pool[k], val_pool[v]] for k, v in zip(keys, vals)], dtype=torch.long)


def write_facts(adapter, V, facts):
    """Delta-write facts [F,2] into bank V [1,N,mem_dim] (functional, returns the updated bank).
    Exactly the trained write path: key/value -> frozen embed -> in_proj -> LayerNorm -> store.write."""
    dev = next(adapter.parameters()).device
    keys = adapter._e(facts[:, 0].unsqueeze(0).to(dev))          # [1,F,mem_dim]
    vals = adapter._e(facts[:, 1].unsqueeze(0).to(dev))
    return adapter.store.write(V, keys, vals)


@torch.no_grad()
def probe_facts(adapter, V, facts, colon, batch=128):
    """Probe each fact against bank V with the SAME read path the published carry numbers use:
    query tokens "<cargo>:" -> _e -> store.read -> readout_q attention-pool -> out_proj -> tied
    unembed; hit = argmax over the FULL vocab == the fact's name token. -> hits float tensor [F]."""
    dev = next(adapter.parameters()).device
    md = adapter.mem_dim
    colon = torch.tensor(colon, dtype=torch.long, device=dev)
    hits = []
    for s in range(0, facts.shape[0], batch):
        f = facts[s:s + batch].to(dev)
        Q = f.shape[0]
        q_ids = torch.cat([f[:, :1], colon.unsqueeze(0).expand(Q, -1)], dim=1)   # [Q, 1+|colon|]
        q = adapter._e(q_ids)                                     # [Q,Lq,mem_dim]
        read, _ = adapter.store.read(V.expand(Q, -1, -1), q)      # read-only: expand is safe
        pq = adapter.readout_q.unsqueeze(0).expand(Q, -1, -1)
        attn = torch.softmax(pq @ read.transpose(1, 2) / (md ** 0.5), dim=-1)
        prefix = adapter.out_proj(attn @ read)                    # [Q,K,base_hidden]
        logits = prefix.mean(dim=1) @ adapter.unembed             # [Q,vocab]
        hits.append((logits.argmax(-1) == f[:, 1]).float().cpu())
    return torch.cat(hits)


@torch.no_grad()
def run_e0(adapter, facts_by_episode, colon):
    """The E0 protocol. facts_by_episode: list of E LongTensors [M,2]. Returns a results dict:
      age_acc[a]        : persistent-bank recall of facts probed a episodes after their write,
                          pooled over every (write-episode, probe-episode) pair with that gap
      per_probe[e][j]   : persistent recall of episode-j facts right after episode e's write
      fresh_acc[e]      : fresh-bank recall of episode e's facts (ceiling, the published setting)
      batch_acc         : recall with all facts written in ONE call into a fresh bank
      empty_acc         : recall against a zero bank (floor)
    """
    dev = next(adapter.parameters()).device
    E = len(facts_by_episode)
    all_facts = torch.cat(facts_by_episode)
    V = adapter.store.init_state(1, dev, dtype=torch.float32)

    per_probe = []                                # per_probe[e][j] = acc of episode-j facts after ep e
    for e in range(E):
        V = write_facts(adapter, V, facts_by_episode[e])
        per_probe.append([float(probe_facts(adapter, V, facts_by_episode[j], colon).mean())
                          for j in range(e + 1)])

    age_hits = {}                                 # age -> [sum_hits, n]
    for e, row in enumerate(per_probe):
        for j, acc in enumerate(row):
            a = e - j
            s, n = age_hits.get(a, (0.0, 0))
            age_hits[a] = (s + acc * facts_by_episode[j].shape[0], n + facts_by_episode[j].shape[0])
    age_acc = {a: s / n for a, (s, n) in sorted(age_hits.items())}

    fresh_acc = []
    for e in range(E):
        Vf = adapter.store.init_state(1, dev, dtype=torch.float32)
        Vf = write_facts(adapter, Vf, facts_by_episode[e])
        fresh_acc.append(float(probe_facts(adapter, Vf, facts_by_episode[e], colon).mean()))

    Vb = adapter.store.init_state(1, dev, dtype=torch.float32)
    Vb = write_facts(adapter, Vb, all_facts)
    batch_acc = float(probe_facts(adapter, Vb, all_facts, colon).mean())

    V0 = adapter.store.init_state(1, dev, dtype=torch.float32)
    empty_acc = float(probe_facts(adapter, V0, all_facts, colon).mean())

    return {"age_acc": age_acc, "per_probe": per_probe, "fresh_acc": fresh_acc,
            "batch_acc": batch_acc, "empty_acc": empty_acc,
            "episodes": E, "facts_per_episode": [int(f.shape[0]) for f in facts_by_episode],
            "n_slots": adapter.store.N}


def report(res):
    E, N = res["episodes"], res["n_slots"]
    total = sum(res["facts_per_episode"])
    print(f"\n[e0] ===== PERSISTENCE PROBE — {E} episodes x {res['facts_per_episode'][0]} facts "
          f"= {total} writes into N={N} slots (fill {total / N:.2f}) =====", flush=True)
    print(f"[e0] controls: fresh-bank (ceiling, published setting) mean {np.mean(res['fresh_acc']):.3f} "
          f"| batch-write-all-at-once {res['batch_acc']:.3f} | empty bank (floor) {res['empty_acc']:.3f}",
          flush=True)
    print(f"[e0] recall vs WRITE-AGE (persistent bank; age = episodes since the fact was written):",
          flush=True)
    print(f"{'age':>5} {'recall':>8}", flush=True)
    for a, acc in res["age_acc"].items():
        print(f"{a:>5} {acc:>8.3f}", flush=True)
    print(f"[e0] age-0 persistent {res['age_acc'].get(0, float('nan')):.3f} vs fresh ceiling "
          f"{np.mean(res['fresh_acc']):.3f} — a gap here means prior accumulation hurts NEW writes too.",
          flush=True)
    oldest = max(res["age_acc"])
    print(f"[e0] verdict inputs: oldest-age recall {res['age_acc'][oldest]:.3f} (age {oldest}) | "
          f"batch {res['batch_acc']:.3f}. If batch holds and old ages decay, the wall is TEMPORAL "
          f"interference; if batch collapses too, it's CAPACITY (fill {total / N:.2f}).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-ckpt", type=str, required=True, dest="load_ckpt",
                    help="saved memory checkpoint (recall_mag --save-ckpt); must be a single-token "
                         "dict pk-store bind — the store whose persistence is being probed")
    ap.add_argument("--episodes", type=int, default=16, help="E: episodes of test-time writes")
    ap.add_argument("--M", type=int, default=8, help="facts written per episode")
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--base1", type=str, default="",
                    help="donor HF model id; default = the donor recorded in the checkpoint")
    ap.add_argument("--out", type=str, default="", help="write the results dict as JSON to this path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # checkpoint gates: E0 is defined for the single-token dict pk store (SLIDING_WINDOW.md E0).
    _meta = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    assert _meta.get("store", "bolt") == "pk", "E0 probes the product-key store; bind with --store pk"
    assert _meta.get("phrasing", "dict") == "dict" and int(_meta.get("cargo_tokens", 1)) == 1, \
        "E0 is single-token dict only (SLIDING_WINDOW.md E0); bind the ckpt accordingly"
    base1_id = args.base1 or _meta.get("base1") or MODEL
    del _meta

    # base-1 supplies ONLY its embedding table/tokenizer (E0 never runs a base forward).
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base1_id)
    m1 = AutoModelForCausalLM.from_pretrained(base1_id, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(DEV).eval()
    embed_weight = m1.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)              # VALUE pool (space-prefixed)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")  # KEY pool (line-initial, dict layout)
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")
    adapter, _injector, _L, ck = load_ckpt(args.load_ckpt, embed_weight, m1, DEV, builder=builder)
    del m1                                                       # embed table is copied; free the base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    key_pool = [t for (_w, t) in cargo]
    val_pool = [t for (_w, t) in names]
    facts = draw_facts(rng, key_pool, val_pool, args.episodes * args.M)
    facts_by_episode = list(facts.split(args.M))

    print(f"[e0] {base1_id} | ckpt {args.load_ckpt} (carry {ck.get('d_carry', float('nan')):.3f}) | "
          f"E={args.episodes} M={args.M} | key pool {len(key_pool)} val pool {len(val_pool)} | "
          f"N={adapter.store.N} slots", flush=True)

    res = run_e0(adapter, facts_by_episode, builder.colon)
    report(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "per_probe"} |
                      {"per_probe": res["per_probe"], "seed": args.seed, "ckpt": args.load_ckpt}, f,
                      indent=1, default=float)
        print(f"[e0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
