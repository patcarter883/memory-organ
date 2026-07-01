"""CAM v0 — the MAG falsifier. Does an additive ZERO-INIT Memory-as-Gate tap deliver the validated
DeepMemory binding through the FROZEN base, where boltA's Memory-as-Context prefix hit the wall
(memory ≈ no_memory)?  Full spec: titans/V0_SPEC.md.

Stage 1 (binding): train the BoltAdapter by the direct tied-unembed loss (reused from recall_boltA);
                    freeze it. This is the validated 0.86-carry binding.
Stage 2 (delivery): freeze base + memory; train ONLY the GatedMemoryTap(s) by LM-loss-through-the-
                    frozen-base on the recall task. Eval mirrors boltA: local_control / memory /
                    no_memory (NLL bits + accuracy). Default = sweep each tap depth independently.

Run:
  python -m cam.recall_mag --bind-steps 3000 --steps 3000 --tap-layers 8,16,24
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# flat package: make sibling modules importable whether run as `python -m cam.X` or `python cam/X.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,  # noqa: E402
                            single_token_ids, DocBuilder)
from recall_boltA import BoltAdapter, eval_direct                         # noqa: E402
from pk_store_adapter import PKStoreAdapter                               # noqa: E402
from gated_tap import MAGInjector                                         # noqa: E402

LN2 = math.log(2.0)
# eval-batch cap: eb = EVAL_BATCH_CAP // (M*Kc), shrinking the eval forward as M/Kc grow so the
# whole-doc ceiling pass fits on a 16GB card. Env-overridable (memory-only, accuracy-neutral): lower
# it (e.g. 48 -> eb=3 at M=8 K=2) if the two-base eval OOMs on ROCm (no expandable_segments here).
EVAL_BATCH_CAP = int(os.environ.get("CAM_EVAL_BATCH_CAP", "128"))


# ---- multi-token answer helpers -----------------------------------------------------------------
# In multi-token mode direct_logits is [B,Kc,vocab] and ans is [B,Kc]; single-token stays [B,vocab]/[B].
# These collapse both shapes so the bind/eval code stays uniform.
def _dlogits(adapter, pref, ans, dec):
    """direct_logits dispatch: pass gold answer ids only to a (teacher-forced AR) decoder readout;
    linear/bolt readouts take prefix only."""
    return adapter.direct_logits(pref, ans) if dec else adapter.direct_logits(pref)


def _seq_ce(logits, ans):
    """teacher-forced CE over the answer token sequence. [B,Kc,V]/[B,Kc] -> scalar; [B,V]/[B] -> scalar."""
    if logits.dim() == 3:
        V = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, V), ans.reshape(-1))
    return F.cross_entropy(logits, ans)


def _seq_metrics(logits, ans):
    """-> (exact_match[B] float, per_token_acc[B] float). single-token: both equal the 0/1 hit."""
    pred = logits.argmax(-1)
    if logits.dim() == 3:
        hit = (pred == ans).float()                     # [B,Kc]
        return hit.min(dim=1).values, hit.mean(dim=1)   # all-correct ; fraction-correct
    h = (pred == ans).float()
    return h, h


def _kc(builder):
    """answer length in tokens (1 single-token; Kc multi-token)."""
    return builder.cargo_tokens if getattr(builder, "multitoken", False) else 1


def _answer_logits(base, ctx_emb, Kc):
    """base forward -> the logits predicting the Kc answer tokens. single-token: [B,V] (last position).
    multi-token: [B,Kc,V] (the last Kc logit positions of a teacher-forced context)."""
    lg = base(inputs_embeds=ctx_emb).logits.float()
    return lg[:, -1] if Kc == 1 else lg[:, -Kc:]


# ---- memory bank: K query-conditioned pooled retrieval vectors (pre out_proj), mirrors BoltAdapter.inject
def memory_bank(adapter, ids, seg_len, qa_start, answer_pos, carry=True):
    """[B,K,mem_dim] — the leak-free memory bank fed to the MAG taps (mem_dim, NOT base-embed space).

    pk-store adapters expose their OWN memory_bank() (the pooled store read, pre out_proj — the
    base-agnostic mem_dim bank); dispatch to it so the Stage-2 tap + v1 translator drive the
    product-key store unchanged. BoltAdapter has no such method -> the DeepMemory path below runs
    byte-identically (the carry/no_memory comparison vs v0 stays exact)."""
    own = getattr(adapter, "memory_bank", None)
    if own is not None:
        return own(ids, seg_len, qa_start, answer_pos, carry=carry)
    emb = adapter._e(ids)                                                 # frozen embed->in_proj->norm
    B = emb.shape[0]
    state = adapter.mem.init_state(B)
    if carry:
        for s in range(0, qa_start, seg_len):
            state = adapter.mem(emb[:, s:s + seg_len], state)             # ingest pre-QA context
    q = emb[:, qa_start:answer_pos]                                       # cargo query (leak-free)
    retrieved = adapter.mem.retrieve(q, state)                           # [B,Lq,mem_dim]
    pq = adapter.readout_q.unsqueeze(0).expand(B, -1, -1)
    attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (adapter.mem_dim ** 0.5), dim=-1)
    return attn @ retrieved                                              # [B,K,mem_dim] pooled


def _leakfree_ctx(base, builder, ids, apos, end=None):
    """header (FORMAT only, no bindings) + query tokens -> base inputs_embeds. Same context boltA used.

    For multi-token TEACHER-FORCING pass end=apos+Kc-1: the context then also includes the first Kc-1
    GOLD answer tokens (ids[:,qa_start:end]) so the base's last Kc logit positions predict the full
    answer sequence ans_0..ans_{Kc-1}. end=None (default) = single-token (context ends at the query)."""
    if end is None:
        end = apos
    hlen = len(builder.bos) + len(builder.header)
    ctx_ids = torch.cat([ids[:, len(builder.bos):hlen], ids[:, builder.qa_start:end]], dim=1)
    return base.get_input_embeddings()(ctx_ids)


# ---- adapter factory: bolt (DeepMemory v0) or pk (product-key store, hub-free) -----------------
def build_adapter(args, embed_weight, H, builder=None):
    """Construct the memory front-end selected by args.store. 'bolt' = BoltAdapter (byte-identical to
    v0). 'pk' = PKStoreAdapter (product-key store) + set_builder(builder) for the bind-block positions,
    with addr-sup active during Stage-1/bind. Both satisfy the same inject/direct_logits/memory_bank
    contract Stage-2 needs."""
    store = getattr(args, "store", "bolt")
    if store == "pk":
        adapter = PKStoreAdapter(
            embed_weight, H, args.mem_dim, args.heads, args.chunk, args.expansion, args.k,
            n_sub=args.n_sub, topk=args.topk, sub_topk=args.sub_topk,
            addr_sup_weight=args.addr_sup_weight,
            read_heads=(args.pk_read_heads if args.pk_read_heads > 0 else None),
            mt_value=getattr(args, "mt_value", "mean"),
            mt_positions=max(2, getattr(args, "cargo_tokens", 1)),
            readout=getattr(args, "readout", "linear"),
            dec_layers=getattr(args, "dec_layers", 2),
            dec_heads=getattr(args, "dec_heads", 4),
            dec_dim=getattr(args, "dec_dim", 256),
            perpos_key=getattr(args, "perpos_key", "additive")).to(DEV)
        assert builder is not None, "pk adapter needs the DocBuilder (bind-block positions)"
        adapter.set_builder(builder)
        return adapter
    return BoltAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk,
                       args.expansion, args.k).to(DEV)


# ---- stage 1: bind (direct tied-unembed; no base in the loop) ----------------------------------
def bind_adapter(adapter, builder, rng, args):
    train_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    for step in range(args.bind_steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        pref = adapter.inject(ids, args.seg_len, builder.qa_start, apos, carry=True)
        # decoder readout is autoregressive + teacher-forced -> needs the gold answer ids; linear/bolt
        # readouts ignore it (their direct_logits takes only prefix).
        _dec = getattr(adapter, "readout", "linear") == "decoder"
        lm = _seq_ce(_dlogits(adapter, pref, ans, _dec), ans)  # CE over answer token SEQUENCE
        # pk-aware addressing-supervision: PKStoreAdapter.aux_loss() returns its weighted write->read
        # InfoNCE term (None when off / not a pk adapter). BoltAdapter has no aux_loss -> untouched.
        aux_fn = getattr(adapter, "aux_loss", None)
        aux = aux_fn() if aux_fn is not None else None
        loss = lm + aux if aux is not None else lm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.bind_steps - 1:
            em, pt = _seq_metrics(_dlogits(adapter, pref, ans, _dec), ans)   # exact-match ; per-token
            astr = f" addr {aux.item():.3f}" if aux is not None else ""
            print(f"[mag] bind step {step:4d} loss {lm.item():.3f}{astr} "
                  f"direct_acc(exact) {em.mean().item():.3f} per_tok {pt.mean().item():.3f}", flush=True)
    d_carry, d_abl, pt_carry, pt_abl = eval_direct(adapter, builder, rng, args)
    print(f"[mag] binding held-out: carry(exact) {d_carry:.3f} | ablated(exact) {d_abl:.3f} | "
          f"carry per-tok {pt_carry:.3f} | ablated per-tok {pt_abl:.3f} | chance {1/args.M:.3f}",
          flush=True)
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    return d_carry


# ---- stage 2: train the MAG tap(s) by LM-loss through the frozen base ---------------------------
def train_taps(base, adapter, injector, builder, rng, args, tag):
    injector.attach().train()
    Kc = _kc(builder)
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=True)
        injector.set_bank(bank)                                          # memory frozen -> bank detached
        # multi-token: teacher-force the answer prefix into the context (end=apos+Kc-1) so the last Kc
        # logit positions predict the full answer sequence.
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        logits = _answer_logits(base, ctx_emb, Kc)                       # [B,V] or [B,Kc,V]
        loss = _seq_ce(logits, ans)
        if not torch.isfinite(loss):                                     # NaN/Inf guard: skip the step
            print(f"[mag][{tag}] step {step:4d} NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(injector.parameters()), 1.0)
        if not torch.isfinite(gn):                                       # NaN grad guard
            print(f"[mag][{tag}] step {step:4d} NON-FINITE grad -> skip", flush=True)
            opt.zero_grad(); continue
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            em, pt = _seq_metrics(logits, ans)
            print(f"[mag][{tag}] step {step:4d} loss {loss.item():.3f} exact {em.mean().item():.3f} "
                  f"per_tok {pt.mean().item():.3f} gate {injector.gate_stats()}", flush=True)
    injector.set_bank(None)


def _nll_bits(logits, ans):
    """teacher-forced answer NLL in bits, mean over the answer tokens. [B,V]/[B] or [B,Kc,V]/[B,Kc]."""
    lp = F.log_softmax(logits, -1)
    if logits.dim() == 3:
        b = -lp.gather(-1, ans.unsqueeze(-1)).squeeze(-1) / LN2   # [B,Kc] per-token bits
        return b.mean(dim=1).tolist()                            # mean per-token bits per row
    return (-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist()


@torch.no_grad()
def eval_generative_mag(base, adapter, injector, builder, rng, args, n=512):
    """Delivery eval through the frozen base. Multi-token: scores TEACHER-FORCED exact-match (all Kc
    answer tokens correct) AND per-token acc; single-token: both equal the 0/1 hit (byte-identical).
    Returns {cond: (nll_bits, exact_match_acc, per_token_acc)}."""
    base_embed = base.get_input_embeddings()
    Kc = _kc(builder)
    # per cond: [nll_list, exact_sum, pertok_sum]
    res = {c: [[], 0.0, 0.0] for c in ("local_control", "memory", "no_memory")}
    injector.eval()
    seen = 0
    while seen < n:
        # eval batch shrinks as M grows: M>=64 docs (qa_seg 9) OOM the full-base ceiling forward at
        # batch 16 on a 16GB card. The CEILING forward processes the WHOLE doc (~bos+header+M*bind_len+qa
        # tokens) at this batch — the memory hog — so cap conservatively for multi-token (the qa_seg-2
        # M=8 K=2 doc + accumulated frag OOM'd at eb=16). Batch size doesn't affect accuracy, only memory.
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M * Kc)))
        cur = min(eb, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # ceiling: full in-context doc up to apos+Kc-1 (teacher-forced), last Kc logits
        injector.set_bank(None)                                          # tap OFF -> ceiling
        lc = _answer_logits(base, base_embed(ids[:, :apos + Kc - 1]), Kc)
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            injector.set_bank(bank)
            lg = _answer_logits(base, ctx_emb, Kc)
            res[cond][0].extend(_nll_bits(lg, ans))
            em, pt = _seq_metrics(lg, ans)
            res[cond][1] += em.sum().item(); res[cond][2] += pt.sum().item()
        injector.set_bank(None)
        res["local_control"][0].extend(_nll_bits(lc, ans))
        em, pt = _seq_metrics(lc, ans)
        res["local_control"][1] += em.sum().item(); res["local_control"][2] += pt.sum().item()
        seen += cur
        # heartbeat: the multi-token eval (small eb -> many batches) is otherwise silent for minutes
        # and trips the watchdog's log-idle STALL guard. Print progress so the run stays alive.
        print(f"[mag] eval progress {seen}/{n}", flush=True)
    return {c: (float(np.mean(res[c][0])), res[c][1] / seen, res[c][2] / seen) for c in res}


# ---- checkpoint: persist the frozen v0 memory front-end (BoltAdapter) + a passing GatedMemoryTap ----
# so v1 reuses ONE fixed memory across bases instead of re-binding it each run. The bank fed to the
# taps ([B,K,mem_dim]) is base-AGNOSTIC (DeepMemory's own mem_dim space), so the same checkpoint drives
# any base; only the per-base translator/tap geometry differs.
def save_ckpt(path, adapter, injector, tap_layer, args, d_carry):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # drop the frozen tied embed/unembed (~3GB) — rebuilt from base-1's table on load
    asd = {k: v for k, v in adapter.state_dict().items()
           if not (k.startswith("embed.") or k == "unembed")}
    torch.save({
        "adapter": asd,
        "taps": injector.taps.state_dict(),
        "tap_layer": tap_layer,
        "tap_heads": args.tap_heads,
        "mem_dim": args.mem_dim, "heads": args.heads, "chunk": args.chunk,
        "expansion": args.expansion, "k": args.k, "d_carry": d_carry,
        # store selector + pk knobs so load_ckpt rebuilds the right adapter (bolt path unchanged:
        # store defaults to 'bolt' and the pk_* keys are ignored when rebuilding a BoltAdapter).
        "store": getattr(args, "store", "bolt"),
        "n_sub": getattr(args, "n_sub", 32), "topk": getattr(args, "topk", 8),
        "sub_topk": getattr(args, "sub_topk", 4),
        "addr_sup_weight": getattr(args, "addr_sup_weight", 0.0),
        "pk_read_heads": getattr(args, "pk_read_heads", 0),
        "cargo_tokens": getattr(args, "cargo_tokens", 1),   # multi-token answer length (v1 rebuilds K)
        "mt_value": getattr(args, "mt_value", "mean"),      # multi-token value mode (mean/perpos)
        "readout": getattr(args, "readout", "linear"),      # Stage-1 value readout (linear/decoder)
        "dec_layers": getattr(args, "dec_layers", 2),
        "dec_heads": getattr(args, "dec_heads", 4),
        "dec_dim": getattr(args, "dec_dim", 256),
        "perpos_key": getattr(args, "perpos_key", "additive"),  # per-position key conditioning
    }, path)
    print(f"[mag] saved v0 memory checkpoint -> {path} (tap L={tap_layer}, carry {d_carry:.3f})", flush=True)


def load_ckpt(path, embed_weight, base, dev, builder=None):
    """Rebuild the frozen memory front-end (BoltAdapter or PKStoreAdapter) + GatedMemoryTap from a
    checkpoint and freeze them. Returns (adapter, injector, tap_layer, meta). For a pk-store ckpt a
    DocBuilder MUST be passed (the store needs the bind-block positions for memory_bank); bolt ckpts
    ignore it (store defaults to 'bolt' for pre-pk checkpoints)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    H = base.config.get_text_config().hidden_size
    store = ck.get("store", "bolt")
    if store == "pk":
        adapter = PKStoreAdapter(
            embed_weight, H, ck["mem_dim"], ck["heads"], ck["chunk"], ck["expansion"], ck["k"],
            n_sub=ck.get("n_sub", 32), topk=ck.get("topk", 8), sub_topk=ck.get("sub_topk", 4),
            addr_sup_weight=ck.get("addr_sup_weight", 0.0),
            read_heads=(ck.get("pk_read_heads", 0) if ck.get("pk_read_heads", 0) > 0 else None),
            mt_value=ck.get("mt_value", "mean"),
            mt_positions=max(2, ck.get("cargo_tokens", 1)),
            readout=ck.get("readout", "linear"),
            dec_layers=ck.get("dec_layers", 2),
            dec_heads=ck.get("dec_heads", 4),
            dec_dim=ck.get("dec_dim", 256),
            perpos_key=ck.get("perpos_key", "additive")).to(dev)
        assert builder is not None, "pk-store ckpt requires a DocBuilder (set_builder) at load_ckpt"
        adapter.set_builder(builder)
    else:
        adapter = BoltAdapter(embed_weight, H, ck["mem_dim"], ck["heads"], ck["chunk"],
                              ck["expansion"], ck["k"]).to(dev)
    # embed/unembed are not in the ckpt (rebuilt from base-1's table); load the rest strictly-ish
    missing, unexpected = adapter.load_state_dict(ck["adapter"], strict=False)
    assert not unexpected, f"unexpected ckpt keys: {unexpected}"
    assert all(k.startswith("embed.") or k == "unembed" for k in missing), \
        f"unexpected MISSING adapter keys: {missing}"
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    L = ck["tap_layer"]
    injector = MAGInjector(base, [L], ck["mem_dim"], n_heads=ck["tap_heads"]).to(dev)
    injector.taps.load_state_dict(ck["taps"])
    for p in injector.parameters():
        p.requires_grad_(False)
    injector.eval()
    print(f"[mag] loaded v0 memory checkpoint <- {path} (tap L={L}, carry {ck.get('d_carry', float('nan')):.3f})",
          flush=True)
    return adapter, injector, L, ck


def verdict(tag, d_carry, gen, chance):
    lc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]            # exact-match (==hit single-token)
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print(f"\n[mag][{tag}] === generative through frozen base (acc=EXACT-MATCH; per_tok shown) ===",
          flush=True)
    print(f"{'condition':>14} {'NLL(bits)':>11} {'exact':>7} {'per_tok':>8}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>11.3f} {gen[c][1]:>7.3f} {gen[c][2]:>8.3f}", flush=True)
    print(f"[mag][{tag}] memory exact {m_acc:.3f} (per_tok {gen['memory'][2]:.3f}) / no_memory "
          f"{nm_acc:.3f} (per_tok {gen['no_memory'][2]:.3f}) / ceiling {lc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits", flush=True)
    if m_acc > nm_acc + 0.15 and m_acc > 0.5:
        v = "MAG WORKS — greenlight v1 (translator + 2nd base)"
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        v = "PARTIAL — go multi-layer / data-dependent gate / unfreeze memory gates"
    else:
        v = "WALL at this depth — escalate to multi-layer; if all depths fail, frozen-base premise is the limit"
    print(f"[mag][{tag}] => {v}\n" + "=" * 64, flush=True)
    return m_acc, nm_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-steps", type=int, default=3000, dest="bind_steps")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mem-dim", type=int, default=512, dest="mem_dim")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expansion", type=float, default=4.0)
    ap.add_argument("--store", type=str, default="bolt", choices=["bolt", "pk"],
                    help="memory mechanism: 'bolt' (DeepMemory v0, default) or 'pk' (product-key "
                         "sparse store, hub-free). pk routes the Stage-2 tap + ckpt through PKStoreAdapter.")
    ap.add_argument("--n-sub", type=int, default=32, dest="n_sub", help="pk: N=n_sub^2 slots")
    ap.add_argument("--topk", type=int, default=8, help="pk: global product-keys kept per query")
    ap.add_argument("--sub-topk", type=int, default=4, dest="sub_topk", help="pk: top-k per half")
    ap.add_argument("--addr-sup-weight", type=float, default=0.0, dest="addr_sup_weight",
                    help="pk: weight of the write->read addressing-supervision (InfoNCE) loss")
    ap.add_argument("--pk-read-heads", type=int, default=0, dest="pk_read_heads",
                    help="pk: override the store's read-head count (0 = use --heads)")
    ap.add_argument("--tap-heads", type=int, default=8, dest="tap_heads")
    ap.add_argument("--tap-layers", type=str, default="", dest="tap_layers",
                    help="comma list of decoder layers to tap; empty -> [n_layers//2]")
    ap.add_argument("--multi", action="store_true",
                    help="train ALL --tap-layers together (escalation) instead of sweeping each")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--save-ckpt", type=str, default="", dest="save_ckpt",
                    help="after a single-layer run, save the frozen BoltAdapter+tap to this path")
    ap.add_argument("--load-ckpt", type=str, default="", dest="load_ckpt",
                    help="reload a saved v0 memory checkpoint instead of re-binding; reproduces V0")
    ap.add_argument("--save-anyway", action="store_true", dest="save_anyway",
                    help="save the ckpt even if the tap didn't pass (smoke-test plumbing only)")
    ap.add_argument("--cargo-tokens", type=int, default=1, dest="cargo_tokens",
                    help="K: multi-token answer cargo phrase length (1=single-token; >1=role-swapped "
                         "'name: <K-token real-word phrase>', answer = the K-token sequence)")
    ap.add_argument("--readout", type=str, default="linear", choices=["linear", "decoder", "perpos"],
                    help="pk multi-token VALUE readout: 'linear' (default, byte-preserved: slot t -> "
                         "answer token t in one projection) or 'decoder' (tiny AR transformer-decoder "
                         "head over the K retrieved store slots; teacher-forced causal decode of the "
                         "answer sequence — gives the value path real sequence capacity) or 'perpos' "
                         "(FACTORIZE the K-token answer into K single-token (name,position) bindings: "
                         "per-position store value slots + per-position read queries + PER-POSITION "
                         "addressing supervision; each slot decoded by the single-token linear readout. "
                         "Shorthand for '--pk-mt-value perpos' with per-position addr-sup.)")
    ap.add_argument("--dec-layers", type=int, default=2, dest="dec_layers", help="decoder readout layers")
    ap.add_argument("--dec-heads", type=int, default=4, dest="dec_heads", help="decoder readout heads")
    ap.add_argument("--dec-dim", type=int, default=256, dest="dec_dim", help="decoder readout model dim")
    ap.add_argument("--pk-mt-value", type=str, default="mean", dest="mt_value", choices=["mean", "perpos"],
                    help="pk multi-token value mode: 'mean' (one store value = mean of the K cargo "
                         "embeds; K-slot head disentangles) or 'perpos' (K position-tagged store "
                         "associations; each answer slot reads its own token)")
    ap.add_argument("--perpos-key", type=str, default="additive", dest="perpos_key",
                    choices=["additive", "gated", "codebook", "disjoint"],
                    help="perpos per-position KEY conditioning (Thrust 1 #3): how answer position t is "
                         "folded into the name key/query. 'additive' (default, byte-preserved): "
                         "key=name+pos_tag[t] (the weak code exp#2 ruled out). 'gated': "
                         "key=name*pos_gate[t]+pos_tag[t] (elementwise per-position scale). 'codebook': "
                         "key=pos_proj[t](name)+pos_tag[t] (per-position LINEAR map ~identity-init, the "
                         "strongest separation — each position addresses a different codebook subspace "
                         "BEFORE the product-key half-split). 'disjoint' (Thrust 1 #4): each position "
                         "owns an ENTIRELY SEPARATE ProductKeyStore (own codebook/heads/bank); position t "
                         "reads/writes/addr-sups against store t ONLY, so the per-position address cannot "
                         "be contaminated by the other position's slots (key = L2(name)+pos_tag[t]). "
                         "Only active under perpos.")
    args = ap.parse_args()

    # '--readout perpos' is shorthand: it selects the per-position VALUE STORE (mt_value=perpos) +
    # per-position addressing supervision, decoded by the single-token LINEAR readout (slot t @ unembed).
    # Normalize to the underlying (mt_value, readout=linear) the adapter/ckpt understand.
    if args.readout == "perpos":
        args.mt_value = "perpos"
        args.readout = "linear"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    n_layers = base.config.get_text_config().num_hidden_layers
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    cargo_words = single_token_ids(tok, MULTITOKEN_WORD_POOL) if args.cargo_tokens > 1 else None
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict",
                         cargo_tokens=args.cargo_tokens, cargo_words=cargo_words)
    if args.cargo_tokens > 1:
        print(f"[mag] MULTI-TOKEN cargo: K={args.cargo_tokens} word_pool={len(cargo_words)} "
              f"(answer = K-token real-word phrase; acc = exact-match)", flush=True)

    # ---- RELOAD path: reuse a fixed v0 memory checkpoint (no re-bind) and reproduce the V0 eval ----
    if args.load_ckpt:
        adapter, injector, L, ck = load_ckpt(args.load_ckpt, embed_weight, base, DEV, builder=builder)
        print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | RELOAD tap L={L} | "
              f"K={ck['k']} mem_dim={ck['mem_dim']} | chance acc={1/args.M:.3f}", flush=True)
        injector.attach()
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(str(L), ck.get("d_carry", float("nan")), gen, 1 / args.M)
        injector.detach()
        print("\n[mag] RELOAD SANITY (tap -> memory / no_memory / ceiling):", flush=True)
        print(f"  L={L:>8}  {m_acc:.3f} / {nm_acc:.3f} / {gen['local_control'][1]:.3f}", flush=True)
        print("[mag] PASS if memory ≫ no_memory (reproduces V0 on base-1 from the saved memory).", flush=True)
        return

    layers = ([int(x) for x in args.tap_layers.split(",") if x != ""]
              if args.tap_layers else [n_layers // 2])
    print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | tap_layers={layers} multi={args.multi} | "
          f"K={args.k} mem_dim={args.mem_dim} | chance acc={1/args.M:.3f}", flush=True)

    # ---- stage 1: bind once ----
    adapter = build_adapter(args, embed_weight, H, builder=builder)
    d_carry = bind_adapter(adapter, builder, rng, args)

    # ---- stage 2: MAG delivery ----
    configs = [layers] if args.multi else [[L] for L in layers]
    summary = []
    for cfg in configs:
        tag = "+".join(map(str, cfg))
        injector = MAGInjector(base, cfg, args.mem_dim, n_heads=args.tap_heads).to(DEV)
        train_taps(base, adapter, injector, builder, rng, args, tag)
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(tag, d_carry, gen, 1 / args.M)
        summary.append((tag, m_acc, nm_acc, gen["local_control"][1]))
        # save the FIRST passing single-layer tap as the reusable v0 memory checkpoint
        if args.save_ckpt and len(cfg) == 1 and (args.save_anyway or (m_acc > nm_acc + 0.15 and m_acc > 0.5)):
            save_ckpt(args.save_ckpt, adapter, injector, cfg[0], args, d_carry)
            args.save_ckpt = ""  # save only once (the first passing depth)
        injector.detach()

    print("\n[mag] SUMMARY (tap -> memory / no_memory / ceiling):", flush=True)
    for tag, m, nm, lc in summary:
        print(f"  L={tag:>8}  {m:.3f} / {nm:.3f} / {lc:.3f}", flush=True)
    print(f"[mag] boltA reference (MAC): memory ≈ no_memory ≈ 0.000 (the wall this run tests against).",
          flush=True)


if __name__ == "__main__":
    main()
