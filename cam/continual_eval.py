"""continual_eval — Axis 4 (HOPE headline): continual knowledge / NO-FORGET via cartridge COMPOSABILITY.

Thesis (Cartridges arXiv:2506.06266 §composition; scorecard axis 4): train N per-fact Cartridges
INDEPENDENTLY, then CONCATENATE them along the prefix dim and serve together. If composition works,
each fact stays answerable as N grows (no interference), and unrelated base knowledge is preserved
(locality / no-forget) — WITHOUT ever touching the base. Contrast = sequential LoRA fine-tuning, which
overwrites earlier facts (catastrophic forgetting).

Reports:
  1. accuracy-vs-N for the COMPOSED cartridge (each fact queried with all N loaded);
  2. general-knowledge PRESERVATION (base answers a held-out QA set; does loading memory degrade it?);
  3. sequential-LoRA-FT contrast (train fact1..factK in order, re-test fact1 -> forgets).

Run (local gfx1201, via the shared arbiter):
  PYTHONPATH=/home/pat/vllm-rocm/lib/python3.12/site-packages \
  CAM_BASE_MODEL=Qwen/Qwen3-4B-Base HF_HUB_OFFLINE=1 \
  gpu-lease -n 1 -- python -m cam.continual_eval --n 5 --p 64 --max-steps 40 --lora-facts 3
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

try:
    from .m2_adapter import load_frozen_base
    from .cartridge import concat_cartridges
    from .cartridge_train import train_cartridge, direct_cloze_delivers
    from .cotrain_lora import attach_cotrain_lora
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import load_frozen_base                       # noqa: E402
    from cartridge import concat_cartridges                       # noqa: E402
    from cartridge_train import train_cartridge, direct_cloze_delivers  # noqa: E402
    from cotrain_lora import attach_cotrain_lora                  # noqa: E402


# A dozen simple factual questions the frozen base answers correctly with NO cartridge. Preservation =
# does loading the composed memory (about UNRELATED facts) degrade these? gold = substring alias(es).
GENERAL_QA = [
    ("The capital of France is", ["Paris"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The largest planet in the solar system is", ["Jupiter"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The author of Romeo and Juliet is", ["Shakespeare"]),
    ("The tallest mountain on Earth is", ["Everest"]),
    ("The number of continents on Earth is", ["seven", "7"]),
    ("The speed of light is approximately 300,000 kilometers per", ["second"]),
    ("The first president of the United States was George", ["Washington"]),
    ("The opposite of hot is", ["cold"]),
    ("The square root of 144 is", ["12", "twelve"]),
]


def load_counterfact_facts(path, n, tok):
    """Distinct counterfactual facts from CounterFact: (subject, cloze prompt, target_new, target_true).
    Pick edits with DISTINCT relations+subjects so the N facts don't alias. Multi-token targets are OK
    (delivery uses substring match, not the single-token cloze)."""
    recs = json.load(open(path))
    facts, seen_subj, seen_rel = [], set(), set()
    for r in recs:
        rw = r["requested_rewrite"]
        subj, rel = rw["subject"], rw["relation_id"]
        if subj in seen_subj:
            continue
        prompt = rw["prompt"].format(subj)
        tgt_new, tgt_true = rw["target_new"]["str"], rw["target_true"]["str"]
        if tgt_new.lower() in prompt.lower():
            continue
        facts.append(dict(subject=subj, prompt=prompt, new=tgt_new, true=tgt_true, rel=rel))
        seen_subj.add(subj)
        seen_rel.add(rel)
        # prefer relation diversity but don't hard-require it once we run low
        if len(facts) >= n:
            break
    return facts[:n]


@torch.no_grad()
def _greedy_after_lora(base, tok, cloze, target, gen_len=10):
    """Greedy cloze delivery with whatever is currently attached to `base` (LoRA hooks). Substring hit."""
    device = next(base.parameters()).device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + tok(cloze, add_special_tokens=False).input_ids
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(gen_len):
        logits = base(input_ids=cur).logits[:, -1]
        nxt = logits.argmax(-1)
        out.append(int(nxt.item()))
        cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
        if "\n" in tok.decode(out):
            break
    gen = tok.decode(out)
    return " ".join(target.split()).lower() in " ".join(gen.split()).lower(), gen


def sequential_lora_ft(base, tok, facts, steps_per_fact=40, lr=1e-3, rank=8, layers=None):
    """CONTRAST baseline: sequential fine-tuning. Attach LoRA to late attn layers, then train fact-by-fact
    (LM cross-entropy on each fact's target tokens given the cloze), continuing the SAME adapters. Returns
    per-fact recall measured AFTER all facts are learned (earlier facts should be overwritten)."""
    cfg = base.config.get_text_config()
    nL = cfg.num_hidden_layers
    layers = layers or list(range(max(0, nL - 8), nL))   # late layers (where the fact resolves)
    adapters, params, handles = attach_cotrain_lora(base, layers, ["q_proj", "k_proj", "v_proj", "o_proj"],
                                                     rank=rank, alpha=2 * rank)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    device = next(base.parameters()).device
    recall_after_each = []           # recall of fact 0 immediately after learning each fact (drift curve)
    for fi, f in enumerate(facts):
        cloze_ids = tok(f["prompt"] + " ", add_special_tokens=False).input_ids
        tgt_ids = tok(f["new"], add_special_tokens=False).input_ids
        seq = bos + cloze_ids + tgt_ids
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        labels = torch.full((1, len(seq)), -100, dtype=torch.long, device=device)
        start = len(bos) + len(cloze_ids)
        labels[0, start:] = torch.tensor(tgt_ids, device=device)
        for _ in range(steps_per_fact):
            logits = base(input_ids=inp).logits
            loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        hit0, _ = _greedy_after_lora(base, tok, facts[0]["prompt"], facts[0]["new"])
        recall_after_each.append(hit0)
    # final recall of EVERY fact
    final = []
    for f in facts:
        hit, gen = _greedy_after_lora(base, tok, f["prompt"], f["new"])
        final.append((f, hit, gen))
    for h in handles:
        h.remove()
    base._cotrain_lora = None
    return final, recall_after_each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pat/code/memory-organ/data/counterfact.json")
    ap.add_argument("--n", type=int, default=5, help="number of facts / cartridges")
    ap.add_argument("--p", type=int, default=64, help="per-cartridge prefix length (composed p = n*p)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--samples-per-seed", type=int, default=1)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--lora-facts", type=int, default=3, help="how many facts in the sequential-FT contrast")
    ap.add_argument("--lora-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    print(f"[continual] loading base {os.environ.get('CAM_BASE_MODEL', '<default>')} ...", flush=True)
    base, tok = load_frozen_base()
    facts = load_counterfact_facts(args.data, args.n, tok)
    print(f"[continual] {len(facts)} facts:", flush=True)
    for i, f in enumerate(facts):
        print(f"  [{i}] {f['prompt']!r} -> {f['new']!r} (true {f['true']!r}, rel {f['rel']})", flush=True)

    # -------- 1) train N cartridges INDEPENDENTLY --------
    carts = []
    t_train = 0.0
    for i, f in enumerate(facts):
        edit_text = f["prompt"] + " " + f["new"]
        t0 = time.time()
        cart, info = train_cartridge(base, tok, edit_text, p=args.p, lr=args.lr, epochs=args.epochs,
                                     max_steps=args.max_steps, n_seeds=args.n_seeds,
                                     samples_per_seed=args.samples_per_seed, gen_len=args.gen_len,
                                     seed=args.seed + i)
        dt = time.time() - t0
        t_train += dt
        carts.append(cart)
        # sanity: does the SOLO cartridge deliver its own fact?
        solo_hit, solo_gen = direct_cloze_delivers(base, tok, cart, f["prompt"], f["new"])
        print(f"  [train {i}] KL={info['final_kl']:.3f} steps={info['steps']} {dt:.0f}s | "
              f"solo deliver={'Y' if solo_hit else '.'} gen={solo_gen.strip()[:40]!r}", flush=True)

    # -------- 2) accuracy vs N: query each fact with the FIRST k cartridges composed --------
    print("\n[continual] === accuracy-vs-N (composed cartridge) ===", flush=True)
    print(f"[continual] {'N':>3s} | {'each-fact recall (composed)':<32s} | acc", flush=True)
    acc_vs_n = []
    for k in range(1, len(carts) + 1):
        composed = concat_cartridges(carts[:k])
        hits = []
        for j in range(k):
            hit, _ = direct_cloze_delivers(base, tok, composed, facts[j]["prompt"], facts[j]["new"])
            hits.append(hit)
        acc = sum(hits) / len(hits)
        acc_vs_n.append((k, acc))
        marks = "".join("Y" if h else "." for h in hits)
        print(f"[continual] {k:3d} | {marks:<32s} | {acc:.3f}", flush=True)

    # -------- 3) preservation / locality: base vs base+composed(all N) on unrelated general QA --------
    print("\n[continual] === preservation (unrelated general knowledge) ===", flush=True)
    composed_all = concat_cartridges(carts)
    base_ok, cart_ok, both = 0, 0, 0
    for q, golds in GENERAL_QA:
        hb = any(direct_cloze_delivers(base, tok, None, q, g)[0] for g in golds)
        hc = any(direct_cloze_delivers(base, tok, composed_all, q, g)[0] for g in golds)
        base_ok += hb
        cart_ok += hc
        both += (hb and hc)
        print(f"  base={'Y' if hb else '.'} +mem={'Y' if hc else '.'}  {q!r}", flush=True)
    n_qa = len(GENERAL_QA)
    retained = both / base_ok if base_ok else float("nan")
    print(f"[continual] base QA acc={base_ok}/{n_qa}={base_ok/n_qa:.3f}  "
          f"+composed-memory acc={cart_ok}/{n_qa}={cart_ok/n_qa:.3f}  "
          f"preserved(of base-correct)={retained:.3f}", flush=True)

    # -------- 4) sequential-LoRA-FT contrast (catastrophic forgetting) --------
    print("\n[continual] === sequential LoRA fine-tuning contrast ===", flush=True)
    kL = min(args.lora_facts, len(facts))
    lora_facts = facts[:kL]
    final, drift = sequential_lora_ft(base, tok, lora_facts, steps_per_fact=args.lora_steps)
    print(f"[continual] fact0 recall AFTER learning each subsequent fact (drift): "
          f"{['Y' if h else '.' for h in drift]}", flush=True)
    lora_hits = [h for _f, h, _g in final]
    print(f"[continual] final LoRA recall of all {kL} facts (after sequential FT):", flush=True)
    for (f, h, g) in final:
        print(f"  {'Y' if h else '.'}  {f['prompt']!r} -> want {f['new']!r} got {g.strip()[:40]!r}", flush=True)
    lora_acc = sum(lora_hits) / len(lora_hits)

    # -------- SUMMARY --------
    composed_kL = concat_cartridges(carts[:kL])
    comp_hits = [direct_cloze_delivers(base, tok, composed_kL, facts[j]["prompt"], facts[j]["new"])[0]
                 for j in range(kL)]
    comp_acc = sum(comp_hits) / len(comp_hits)
    print("\n[continual] " + "=" * 68, flush=True)
    print(f"[continual] SUMMARY (n={len(facts)}, p={args.p}, composed p={args.p*len(carts)})", flush=True)
    print(f"[continual]   composed accuracy-vs-N : {[f'{k}:{a:.2f}' for k, a in acc_vs_n]}", flush=True)
    print(f"[continual]   preservation (base-correct QA still correct w/ memory): {retained:.3f}", flush=True)
    print(f"[continual]   NO-FORGET contrast @N={kL}:  composed cartridges acc={comp_acc:.3f}  "
          f"vs  sequential-LoRA-FT acc={lora_acc:.3f}  (fact0: composed="
          f"{'Y' if comp_hits[0] else '.'} vs LoRA={'Y' if lora_hits[0] else '.'})", flush=True)
    print(f"[continual]   cartridge train time total={t_train:.0f}s ({t_train/max(1,len(facts)):.0f}s/fact)",
          flush=True)
    print("[continual] " + "=" * 68, flush=True)


if __name__ == "__main__":
    main()
