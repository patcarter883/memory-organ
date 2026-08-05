"""moc_eval — Axis 4 (HOPE no-forget) via MIXTURE-OF-CARTRIDGES: sparse ROUTING over per-fact cartridges.

Continual_eval showed that naive `concat_cartridges` (all N cartridges as one always-on prefix) collapses
beyond N~2-3 (mutual interference) AND corrupts unrelated knowledge (general-QA 0.75 -> 0.17 at N=10). The
structural fix is SPARSE ACTIVATION: a lightweight, NON-PARAMETRIC router that per-query activates ONLY the
relevant cartridge(s). Then there is ZERO cross-fact interference by construction (each fact isolated in its
own memory), and an unrelated query activates NO corrupting memory. This is the one axis where a routed bank
of frozen, isolated cartridges can STRUCTURALLY beat a monolithic model (Titans/HOPE share weights across
every fact; a router of isolated cartridges does not).

Router = non-parametric retrieval (no training, frozen base): KEY per fact = the frozen base's mean-pooled
last-hidden-state of the fact's subject/prompt text; at query time embed the query the same way and pick the
top-1 (or top-2) cartridge by cosine similarity. Retrieval accuracy is reported SEPARATELY from end-task
accuracy so failures are attributable (router mis-pick vs cartridge non-delivery).

Reports:
  (a) router retrieval accuracy (top-1 / top-2), key=subject vs key=prompt;
  (b) accuracy-vs-N: ROUTING (attach only top-1) vs naive CONCAT (all N always-on);
  (c) locality/preservation on unrelated general-QA: ROUTING (top-1-anyway, and thresholded: attach only if
      sim>thr else NO cartridge) vs CONCAT;
  (d) cost.

Run (local gfx1201, via the shared arbiter):
  PYTHONPATH=/home/pat/vllm-rocm/lib/python3.12/site-packages \
  CAM_BASE_MODEL=Qwen/Qwen3-4B-Base HF_HUB_OFFLINE=1 \
  gpu-lease -n 1 -- python -m cam.moc_eval --n 10 --schedule 1,2,4,6,8,10 --p 128 --max-steps 180
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

try:
    from .m2_adapter import load_frozen_base
    from .cartridge import concat_cartridges
    from .cartridge_train import train_cartridge, direct_cloze_delivers
    from .continual_eval import load_counterfact_facts, GENERAL_QA
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import load_frozen_base                            # noqa: E402
    from cartridge import concat_cartridges                            # noqa: E402
    from cartridge_train import train_cartridge, direct_cloze_delivers  # noqa: E402
    from continual_eval import load_counterfact_facts, GENERAL_QA      # noqa: E402


# ------------------------------------------------------------------------------------------------------
# Router: non-parametric retrieval over the frozen base's own mean-pooled hidden states (no training).
# ------------------------------------------------------------------------------------------------------
@torch.no_grad()
def embed_text(base, tok, text):
    """Mean-pooled LAST-hidden-state of `text` under the FROZEN base (NO cartridge attached). A cheap,
    model-internal sentence embedding. Returns the UN-normalized [H] fp32 vector (caller centers/normalizes).
    Mean-pooled LM states are strongly ANISOTROPIC (a big shared component), so raw cosine barely separates
    unrelated text; centering (subtracting a background mean) before normalizing restores discrimination."""
    device = next(base.parameters()).device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + tok(text, add_special_tokens=False).input_ids
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = base(input_ids=inp, output_hidden_states=True)
    h = out.hidden_states[-1][0].float()          # [T, H]
    return h.mean(dim=0)                          # mean-pool over tokens (un-normalized)


def _unit(v):
    return F.normalize(v, dim=-1)


def _center_unit(x, mean):
    """Remove the anisotropy component `mean`, then L2-normalize. x: [...,H], mean: [H]."""
    return _unit(x - mean)


def route_topk(query_vec, keys, k=1):
    """keys: [N, H] unit rows. Returns (indices[k], sims[k]) by descending cosine similarity."""
    sims = keys @ query_vec                       # cosine (both unit-norm)
    k = min(k, keys.shape[0])
    vals, idx = torch.topk(sims, k)
    return idx.tolist(), vals.tolist(), sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pat/code/memory-organ/data/counterfact.json")
    ap.add_argument("--n", type=int, default=10, help="number of facts / cartridges (bank size)")
    ap.add_argument("--schedule", default="1,2,4,6,8,10", help="N values for accuracy-vs-N")
    ap.add_argument("--p", type=int, default=128, help="per-cartridge prefix length")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=180, help="converged budget ~150-180 steps/fact")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--samples-per-seed", type=int, default=2)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7", help="sim thresholds for locality gating")
    ap.add_argument("--router-only", action="store_true",
                    help="cheap validation: build router keys + report retrieval accuracy, SKIP training/eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    schedule = [int(x) for x in args.schedule.split(",") if int(x) <= args.n]
    if args.n not in schedule:
        schedule.append(args.n)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"[moc] loading base {os.environ.get('CAM_BASE_MODEL', '<default>')} ...", flush=True)
    base, tok = load_frozen_base()
    facts = load_counterfact_facts(args.data, args.n, tok)
    print(f"[moc] {len(facts)} facts:", flush=True)
    for i, f in enumerate(facts):
        print(f"  [{i}] subj={f['subject']!r} {f['prompt']!r} -> {f['new']!r} "
              f"(true {f['true']!r}, rel {f['rel']})", flush=True)

    # -------- 1) build the router keys + query embeddings (frozen base, no training) --------
    print("\n[moc] === building router keys/queries (mean-pooled base hidden states) ===", flush=True)
    subj_raw = torch.stack([embed_text(base, tok, f["subject"]) for f in facts])        # [N,H]
    prompt_raw = torch.stack([embed_text(base, tok, f["prompt"]) for f in facts])       # [N,H]
    fq_raw = prompt_raw.clone()                    # query = the cloze prompt (same text as prompt-key)
    qa_raw = torch.stack([embed_text(base, tok, q) for q, _g in GENERAL_QA])            # [Q,H]
    # anisotropy-removal mean: estimated ONLY from what the router legitimately owns (its cartridge keys),
    # so it is causal (no peeking at test queries). Subtracting it de-cones the embeddings.
    bg_mean = subj_raw.mean(dim=0)

    def _report_retrieval(tag, subj_keys, prompt_keys, fact_queries, qa_queries):
        """Print retrieval acc (key=subject vs key=prompt) + similarity separation for one embedding variant.
        Returns (subject-keys, fact-queries, qa-queries) all unit-norm, for downstream use."""
        print(f"[moc]   -- variant: {tag} --", flush=True)
        for name, keys in (("subject", subj_keys), ("prompt", prompt_keys)):
            top1 = top2 = 0
            marks = []
            for j in range(len(facts)):
                idx, _v, _s = route_topk(fact_queries[j], keys, k=2)
                hit1 = idx[0] == j
                hit2 = j in idx[:2]
                top1 += hit1
                top2 += hit2
                marks.append("Y" if hit1 else ("2" if hit2 else "."))
            print(f"[moc]      key={name:<8s} top-1={top1}/{len(facts)}={top1/len(facts):.3f}  "
                  f"top-2={top2}/{len(facts)}={top2/len(facts):.3f}  {''.join(marks)}", flush=True)
        self_sims = torch.stack([subj_keys[j] @ fact_queries[j] for j in range(len(facts))])
        off = subj_keys @ fact_queries.T
        off_max = torch.stack([torch.cat([off[:j, j], off[j+1:, j]]).max() if len(facts) > 1
                               else torch.tensor(-1.0) for j in range(len(facts))])
        qa_max = (subj_keys @ qa_queries.T).max(dim=0).values
        print(f"[moc]      sim(query,own subj-key):  min={self_sims.min():.3f} mean={self_sims.mean():.3f} "
              f"max={self_sims.max():.3f}", flush=True)
        print(f"[moc]      sim(query,other subj-key):mean-of-max={off_max.mean():.3f} "
              f"max={off_max.max():.3f}", flush=True)
        print(f"[moc]      sim(general-QA,any subj-key) [want LOW]: min={qa_max.min():.3f} "
              f"mean={qa_max.mean():.3f} max={qa_max.max():.3f}", flush=True)

    # -------- 2) router retrieval accuracy (attribution: does it pick the right cartridge?) --------
    print("\n[moc] === (a) router retrieval accuracy ===", flush=True)
    _report_retrieval("RAW normalized", _unit(subj_raw), _unit(prompt_raw), _unit(fq_raw), _unit(qa_raw))
    _report_retrieval("CENTERED (anisotropy removed)", _center_unit(subj_raw, bg_mean),
                      _center_unit(prompt_raw, bg_mean), _center_unit(fq_raw, bg_mean),
                      _center_unit(qa_raw, bg_mean))
    # Downstream router uses CENTERED subject keys (the honest per-fact index + de-coned for thresholding).
    router_keys = _center_unit(subj_raw, bg_mean)
    fact_queries = _center_unit(fq_raw, bg_mean)
    qa_queries = _center_unit(qa_raw, bg_mean)

    if args.router_only:
        print("\n[moc] --router-only: retrieval validated; skipping cartridge training + task eval.",
              flush=True)
        return

    # -------- 3) train N cartridges INDEPENDENTLY (the converged p=128 budget) --------
    carts, solo = [], []
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
        solo_hit, solo_gen = direct_cloze_delivers(base, tok, cart, facts[i]["prompt"], facts[i]["new"])
        solo.append(solo_hit)
        print(f"  [train {i}] KL={info['final_kl']:.3f} steps={info['steps']} {dt:.0f}s | "
              f"solo deliver={'Y' if solo_hit else '.'} gen={solo_gen.strip()[:40]!r}", flush=True)
    solo_acc = sum(solo) / len(solo)
    print(f"[moc] solo per-fact delivery (each cartridge alone): "
          f"{''.join('Y' if h else '.' for h in solo)} = {solo_acc:.3f}", flush=True)

    # -------- 4) accuracy-vs-N: ROUTING (top-1) vs CONCAT (all N) --------
    print("\n[moc] === (b) accuracy-vs-N: routing vs naive concat ===", flush=True)
    print(f"[moc]   {'N':>3s} | {'route acc':>9s} | {'concat acc':>10s} | {'router@N':>8s} | route-marks", flush=True)
    rows_accN = []
    for N in schedule:
        keysN = router_keys[:N]
        # routing: for each fact j<N, route top-1 among the N keys, attach ONLY that cartridge, deliver
        route_hits, router_ok = [], 0
        for j in range(N):
            idx, _v, _s = route_topk(fact_queries[j], keysN, k=1)
            chosen = idx[0]
            router_ok += (chosen == j)
            hit, _g = direct_cloze_delivers(base, tok, carts[chosen], facts[j]["prompt"], facts[j]["new"])
            route_hits.append(hit)
        route_acc = sum(route_hits) / N
        # concat: all N as one always-on prefix, deliver each fact with all N loaded
        composed = concat_cartridges(carts[:N])
        concat_hits = [direct_cloze_delivers(base, tok, composed, facts[j]["prompt"], facts[j]["new"])[0]
                       for j in range(N)]
        concat_acc = sum(concat_hits) / N
        rows_accN.append((N, route_acc, concat_acc, router_ok / N))
        marks = "".join("Y" if h else "." for h in route_hits)
        print(f"[moc]   {N:3d} | {route_acc:9.3f} | {concat_acc:10.3f} | {router_ok/N:8.3f} | {marks}",
              flush=True)

    # -------- 5) locality / preservation on unrelated general-QA --------
    print("\n[moc] === (c) locality: general-QA preservation, routing vs concat ===", flush=True)
    # base (no memory)
    base_hits = [any(direct_cloze_delivers(base, tok, None, q, g)[0] for g in golds)
                 for q, golds in GENERAL_QA]
    base_ok = sum(base_hits)
    # concat all N (always-on)
    composed_all = concat_cartridges(carts)
    concat_hits = [any(direct_cloze_delivers(base, tok, composed_all, q, g)[0] for g in golds)
                   for q, golds in GENERAL_QA]
    # routing top-1-anyway (always attach the best cartridge, even for unrelated queries)
    route_any_hits = []
    for qi, (q, golds) in enumerate(GENERAL_QA):
        idx, _v, _s = route_topk(qa_queries[qi], router_keys, k=1)
        c = carts[idx[0]]
        route_any_hits.append(any(direct_cloze_delivers(base, tok, c, q, g)[0] for g in golds))
    # routing thresholded: attach only if best sim > thr, else NO cartridge (pure base)
    thr_hits = {}
    for thr in thresholds:
        hs = []
        for qi, (q, golds) in enumerate(GENERAL_QA):
            idx, vals, _s = route_topk(qa_queries[qi], router_keys, k=1)
            c = carts[idx[0]] if vals[0] > thr else None
            hs.append(any(direct_cloze_delivers(base, tok, c, q, g)[0] for g in golds))
        thr_hits[thr] = hs

    nqa = len(GENERAL_QA)
    def _pres(hits):    # preserved fraction OF base-correct
        return sum(h and b for h, b in zip(hits, base_hits)) / base_ok if base_ok else float("nan")
    print(f"[moc]   base QA acc              = {base_ok}/{nqa} = {base_ok/nqa:.3f}", flush=True)
    print(f"[moc]   concat (all N, always-on): acc={sum(concat_hits)}/{nqa}={sum(concat_hits)/nqa:.3f}  "
          f"preserved={_pres(concat_hits):.3f}", flush=True)
    print(f"[moc]   routing top-1-ANYWAY:      acc={sum(route_any_hits)}/{nqa}={sum(route_any_hits)/nqa:.3f}  "
          f"preserved={_pres(route_any_hits):.3f}", flush=True)
    for thr in thresholds:
        hs = thr_hits[thr]
        n_attached = sum(1 for qi in range(nqa)
                         if route_topk(qa_queries[qi], router_keys, 1)[1][0] > thr)
        print(f"[moc]   routing thr>{thr:.2f}:          acc={sum(hs)}/{nqa}={sum(hs)/nqa:.3f}  "
              f"preserved={_pres(hs):.3f}  (mem attached on {n_attached}/{nqa} QA queries)", flush=True)

    # -------- SUMMARY (the money table) --------
    print("\n[moc] " + "=" * 72, flush=True)
    print(f"[moc] SUMMARY (n={len(facts)}, p={args.p}, steps<= {args.max_steps})", flush=True)
    print(f"[moc]   solo per-fact delivery: {solo_acc:.3f}", flush=True)
    print(f"[moc]   accuracy-vs-N  (N: route / concat / router@N):", flush=True)
    for (N, ra, ca, ro) in rows_accN:
        print(f"[moc]      N={N:2d}:  route={ra:.3f}  concat={ca:.3f}  router-retrieval={ro:.3f}", flush=True)
    print(f"[moc]   locality @N={len(facts)} (preserved of base-correct):", flush=True)
    print(f"[moc]      concat(always-on)={_pres(concat_hits):.3f}  "
          f"route(top1-anyway)={_pres(route_any_hits):.3f}  "
          f"route(best thr)={max(_pres(thr_hits[t]) for t in thresholds):.3f}", flush=True)
    print(f"[moc]   cost: train {t_train:.0f}s total ({t_train/max(1,len(facts)):.0f}s/fact); "
          f"router = non-parametric (0 train, {base.config.get_text_config().hidden_size}-d embeddings)",
          flush=True)
    print("[moc] " + "=" * 72, flush=True)


if __name__ == "__main__":
    main()
