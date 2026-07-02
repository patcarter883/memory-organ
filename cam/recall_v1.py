"""CAM v1 — the base-agnostic translator FALSIFIER (CAM_DESIGN §2.2, §6 v1).

V0 proved: a zero-init MAG tap delivers the validated DeepMemory binding through a FROZEN base
(Qwen3.5-4B). v1 asks the product question: does ONE frozen memory serve a SECOND base (different
d_base) through only a TINY learned translator?

Pipeline:
  1. Load the frozen v0 memory checkpoint (BoltAdapter + a passing GatedMemoryTap @ L=8 or L=24).
     The bank it produces ([B,K,mem_dim]) is base-AGNOSTIC — DeepMemory's own mem_dim retrieval, built
     from the adapter's OWN frozen base-1 embedding, never base-2's space. Same memory, any base.
  2. Load base-2 (frozen, different hidden dim) via load_frozen_base2().
  3. Fit a TINY affine translator (A: d_base2->d_base1, B: d_base1->d_base2 + zero-init gamma2) that
     stitches base-2's residual stream into the frozen tap and back. Train ONLY the translator by
     LM-loss through frozen base-2 on the same recall task. base-2 + memory + tap all frozen.
  4. Eval = same memory/no_memory/ceiling on base-2. PASS = memory >> no_memory with only a tiny fit.

Run:
  python -m cam.recall_v1 --load-ckpt ckpt/cam_v0_L24.pt --steps 3000
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# flat package: sibling imports resolve relatively when imported as cam.X (`python -m cam.X`,
# `import cam.X`) and fall back to a path-hacked absolute import when run as a file (`python cam/X.py`).
try:
    from .m2_adapter import MODEL, DEV
    from .recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,
                                 single_token_ids, DocBuilder, derange_capitals)
    from .recall_mag import (memory_bank, load_ckpt, EVAL_BATCH_CAP,
                             _kc, _answer_logits, _seq_ce, _seq_metrics, _nll_bits, probe_and_filter)
    from .translator import TranslatedInjector, save_translator
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import MODEL, DEV                                      # noqa: E402
    from recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,  # noqa: E402
                                single_token_ids, DocBuilder, derange_capitals)
    from recall_mag import (memory_bank, load_ckpt, EVAL_BATCH_CAP,  # noqa: E402
                            _kc, _answer_logits, _seq_ce, _seq_metrics, _nll_bits, probe_and_filter)
    from translator import TranslatedInjector, save_translator            # noqa: E402

LN2 = math.log(2.0)
# 2nd base, overridable via --base2. Default = the v1 same-family base (Qwen3-0.6B, d=1024).
# Cross-family falsifier base = unsloth/Llama-3.2-3B (d=3072, Llama tiktoken vocab, bos=128000,
# plain LlamaForCausalLM) — a genuinely DIFFERENT tokenizer + architecture, the decisive test that
# the translator isn't exploiting Qwen-family vocab/embedding similarity.
MODEL2 = "Qwen/Qwen3-0.6B"


def _cf_facts_single_token(tok, cf_facts):
    """Filter saved (country, capital) WORD pairs to those single-token (space-prefixed) under `tok`.
    Returns [(country, capital, country_tid, capital_tid)] in the SAME order as cf_facts."""
    out = []
    for country, capital in cf_facts:
        c = tok(" " + country, add_special_tokens=False).input_ids
        k = tok(" " + capital, add_special_tokens=False).input_ids
        if len(c) == 1 and len(k) == 1:
            out.append((country, capital, c[0], k[0]))
    return out


def load_base(model_id):
    """Load + freeze any HF causal LM (pure torch). Loader selection (CausalLM vs
    ImageTextToText) is isolated from the device move / grad-ckpt setup so a real GPU error
    (OOM, HIP) surfaces instead of being masked as a bogus 'unrecognized config' fallback."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    m, load_err = None, None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = loader.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            break
        except (ValueError, KeyError) as e:  # config not recognized by THIS AutoModel -> try the next
            load_err = e
    if m is None:
        raise load_err
    m = m.to(DEV).eval()                                    # device move OUTSIDE the loader fallback
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m, tok


def _leakfree_ctx(base2, builder2, ids2, apos2, end=None):
    """BOS + header (FORMAT only, no bindings) + query tokens -> base-2 inputs_embeds (base-2 vocab).
    Multi-token teacher-forcing: end=apos2+Kc-1 also includes the first Kc-1 gold answer tokens so the
    last Kc base-2 logit positions predict the full answer sequence. end=None = single-token.

    KEEP THE BOS. A prior version stripped it (slice started at len(bos)); base-2's like Gemma are
    highly BOS-sensitive — without the leading <bos> the frozen base's PARAMETRIC next-token recall
    collapses, so the no_memory PRIOR-acc(base-2) validity gate read ~0.000 even though the base-2
    PROBE (which includes bos, exactly as it was trained) recalls the same priors at ~1.000. That was
    a context-format artifact, not a base-2 knowledge gap. Include the bos so the leak-free eval
    context matches the probe's eliciting format and the validity gate is measured honestly."""
    if end is None:
        end = apos2
    hlen = len(builder2.bos) + len(builder2.header)
    ctx_ids = torch.cat([ids2[:, :hlen], ids2[:, builder2.qa_start:end]], dim=1)   # bos + header + query
    return base2.get_input_embeddings()(ctx_ids)


def train_translator(base2, adapter, injector, builder1, builder2, rng, args):
    injector.attach().train()
    Kc = _kc(builder2)                                                   # answer length (base-2 vocab)
    opt = torch.optim.AdamW(injector.A_params(), lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        # SAME random recall instance for base-1 (bank) and base-2 (context): build with shared rng
        # state so the bindings/query match; each base tokenizes with its own DocBuilder.
        seed = int(rng.integers(0, 2**31 - 1))
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed)
        ids1, ans1, apos1 = builder1.build(r1, args.batch, local=False)
        ids2, ans2, apos2 = builder2.build(r2, args.batch, local=False)
        ids1, ids2, ans2 = ids1.to(DEV), ids2.to(DEV), ans2.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids1, args.seg_len, builder1.qa_start, apos1, carry=True)
        injector.set_bank(bank)                                          # memory frozen -> bank detached
        ctx_emb = _leakfree_ctx(base2, builder2, ids2, apos2, end=apos2 + Kc - 1)
        logits = _answer_logits(base2, ctx_emb, Kc)                      # [B,V] or [B,Kc,V]
        loss = _seq_ce(logits, ans2)
        if not torch.isfinite(loss):                                     # NaN/Inf guard: skip the step
            print(f"[v1] step {step:4d} NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(injector.A_params(), 1.0)
        if not torch.isfinite(gn):                                       # NaN grad guard
            print(f"[v1] step {step:4d} NON-FINITE grad -> skip", flush=True)
            opt.zero_grad(); continue
        opt.step()
        injector.tt.clamp_gate(-6.0, 6.0)                               # gamma guard against divergence
        if step % 200 == 0 or step == args.steps - 1:
            em, pt = _seq_metrics(logits, ans2)
            print(f"[v1] step {step:4d} loss {loss.item():.3f} exact {em.mean().item():.3f} "
                  f"per_tok {pt.mean().item():.3f} gate {injector.gate_stat():.4f} |g|grad {float(gn):.2f}",
                  flush=True)
    injector.set_bank(None)


@torch.no_grad()
def eval_v1(base2, adapter, injector, builder1, builder2, rng, args, n=512):
    """Transfer eval on base-2. Multi-token: TEACHER-FORCED exact-match (all Kc tokens) + per-token acc;
    single-token byte-identical. Returns {cond: (nll_bits, exact_match, per_token)}."""
    base_embed = base2.get_input_embeddings()
    Kc = _kc(builder2)
    res = {c: [[], 0.0, 0.0] for c in ("local_control", "memory", "no_memory")}
    injector.eval()
    seen = 0
    while seen < n:
        # eval batch shrinks as M (and Kc) grow: large docs OOM the two-base eval forward on 16GB.
        # Conservative cap for multi-token (matches eval_generative_mag). Batch doesn't affect accuracy.
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M * Kc)))
        cur = min(eb, n - seen)
        seed = int(rng.integers(0, 2**31 - 1))
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed)
        ids1, ans1, apos1 = builder1.build(r1, cur, local=False)
        ids2, ans2, apos2 = builder2.build(r2, cur, local=False)
        ids1, ids2, ans2 = ids1.to(DEV), ids2.to(DEV), ans2.to(DEV)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        injector.set_bank(None)                                          # tap OFF -> ceiling on base-2
        lc = _answer_logits(base2, base_embed(ids2[:, :apos2 + Kc - 1]), Kc)
        ctx_emb = _leakfree_ctx(base2, builder2, ids2, apos2, end=apos2 + Kc - 1)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids1, args.seg_len, builder1.qa_start, apos1, carry=carry)
            injector.set_bank(bank)
            lg = _answer_logits(base2, ctx_emb, Kc)
            res[cond][0].extend(_nll_bits(lg, ans2))
            em, pt = _seq_metrics(lg, ans2)
            res[cond][1] += em.sum().item(); res[cond][2] += pt.sum().item()
        injector.set_bank(None)
        res["local_control"][0].extend(_nll_bits(lc, ans2))
        em, pt = _seq_metrics(lc, ans2)
        res["local_control"][1] += em.sum().item(); res["local_control"][2] += pt.sum().item()
        seen += cur
        print(f"[v1] eval progress {seen}/{n}", flush=True)   # heartbeat vs the watchdog STALL guard
    return {c: (float(np.mean(res[c][0])), res[c][1] / seen, res[c][2] / seen) for c in res}


def verdict(gen, chance, xlator="affine"):
    lc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print(f"\n[v1] === one frozen memory, SECOND base ({MODEL2}) via {xlator} translator "
          f"(acc=EXACT-MATCH; per_tok shown) ===", flush=True)
    print(f"{'condition':>14} {'NLL(bits)':>11} {'exact':>7} {'per_tok':>8}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>11.3f} {gen[c][1]:>7.3f} {gen[c][2]:>8.3f}", flush=True)
    print(f"[v1] memory exact {m_acc:.3f} (per_tok {gen['memory'][2]:.3f}) / no_memory {nm_acc:.3f} "
          f"(per_tok {gen['no_memory'][2]:.3f}) / ceiling {lc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits (chance {chance:.3f})", flush=True)
    if m_acc > nm_acc + 0.15 and m_acc > 0.5:
        v = "TRANSLATOR WORKS — one memory serves TWO bases. v1 PASSES (Modular Memory Organ proven)."
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        v = "PARTIAL — affine translator helps but doesn't fully transfer; try a wider/nonlinear translator."
    else:
        v = "WALL — affine residual-stitch did not transfer the memory to base-2; escalate translator."
    print(f"[v1] => {v}\n" + "=" * 64, flush=True)
    return m_acc, nm_acc


@torch.no_grad()
def eval_v1_counterfactual(base2, adapter, injector, builder1, builder2, rng, args, n=512):
    """COUNTERFACTUAL transfer eval on base-2. Same 4 knowledge-editing metrics as recall_mag, but the
    memory bank comes from base-1 (builder1) and the query context + answers are base-2's (builder2), via
    the translator. The queried fact is shared (same rng draws), so the base-1 bank retrieves the memory
    for the SAME country the base-2 context queries. mem/no_mem scored against BOTH the counterfactual
    capital and base-2's true prior at the query position."""
    base_embed = base2.get_input_embeddings()
    res = {"memory_cf": 0.0, "no_memory_cf": 0.0, "memory_prior": 0.0, "no_memory_prior": 0.0,
           "ceiling_cf": 0.0, "nll_mem_cf": [], "nll_nomem_prior": []}
    injector.eval()
    seen = 0
    while seen < n:
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M)))
        cur = min(eb, n - seen)
        seed = int(rng.integers(0, 2**31 - 1))
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed)
        ids1, ans1_cf, ans1_prior, apos1 = builder1.build_cf(r1, cur, local=False)
        ids2, ans2_cf, ans2_prior, apos2 = builder2.build_cf(r2, cur, local=False)
        ids1, ids2 = ids1.to(DEV), ids2.to(DEV)
        ans2_cf, ans2_prior = ans2_cf.to(DEV), ans2_prior.to(DEV)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        injector.set_bank(None)
        lc = _answer_logits(base2, base_embed(ids2[:, :apos2]), 1)     # in-context ceiling on base-2
        res["ceiling_cf"] += (lc.argmax(-1) == ans2_cf).sum().item()
        ctx_emb = _leakfree_ctx(base2, builder2, ids2, apos2)          # "...The capital of <Country> is"
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids1, args.seg_len, builder1.qa_start, apos1, carry=carry)
            injector.set_bank(bank)
            lg = _answer_logits(base2, ctx_emb, 1)
            res[f"{cond}_cf"] += (lg.argmax(-1) == ans2_cf).sum().item()
            res[f"{cond}_prior"] += (lg.argmax(-1) == ans2_prior).sum().item()
            if carry:
                res["nll_mem_cf"].extend(_nll_bits(lg, ans2_cf))
            else:
                res["nll_nomem_prior"].extend(_nll_bits(lg, ans2_prior))
        injector.set_bank(None)
        seen += cur
        print(f"[v1][cf] eval progress {seen}/{n}", flush=True)
    return {
        "memory_cf_acc": res["memory_cf"] / seen,
        "no_memory_cf_acc": res["no_memory_cf"] / seen,
        "memory_prior_acc": res["memory_prior"] / seen,
        "no_memory_prior_acc": res["no_memory_prior"] / seen,
        "ceiling_cf_acc": res["ceiling_cf"] / seen,
        "nll_mem_cf": float(np.mean(res["nll_mem_cf"])),
        "nll_nomem_prior": float(np.mean(res["nll_nomem_prior"])),
    }


def verdict_v1_counterfactual(cf, chance, valid_thresh=0.6):
    m_cf = cf["memory_cf_acc"]; nm_cf = cf["no_memory_cf_acc"]
    m_pr = cf["memory_prior_acc"]; nm_pr = cf["no_memory_prior_acc"]
    print(f"\n[v1] === COUNTERFACTUAL transfer to base-2 ({MODEL2}) — one memory, knowledge edit ===",
          flush=True)
    print(f"  (a) mem-on   counterfactual-acc : {m_cf:.3f}", flush=True)
    print(f"  (b) no_mem   counterfactual-acc : {nm_cf:.3f}", flush=True)
    print(f"  (c) no_mem   PRIOR-acc (base-2) : {nm_pr:.3f}   (base-2 must hold the priors for validity)",
          flush=True)
    print(f"  (d) mem-on   PRIOR-acc          : {m_pr:.3f}", flush=True)
    print(f"  ceiling (in-context cf, tap off): {cf['ceiling_cf_acc']:.3f} | chance {chance:.3f}",
          flush=True)
    print(f"  NLL bits: mem cf {cf['nll_mem_cf']:.3f} | no_mem prior {cf['nll_nomem_prior']:.3f}",
          flush=True)
    valid = nm_pr >= valid_thresh
    if not valid:
        v = (f"INVALID on base-2 — no_mem prior-acc {nm_pr:.3f} < {valid_thresh:.2f}: base-2 does not "
             f"hold these priors, so the override claim is meaningless on base-2.")
    elif m_cf > nm_cf + 0.15 and m_cf > 0.5:
        v = "TRANSFER + EDIT WORKS — one memory edits a SECOND base's knowledge through the translator."
    elif m_cf > nm_cf + 0.10:
        v = "PARTIAL transfer — some override on base-2; escalate the translator."
    else:
        v = "NO transfer-override — the translator did not deliver the edit to base-2; escalate."
    print(f"[v1] GATE: {'VALID' if valid else 'INVALID'} | => {v}\n" + "=" * 64, flush=True)
    return m_cf, nm_cf, nm_pr


def main():
    global MODEL2
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-ckpt", type=str, required=True, dest="load_ckpt")
    ap.add_argument("--base2", type=str, default=MODEL2,
                    help="2nd (frozen) base; default Qwen3-0.6B same-family, "
                         "or unsloth/Llama-3.2-3B for the cross-family falsifier")
    ap.add_argument("--base1", type=str, default="",
                    help="donor (frozen base-1) HF model id; default = the donor recorded in the "
                         "checkpoint (pre-flag checkpoints fall back to the historical default). "
                         "Must match the base the memory was bound on — its embedding table "
                         "rebuilds the adapter.")
    ap.add_argument("--save-translator", type=str, default="", dest="save_translator",
                    help="save the fitted translator card (A,B,gamma2 + meta) to this path")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--cargo-tokens", type=int, default=0, dest="cargo_tokens",
                    help="K multi-token answer length; 0 = inherit from the loaded checkpoint")
    ap.add_argument("--cf-probe-batch", type=int, default=16, dest="cf_probe_batch",
                    help="batch size for the BASE-2 counterfactual probe/filter (the cross-base "
                         "validity gate: keeps only facts base-2 demonstrably knows)")
    ap.add_argument("--xlator", type=str, default="affine",
                    choices=["affine", "perpos", "mlp", "perpos-mlp"],
                    help="translator variant: affine (shared linear, byte-preserved baseline); "
                         "perpos (a separate affine map per answer position — mirrors the disjoint "
                         "store fix); mlp (1-hidden-GELU non-linear, shared); perpos-mlp (both).")
    ap.add_argument("--mlp-mult", type=float, default=2.0, dest="mlp_mult",
                    help="hidden-width multiplier for the mlp/perpos-mlp translator variants")
    args = ap.parse_args()
    MODEL2 = args.base2

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # checkpoint metadata peek FIRST (cheap; embed/unembed are not in the ckpt): the multi-token
    # answer length K (builder1 needs it BEFORE load_ckpt), the doc format, and the DONOR id the
    # memory was bound on (--base1 overrides; pre-flag checkpoints fall back to MODEL).
    K = args.cargo_tokens
    _meta = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    if K == 0:
        K = int(_meta.get("cargo_tokens", 1))
    phrasing = _meta.get("phrasing", "dict")   # rebuild the SAME doc format the memory was bound on
    cf_facts = _meta.get("cf_facts", None)     # counterfactual: filtered (country,capital) word pairs
    if args.base1 and _meta.get("base1") and args.base1 != _meta["base1"]:
        print(f"[v1] WARNING: --base1 {args.base1} != ckpt-recorded donor {_meta['base1']} — the "
              f"memory was bound on {_meta['base1']}; a different donor's embedding table makes the "
              f"transfer numbers meaningless (load_ckpt will reject a shape mismatch, but a "
              f"same-size donor swap it cannot catch).", flush=True)
    base1_id = args.base1 or _meta.get("base1") or MODEL
    del _meta

    # base-1 ONLY supplies its frozen embedding table to rebuild the adapter; we never run base-1 fwd.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok1 = AutoTokenizer.from_pretrained(base1_id)
    m1 = AutoModelForCausalLM.from_pretrained(base1_id, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(DEV).eval()
    for p in m1.parameters():
        p.requires_grad_(False)
    embed_weight = m1.get_input_embeddings().weight.detach().float().clone()
    n1 = m1.config.get_text_config().num_hidden_layers   # base-1 depth (for the proportional tap map)
    counterfactual = phrasing == "counterfactual"
    # base-1 DocBuilder must exist BEFORE load_ckpt: a pk-store ckpt needs the builder (bind-block
    # positions) at construction. Bolt ckpts ignore it. (builder2 is built after base-2 loads, below.)
    # natural phrasing puts the object mid-sentence (space-prefixed); dict is line-initial (no-space).
    cargo_prefix = " " if phrasing in ("natural", "varied", "counterfactual") else ""
    if counterfactual:
        # COUNTERFACTUAL transfer: rebuild the filtered fact table saved at bind time. builder1 is a
        # placeholder here (its FINAL aligned fact set is set below, after base-2's single-token subset is
        # known — both builders must draw the SAME facts in the SAME order under the shared rng). Provide a
        # temporary single-token subset so load_ckpt (pk-store) has a valid builder.
        assert cf_facts is not None, "counterfactual ckpt is missing cf_facts (re-bind with the CF build)"
        tmp1 = _cf_facts_single_token(tok1, cf_facts)
        assert len(tmp1) >= args.M, f"base-1 single-token CF facts ({len(tmp1)}) < M ({args.M})"
        builder1 = DocBuilder(tok1, None, None, args.M, args.seg_len, args.qa_seg,
                              phrasing="counterfactual", facts=tmp1)
        builder1.set_counterfactual([f[3] for f in tmp1])   # placeholder; re-set on the aligned set below
    else:
        names1 = single_token_ids(tok1, NAME_CANDIDATES); cargo1 = single_token_ids(tok1, CARGO_CANDIDATES, prefix=cargo_prefix)
        cargo_words1 = single_token_ids(tok1, MULTITOKEN_WORD_POOL) if K > 1 else None
        builder1 = DocBuilder(tok1, names1, cargo1, args.M, args.seg_len, args.qa_seg, phrasing=phrasing,
                              cargo_tokens=K, cargo_words=cargo_words1)

    # Build the v0 memory front-end (adapter + tap) against base-1's embedding, THEN free base-1's full
    # weights BEFORE loading base-2 — base-1 forward is never used at v1, and a large cross-family base-2
    # (e.g. Llama-3.2-3B ~6GB) won't fit on a 16GB card alongside base-1 (~8GB). load_ckpt only reads
    # base-1's hidden_size off the model, so m1 is sufficient here.
    adapter, injector_tap, tap_layer, ck = load_ckpt(args.load_ckpt, embed_weight, m1, DEV, builder=builder1)
    frozen_tap = injector_tap.taps[str(tap_layer)]
    # CRITICAL: MAGInjector.__init__ did `self.layers = decoder_layers(base)` — it holds references to
    # base-1's decoder modules, so the whole Qwen-4B (~8GB) stays alive unless we drop the injector too.
    # At v1 the injector_tap is never attached (we only need the standalone frozen tap); cut its base ref.
    injector_tap.layers = None
    del injector_tap
    del m1                              # free base-1 weights NOW (forward never used at v1)
    del embed_weight                   # the adapter holds its own fp32 self.embed copy now
    # the adapter's tied UNEMBED buffer (~1.5GB fp32) is ONLY used by the DIRECT bind loss, never at v1
    # (memory_bank reads adapter.embed -> in_proj -> mem; it never unembeds). Drop it so a large
    # cross-family base-2 fits on a 16GB card alongside the adapter's fp32 embed.
    if hasattr(adapter, "unembed"):
        adapter.unembed = None
    torch.cuda.empty_cache()

    # base-2: the SECOND base (different d_base), frozen — loaded AFTER base-1 is freed
    base2, tok2 = load_base(MODEL2)
    H2 = base2.config.get_text_config().hidden_size
    n2 = base2.config.get_text_config().num_hidden_layers
    # map the base-1 tap depth to a base-2 depth proportionally (cards carry the tap-layer as metadata)
    tap_layer2 = min(int(round(tap_layer / n1 * n2)), n2 - 1)

    print(f"[v1] base-1={base1_id} (d_base1={frozen_tap.H}, tap L={tap_layer}) | "
          f"base-2={MODEL2} (d_base2={H2}, n_layers={n2}, tap L={tap_layer2}) | "
          f"K={ck['k']} mem_dim={ck['mem_dim']} | chance {1/args.M:.3f}", flush=True)

    # COUNTERFACTUAL transfer: align the fact set across base-1 and base-2. Both builders must draw the
    # SAME facts in the SAME order under the shared rng, so intersect the saved fact table to those
    # single-token in BOTH tokenizers (in saved order), rebuild BOTH builders on that shared set, and
    # re-derange it deterministically (a fresh derangement over the shared indices — the saved cf_perm was
    # over base-1's kept indices, which shrink under intersection). The memory holds base-1's DeepMemory
    # bank keyed by base-1 token embeds; base-2 answers in base-2's vocab. The country->counterfactual
    # capital association is identical (same word), so the transfer test is well-posed.
    if counterfactual:
        f1 = _cf_facts_single_token(tok1, cf_facts)          # base-1 single-token facts (saved order)
        w1 = {c for (c, _cap, _ct, _kt) in f1}
        f2_all = _cf_facts_single_token(tok2, cf_facts)      # base-2 single-token facts
        w2 = {c for (c, _cap, _ct, _kt) in f2_all}
        shared = [(c, cap) for (c, cap) in cf_facts if c in w1 and c in w2]   # saved order, both-single
        assert len(shared) >= args.M, \
            f"cross-base shared CF facts ({len(shared)}) < M ({args.M}); pick a closer base-2 or lower M"
        # ---- BASE-2 PROBE -> FILTER (the cross-base validity fix) -----------------------------------
        # Mirror the base-1 probe that MADE the same-base test valid: BEFORE fitting/measuring, run a
        # NO-MEMORY forward of the FROZEN BASE-2 over the shared candidate facts in base-2's OWN best
        # eliciting format (the SAME "The following facts are given.\nThe capital of <Country> is"
        # context the eval reconstructs on base-2, tokenized in base-2's vocab), and KEEP ONLY the facts
        # base-2 demonstrably knows (predicts the TRUE capital). Without this, the shared set only proved
        # single-token-ness in base-2, NOT that base-2 HOLDS the prior — so no_mem prior-acc(base-2) came
        # out 0.000 and the override claim on base-2 was meaningless. probe_and_filter uses tok2's own
        # single-token capital ids, so it measures base-2's genuine parametric recall.
        shared_facts2 = _cf_facts_single_token(tok2, shared)  # (country,capital,ctid2,ktid2) in base-2 vocab
        kept2, prior_acc_b2_full = probe_and_filter(base2, tok2, shared_facts2,
                                                    batch=getattr(args, "cf_probe_batch", 16))
        print(f"[v1][cf] BASE-2 PROBE/FILTER: base-2 prior-acc over all {len(shared_facts2)} shared "
              f"candidates = {prior_acc_b2_full:.3f}", flush=True)
        print(f"[v1][cf] BASE-2 FILTERED-SET SIZE = {len(kept2)} facts base-2 demonstrably knows "
              f"(measure the override on THESE)", flush=True)
        assert len(kept2) >= args.M, \
            (f"base-2 filtered CF facts ({len(kept2)}) < M ({args.M}): base-2 knows too few of the "
             f"shared priors in ANY single-token format — the editing transfer cannot be validly "
             f"tested on this base-2 (report honestly / pick a closer base-2 / lower --M).")
        kept2_countries = {c for (c, _cap, _ct, _kt) in kept2}
        shared = [(c, cap) for (c, cap) in shared if c in kept2_countries]    # base-2-KNOWN subset only
        facts1 = _cf_facts_single_token(tok1, shared)
        facts2 = _cf_facts_single_token(tok2, shared)
        assert [f[0] for f in facts1] == [f[0] for f in facts2], "cross-base CF fact misalignment"
        # deterministic derangement over the SHARED set (same for both builders -> aligned memory VALUES)
        rd = np.random.default_rng(args.seed)
        perm = derange_capitals(rd, len(facts1))
        cf1 = [facts1[perm[i]][3] for i in range(len(facts1))]   # base-1 counterfactual capital tids
        cf2 = [facts2[perm[i]][3] for i in range(len(facts2))]   # base-2 counterfactual capital tids
        builder1 = DocBuilder(tok1, None, None, args.M, args.seg_len, args.qa_seg,
                              phrasing="counterfactual", facts=facts1)
        builder1.set_counterfactual(cf1)
        builder2 = DocBuilder(tok2, None, None, args.M, args.seg_len, args.qa_seg,
                              phrasing="counterfactual", facts=facts2)
        builder2.set_counterfactual(cf2)
        print(f"[v1] counterfactual transfer: shared single-token facts={len(shared)} "
              f"(aligned base-1/base-2, re-deranged)", flush=True)
        Kc_x = 1
        injector = TranslatedInjector(base2, frozen_tap, tap_layer2,
                                      xlator=args.xlator, kc=Kc_x, mlp_mult=args.mlp_mult).to(DEV)
        nparam = sum(p.numel() for p in injector.A_params())
        print(f"[v1] translator variant={args.xlator} (Kc={Kc_x}, mlp_mult={args.mlp_mult}) "
              f"trainable params: {nparam/1e6:.3f}M (base-2 d={H2} <-> tap d={frozen_tap.H})", flush=True)
        train_translator(base2, adapter, injector, builder1, builder2, rng, args)
        gen = eval_v1_counterfactual(base2, adapter, injector, builder1, builder2, rng, args)
        verdict_v1_counterfactual(gen, 1 / args.M)
        if args.save_translator:
            save_translator(args.save_translator, injector, {
                "base2": MODEL2, "tap_layer2": tap_layer2, "steps": args.steps, "lr": args.lr,
                "phrasing": "counterfactual",
                "memory_cf_acc": gen["memory_cf_acc"], "no_memory_prior_acc": gen["no_memory_prior_acc"],
            })
        injector.detach()
        print(f"\n[v1] base-2 ({MODEL2}) COUNTERFACTUAL SUMMARY: mem cf {gen['memory_cf_acc']:.3f} / "
              f"no_mem cf {gen['no_memory_cf_acc']:.3f} / no_mem PRIOR {gen['no_memory_prior_acc']:.3f}",
              flush=True)
        return

    # base-2 DocBuilder (builder1 was built above, before load_ckpt). Same single-token NAME/CARGO
    # words, each tokenized in its own base's vocab.
    names2 = single_token_ids(tok2, NAME_CANDIDATES); cargo2 = single_token_ids(tok2, CARGO_CANDIDATES, prefix=cargo_prefix)
    cargo_words2 = None
    if K > 1:
        # CRITICAL for cross-base alignment: builder1/builder2 draw cargo words by INDEX under a shared
        # rng, so both must hold the SAME word list in the SAME order. A word that is single-token in
        # base-1 but NOT in base-2 (or vice-versa) would misalign the draws. Intersect by word (in
        # MULTITOKEN_WORD_POOL order) and REBUILD builder1's pool to that intersection too.
        w1 = {w for (w, _t) in cargo_words1}
        w2set = {w for (w, _t) in single_token_ids(tok2, MULTITOKEN_WORD_POOL)}
        shared = [w for w in MULTITOKEN_WORD_POOL if w in w1 and w in w2set]
        cargo_words1 = single_token_ids(tok1, shared)
        cargo_words2 = single_token_ids(tok2, shared)
        assert [w for w, _ in cargo_words1] == [w for w, _ in cargo_words2] == shared, \
            "cross-base cargo-word pool misalignment"
        builder1.cargo_word_tids = [t for (_w, t) in cargo_words1]   # re-point builder1 to the shared set
        print(f"[v1] multi-token cargo: K={K} shared word pool={len(shared)} (aligned base-1/base-2)",
              flush=True)
    builder2 = DocBuilder(tok2, names2, cargo2, args.M, args.seg_len, args.qa_seg, phrasing=phrasing,
                          cargo_tokens=K, cargo_words=cargo_words2)

    Kc_x = _kc(builder2)                            # answer length for the per-position translator
    injector = TranslatedInjector(base2, frozen_tap, tap_layer2,
                                  xlator=args.xlator, kc=Kc_x, mlp_mult=args.mlp_mult).to(DEV)
    nparam = sum(p.numel() for p in injector.A_params())
    print(f"[v1] translator variant={args.xlator} (Kc={Kc_x}, mlp_mult={args.mlp_mult}) "
          f"trainable params: {nparam/1e6:.3f}M "
          f"(base-2 d={H2} <-> tap d={frozen_tap.H})", flush=True)

    train_translator(base2, adapter, injector, builder1, builder2, rng, args)
    gen = eval_v1(base2, adapter, injector, builder1, builder2, rng, args)
    m_acc, nm_acc = verdict(gen, 1 / args.M, xlator=args.xlator)
    if args.save_translator:
        save_translator(args.save_translator, injector, {
            "base2": MODEL2, "tap_layer2": tap_layer2, "steps": args.steps, "lr": args.lr,
            "memory_acc": m_acc, "no_memory_acc": nm_acc, "ceiling": gen["local_control"][1],
        })
    injector.detach()
    print(f"\n[v1] base-2 ({MODEL2}) SUMMARY: memory {m_acc:.3f} / no_memory {nm_acc:.3f} / "
          f"ceiling {gen['local_control'][1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
