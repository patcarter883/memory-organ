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

# flat package: make sibling modules importable whether run as `python -m cam.X` or `python cam/X.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from m2_adapter import MODEL, DEV                                          # noqa: E402
from recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,  # noqa: E402
                            single_token_ids, DocBuilder)
from recall_mag import (memory_bank, load_ckpt, EVAL_BATCH_CAP,  # noqa: E402
                        _kc, _answer_logits, _seq_ce, _seq_metrics, _nll_bits)        # noqa: E402
from translator import TranslatedInjector, save_translator                # noqa: E402

LN2 = math.log(2.0)
# 2nd base, overridable via --base2. Default = the v1 same-family base (Qwen3-0.6B, d=1024).
# Cross-family falsifier base = unsloth/Llama-3.2-3B (d=3072, Llama tiktoken vocab, bos=128000,
# plain LlamaForCausalLM) — a genuinely DIFFERENT tokenizer + architecture, the decisive test that
# the translator isn't exploiting Qwen-family vocab/embedding similarity.
MODEL2 = "Qwen/Qwen3-0.6B"


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
    """header (FORMAT only, no bindings) + query tokens -> base-2 inputs_embeds (base-2 vocab).
    Multi-token teacher-forcing: end=apos2+Kc-1 also includes the first Kc-1 gold answer tokens so the
    last Kc base-2 logit positions predict the full answer sequence. end=None = single-token."""
    if end is None:
        end = apos2
    hlen = len(builder2.bos) + len(builder2.header)
    ctx_ids = torch.cat([ids2[:, len(builder2.bos):hlen], ids2[:, builder2.qa_start:end]], dim=1)
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


def main():
    global MODEL2
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-ckpt", type=str, required=True, dest="load_ckpt")
    ap.add_argument("--base2", type=str, default=MODEL2,
                    help="2nd (frozen) base; default Qwen3-0.6B same-family, "
                         "or unsloth/Llama-3.2-3B for the cross-family falsifier")
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

    # base-1 ONLY supplies its frozen embedding table to rebuild the adapter; we never run base-1 fwd.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok1 = AutoTokenizer.from_pretrained(MODEL)
    m1 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(DEV).eval()
    for p in m1.parameters():
        p.requires_grad_(False)
    embed_weight = m1.get_input_embeddings().weight.detach().float().clone()
    n1 = m1.config.get_text_config().num_hidden_layers   # base-1 depth (for the proportional tap map)

    # determine the multi-token answer length K up-front (builder1 needs it BEFORE load_ckpt): CLI
    # override else inherit from the checkpoint metadata (cheap metadata peek; embed/unembed are not
    # in the ckpt). cargo_words for base-1; base-2's pool is intersected below for index alignment.
    K = args.cargo_tokens
    _meta = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    if K == 0:
        K = int(_meta.get("cargo_tokens", 1))
    phrasing = _meta.get("phrasing", "dict")   # rebuild the SAME doc format the memory was bound on
    del _meta
    # base-1 DocBuilder must exist BEFORE load_ckpt: a pk-store ckpt needs the builder (bind-block
    # positions) at construction. Bolt ckpts ignore it. (builder2 is built after base-2 loads, below.)
    # natural phrasing puts the object mid-sentence (space-prefixed); dict is line-initial (no-space).
    cargo_prefix = " " if phrasing in ("natural", "varied") else ""
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

    print(f"[v1] base-1={MODEL} (d_base1={frozen_tap.H}, tap L={tap_layer}) | "
          f"base-2={MODEL2} (d_base2={H2}, n_layers={n2}, tap L={tap_layer2}) | "
          f"K={ck['k']} mem_dim={ck['mem_dim']} | chance {1/args.M:.3f}", flush=True)

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
