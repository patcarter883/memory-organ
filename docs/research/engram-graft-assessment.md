# Engram-style n-gram tables as a frozen-base graft — an assessment

*A research note, 2026-11-09. This note has no new measurements. It checks the "graft an Engram table
instead of Titans" pitch against the papers it cites and against what this repo has already measured.
It closes with a falsifiable experiment ladder. Companion to
[frozen-base-titans-scorecard.md](frozen-base-titans-scorecard.md) and
[key-encoding-for-editing-memory.md](key-encoding-for-editing-memory.md).*

## TL;DR

- **The comparison is with a design this repo no longer uses.** memory-organ stopped relying on a Titans
  recurrent memory long ago. The Titans DeepMemory store hit a capacity wall at M≈8–12 (RESULTS §2) and was
  replaced by a product-key store. The *deployed* store is a whitened-GTE **pointer** id-bank (RESULTS §7,
  A/B/C bake-off). An Engram table has to beat **that** store, plus the conf-gated logit injection. Beating
  inner-loop Titans is not the test, because that problem is already gone here.
- **The cited papers are real, but none of them shows a frozen-backbone graft *for fact editing*.**
  *(Correction, 2026-11-09: an earlier version said no frozen-backbone Engram graft existed at all. That
  was wrong. The [relaunch proposal](engram-relaunch-proposal.md) review found frozen-base Engram-style
  adapters for **task gains**, e.g. Engram Adapter [2608.29327](https://arxiv.org/abs/2608.29327) on
  frozen Qwen3-4B/8B, whose injected residual is attenuated to ~0.08 % of the hidden-state norm. None of
  them edits facts. The per-fact-vector design in §3 also has direct prior art: NeuralDB
  [2507.18028](https://arxiv.org/abs/2507.18028) and SWEA [2401.17809](https://arxiv.org/abs/2401.17809).)*
  - STAPLE (zero-init tables) trains the backbone for ~200B tokens on a 1B model.
  - EngramEdit's headline numbers (99.5 % efficacy) come from a model whose 31.4B-param n-gram memory was
    **pretrained natively**.
  - "Mode A" in the pitch combines EngramEdit's numbers with a frozen graft that neither paper ran. **No
    paper we found demonstrates it.**
- **The likely failure point is not "residual shock".** Zero-init already solves residual shock, and this
  repo already does it: `GatedMemoryTap` with γ=0 is a byte-exact no-op. The likely failure is the one this
  repo has already measured. A frozen downstream circuit was never trained to *read* an extra additive
  channel at layer 2. Every frozen residual-site write so far has stalled at the ~0.7 single-site ceiling
  or below the in-context ceiling, and early layers (L3) *garble* (scorecard; key-encoding §0(C)). In
  EngramEdit's model the backbone learned to consume the memory channel during pretraining. In a graft it
  hasn't.
- **Some parts are still worth taking, as components inside the existing routed bank:**
  1. EngramEdit's **write rule**: solve for a target perturbation δ\* with the base frozen, then do a
     regularized least-squares allocation with a reuse/frequency penalty.
  2. **Exact hashed token-id keys.** They give a free *presence* bit, which is the store-presence/provenance
     gate already shown to rescue confident-wrong facts (+0.464). They also give O(1) CPU-side prefetch with
     no encoder forward.
  3. **Per-row isolated values.** These may relieve the one open limit, *efficacy-at-scale = shared readout
     capacity*, because there is no shared `out_proj`/`W_V` bottleneck.
  4. N-gram keys over the **already-emitted object tokens**, a natural per-position addressing scheme for
     the multi-token wall (#100).
- **Mode B (FFN de-memorization) is out of scope** under the project's own invariant that no base weights
  are trained. It is the "co-trained memory" bet the scorecard already split out as a separate,
  from-scratch-class effort.

## 1. Fact-check of the pitch

| Pitch claim | What the source actually shows | Verdict |
|---|---|---|
| STAPLE solved retrofitting with a zero-init W_V parallel residual branch at L2 | STAPLE ([2610.12599](https://arxiv.org/abs/2610.12599), Infini-AI-Lab) zero-inits **the embedding tables themselves**. The tables are **token-id indexed (not n-grams)** and sit **inside the FFN, added to the up-projection** in layers 1–12 (1B) or 1–16 (7B). It is not a parallel L2 branch. Several non-zero inits "did not improve long-term performance". | Mechanism misdescribed. The zero-init identity property is real. |
| Frozen-backbone contrastive warmup; only fusion params train | STAPLE freezes **only the FFN up-projections, and only during warmup**. All groups then train: ~100B warmup + ~100B mid-training tokens on OLMo-2 1B (~5 % of its 4T pretraining), and 30B + 70B on 7B. Gains are +1.2 avg (1B) and +0.9 avg (7B) over dense continued training. | **Not frozen-backbone.** This is Mode B, at a much larger budget than "days on a workstation". |
| STAPLE's Engram baseline shows grafted Engram works | That baseline is hashed 2/3-gram tables at L2 and L7, ~9.9B params, **trained jointly with the backbone**. It averages 44.77, against 45.59 for STAPLE and 44.39 for dense. | A trained-backbone result. Engram-at-L2 is only marginally above dense. |
| EngramEdit: 5,000+ sequential edits, 99.5 % efficacy | EngramEdit ([2610.10533](https://arxiv.org/abs/2610.10533)) reports 99.5 / 97.0 / 85.2 efficacy / generalization / specificity on CounterFact at **2,000** edits. Stability and general-capability tests run to 5,000 (>96 % of F1 retained). On ZsRE, specificity is **38.3 vs 41.6 pre-edit**. | Numbers are accurate, but the base is **LongCat-Flash-Lite with 31.4B natively pretrained n-gram params** ([2601.21204](https://arxiv.org/abs/2601.21204)). Not a graft. |
| δ\* computed at Layer 2 and written into the table | δ is added to the memory representation **before the first decoder block**, at the last subject token, using ROME/MEMIT-style NLL + regularizer + norm clamp. The write is a closed-form ridge solve `U* = (AᵀA+Λ)⁻¹AᵀB`, with per-n-gram penalty λⱼ = 0.05·wⱼ + 0.01 (shorter / more frequent n-grams are penalized more). Main results use **sum** fusion. The gated (Engram-style) case is an appendix derivation with **no results**. | Close, but at the input-embedding level, ungated, natively trained. |
| ~3× multi-hop CoT accuracy | True relative to the strongest baseline. In absolute terms, MQuAKE-CoT is **36.9 / 20.1 / 15.1 %** at 2/3/4 hops. | Consistent with our scorecard: injection does not integrate. |
| Frozen-backbone Engram grafting studies exist | None found. "Memory Grafting" ([2605.20948](https://arxiv.org/abs/2605.20948)) builds memory values from a *frozen donor's* hidden states, but the **recipient is pretrained** with them. *Correction:* frozen-base Engram-style adapters for task gains do exist (Engram Adapter [2608.29327](https://arxiv.org/abs/2608.29327)); none for fact editing. | Unsupported for editing. |
| Fusion params < 8M per layer at d=5120, r=512 | 3·d·r ≈ 7.9M, plus a tiny conv. | Correct arithmetic. |

The honest summary: **every published Engram success has a backbone that learned to read the memory.** The
one setting this project cares about, *fact editing* through a channel a frozen backbone never saw, is
untested in the literature. (Frozen-base Engram adapters for task gains exist; see the correction above.)

## 2. Mapping onto what this repo has already measured

An Engram module combines three design choices: a **key** (hash of the last n token ids), a **value**
(a row in an r- or d-dim table), and a **site** (additive at an early layer, context-gated). memory-organ has
run campaigns on each of these separately.

**Site: the most likely point of failure.**
- Zero-init identity is not new here. `cam/gated_tap.py` sets γ=0, which makes the tap an exact no-op, cast
  back to the base dtype.
- What the repo *did* find is that a frozen-residual gated tap at a single site delivers about **0.7** even
  for one collision-free fact. That ceiling held across gate calibration, multi-layer injection, depth and
  encoder (key-encoding §3.7–3.11), and it matches WISE (0.70–0.77) and MEMIT (0.66).
- The edit-ripple sweep put **early L3 as garbling**, mid layers at ~0.2–0.3 and late at ~0.45. All are
  below RAG (~0.5).
- The escapes that worked moved off the residual: conf-gated **logit injection** (0.65 → 0.88 solo) and the
  **pointer**, which forces exact object tokens.
- An Engram graft at L2 moves *back* to the earliest, worst-measured site. EngramEdit avoids this only
  because its backbone was trained to consume an input-level memory channel.

**Key: genuinely interesting, partly already here.**
- `CAM_DISJOINT_BANKS` already routes by `md5(subject_tids) % B`, which is a hashed token-id key used for
  sharding. Addressing is already solved: below-gate 0/441 with K1 write-at-read.
- So exact hashing does not fix an open problem. It does have three properties the GTE pointer lacks:
  1. **No encoder forward.** The key is a pure function of input ids, so it can be prefetched on the CPU
     before layer 1. For decode it is one step ahead, which is still just a hash.
  2. **A binary presence signal.** Hit or miss is the store-presence gate that beats the
     confidence-separability wall (RESULTS §8). Out-of-store false-fire becomes collision rate, which
     multi-head hashing controls.
  3. **Determinism.** No whitening is needed, and there are no anisotropy failures (raw GTE: 100 %
     false-fire at 0.90).
- The cost is **paraphrase brittleness on the key side**. A key that is the subject's last n tokens only
  fires when those exact tokens appear. EngramEdit's 97 % generalization is over paraphrases that keep the
  subject string. The GTE key also covers surface variants ("NYC" / "New York City"), and an n-gram key
  cannot.
- The n-gram key is also **context-blind**. "of Paris" fires everywhere. Locality then rests entirely on the
  context gate α_t, which a graft has to *train*. EngramEdit itself finds that >92 % of new unrelated
  errors involve updated rows.

**Value: the most promising angle.**
- The precisely located open limit is **efficacy-at-scale**, about 0.81 → 0.55 from N=137 → 441 with
  addressing perfect, and it is attributed to the *shared readout/injection capacity* (key-encoding §0(D)).
- An Engram row holds its own value. If δ\* is stored in **d_model space** (or behind a *fixed*, untrained
  projection), there is no shared `out_proj`/`W_V` that every fact squeezes through. This is a direct test
  of the shared-readout hypothesis, and the repo has never run it.
- Prior art: this is close to **GRACE** (Hartvigsen et al. 2023, discrete key→activation codebook at one
  layer) with hashed-token keys instead of activation keys. It should be positioned that way, not as new.

**Multi-token (#100).**
- The wall sits at positions t≥1. N-gram keys over the *already-emitted object prefix*
  (`hash(o_{t-2}, o_{t-1})`) give per-position addresses for free. That is the same idea as the disjoint
  per-position stores the continuation doc proposes, but it needs no learned codebook.
- Risk: once t>n, the key no longer contains the subject. That is fine for a novel phrase ("Sindarin")
  whose own prefix is unique, but it collides for common object-internal n-grams. EngramEdit's
  reuse/frequency penalty is the right tool for that.

**Portability (the north-star axis).**
- Token-id hashes are **tokenizer-specific**, and δ\* in d_model space is **base-specific**. An Engram
  cartridge does not transfer to another base.
- Mitigations: key on *normalized text* n-grams (tokenizer-agnostic, like the GTE pointer), and hand the
  value to the existing translator. Still, this would be a step back on the axis that motivated the project.

**No-forget by routing.** An exact-hash table is a routed bank of isolated rows, which is the project's
structural win, so it is philosophically aligned. Interference comes back only through *shared* n-grams,
which is the n-gram analogue of the "same-structure alias" router misses in the scorecard.

## 3. Recommendation

**Do not replace the current system. Run an Engram-style write path as arm D against the deployed
pointer and conf-gated logit injection.** Do it on the existing CounterFact harness and its triad
(efficacy / locality-keep / generality) at N=441 and N=1,105, with 3 reps.

| Step | What | Trains | Kill criterion |
|---|---|---|---|
| **E0** | **Training-free δ\*-bank.** At layer ℓ ∈ {embed, 2, 8, 16, 24}, solve the EngramEdit/ROME δ\* at the last subject token, base frozen. Store it raw (d_model) in a dict keyed by `hash(last-n subject ids)`, n ∈ {1,2,3}. Inject additively on an exact hit, gated by presence. | nothing | Efficacy below the conf-gated logit inject at N=441, **or** locality-keep below the pointer. If every ℓ sits at or below ~0.7, the site wall holds and the Engram site is dead on a frozen base. |
| **E1** | **Shared-readout test.** Same as E0, but compare raw-d rows against a fixed r=512 projection and against the repo's trained `out_proj`. | nothing / probe | If raw-d rows do **not** hold efficacy flat from N=137 → 441 while `out_proj` drops, the "shared readout capacity" diagnosis is wrong. That result is worth knowing either way. |
| **E2** | **EngramEdit ridge allocation.** Spread δ\* over the 1/2/3-gram rows with the reuse penalty, so paraphrases sharing sub-n-grams generalize. | nothing | Generality is no better than E0's single-key rows. |
| **E3** | **Calibrated fusion (Mode A proper).** Train W_Q/W_K/W_V plus the gate on a few hundred MB, base frozen, then freeze and write via E2. | ~8M/layer | Run **only if** E0 shows the site delivers but leaks locality, because a trained context gate is the only thing E3 adds. |
| **E4** | **#100 per-position keys.** Key position t on `hash(o_{t-2}, o_{t-1})` (plus a subject salt for t<n) on the private-demo invented objects. | nothing | Isolated per-token round-trip does not beat the 0.50 wall. |
| **E5** | **Portability.** Normalized-text n-gram keys, with values carried by the existing translator. | translator | Only if E0–E2 win same-base. |

**Order and cost.** E0 → E1 → E2 are training-free and reuse the harness, eval and gates. Each is one
δ-solve per fact plus a dict. They answer the decisive question, *does the early-site wall bind a hashed
row the way it binds the tap?*, before any spend on fusion training. If E0 dies at every layer, the
pitch's Mode A is falsified on a frozen base, and the only parts worth keeping are the **key** properties
(prefetch, presence bit), bolted onto the existing pointer/logit-inject value path.

## Sources

- STAPLE — [arXiv 2610.12599](https://arxiv.org/abs/2610.12599), [code](https://github.com/Infini-AI-Lab/STAPLE)
- EngramEdit — [arXiv 2610.10533](https://arxiv.org/abs/2610.10533),
  [checkpoint](https://huggingface.co/ModalityDance/EngramEdit-LongCat-Flash-Lite-MCF-2K)
- Engram (DeepSeek) — [arXiv 2601.07372](https://arxiv.org/abs/2601.07372), [code](https://github.com/deepseek-ai/Engram)
- LongCat-Flash-Lite (native n-gram embeddings) — [arXiv 2601.21204](https://arxiv.org/abs/2601.21204)
- Memory Grafting — [arXiv 2605.20948](https://arxiv.org/abs/2605.20948)
- GRACE — Hartvigsen et al., *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors*, NeurIPS 2023
