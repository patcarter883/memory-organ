# Research notes — key encoding for a quantized associative editing-memory

**Status:** active · **Tracks:** issue #19 (Track 4, persistent/online memory) · **Started:** 2026-07-03

Living research log. Newest findings at the top of each section; the running Log at the bottom is
chronological. Numbers are 3-rep means unless noted (the delivery metric noise is ±0.10 at cohort=10, so
single runs mislead — see Methodology).

---

## 0. CAMPAIGN SYNTHESIS & CONCLUSIONS (2026-07-04)

The full arc — persistent knowledge-editing memory, N=137 edits, delivery from ~0.24 → the ceiling —
resolves into two solved problems and one characterized wall:

**(A) ADDRESSING — SOLVED.** The N=137 delivery ceiling was **key collision under quantized
product-key addressing**, not store capacity (4× store: no change) and not the value side. Fix:
**disjoint value banks** (route each subject to one of B banks by a stable token-id hash → ~B parallel
low-crowding stores). **0.24 → 0.66 at B=32** (knee ~4 subjects/bank). Encoder-side transforms (soft-ZCA
whitening, query BatchNorm) reduce collision but are **marginal in delivery (≤+0.11) and do NOT substitute
for banks**. A semantic retriever (GTE-ModernColBERT) is the wrong tool (semantic clustering is an
anti-feature) — but **whitening *revived* it from 0.000 → 0.589**, confirming interaction effects are real
and that OFAT verdicts were confounded. Shipped: `CAM_DISJOINT_BANKS` (default 32).

**(B) METHODOLOGY — VALIDATED.** A **quantization-aware CPU proxy** (per-key product-key slot-overlap
load), gated on reproducing the known raw-B-sweep + GTE-death, let us screen the combinatorial factor
space cheaply and catch the OFAT-confounded GTE kill. Interaction-aware, proxy-screened design worked.

**(C) RETRIEVAL FIDELITY — WALL CHARACTERIZED, THEN ESCAPED.** With addressing solved, a **single
collision-free fact still delivers only ~0.7** (R0) through the **frozen-residual gated tap** — the
documented single-site residual-injection ceiling (WISE 0.70–0.77, MEMIT 0.66), robust to gate
calibration, multi-layer injection, depth, and encoder (§3.7–3.11). **The escape is to leave the residual
site: `CAM_LOGIT_INJECT` adds the retrieved value's contribution (out_proj→lm_head) straight to the OUTPUT
logits.** Solo fidelity 0.65 → 0.88 (§3.14). The blunt version has a **locality wall** (it fires on every
prompt → wrecks neighbours, keep 0.47→0.10), BUT the store's retrieval confidence separates in-store
edited subjects (median ≈122) from out-of-store neighbours (≈0.04) almost perfectly, so a **HARD conf-gate
on the injection recovers full baseline locality (keep flat at 0.47) while still delivering +0.12** — the
usable operating point (§3.15). The wall is a property of the *site*, not of frozen-ness; logit-space with
a conf-gate escapes it deployably. Shipped: `CAM_LOGIT_INJECT`, `CAM_LOGIT_GATE_C0`/`_K`/`_HARD`.

**BOTTOM LINE (product):** the per-fact **~0.7 residual ceiling is NOT fundamental** — hard-conf-gated
logit injection escapes it at ~zero locality cost, and the residual gap was **addressing-limited**. That
addressing frontier is now **CLOSED by K1 write-where-you-read** (§3.16, `CAM_WRITE_AT_READ`, persistent-
path only): addressing the write with the *read* query eliminated every self-addressing failure
(below-gate 15–25 → **0** in all 3 reps), collapsed the neighbour-collision tail (conf p95 118 → 2), and
lifted end-to-end delivery to **~0.90 (n=3) / neighbour-keep ~0.53 / leak ~0.03** — a +0.18 jump over the
B=137 baseline (0.72), past both the ~0.66 addressing plateau and the ~0.7 residual wall. The C0 gate
calibration is now moot (in/out conf separate at any threshold). The retrain tier (K4–K6) is obviated; the
only residual is the ~10% genuine readout/value-miss floor (a value-side lever, not addressing). No
frozen-base architecture cap remains.

## 1. The question

*What is the right key representation for a **quantized associative editing-memory**, and how do the
levers we've found — key transform (whitening / learned rotation), disjoint-bank hashing, encoder choice —
interact?*

Concretely: our store is a **product-key memory** (Lample et al. 2019, "Large Memory Layers with Product
Keys"): a query is split in half, each half does top-k over a sub-codebook, the cartesian product selects
sparse slots; we write `(key→value)` with an error-correcting delta and read by nearest-key. Keys are
entity **subjects** (often multi-token names), values are new facts, injected into a frozen LLM's residual
by a trained tap. The empirical bottleneck is **key collision under the store's quantized addressing** as
the number of stored edits N grows.

## 2. Executive summary (what we believe now)

- The N=137 delivery ceiling (~0.24) was **addressing / key-collision**, *not* store capacity (4× store =
  no change) and *not* the value side. **[established]**
- **Disjoint value banks** (route each subject to one of B banks by a stable token-id hash → ~B parallel
  low-crowding stores) break the ceiling: N=137 delivery **0.24 → 0.66 at B=32**, knee ~B=32 (~4
  subjects/bank). This is hash-sharding to reduce collision. **[established, shipped: CAM_DISJOINT_BANKS]**
- A **semantic retrieval encoder is the wrong tool** for keys: GTE-ModernColBERT pooled keys are
  catastrophically anisotropic (mean pairwise cos 0.968) and can't bind at all (flat chance loss, 0
  delivery). Semantic clustering (pulling *similar* entities together) is an **anti-feature** — editing
  keys need the opposite (distinct entities far apart). **[established]**
- **Whitening (ZCA, no training) is the lever on the encoder side.** It restores isotropy and slashes
  nearest-neighbor confusability for every encoder; **whitened Qwen input-embeddings win the bake-off**
  and need no new model/dependency. **[established on the proxy metric; delivery test pending]**
- **Reframe:** this is the classic **product-quantization addressing** problem from ANN/vector search.
  "Transform the keys so product quantization separates them" = Optimized Product Quantization (learned
  rotation). Our whitening win looks like a special case; the right transform may be a *learned rotation*
  aligned to the product-key sub-codebooks, not generic ZCA. **[lit grounding in progress]**

## 3. Results

### 3.1 The N=137 ceiling is addressing, not capacity (issue #19, PRs #44)
| Lever | N=137 delivery |
|---|---|
| bigger store (4× slots + heads) | ~0.23 (no change) |
| single-key baseline | ~0.24 |

→ Quadrupling slots/heads does nothing; the limit is key separation under quantized addressing.

### 3.2 Disjoint value banks break it (PR #48; B-sweep #48)
| B (banks) | ~subj/bank | N=137 | N=34 |
|---|---|---|---|
| 1 | 137 | 0.255 | 0.412 |
| 8 | 17 | 0.421 | 0.578 |
| 16 | 9 | 0.526 | 0.677 |
| **32** | **4.3** | **0.655** | 0.755 |
| 64 | 2.1 | 0.606* | 0.716 |

Knee ~B=32; at B=32, N=137 (0.66) exceeds the *original* single-bank N=34 rate. Delivery becomes bounded
by per-bank store quality (~0.7), not by N.

### 3.3 Encoder bake-off — separability (nearest-neighbor cosine, lower = better; 2936 subjects)
| Encoder | NN-cos | mean-pair-cos |
|---|---|---|
| **Qwen input-embed — WHITENED** | **0.146** | −0.001 |
| Qwen3-Embedding-0.6B — whitened | 0.173 | −0.001 |
| GTE-ColBERT pooled — whitened | 0.441 | −0.001 |
| Qwen input-embed — RAW *(incumbent)* | 0.505 | 0.067 |
| Qwen3-Embedding-0.6B — raw | 0.857 | 0.573 |
| GTE MaxSim (nn/self ratio) | 0.962 | — |
| GTE-ColBERT pooled — RAW | 0.988 | 0.968 |

- Whitening cuts NN-confusability across the board (−71% on input-embeds, −80% on the dense embedder) and
  makes every encoder isotropic (mean-pair-cos → 0).
- The incumbent input-embeddings, **whitened**, win — no new model needed.
- Semantic encoders lose raw (clustering anti-feature); GTE stays worst even whitened (ColBERT pooling is
  pathologically anisotropic). *(SPLADE-v3 untested — gated HF repo, 403.)*

### 3.4 GTE-ModernColBERT in the store — NEGATIVE (definitive)
Pooled GTE keys: mean pairwise cos 0.968, NN-cos 0.987 (near-parallel) → store can't bind (bind loss flat
at chance 12.42; delivery 0.000). 2936/2936 lookup hits, so not a bug — the anisotropy is the cause.
Kills pooled/single-vector GTE. GTE-MaxSim (multi-vector) untested but would need the late-interaction
store redesign, and its raw margin already trails whitened input-embeds.

## 3.5 P0 GATE PASSED — quantization-aware proxy validated (2026-07-03)

Proxy = per-key total product-key **slot-overlap load** within disjoint banks (`tools/pk_proxy.py`).
Metric v1 (mean-per-pair) FAILED (flat across B — caught by the gate); fixed to per-key total load
(÷N not ÷pairs). Validated table (lower=better):

| encoder | BN | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| inembed raw | off | 2.225 | 0.263 | 0.137 | 0.067 |
| inembed whitened | off | 1.464 | 0.186 | 0.089 | 0.045 |
| inembed raw | on | 1.477 | 0.194 | 0.101 | 0.057 |
| GTE raw | off | 85.9 | 10.5 | 5.1 | 2.67 |
| **GTE raw** | **on** | 1.356 | 0.156 | 0.073 | **0.038** |
| GTE whitened | off | 1.369 | 0.164 | 0.072 | 0.041 |

**Validation:** raw-inembed collision 2.225→0.067 tracks delivery 0.255→0.655 (rank-inverse, monotone);
GTE-raw catastrophic (85.9) matches delivery 0.000. → proxy trustworthy for the combinatorial screen.
**Interactions surfaced (proxy):** (1) whitening ≈ query-BatchNorm (both −30-35% collision, substitutive —
both de-anisotropize). (2) **negative×positive confirmed: GTE raw 85.9 (dead) → 0.038 with BatchNorm /
0.041 whitened** — de-anisotropized GTE is the BEST cell. **We killed GTE prematurely (tested only raw);
OFAT confounding, empirically demonstrated.** Delivery confirmation of these predictions = P2.

## 3.6 P2 DELIVERY GRID — resolved (2026-07-03)

cf-delivery, 3 reps (`tools/p2_grid.sh`):
| cell | N=137 | N=34 |
|---|---|---|
| raw · B=1 | 0.209 | 0.510 |
| raw · B=32 | 0.608 | 0.735 |
| white · B=1 | 0.255 | 0.490 |
| white · B=32 | 0.567 | 0.774 |
| bn · B=1 | 0.317 | 0.588 |
| whiteGTE · B=32 | 0.589 | 0.667 |

- **GTE REVIVAL CONFIRMED (negative×positive, end-to-end):** raw GTE 0.000 → **whitened GTE 0.589** @B=32.
  OFAT killed GTE prematurely; de-anisotropized it's competitive. Proxy predicted it; delivery confirms.
  Validates the interaction-aware / proxy-screened methodology.
- **H1 = FALSE (whitening does NOT substitute for banks):** banks move delivery +0.40 (0.21→0.61); every
  encoder transform ≤ +0.11, and white@B1 (0.255) ≪ raw@B32 (0.608). Keep the disjoint banks.
- **H4 = modest+:** query BatchNorm is the best *encoder-side* cheap lever (+0.11 @B=1, native, no key
  change; beats whitening there) — but still ≪ banks.
- **Proxy fidelity:** correctly ranked BN>raw@B1 and the GTE revival; OVERSTATED whitening@B32 (proxy
  0.045<0.067 → delivery 0.567≈0.608). ⇒ proxy is a good *screen* (ranks, catches death/revival) but small
  proxy gaps don't predict small delivery gaps — both in the ±0.10 noise.
- **FRONTIER MOVED:** at B=32 all encoders converge ~0.57–0.61 (and B=64 was ~0.61) → addressing is
  SATURATED. The ~0.6–0.7 ceiling is now **per-bank RETRIEVAL FIDELITY** (store/value/tap readout), not
  key collision. Untouched all campaign — the next research phase.

## 3.7 R0 RESULT — the ceiling is the injection mechanism, confirmed (2026-07-03)

Single-fact fidelity (`--persistent-solo`, each edit ALONE, no collision possible), 3 reps:
**solo-delivery 0.657 / 0.657 / 0.745 → mean 0.69**, vs same-run N=137@B=32 = 0.606/0.628/0.715 → 0.65.

- Solo (zero collision) ≈ **0.69** — dead in the WISE (0.70–0.77) / MEMIT (0.66) single-site-injection
  band. Removing ALL collision buys only **+0.04** over B=32. ⇒ disjoint banks already captured ~all the
  collision headroom; the remaining ~0.31 to 1.0 is the **single additive gated residual injection**, NOT
  the store/addressing. Phase-R thesis confirmed. Next: **P-R1 = GCAV calibrated gate.**

## 3.8 P-R1 (norm-relative gate) — NEGATIVE; pivot to P-R2 multi-layer (2026-07-03)

`CAM_NORM_GATE` (inject value DIRECTION at α·‖h‖) **collapsed**: solo 0.002 (vs baseline 0.674), N=137
0.000, prior ~0.99 — the tap learned nothing. Diagnosis: normalizing `y` discards the tap's learned
per-dim magnitude, and one global α scaling by ‖h‖ can't calibrate it → optimizer drives α→0 (no-op). The
crude norm-relative swap destabilizes training; the real GCAV lever needs to target the OBJECT LOGIT
(needs the object direction) — a bigger change, deferred. **Pivot to P-R2 (multi-layer injection)** — the
theory's robust "past the single-site ceiling" lever, reusing `--tap-layers`/`--multi` (no gate surgery):
tap at ~3 layers (e.g. 16,20,24) with `--multi`, and a single-layer sweep (L18/L20 vs L24 — is L24 too
deep for a 4B model?). Metric: solo-fidelity vs the 0.69 single-L24 ceiling.

## 3.9 P-R2 (multi-layer) NEGATIVE — the ~0.7 wall is ROBUST (2026-07-03)

solo-fidelity vs single-L24 (~0.69): **multi[16,20,24] = 0.698** (0.745/0.686/0.664, ≈ baseline);
single-L18 = 0.565 (worse — L24 not too deep). Multi-layer injection did NOT break the ceiling.

**Depth sweep completed (P-R2b):** L18 0.565 · L24 ~0.69 · **L28 0.701** · L30 0.623. Fidelity peaks
mid-late (L24–28 ~0.70), no depth escapes the wall; L30 drops (matches "final quarter reverts"). 5th
confirmation. **Next (R-mech): what IS the ~30% that fails solo — a consistent un-editable subset (⇒
characterizable editability boundary, a real lever) or uniform noise (⇒ stochastic floor)?** Per-fact
logging (CAM_SOLO_LOG) + 3-rep consistency analysis. Reframes the wall rather than adding a 6th knob.

**CONVERGED CONCLUSION:** ~0.7 is the robust reliability ceiling of a trained gated *residual* injection
into a frozen LM — stable across encoder (P2), gate calibration (P-R1), layer count + depth (P-R2), and
matching WISE (0.70–0.77) / MEMIT (0.66). Four independent confirmations of the lit's "single additive
nudge has a hard reliability wall no store tuning removes." **Remaining lever = leave residual space:**
**P-R3 logit-level / unembedding-direction injection** (the theory's guaranteed-to-flip fallback — directly
boost the object token's logit) — bigger architecture change, with a locality trade-off to watch. Decision
point: attempt P-R3, or conclude Phase R with the characterization (the ceiling is now well-established and
matches the literature).

## 3.10 R-mech — the wall CRACKED OPEN: it's a mixture, failures = strong base prior (2026-07-04)

Per-fact solo outcomes, 137 facts × 3 reps (mean hit 0.69):
- always-hit 60 (44%) · **always-FAIL 16 (12%, vs 3% i.i.d.-noise expectation = 4×)** · variable 61 (45%).
- ⇒ NOT uniform noise: a real **un-editable subset** + a stochastic borderline + a reliable core.
- Strongly **relation-dependent**: P159 0.47 → P140 0.93.
- **Decisive correlate:** always-FAIL base-prior-recall **0.40** (model still emits the ORIGINAL after the
  edit) vs always-HIT **0.00**. **The un-editable facts are the ones the frozen base is most confident
  about** — the edit fights a strong prior (matches AlphaEdit).

**LEVER REVEALED (R1-prior):** the tap only pushes TOWARD the new object; it never SUPPRESSES the original,
so confident-prior facts revert. Two-sided / prior-aware injection (promote new AND damp the base's own
next-token direction) should recover the strong-prior failures — a targeted lever the failed
global-strength/depth levers couldn't be, because it attacks the *specific* 12–30% that fails, not the
average. Metric: solo-fidelity, esp. on the always-FAIL / high-prior subset; watch locality.

**R1-prior-v1 (CAM_TWOSIDED, blind residual-damp) NEGATIVE:** ON 0.616 ≈ OFF 0.613; recovery of the
always-FAIL subset = 0.125 (≈ noise). Damping ALL of h (not the original object's direction) → optimizer
keeps it ~0 (no-op). **v2 (CAM_VALUE_SUPPRESS=λ): compose the stored value = new − λ·original** (we have
r.true_tid) so the injection promotes new AND damps the ORIGINAL logit specifically — eval-only, no
retrain. Directly attacks the strong-prior failures. A/B λ∈{0,0.5,1.0} on solo + always-FAIL recovery.
**R1-v2 NEGATIVE:** λ=0.5 = 0.611 (no change, recovery 0.125 ≈ noise); λ=1.0 = 0.246 (over-suppresses,
hurts). Even targeted logit-suppression of the actual original doesn't recover strong-prior fails — the
frozen tap doesn't cleanly translate a composed value into a two-sided logit effect. **7 negatives total ⇒
the ~0.7 wall is intrinsic to single-site frozen-base editing.**

## 3.11 UNIVERSALITY — the wall is general, and its level tracks base knowledge (2026-07-04)

Solo-fidelity across bases (`--base1`, harness is base-agnostic):
| base | solo-fidelity | base prior-acc (knowledge) | edits |
|---|---|---|---|
| Qwen3.5-4B | ~0.69 | 0.138 | 137 |
| Llama-3.2-3B (diff family, no GDN) | 0.574 | 0.222 | 173 |

- **The ~0.5–0.7 wall is GENERAL** (holds across architecture families) → a property of the frozen-base
  single-site editing METHOD, not Qwen-specific.
- **Level is model-dependent AND the mechanism predicts the direction:** Llama KNOWS MORE (prior-acc 0.222
  vs 0.138) and edits WORSE (0.57 < 0.69) — cross-model confirmation of R-mech ("confident priors resist
  editing"). **More capable base ⇒ harder to edit** — a real product tension.
- **HYPOTHESIS (R-univ):** solo-fidelity is negatively predicted by base prior-acc. Sweep more cached bases
  → (prior-acc, fidelity) scatter; a clean correlation = a quantitative law + a base-selection lever.

## 3.12 MECHANISM QUANTIFIED — editability is governed by base-prior confidence (2026-07-04)

Within Qwen3.5-4B (architecture-controlled), per-fact PRE-EDIT base P(original) vs edit success:
- Dose-response: P(orig) 0.10-0.25 -> edit 0.75; 0.50-0.75 -> 0.48; 0.75+ -> 0.46. **Pearson r=-0.23**
  (n=137, p~0.007) — confident priors resist editing, significant but modest at the fact level.
- **Relation-level (noise averaged): r=-0.60.** P140/P276 (P(orig) 0.2-0.3) edit 1.00; P37 (0.55) edits
  0.58. **Base prior-confidence is the DOMINANT SYSTEMATIC driver of editability.**
- **Residual relation-specific factor:** P159 (edit 0.20 vs 0.41-prior prediction) and P364 (0.58 vs
  P103's 0.81 at equal prior) are harder than confidence explains -> a SECOND factor (prompt format /
  object vocabulary), not prior-confidence.

**RESIDUAL FACTOR IDENTIFIED = OBJECT-VOCABULARY SIZE.** corr(relation edit-success, #distinct objects)
= **-0.89**: large answer spaces (P159 headquarters -> 88 cities, edit 0.20) are far harder than small
closed sets (P140 religion -> 7 options, edit 1.00) — the injected value must win against more candidates.

**FINAL MECHANISM (fully decomposed):** the ~0.7 frozen-base single-site wall's variance = (1) base-prior
confidence (r=-0.60/relation, confident priors resist override) + (2) object-vocabulary size (r=-0.89/
relation, large answer spaces dilute the injection) + (3) stochastic per-attempt noise. P159 hard on both,
P140 easy on both.

## 3.13 EDITABILITY IS PREDICTABLE (practical capstone, 2026-07-04)

Joint fit hit ~ base-P(original) + log(object-vocab) on 137 facts: **R2=0.20** (modest — most residual is
the irreducible single-site stochastic reliability), but the RANKING is strongly usable: **top-predicted
quartile edits 0.94 vs bottom quartile 0.27** (3.5x separation). Object-vocab is the stronger fact-level
driver (beta -0.19), prior-confidence secondary (-0.06). ⇒ We can't BREAK the wall but we can PREDICT it:
two PRE-MEASURABLE quantities flag which facts edit reliably vs not — turning "~66% reliable, unknown
which third fails" into a routable editability score. PRODUCT LEVER (no paradigm change needed). Confirmed within-model (architecture-controlled) and directionally
cross-model (more-knowledge bases edit worse). This is the quantitative capstone; the campaign has
characterized the wall, its universality, and its mechanism.

## 3.14 PARADIGM CRACK — logit-level injection breaks the residual wall (2026-07-04)

`CAM_LOGIT_INJECT=α` adds the retrieved value's contribution (out_proj(bank)→lm_head) straight to the
OUTPUT logits, bypassing the residual site. Solo-fidelity: α=0 **0.65** → α=2 **0.876** → α=8 0.861 →
α=20 0.883 (prior-reversion 0.22→0.00).
- **The ~0.7 wall was the injection SITE (residual), not frozen-ness** — same value, logit-space, escapes
  it (+0.22). Diagnosis confirmed.
- **New ceiling ~0.87, NOT prior-reversion (prior≈0) and NOT strength (flat α=2→20):** the residual ~13%
  output neither original nor new → the **value→logit READOUT fidelity** (out_proj→lm_head not perfectly
  peaked on the target) — a different, addressable limit.
- **CRITICAL open follow-up = LOCALITY.** Logit injection is blunt: it forces the edited object's logit
  wherever the tap fires, so it likely damages NEIGHBOR facts (queries about the edited subject that
  should NOT change). Delivery ↔ locality trade-off across α is the experiment that decides whether this
  is a usable escape or just trades the fidelity wall for a locality wall. → RESOLVED in §3.15.

## 3.15 LOGIT-INJECTION LOCALITY — the escape is USABLE, but only conf-gated (2026-07-04)

The decisive follow-up to §3.14. `--persistent-locality` writes all 137 edits, then sweeps α and scores
edit **delivery** together with neighbour **keep** (gold = the neighbour's TRUE object) / **leak** (flips
to the EDIT's counterfactual). Neighbours are scored two ways: **DEP** (deployment-faithful — the
neighbour's OWN parsed subject drives store retrieval, so the store's *addressing* is in play) and **ADV**
(adversarial upper bound — neighbour forced onto the edit's own bank). *(Absolute delivery wanders ±0.15
run-to-run — the 150-step tap fit is stochastic under native-GDN nondeterminism — but the locality
STRUCTURE below is identical across all runs. Numbers shown are the hard-gate run.)*

**Unconditional logit injection has a real LOCALITY WALL.** At α=2 (DEP): delivery 0.55→0.77 but
neighbour-keep **collapses 0.47→0.10** — a locality loss *larger* than the delivery gain. Blunt injection
trades the fidelity wall for a locality wall. Mechanism: the product-key store returns *some* bank for
every query, and unconditional injection fires at fixed α on **every** prompt, corrupting unrelated facts.
The residual tap has a trained retrieval-strength conf-gate; the raw logit injection had **none**.

**The retrieval-confidence signal separates in/out-of-store PERFECTLY.** Factual-head retrieval magnitude
(`conf`): **edited (in-store) median ≈ 122, neighbour (out-of-store) median ≈ 0.04** (separation +122).
This is the enabling fact — the store *knows* when a query is a real key vs a stranger.

**Conf-GATING the injection (scale α by conf) recovers locality.** `CAM_LOGIT_GATE_C0` (+`_K`, or
`_HARD=1`) gates the injection on retrieval strength. Self-calibrated C0 = midpoint of the two medians.
Deployment (DEP) numbers, α=2:

| injection            | delivery | nbr-keep | nbr-leak |
|----------------------|----------|----------|----------|
| baseline (α=0)       | 0.547    | 0.466    | 0.110    |
| unconditional        | 0.766    | **0.096**| 0.096    |
| conf-gated (soft)    | 0.737    | 0.233    | 0.096    |
| conf-gated (**HARD**)| 0.664    | **0.466**| 0.068    |

- **HARD gate = the usable operating point.** Neighbour-keep stays **exactly at baseline (0.466) across
  ALL α** (neighbours' conf ≈0.04 ≪ C0=61 → hard-zeroed) while delivery rises **+0.117** and leak *drops*
  to 0.068. Logit injection breaks the residual wall **at zero locality cost** — it is a genuine escape,
  not a wall-trade, provided it is hard-gated on retrieval confidence.
- The soft sigmoid gate is an intermediate (transition width ~1/K wider than the gap → partial leak to
  mid-conf rows); the **hard step exploits the 122-vs-0 separation** and is strictly better for locality.
- **Residual gap = addressing, not locality.** Hard-gate delivery (0.664) < unconditional (0.766) because
  some *edited* subjects retrieve with conf < C0 (false negatives — the hard gate zeros their injection
  too). That is a **store-addressing / write-strength** lever (raise in-store conf so more edits clear the
  threshold, or a per-edit adaptive threshold), NOT a locality one. This reconnects to §3.2 (banks) — the
  fidelity and addressing frontiers are now coupled through the gate threshold.

**Threshold sweep — the midpoint C0 is over-conservative; the knob is the neighbour TAIL, not the median.**
Sweeping the hard-gate C0 downward (α=20; edited conf ~118, neighbour median ~0.02) shows a clean,
favourable frontier — and one subtlety:

| C0 | delivery | DEP-keep | DEP-leak |
|----|----------|----------|----------|
| 59 (midpoint) | 0.628 | 0.479 | 0.027 |
| 20 | 0.650 | 0.466 | 0.027 |
| **10** | **0.672** | **0.466** | 0.027 |
| 5  | 0.686 | 0.452 | 0.027 |
| 2  | 0.708 | 0.425 | 0.027 |
| 0.5 | 0.715 | 0.384 | 0.027 |

- **DEP-leak is pinned at 0.027 for EVERY threshold** — neighbours never flip to the edit's object,
  because even C0=0.5 ≫ the neighbour conf median (0.02). The gate is leak-proof across its whole range.
- **Lowering C0 monotonically recovers delivery** (0.628 → 0.715) by admitting sub-threshold *edited*
  subjects — confirming the gap is edited-side false-negatives, not a locality limit.
- **But keep degrades gradually** (0.479 → 0.384): the neighbour conf *distribution* has a TAIL (a few
  neighbours token-overlap an edit in the same bank), so pushing C0 to the median level injects into that
  tail and corrupts those neighbours to *random* wrong tokens (keep-loss WITHOUT leak-gain).
- **Sweet spot C0 ≈ 10** — roughly a decade below the edited median, above the neighbour tail: delivery
  0.672 (+0.044 over the midpoint) at keep 0.466 (−0.013 from baseline 0.479). So the right calibration is
  **2× the neighbour conf p95** (exclude the tail, include everything else), NOT the in/out midpoint. Gate
  default updated accordingly; `CAM_LOGIT_GATE_C0` is the production delivery↔locality knob.
- There is an **irreducible tail-trade**: the gate cannot fully match unconditional delivery (0.77) without
  touching the neighbour tail, but at MATCHED locality (keep ~0.47) gated delivers 0.67 vs ungated's ~0.65
  baseline, and at matched delivery gated has *far* better locality. The gate **dominates the ungated
  frontier** everywhere.

**Per-edit conf diagnostic — the gap decomposes into ADDRESSING, and the neighbour tail is bimodal.**
`CAM_CONF_DIAG=1` logs each edit's retrieval conf vs subject length / relation / whether it delivers under
unconditional max-α (readable?) and residual-only (base?):
- **The delivery gap is ~17/137 addressing false-negatives.** 25 edits fall below C0≈30; **19 of them still
  DELIVER under unconditional injection** — i.e. their value is perfectly *readable*, they just *retrieve
  weakly*. Histogram: conf 1–10 = 8 edits, **all 8 deliver**; conf<1 = 15 edits, 9 deliver. So the
  hard-gate gap is dominated by real edits whose own subject key retrieves their value at low magnitude —
  an **addressing / write-strength** problem, NOT readout. Mean conf is ~uniform across the 6 relations
  (90–110), so it's not relation-specific → points to bank-crowding / key-collision (the §3.2/§3.4 levers:
  more banks, better keys). ~6 of the 25 fail even unconditionally = genuine readout/value misses.
- **The neighbour conf distribution is BIMODAL** — median ~0.001 but **p95 ≈ 118 ≈ the edited median**.
  A ~5% tail of neighbours *fully collide* with an edit in the same disjoint bank and retrieve at in-store
  strength; **no conf threshold can exclude them** (they're indistinguishable from real edits by retrieval
  magnitude). These are the irreducible keep-loss cases and set the locality floor. (Consequence: the p95
  is useless for calibration — C0 is set from the edited side, em/12, to include the weak-but-real edits.)
- **Next lever (addressing):** raise in-store retrieval conf for the ~17 weak-but-deliverable edits — more
  banks (B>32) / better keys to de-crowd their bank — so a *safe* high C0 captures full delivery AND the
  bimodal neighbour tail shrinks (fewer same-bank collisions). Fidelity is now fully recoupled to the
  addressing frontier; there is no separate frozen-base fidelity cap left to attack.

**CAPSTONE — scaling disjoint banks lifts BOTH delivery and locality at once (no trade), via readout
PURITY not retrieval magnitude.** `CAM_BANK_SWEEP` re-routes the standing store into B banks (bind/tap
unchanged — persistent-path only), per-B hard gate C0=em/12, α=20:

| B (edits/bank) | edit-conf | below-gate | delivery | DEP-keep | DEP-leak |
|----------------|-----------|------------|----------|----------|----------|
| 32 (~4)        | 109       | 18         | 0.635    | 0.507    | 0.014    |
| 64 (~2)        | 109       | 20         | 0.679    | 0.534    | 0.014    |
| 137 (~1)       | 109       | 19         | **0.723**| **0.575**| 0.014    |

- **Both axes improve together:** delivery 0.635→0.723 AND neighbour-keep 0.507→0.575, leak pinned 0.014.
  Scaling addressing is a *pure* win here — no delivery↔locality trade.
- **Mechanism CORRECTION:** edit-conf median (~109) and below-gate count (~19) are **FLAT across B** — so
  the gain is NOT de-crowding raising retrieval magnitude (the §3.15 hypothesis). It's **readout PURITY**:
  at B=32 an edit's bank holds ~4 mixed values, so the retrieved+injected value is contaminated *at the
  same conf*; at B=137 (1 edit/bank) the bank is pure → cleaner delivery, and a neighbour that collides
  into a bank finds 1 value not 4 → cleaner separation → higher keep. **Bank crowding degrades the VALUE
  READOUT, not the retrieval STRENGTH.** (The ~19 below-gate edits are a stable, B-independent subset —
  genuinely weak self-retrieval / the ~6 unreadable — a *separate* residual lever: keys/writes, not banks.)
- **Usable end-to-end operating point:** B=137 + hard-conf-gated logit injection → delivery 0.72, keep
  0.58, leak 0.01. The persistent editing memory now delivers ~0.72 of edits AND preserves neighbours,
  well past the old ~0.66 addressing plateau and the ~0.7 residual fidelity wall — both escaped, together.

**Weak-edit decomposition — value-norm artifact vs genuine self-addressing failure.** At B=137 (each edit
alone in its bank, NO crowding) ~25/137 edits still fall below the gate, and ~23 deliver unconditionally
(readable, weakly RETRIEVED). The retrieval conf = ‖ctx‖ = slot-weight × ‖stored value‖, and the value is
the object token's embedding — so conf mixes *addressing quality* with *token-embedding norm*.
`CAM_VALUE_UNIT_NORM=1` writes a unit value to isolate the two:
- Neither driver is **subject length** (mean conf flat 4–5 across slen 2–5) nor **relation** (weak edits
  scattered 9–27% per relation).
- Unit-norm drops the below-gate count **25→15** — so **~10 weak edits were a pure value-norm artifact**
  (low-embedding-norm object tokens; their retrieval direction was right, only the magnitude was small →
  now in the tight high-conf cluster).
- **~15 edits stay low-conf even with unit values** (conf < 0.5 while the bulk sits at 5.5–6.9, tight) —
  a **genuine key SELF-ADDRESSING failure**: the learned pooled query does not peak on its own written
  key. Intrinsic (no crowding at B=137, no value-norm, no length/relation signal); 11/15 still deliver.
- **CAM_VALUE_UNIT_NORM is a DIAGNOSTIC, not a production setting.** It (a) breaks the residual tap, which
  was trained on natural value magnitudes (α=0 delivery 0.02), and (b) destroys the conf-gate's separation
  — with unit values the neighbour conf p95 (5.79) *overlaps* the edit median (5.56), so hard-gated
  locality drops (keep 0.58→0.49, leak 0.01→0.03). The value MAGNITUDE is load-bearing for the gate. Ship
  off-by-default.
- **The final residual lever is KEY ENCODING for self-addressing** — make each subject's query peak on its
  own written key (the §3.4 whitening / key-repulsion / OPQ family), now with a precise target: the ~15
  non-self-peaking subjects. This is the last open thread; everything else in the fidelity/addressing
  frontier is closed.

**Verdict:** §3.14's paradigm crack is real AND deployable. The ~0.7 residual wall is escaped by
logit-space injection; the locality cost that made it look like a mere trade is **eliminated by a hard
conf-gate**, which the store's near-binary in/out retrieval confidence makes almost free. Shipped:
`CAM_LOGIT_GATE_C0` / `_K` / `_HARD`, `--persistent-locality` (dep+adv cohorts, self-calibrated gate).

## 3.16 PHASE K / K1 — the addressing story CLOSES: write-where-you-read (2026-07-04)

The last open lever (§3.15 / §6e). Root cause of the ~15 weak-retrieval edits: WRITE addresses via
`_address(to_wkey(key))` but READ addresses via `_address(read_q[0](q)+head_bias[0])` — two learned
projections coupled only *softly* (cosine addr-sup), so a subject near a product-key top-k boundary lands
in **different cells** at write vs read → its read query misses its own value → conf≈0. **K1
(`CAM_WRITE_AT_READ=1`, persistent-path only, NO retrain):** address the write with the *read* query
`head_query(key)` so the value lands at the exact slot the read selects. Result at B=137 (n=3 reps;
means, with the structural wins reproducing exactly in every run):

| metric | B=137 baseline | **K1 (n=3)** |
|--------|----------------|--------------|
| below-gate edits | ~15–25 | **0 / 137 (all 3 reps)** |
| edit conf (median / min) | ~109 / <1 | **~133 / 68** |
| neighbour conf p95 | 22–118 | **~2** |
| hard-gate delivery (α=8) | ~0.72 | **~0.90** (0.87 / 0.91 / 0.93) |
| DEP-keep / leak | ~0.58 / 0.01 | ~0.53 / 0.027 |

- **below-gate 15–25 → 0.** Every edit now self-retrieves strongly (min conf 68 ≫ gate); the addressing
  false-negatives are *gone*. By construction: write and read address the identical vector.
- **Neighbour-collision tail COLLAPSES (p95 118 → 2.14).** The value now sits at the edit's *exact*
  read-address, not the broader `to_wkey` cell, so a different subject's read query almost never hits it →
  the bimodal tail (§3.15) that set the irreducible locality floor is largely eliminated too. K1 fixes
  BOTH sides — delivery AND the locality floor — with one change.
- **The C0-threshold sweep goes FLAT** (0.934 / 0.548 for C0 = 0.5…20): with perfect self-addressing,
  in-store (133) and out-of-store (~0–2) conf are cleanly separated at *any* threshold — the gate
  calibration problem of §3.15/§3.16-prior is now moot. The hard gate still beats unconditional on
  locality (keep 0.548 vs 0.425 at α=8) by suppressing the rare neighbour hit, but C0 placement is free.
- **Delivery ~0.90 (n=3: 0.87/0.91/0.93) at held locality (keep ~0.53, leak ~0.027)** — a +0.18 jump over
  the B=137 baseline (0.72), at/near the ~0.88 solo ceiling, for a small locality cost (keep −0.05, leak
  +0.013 vs baseline — within run noise). The residual ~10% non-delivery is the genuine readout/value-miss
  floor (`below-gate DELIVER-under-unconditional = 0` → not addressing), a separate value-side lever.

**Consequence: K1 alone closes the addressing frontier — the retrain tier (K4–K6) is OBVIATED** (there is
no residual self-addressing failure to retrain away; below-gate = 0 in all 3 reps). Shipped:
`CAM_WRITE_AT_READ` (K1), `CAM_WRITE_REDUNDANT` (K2), `CAM_READ_SUB_TOPK` (K3, unneeded).

## 3.19 M0 CORRECTION — multi-token is NOT the gate; the cap is a subject-length BATCHING artifact (2026-07-04)

§3.18 concluded multi-token objects were the scale gate. **M0 (a cheap `--probe-only` sweep of
`CAM_MAX_OBJ_TOK` 1→4) REFUTED that** — and found the real, much easier lever:

| K | candidate facts | base-known facts |
|---|-----------------|------------------|
| 1 | 21,321 | **2,936** |
| 2–4 | 21,919 | 2,905 |

- **Single-token objects were NEVER the binding constraint:** 21,321 of 21,919 CounterFact facts (97%)
  already have single-token objects; multi-token *supply* is only **598** (2.7%), and base-known even drops
  slightly with them (2905 < 2936 — the base predicts multi-token objects' first token a bit worse). So
  multi-token objects add ≈0 scale.
- **The real cap: ~2,936 base-known editable facts collapse to 147 via the per-relation
  single-subject-length grouping** (`setup_counterfact_multi`). Each relation keeps only its ONE largest
  `(prompt, subject-length)` bucket and only relations with ≥`per_rel_min` facts survive → 9 relations ×
  one length each = 147 (e.g. P103 keeps len-4/37-facts, *discards* its len-2/3/5 facts). That grouping is a
  **rectangular-bind-batching artifact**, not a store limit — the store keys on the subject's last-token /
  pooled span, which is length-agnostic.
- **⇒ the scale lever is binding across ALL subject-lengths and ALL relations** (length-bucketed
  sub-batches, exactly as the eval probe already does). Headroom: **N 147 → up to ~2,936 (~20×)** on
  Qwen3.5-4B, with NO multi-token work. This is almost certainly a smaller change than multi-token objects.
- **Methodology win:** a ~15-min probe caught a wrong hypothesis before we built it. §3.18's "multi-token is
  the gate" is **superseded**; Phase M (multi-token) is demoted to a ≤+2.7% nice-to-have. See §6f (revised).

## 3.18 SCALE-N de-risk — [SUPERSEDED by §3.19] single-token cap hypothesis (2026-07-04)

Before productionizing, we asked: does the triad hold as N grows (137 → 1000)? The sweep (`--multi-relations`
6/15/30/60, the N knob) delivered a **more decisive finding than a triad-vs-N curve**:

- **N caps at ~147 on single-token CounterFact.** R6 → 137 edits / 6 relations; R15/R30/R60 → **147 / 9
  relations** (identical — 9 is the ceiling). `cf-probe-cap 21500` probes ~all 21k records, so this is not
  probe-limited: only ~147 CounterFact facts have a single-token subject AND single-token both-objects AND
  are base-known by Qwen3.5-4B. The store's single-token-VALUE design is the cap. **We cannot test true
  deployment scale (500–1000+) on this data at all** without multi-token object support.
- **Consequence for productionization:** multi-token object support is a **prerequisite**, not a
  refinement — it is needed both to reach real scale AND because real edited facts have multi-token
  objects. The scale-N question is *gated* on it; there is no point writing serving code for a
  single-token-capped store.
- Two harness issues surfaced: (1) `--multi-relations R` with R > available-base-known-relations **crashes**
  (`ValueError: '<rid>' not in slot_relid`) — must clamp R to the available set; (2) the paraphrase-cohort
  forward **intermittently HANGS on RDNA4** even with cohort/length caps + debug flushes (a genuine
  kernel-sync flake, not a logic bug) — the triad eval needs a watchdog/retry, or the cohort scoring moved
  off the flaky path, before it's a reliable production gate. Added robustness knobs (`CAM_COHORT_CAP`,
  `CAM_PROMPT_MAXTOK`, `ALPHA_SWEEP`) that bound the work but do not eliminate the flake.
- **What we CAN say at N≈137–147:** the triad (§3.16/§3.17) holds — efficacy ~0.81, locality-keep ~0.55,
  generality ~0.82, below-gate 0. But 147 ≈ 137, so this is not a scaling *curve*, just stability in a
  narrow band.

**Verdict: do multi-token object support BEFORE productionizing** — it is the true gate for scale and
realism, and it unblocks the scale-N question that this sweep could not answer.

## 3.17 GENERALITY — the third leg: the edit fires on PARAPHRASES too (2026-07-04)

We had measured efficacy (delivery) and locality but not **generality** (does the edit fire on
*rephrasings* of the fact, not just the exact prompt?). Added the GEN cohort to `--persistent-locality`:
each edit's `paraphrase_prompts`, keyed on the edit's OWN subject (deployment-faithful — a paraphrase is
about the same subject, so it routes to the edit's bank and retrieves strongly), gold = new_tid.

Full triad, K1-on, B=137, hard gate (below-gate 0/137, conf p95 ~5):

| α | efficacy (delivery) | locality (DEP-keep / leak) | **generality (GEN-hit / prior)** |
|---|---------------------|----------------------------|----------------------------------|
| 0 | 0.715 | 0.548 / 0.041 | 0.635 / 0.055 |
| 2 | 0.810 | 0.548 / 0.027 | **0.818** / 0.011 |
| 8 | 0.810 | 0.548 / 0.027 | **0.818** / 0.007 |

- **Generality ≈ efficacy at every α** (0.818 vs 0.810 at α=2), and paraphrase-reversion to the original
  answer is ~0.01. The edit fires on rephrasings as reliably as on the exact prompt.
- **Mechanism: retrieval is SUBJECT-keyed, so generality is free by design.** A paraphrase shares the
  edit's subject → routes to the same bank → retrieves the same value → gets the same logit injection.
  The "0.8–0.9 delivery" is NOT an exact-prompt artifact; the memory generalises across phrasings.
- Closes the standard editing triad (efficacy / locality / generality) for the persistent logit-injection
  + K1 design: **~0.81 efficacy, ~0.55 locality-keep, ~0.82 generality** (n=1 this run; delivery/keep
  carry ±0.15 tap-fit noise, but GEN≈efficacy is structural). Shipped: GEN cohort in
  `eval_persistent_locality`; `CAM_WRITE_AT_READ` now DEFAULT-ON (K1); `CAM_TRIAD_DEBUG` flag.
- *Op note:* the triad sweep intermittently HANGS after the header under K1-on without the debug flushes
  (a GPU-side sync flake on RDNA4, observed single-job — not contention); `CAM_TRIAD_DEBUG=1` (extra
  flushed prints = more sync points) reliably completes it. Known flake, workaround in place.

## 4. Theory connections *(from the 2026 literature pass)*

One-line map: **whitening = the "make quantization error data-independent" half of modern PQ (OPQ/RaBitQ);
disjoint-bank hashing = the "spread load, kill collision" half of learned-hashing / MoE balancing.** Both
of our empirical wins are established theory with cheaper-and-better modern successors — and there's one
native product-key lever we never tried.

- **OPQ → RaBitQ (the transform).** OPQ (Ge et al. 2013) learns a rotation aligning data to the PQ
  sub-codebook axes; our ZCA whitening is a *special case* (decorrelate+equalize variance globally, but not
  rotated to the product-key half-split). The 2024 successor **RaBitQ** (Gao & Long, SIGMOD 2024,
  [2405.12497](https://arxiv.org/abs/2405.12497)) applies a **random orthogonal rotation** after
  normalization → removes the codebook's directional preference → **unbiased, data-distribution-independent
  error bound O(1/√D)**. That is the theorem behind our "whitening made every encoder isotropic and stopped
  collisions." **Zero training** (vs OPQ's alternating opt — bad for a growing store). Ranked for us:
  ZCA whitening (have it) < **ZCA + fixed random Hadamard rotation** (free, RaBitQ guarantee) < OPQ
  (moderate, must retrain as store grows) < neural PQ (QINCo, [2401.14732] — major, overkill).
- **Isotropy IS the quantized-retrieval mechanism** (not cosine-only). RaBitQ's bound depends on
  directional uniformity; anisotropy → dominant eigen-directions over-crowd their cells → collision.
  Isotropizing equalizes per-cell occupancy → min expected collision at fixed cell budget. The discrete
  analog of product-key "usage collapse" (below).
- **Online/streaming whitening generalizes — with a cheap safety.** A whitening/mean-bias fit on a fixed
  population transfers to new items from the same encoder distribution (mean-bias fit once on Wikipedia
  generalizes across 38 models, [2511.11041](https://arxiv.org/html/2511.11041v1)). Failure modes:
  covariance under-sampling early + domain shift → amplified noise in small eigen-directions. Fix:
  **Ledoit-Wolf/OAS shrinkage covariance** (analytic, no tuning) + an **ε eigenvalue floor** (also the
  "don't over-whiten" knob). Periodic **incremental PCA** refit for drift.
- **Don't over-whiten.** Soft-ZCA (ε∈{0.01,0.1}) beats full ZCA, which "collapses the eigenvalue
  hierarchy" ([2411.17538](https://arxiv.org/pdf/2411.17538)); "All-but-the-Top" (Mu & Viswanath, ICLR'18,
  [1702.01417](https://arxiv.org/pdf/1702.01417)) — removing mean + top-few directions captures most of the
  win. ⇒ use **soft-ZCA with an ε floor**, not full ZCA.
- **The lever we missed — native product-key usage balancing.** The ORIGINAL PK paper (Lample et al. 2019,
  [1907.05242](https://arxiv.org/pdf/1907.05242)) hit *our exact collision problem* and fixed it with
  **query BatchNorm**, raising key utilization **25.8% → 80.3%** (ppl 19.8→18.0). This is native, cheap,
  and directly maximizes distinct-slot coverage — a quantization-aware objective, not a cosine proxy.
  Complementary: **ScaNN's anisotropic loss** (Guo et al., ICML'20, [1908.10396](https://arxiv.org/abs/1908.10396))
  weights the query-*parallel* residual more (parallel error is what flips top-k ranking / causes slot
  collision).
- **Editing-memory analogs.** **MEMOIR** (NeurIPS'25, [2506.07899](https://arxiv.org/abs/2506.07899)) =
  our disjoint banks but with *content-derived sparse masks* (a learned router), scaling to thousands of
  edits. **GRACE** ε-ball = *reactive* collision fix (split on collision) — combine with whitening
  (*preventive*). **WISE** = side-memory + router + sharding. None use rotation/quantization-aware keys ⇒
  **whitening+rotation is a genuinely novel, portable upgrade to the editing-memory family.**
- **Anti-recommendation (theory-confirmed):** do NOT use a semantic embedder/retriever or neural PQ for
  keys — semantic clustering is the anti-feature we observed; representation-degeneration / "All-but-the-Top"
  explain *why* those spaces are anisotropic cones. Stay with **whitened input-embeddings + rotation**.

## 5. Open questions & hypotheses

- **H1 (whitening × banks interaction).** Complementary (whitening lifts every B), substitutive (whitening
  at B=1 recovers most of B=32 → *drop the banks*, simpler store), or redundant (banks already saturate
  addressing). Each is decision-relevant.
- **H2 (rotation > whitening).** ZCA + a fixed **random orthogonal rotation** (RaBitQ) beats plain ZCA for
  *quantized* addressing (data-independent uniform-collision bound).
- **H3 (generalization).** Soft-ZCA (+shrinkage covariance +ε floor) fit on an initial subject set holds
  for **held-out** subjects (a real memory grows). Test on a train/test split.
- **H4 (native objective beats encoder tricks) — PROMOTED, likely the biggest lever.** **Query BatchNorm /
  product-key usage balancing** (native PK collision fix, 25.8%→80.3% in the original paper) beats or
  subsumes the encoder-side whitening, because it directly maximizes distinct-slot coverage under the
  store's own quantizer. We never tried it. Cheap to add.
- **H5 (soft not full).** Soft-ZCA (ε∈{0.01,0.1}) ≥ full ZCA (avoid eigenvalue-hierarchy collapse).

## 6. Experimental design — factorial, not one-off A/Bs

Two orthogonal axes now: the **key transform** (encoder-side) and the **native store objective** (query
BatchNorm) — H4 says the latter may dominate, so test both.

**Grid A — transform × disjoint-banks** (delivery @ N=137, 3 reps each):

| transform \ B | 1 | 8 | 32 |
|---|---|---|---|
| raw keys | 0.26 | 0.42 | 0.66 |  ← known
| soft-ZCA whitened | ? | ? | ? |  ← H1
| soft-ZCA + random rotation (RaBitQ) | ? | ? | ? |  ← H2

**Grid B — query BatchNorm (H4), the native lever**, crossed with {raw, whitened} keys at B∈{1,32}. If
BatchNorm alone at B=1 approaches the whitened/B=32 numbers, it's the cheapest, most principled fix and
subsumes the encoder work.

Protocol — **screen cheap → confirm expensive**:
1. Proxy screen (CPU): NN-confusability **+ a quantization-aware metric** = expected **top-k slot overlap**
   under the actual product-key codebook (much closer to the store than raw cosine; = the ScaNN/PK-usage
   target). Rank transforms before spending GPU.
2. Delivery confirmation (GPU): proxy survivors on Grids A/B, 3 reps, N=137 + retention curve.
3. Generalization (H3): fit soft-ZCA (Ledoit-Wolf + ε floor) on a **train split**, evaluate delivery on
   **held-out** subjects.

Sequencing per the lit ranking: **(1) query BatchNorm (H4)** — cheapest, native, possibly biggest; then
**(2) soft-ZCA + random rotation (H2)** on the survivors; banks scale + load-balance monitor throughout.

## 6b. Interaction effects & the combinatorial strategy

OFAT (one-factor-at-a-time) — which we've used throughout — **provably misses interactions**, and we have
direct evidence we've been bitten: **multi-vector keys were killed in a *shared* store (Phase B, 0.14)
before disjoint banks existed** — their failure mode was crowding, which banks fix, so that verdict is
**confounded** and untested-in-context. GTE went cos-0.968-dead → 0.441 under whitening (encoder×transform
interaction). The dangerous class is **two individually-negative levers that combine positive** — invisible
to main effects *and* to mechanism reasoning (that's what makes it "strange"), catchable only by sampling
the joint space.

Full factorial in *delivery* is unaffordable (~6 factors × multi-level × 3 reps × ~3 min on 2 shared cards).
Resolution:
- **Run the full factorial in the CPU PROXY, not delivery.** The quantization-aware proxy (top-k slot
  overlap under the product-key codebook) is seconds/cell → screen hundreds of combinations cheaply,
  promote only promising **and surprising** cells (incl. negative×negative) to GPU delivery.
- **GATE: validate proxy↔delivery correlation first** on the cells we already have (raw × B∈{1,8,16,32}
  = 0.255→0.655 monotone; GTE = 0.000). If the proxy reproduces that ordering, the screen is trustworthy;
  else fix the proxy before trusting any screen. (Garbage-in guard.)
- **Fractional-factorial / screening DOE** (Plackett-Burman / definitive screening) for main + 2-way
  interactions in few *delivery* runs.
- **Mechanism-guided revivals** of OFAT-confounded kills (predictable synergies) + **broad proxy screen**
  as the safety net for the unpredictable negative×negative class.

**Revival list (OFAT-confounded kills to re-test in context):**
- multi-vector keys **× disjoint banks** (crowding co-factor now exists).
- GTE-MaxSim **× whitening × banks** (anisotropy + crowding co-factors).
- repulsion loss **× query BatchNorm / whitening** (all touch slot occupancy).

## 6c. Campaign plan (phased, queued)

- **P0 — GATE (CPU, now):** build the quantization-aware proxy (product-key top-k slot-overlap within
  banks; model query-BatchNorm as pre-top-k normalization) and validate it reproduces the known
  raw-B-sweep + GTE-death ordering. Blocks everything downstream.
- **P1 — combinatorial proxy screen (CPU):** factor grid over {encoder/transform: raw, soft-ZCA,
  soft-ZCA+rotation, GTE} × {key-structure: single, multi-vector} × {query-BatchNorm: off/on} ×
  {B: 1,8,32}. Rank by proxy collision; surface main effects + interactions + negative×negative surprises.
- **P2 — delivery confirmation (GPU, 3 reps):** the proxy top cells + the revival list + suspected
  interactions. Ordered by proxy promise; native lever (query BatchNorm) first per lit ranking.
- **P3 — generalization (H3):** fit transform on a train split, delivery on held-out subjects; Ledoit-Wolf
  + ε floor.
- **P4 — synthesize:** update §3/§5, pick the production key-encoding recipe, decide banks-vs-transform
  (H1 substitutive → simpler store).

## 6d. Phase R — the retrieval-fidelity frontier (NEW, opened by P2)

The addressing question is **resolved**: disjoint banks (B=32) dominate; better keys (whitening/BN) give
≤+0.11 and don't substitute; GTE is revivable but not superior. At B=32 every encoder converges to
**~0.6–0.7**, and B=64 doesn't beat B=32 → the ceiling is **NOT collision**. It's **per-bank retrieval
fidelity**: a subject nearly alone in its bank still delivers only ~0.7. This is the store/value/tap
readout side — untouched all campaign. New research question:

*What caps per-fact retrieval fidelity at ~0.7, and can we push it toward 1.0?*

Candidate levers (never investigated — all store/tap, not keys):
- **Value encoding** — value is `_e([new_tid])`, one mem vector; multi-token / richer value? value capacity?
- **The tap / injection** — 1 tap layer (L24); more tap layers, stronger gate, conf-gate calibration?
- **Readout** — `readout_q` K-slot pool; read-head count per bank; the delta-write β / write fidelity.
- **The frozen-base bottleneck** — maybe ~0.7 is how reliably a single residual nudge flips the argmax at
  all (a ceiling of the *mechanism*, not the store) — needs an isolation test.

**R0 diagnostic (cheapest, foundational):** measure **single-fact fidelity** — write ONE edit into an
empty store, query it; average over edits. If ~0.85 → the floor is tap/value fidelity (gap to 1.0 = the
lever); if ~1.0 → the B=32 ceiling was still residual collision after all. Establishes the true ceiling and
which side to attack. Then sweep the top store/tap lever at fixed B=32.

### Phase-R theory (2026 lit pass) — the ceiling is the SINGLE-SITE INJECTION mechanism, not the store

**~0.6–0.7 is a documented, cross-literature single-site-injection ceiling.** WISE ([2405.14768](https://arxiv.org/abs/2405.14768)),
a *trained gated side-memory* — our exact analog — tops out at reliability **0.70–0.77** on single-fact
edits (locality 1.0). MEMIT's unconstrained single-site update = **0.6565**. Our ~0.66 is that ceiling.
⇒ it's intrinsic to a single additive gated residual edit, NOT our addressing/store. Store tuning can't
remove it; the ceiling-breakers are architectural. Ranked fidelity levers:

1. **Calibrated closed-form gate (GCAV, [2501.05764](https://arxiv.org/pdf/2501.05764)) replacing the learned scalar** —
   *high impact, LOW cost.* Solve per-token for the minimal ε that pushes the object's logit past the flip
   threshold, capped at a fraction of ‖h‖ (a per-token, norm-relative target a scalar gate can't represent;
   +31% rel. over fixed coeffs). A gate-formula swap, no new params. **← do first.**
2. **Project the value onto the OBJECT'S UNEMBEDDING direction** — *high, low.* Spend the nudge on the
   argmax, not generic residual displacement; makes "reach the flip threshold" well-defined.
3. **Multi-layer injection** (~3 adjacent layers at ~65–80% depth) + sweep L24 DOWNWARD (L24 may be slightly
   deep for a 4B model) — *high, moderate.* Single-site has the ~0.7 wall; layer-separated injection is the
   field-standard way past it.
4. **Multi-vector value bank (k>1, softmax-pooled) + subject-final-token keying + KL/norm penalty** —
   *moderate, moderate.* k=1 is a degenerate product-key readout (memory layers pool top-k, [2412.09764]);
   the keying + distribution-preservation constraint are AlphaEdit's 65→99% efficacy levers ([2410.02355]).

Logit-level injection (add the calibrated direction at the final layer) is the guaranteed-to-flip fallback
if residual-space stays capped. R0 should read ~0.7 (confirming the WISE/MEMIT ceiling); then **P-R1 = the
GCAV calibrated gate.**

## 6e. Phase K — key SELF-ADDRESSING (NEW, the last open lever; opened by §3.15 decomposition)

**The problem, precisely.** After the logit-injection arc (§3.14–3.15) the only residual gap is ~15/137
edits that self-retrieve weakly *even alone in their own bank* (B=137, no crowding; not value-norm, not
length, not relation). Root cause, from the code: the **write and read address slots through DIFFERENT
learned projections** —
- WRITE (`store.write` ← `persistent_write`): `slot = _address(to_wkey(key))` (`pk_store.py:168-170`).
- READ (`store.read` ← `persistent_bank`): `slot = _address(read_q[0](q) + head_bias[0])` (`pk_store.py:209`).

`to_wkey` and `read_q[0]` are coupled only *softly*, by the addr-sup InfoNCE during bind (which pulls
`read_q[0](q)` toward `to_wkey(key)`, `pk_store_adapter.py:536`). For a subject whose pooled key lands near
a product-key sub-codebook **top-k boundary**, the two projections fall into *different* product cells → the
read query never selects the slot the value was written to → `conf≈0`. The addr-sup aligns them in
aggregate but cannot guarantee per-subject cell agreement at boundaries. So the fix is to **close the
write↔read addressing gap**, ordered impact×cost:

**Persistent-path-only (no retrain — the disjoint-banks-style cheap class; test immediately on the frozen
ckpt via the `logit_locality.sh` + `CAM_CONF_DIAG` harness):**
- **K1 — WRITE-WHERE-YOU-READ (primary).** Address the persistent write with the *read* query
  `head_query(key)=read_q[0](key)+head_bias[0]` instead of `to_wkey(key)`, so the value lands exactly at the
  slot the read will select → self-match **exact by construction** (modulo the shared query-BN). Value is
  still `to_wval(value)` (tap-compatible; only the slot location moves). New `store.write_at(V, addr_q,
  values)` + `CAM_WRITE_AT_READ=1` routing in `persistent_write`. Expected: below-gate count 15 → the
  ~4-6 genuinely-unreadable floor; gated delivery up; **locality NEUTRAL-or-better** (the value is now at
  the edit's *exact* read-address, so only that query retrieves it → tighter specificity). Predicted the
  single highest-value experiment.
- **K2 — write redundancy** *(scaffolded: `CAM_WRITE_REDUNDANT=1`)*. Write the value to BOTH `to_wkey` and
  `read_q[0]` slots (two delta writes into the same V) — a softer K1 that keeps the trained write-address
  too. Fallback if K1's pure relocation hurts anything.
- **K3 — widen read selection** *(scaffolded: `CAM_READ_SUB_TOPK=N`)*. Grow the read-side `sub_topk`
  candidate pool (`_address(qh, sub_topk=N)`) so a boundary subject's write-slot is more likely to be a
  candidate — WITHOUT changing the final `topk` slots mixed (no added readout contamination). Cheap; watch
  the DEP-leak/keep cost anyway; likely dominated by K1.

**Requires retrain (re-run the bind stage; NOTE — bind is only 1000 steps + 150 tap, so a "retrain" run
is ~the same ~2 min cached cost as any K1/K2/K3 locality run — the retrain tier is cheap here, its only
cost is that it touches the trained ckpt, so K1's persistent-only property is preferred *if it suffices*).
The §3.4 / §4 theory family — pursue only if K1–K3 leave a residual:**

- **K4 — TIE the write-key and factual-read-query projections** *(recommended first retrain lever)*.
  - *Mechanism.* Today `to_wkey` (write address) and `read_q[0]` (factual read query) are **separate**
    `Linear(d_hub,d_hub)` maps (`pk_store.py:87,91`) coupled only by the addr-sup `(a)` InfoNCE, which
    aligns them by **cosine** (`pk_store_adapter.py:571`) — looser than product-*cell* agreement, so
    boundary subjects still split. Tie them: `to_wkey ≡ read_q[0]`. Then for the persistent path
    (key == query == pooled subject) the write address `to_wkey(s)` and read address
    `read_q[0](s)+head_bias[0]` are **identical** — because the factual **head 0 bias is exactly zero**
    (`pk_store.py:104`, "head 0 (factual) has no bias") — so `_address` sees the same vector at write and
    read → same product cell **by construction, for every subject** (the permanent, retrained form of K1).
  - *Change.* `pk_store.__init__`: under `CAM_TIE_WKEY_READQ=1`, set `self.to_wkey = self.read_q[0]`
    (share the Module; drop the separate param). `to_wval` stays independent (values must differ from
    addresses). Assert `head_bias[0]` stays frozen at 0. ~5 lines.
  - *Cost/risk.* One bind re-run. Risk = capacity: read query and write key can no longer specialise
    apart — but for a symmetric key↔key store they *should* be one map (the addr-sup was already forcing
    it). Validate episodic delivery (the QA-cargo read also flows through `read_q[0]`) doesn't regress;
    the persistent subject read is the production target and is exactly what tying optimises.
  - *Expected.* below-gate → the genuinely-unreadable floor (~4–6, the readout/value misses), delivery ↑
    toward the ~0.88 solo ceiling, locality flat. Distinguishes from K1: K1 relocates the value at eval
    for a fixed ckpt; K4 makes the ckpt itself self-consistent (so even the episodic path benefits).

- **K5 — RaBitQ-style rotation before addressing** (§4 theory).
  - *Mechanism.* Insert a **fixed random orthogonal rotation** (fast Hadamard + random sign, or a frozen
    random `R ∈ O(d_hub)`) on the address vector inside `_address`, after the query-BN, *before* the
    sub-codebook scoring. Applied **identically to write and read** it preserves self-match while
    isotropising the space the product cells quantise → data-independent, unbiased cell assignment
    (RaBitQ O(1/√D) bound, [2405.12497]) → fewer subjects sit on a top-k boundary. Strictly better than
    the ZCA whitening we have; **composes with K1/K4** (rotation reduces boundary *count*, K1/K4 fixes the
    write/read *gap*).
  - *Change.* `_address`: `if CAM_RABITQ_ROT: q = q @ self._rot` where `self._rot` is a frozen buffer set
    once at init (Hadamard/random-orthogonal). ~8 lines + the buffer.
  - *Cost/risk — the catch.* The **codebooks were fit (`anchor_keys`) in the UNrotated space**, so a
    rotation applied at eval-only would mis-match them and *degrade* addressing. K5 therefore needs the
    **codebooks refit in the rotated space** → retrain tier (not persistent-only, unlike K1). Zero
    training for the rotation *itself*, but the store must be re-bound with it on.
  - *Expected.* Reduces the *incidence* of boundary subjects (a distributional win), not a
    by-construction guarantee like K4 — so its value is as a **compounding** lever under K4, or standalone
    if tying proves too restrictive.

- **K6 — hard self-consistency addr-sup term** *(targeted loss; use if K4's weight-tie is too restrictive
  for delivery)*.
  - *Mechanism.* Keep `to_wkey` and `read_q[0]` separate but add a training loss that forces their
    product-**cell** distributions to agree (not just their cosine). For each binding compute the soft
    candidate-cell scores (the `cand` tensor in `_address`, pre-top-k) for both `to_wkey(k)` and
    `read_q[0](k)`, form softmax distributions `p_w`, `p_r` over the sub_topk² candidates, and add
    `λ·½(KL(p_r‖p_w)+KL(p_w‖p_r))` to the aux loss. This trains the two maps to select the **same cells**
    at the boundary — precisely the failure mode — while leaving them free to differ elsewhere (more
    capacity than K4's hard tie). Straight-through is unnecessary; the soft cell scores are differentiable.
  - *Change.* Expose the pre-top-k `cand`+`slot` from `_address` (small refactor), add
    `_compute_addr_sup` term under `CAM_ADDR_SELFCONSIST_W>0`. ~20 lines.
  - *Cost/risk.* One bind re-run. Risk = the extra term competes with the delivery/value losses; sweep λ.
  - *Expected.* Between K4 (hard, by-construction) and the status quo (soft cosine): closes most of the
    boundary gap while retaining projection capacity. Preferred if K4 regresses episodic delivery.

**Recommended order (retrain tier):** **K4 first** (cheapest, by-construction, ~5 lines) → if it regresses
episodic delivery, **K6** (soft cell-consistency, keeps capacity) → **K5** to *compose* on top of either
(reduce boundary incidence). Only enter this tier if K1–K3 (persistent-only) leave a residual below-gate
count above the ~4–6 unreadable floor. The unreadable floor itself (values whose object token the readout
can't peak) is a *different* lever — value/readout, tracked separately from Phase K.

**Measurement (same harness, all runs):** `CAM_CONF_DIAG=1` below-gate count + deliverable-weak split;
hard-gated delivery / DEP-keep / DEP-leak at the §3.15 operating point (B=137, C0=em/12). Success = below-gate
→ the unreadable floor, gated delivery ↑ toward the ~0.88 solo-fidelity ceiling, locality flat. n≥3 reps
(±0.15 tap-fit noise); the within-run below-gate count is the low-noise primary signal. **Start with K1.**

## 6g. Phase N — SCALE via variable-length subject binding (the REAL gate; M0-corrected 2026-07-04)

**This supersedes Phase M as the scale priority.** M0 (§3.19) showed the N=147 cap is NOT multi-token
objects (those add ≤+2.7%) but the **per-relation single-subject-length grouping** in
`setup_counterfact_multi`: ~2,936 base-known facts collapse to 147 because each relation keeps only its ONE
largest `(prompt, subject-length)` bucket. The store keys on the subject's last-token / pooled span
(length-agnostic), so the fixed length is purely a **rectangular-bind-batching** shortcut.

- **The fix:** bind across ALL subject-lengths per relation (and drop the `per_rel_min`-at-one-length
  gate), using **length-bucketed sub-batches** in the DocBuilder bind — the exact pattern the eval probe
  already uses (`buckets = defaultdict(list)` by tokenized length). Keys stay last-token/pooled;
  nothing else changes.
- **Headroom:** N 147 → up to ~2,936 (~20×) on Qwen3.5-4B, **no multi-token work**. This is the real
  scale unlock and almost certainly a smaller change than Phase M.
- **Phasing:** **N0** — relax the grouping in `setup_counterfact_multi` to keep all length buckets; measure
  the new N (probe-only). **N1** — DocBuilder length-bucketed bind so training handles the mixed-length set;
  re-run the triad + the scale-N curve §3.18 couldn't run (now with real N). **N2** — generation-coherence
  check at scale.
- **Risks:** (a) more subjects/relation → more per-bank crowding at fixed B; scale `CAM_DISJOINT_BANKS` with
  N (K1 makes crowding cheap). (b) mixed-length bind batches complicate the DocBuilder — length-bucketing
  is the mitigation. (c) the RDNA4 cohort-forward flake (§3.18) recurs at larger N → watchdog first.
  (d) still fix `--multi-relations R > available` crash (clamp R).

## 6f. Phase M — MULTI-TOKEN OBJECTS ([DEMOTED by §3.19] ≤+2.7% supply; realism nice-to-have, not scale)

**Demoted:** M0 (§3.19) showed multi-token objects add only 598 candidate facts (2.7%) and *reduce*
base-known slightly — they are NOT the scale gate (Phase N is). Keep this as a **realism/quality** item
(real answers are sometimes multi-token) for AFTER scale is unlocked, not a prerequisite. The store already
has a **multi-token value subsystem** from an earlier thrust — the work is mostly INTEGRATION into the
editing + persistent + logit-injection path, not a from-scratch build.

**Already built & reusable (do NOT rebuild):**
- **Multi-token subjects** — `EditRecord.subject_tids` is a full token list; pooled/K1 keying already
  multi-token. (Only VALUES are single-token.)
- **Store side** (`pk_store_adapter.py`): `mt_value='perpos'` stores the K object tokens as K
  **position-tagged associations** (key = subject + learned `pos_tag[t]`, value = object-token t);
  `perpos_key` ∈ {additive, gated, codebook, disjoint} resolves per-position addressing;
  `mt_positions`, `pos_tag`, per-position disjoint sub-codebooks all exist.
- **Readout** (`direct_logits`): `readout='linear'` (slot t → answer token t → [B,Kc,V]) and
  `readout='decoder'` (AR teacher-forced transformer head, [B,Kc,V]).
- **Metrics/loss** (`recall_mag.py:56–95`): `_seq_ce`, `_seq_metrics` (exact-match = all-K-correct +
  per-token acc), `_kc`, `_answer_logits` (Kc-position logits, OOM-safe). `--cargo-tokens K` flag.

**Single-token assumptions to LIFT (the integration gaps), by site:**
1. **Records/loader** (`realedit.py`): add `new_ids` / `true_ids` (full token lists) to `EditRecord`
   (currently only single `new_tid`/`true_tid`, = −1 when multi-token → these facts are *dropped* by the
   `single_token_only` filter, `recall_mag.py:605`). Stop dropping them; carry the K-token objects.
2. **Persistent write** (`_persistent_write_val`, `recall_mag.py:1036`): today writes one
   `_e([[val_tid]])`. → write the K object tokens via the **perpos** path (K position-tagged associations
   per subject) — the same primitive the cargo/dict path already uses at bind, exposed for
   `persistent_write`.
3. **Persistent read + score** (`_persistent_preds`/`_persistent_score`, `recall_mag.py:1052–1129`):
   today single `argmax(-1)` at the last position vs `new_tid`. → read the K perpos values, produce K
   answer-token predictions (teacher-forced against `new_ids` for scoring, or AR for generation), score
   with `_seq_metrics` (exact-match over K).
4. **Logit injection** (the §3.14 crack, `recall_mag.py:1099–1117`) — **the one genuinely new piece.**
   Today adds `α·out_proj(bank)@lm_head` at the **single** last position. → make it **multi-position**:
   at each of the K answer positions t, inject position-t's retrieved value contribution. Teacher-forced
   this is K parallel injections; free-running it interleaves with generation. The **conf-gate** (§3.15)
   generalises per-position (each position has its own retrieval conf).
5. **Metric wiring**: delivery/locality/generality become **exact-match over the K-token object** (strictly
   harder than single-token argmax — expect the headline numbers to drop; report per-token acc too).

**Phasing (each a testable increment; bind is cheap ~2 min):**
- **M0** — carry multi-token objects through the loader/records; measure the new N (should jump well past
  147 once the single-token filter is lifted — the actual deployment-scale unlock).
- **M1** — perpos persistent write + teacher-forced multi-position readout scoring (efficacy first, on the
  bigger N). Reuse `readout='decoder'` or `'linear'`.
- **M2** — multi-position logit injection + per-position conf-gate (port the §3.14/§3.16 wins to K>1).
- **M3** — re-run the triad (efficacy/locality/generality) at real N (now unblocked) + the scale-N curve
  the §3.18 sweep couldn't run; add a generation-coherence check (the other original de-risk).

**Risks / unknowns:** (a) per-position addressing is the historically hard part (the thrust notes
`exp#2/#3` "per-position address never resolved" until `perpos_key='codebook'/'disjoint'`) — K1
write-where-you-read should help here too and is worth combining; (b) exact-match over K tokens is a
stricter bar — the ~0.8 single-token efficacy will not transfer 1:1; (c) the RDNA4 cohort-forward flake
(§3.18) will recur at larger N — needs a watchdog/retry in the harness first. **Also fix the found bug:**
`--multi-relations R > available` crashes (`slot_relid.index` ValueError) — clamp R.

## 7. Methodology notes

- **Metric noise:** persistent-sweep delivery at cohort=10 swings ±0.10 run-to-run (GPU tap-fit
  nondeterminism + quantization). **n ≥ 3 reps mandatory.** A single ON run once showed a phantom 0.80.
- Trust the larger-cohort points (N=34/137 cumulative) for direction; the fixed 10-cohort early curve is
  quantized to 0.1 steps.
- Runs require `CAM_NATIVE_GDN=1` (fla segfaults stage-2 on RDNA4). Probe cache
  (`CAM_PROBE_CACHE`) skips the ~21k-record base-known probe → a full run is ~3 min single card.
- Reproducers: `tools/{phasec,rep,mk,globalrep,maxsim}_sweep.sh`, `tools/keyenc_bakeoff.py`,
  `tools/colbert_sep_spike.py`, `tools/gte_precompute.py`.
- GTE-key plumbing (`CAM_GTE_KEYS` + `{subject_tids→vector}` table + `gte_proj`) is reusable to inject
  ANY precomputed key vectors (e.g. whitened input-embeds) into the store — the whitening delivery test
  reuses it directly.

## 8. Log

- **2026-07-03** — Started notes. Consolidated: N=137 ceiling = addressing (#44); disjoint banks fix +106%,
  knee B=32 (#48); GTE-ModernColBERT killed (anisotropy); whitening bake-off (whitened input-embeds win);
  reframed as product-quantization addressing.
- **2026-07-03 (lit pass)** — Grounded §4. Whitening = OPQ/**RaBitQ** special case → upgrade: **soft-ZCA +
  random rotation + Ledoit-Wolf + ε floor** (data-independent O(1/√D) collision bound, zero training,
  streaming-safe). Isotropy IS the quantized-retrieval mechanism. Whitening generalizes to held-out items
  (with shrinkage). **Missed native lever surfaced: query BatchNorm / product-key usage balancing** (orig
  PK paper: utilization 25.8%→80.3%) — promoted to **H4, likely biggest+cheapest**. Disjoint banks =
  **MEMOIR** analog (theory-validated). Anti-rec: no semantic embedders / neural PQ (theory-confirmed).
  Next: **run Grid B (query BatchNorm) first**, then Grid A (soft-ZCA+rotation); start with the CPU
  quantization-aware proxy screen (top-k slot overlap).
- **2026-07-03 (P0 gate + P2 build)** — Proxy GATE **PASSED** (§3.5): quantization-aware proxy (per-key
  slot-overlap load, `tools/pk_proxy.py`) reproduces the raw B-sweep + GTE death → combinatorial screen
  trustworthy. Proxy surfaced whitening≈query-BatchNorm (substitutive) and **empirically confirmed the
  negative×positive class: de-anisotropized GTE (BatchNorm/whitened) = BEST cell** (OFAT killed GTE
  prematurely). Built: query BatchNorm (`CAM_QUERY_BATCHNORM`, native H4 lever, in `pk_store._address`);
  whitened key tables (`whiten_inembed_keys.pkl` 2560-d, `whiten_gte_keys.pkl` 128-d, soft-ZCA ε=0.05);
  P2 delivery grid (`tools/p2_grid.sh`, 6 cells × 3 reps: raw/whitened/BatchNorm/whitened-GTE × banks).
  **P2 BLOCKED on GPU contention** (another agent's `lease-serve-serve` container co-resident on the
  leased card → OOM at model load; gpu-status shows FREE but VRAM occupied — the known hazard). Queued
  behind a GPU-clear waiter (`tools/p2_wait_run.sh`). Next: harvest P2 delivery, confirm/deny H1/H4 + the
  GTE revival, update §3.
