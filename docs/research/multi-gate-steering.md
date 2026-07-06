# Multi-gate memory-steering — design space (Track 5, #99)

Delivery is `Δlogits = G · α · out_proj(bank)@lmᵀ`. Today `G` is ONE scalar per query (store retrieval
conf, §3.15). This note maps where a **multi-gate** delivery can go, grounded in the Track-5 results and
the 2021–2026 conditional-steering / knowledge-editing literature. The organising idea: **a gate is not an
on/off switch — it is a factor that controls one DEGREE OF FREEDOM of the injection and neutralises one
FAILURE MODE.** More gates = more DOF controlled, not a bigger switch.

## The DOF ↔ signal ↔ failure-mode table

| DOF controlled | gate signal | neutralises | Track-5 status | literature anchor |
|---|---|---|---|---|
| **strength** (scalar gain) | store retrieval conf | neighbour leak (§3.15) | shipped | CAST cosine-θ gate `arxiv 2409.05907` |
| **strength** (dose) | headroom `1−p_base(target)` | confident-fact harm (§3.24) | **margin gate, validating** | CD Adaptive-Plausibility mask `2210.15097`; CAD self-null `2305.14739` |
| **support** (which tokens/dims) | top-k / SAE feature mask | blunt collateral KL (§3.24, KL 24–38) | OPEN — highest leverage | ITI KL=0.27 `2306.03341`; SAE-TS `2411.02193` |
| **scope** (in/out of distribution) | readout recon-error / relation novelty | novel-relation corruption (§3.23, delivers 0 **and** wrecks prior) | OPEN | SERAC scope clf `2206.06520`; GRACE ε-ball `2211.11031` |
| **sign** (add vs suppress) | base↔store disagreement | — | OPEN | DExperts `2105.03023`; contrastive decoding `2210.15097` |
| **arbitration** (which fact) | softmax over addressed banks | N-scale interference (#17) | OPEN | WISE routing `2405.14768` |

## What the Track-5 runs already established

- **Bind-to-true + graded metric ⇒ a real lean**, monotonic in base uncertainty (§3.24): +0.53 @P<0.1 →
  −0.41 @P>0.6 (blunt α=0 tap). Counterfactual bind −0.289 (clean metric sanity).
- **Surgical logit path (tap off, α=2) removes the confident-fact HARM** (−0.41 → +0.014) — aiming the
  delta along the target direction, not the tap's broad perturbation. But α=2 is a *precise override*
  (KL 24–38, ΔP +0.88), not a gentle lean → needs low α.
- **Hard argmax agreement gate FAILS on this eval set** (killed the whole lean, +0.353 → −0.066): the base
  already argmaxes the true token at *low probability*, so "top==target" wrongly reads as "satisfied." The
  literature predicts exactly this — O'Brien & Lewis (`2309.09117`) and CAD show contrast helps under
  disagreement and *hurts where the base was already right*; the fix is a **soft margin** gate keyed on the
  base's probability mass on the target, not the argmax. (This is the run currently validating.)

## The novel multi-gate directions (ranked)

1. **When × Where × How-much, learned jointly.** Each factor is validated *in isolation* — CAST (when),
   ITI/SAE-TS (where), PIXEL `2510.10205` (per-token how-much) — but nothing composes all three for one
   steering signal. A memory push with a relevance trigger + a sparse target support + a per-token gain is
   a clean novelty *and* the concrete fix for the blunt-KL problem (§3.24). **Highest leverage.**

2. **Explicit per-token agreement gate (formalise what CAD/proxy-tuning do by accident).** Our push is
   structurally DExperts' `α·(expert−base)`; those methods only *incidentally* self-null on agreement.
   Making it explicit — shrink δ per token by `1 − sim(base, steered)` / base↔steer KL — turns
   "don't double-push an already-right token" into a guarantee, per token. The **soft-margin gate is the
   scalar version of this**; the per-token version is the generalisation.

3. **Learned MLP router → a CONTINUOUS, CALIBRATED gain.** Fuse [retrieval conf, base entropy/convergence,
   agreement margin, readout recon-error, relation-match] through a small MLP → a continuous gain, then
   temperature-scale so the gain reads as a true trust probability. Every router in the literature
   (Self-RAG `2310.11511`, CRAG `2401.15884`, `2510.01237`) is linear/thresholded and **uncalibrated** —
   calibration is the field's universal blind spot, so this is beatable.

4. **Outcome-supervised gain (Adaptive-RAG applied to steering).** No steering-gate paper trains its gate
   on "did injecting here actually improve the output?" Borrow Adaptive-RAG's `2403.14403` outcome-derived
   labels to supervise the gain — removes hand-tuned α entirely.

5. **Scope gate for §3.23 (SERAC/GRACE imported into steering).** A readout-recon / relation-novelty gate
   (or a GRACE ε-ball around the memory key) that stays inert on relation shapes the readout never trained
   on — converting the documented "novel relation delivers garbage AND corrupts the prior" total-failure
   into a safe no-op. Turns §3.23 from a wall into a caught exception.

6. **Sparse/SAE-basis injection.** SAE-TS solves for a vector that hits the target feature while
   *subtracting predicted spillover*; expressing the retrieved value in an SAE basis and injecting there
   bounds collateral KL by construction — combining editing-locality goals with sparse-steering mechanics,
   which no memory-steering system currently does.

**The composition to build toward:** `δ = router([signals]) ⊙ sparse_mask ⊙ α · value`, i.e. a learned,
calibrated *gain* (DOF: strength) times a sparse *support* (DOF: where) times the value — with a scope gate
as an outer veto. That single object neutralises all three documented failure modes (leak, harm, corruption)
at once. Attribution notes: CAST = Lee et al. (ICLR'25), headline gate-beats-scalar number 96.4%→2.2%
false-activation; ITI KL=0.27; SAE-TS 0.36 vs CAA 0.22 collateral.

## EMPIRICAL correction (Track-5 runs) — sparse support does NOT buy gentleness; α does
The multigate composes and the self-dosing margin gate works (confident bucket +0.065 → −0.035, mean
composite gain 0.80). **But the sparse top-k mask did NOT reduce collateral KL** (buckets ~15/15/9/4,
gated ≈ ungated at α=1). Mechanism: `KL(off‖on)` is driven by the softmax CONCENTRATION MAGNITUDE (set by
α), not by how many logits are perturbed — masking to 16 tokens still lets the target grab the mass after
renormalisation. **The gentleness knob is α, confirmed by the dose-response** (full multigate): α=0.5 gives
the SAME lean as α=1 (ALL ΔP +0.386) at ~half the KL (confident-bucket KL 0.79 vs 3.7) and POSITIVE d_logp
(no tail catastrophes) — a genuine sweet spot, matching Anthropic's feature-steering "sweet spot" finding.
So sparsity governs *where/interpretability/locality*, not magnitude; the gates decide when/whether/how-much
-per-fact, and α sets gentleness. (Curve: α ∈ {0.1, 0.25, 0.5, 1, 2, 4}.)

## How far can the concept be pushed — the ceiling and the walls (2024–2026 lit pass)

**Ceiling (ambitious but buildable), ranked:**
1. **One outcome-supervised, temperature-calibrated gate with a conformal "inject/defer" guarantee** —
   collapse the five hand gates into a single learned head with a formal risk–coverage curve. Guo et al.
   calibration `1706.04599`; selective prediction / learning-to-defer `2506.20650`; Conformal LM `2306.10193`.
2. **Value-function dose controller** — replace hand "headroom" with a controlled-decoding prefix scorer
   (KL-regularised RL value head), provably optimal, composable across rewards, transfers across bases.
   Mudgal et al. `2310.17022`; FUDGE `2104.05218`; GeDi `2009.06367`; PPLM `1912.02164`.
3. **Closed-loop PID/setpoint controller** over the generated sequence — regulate injection gain to a target
   output property with anti-windup + bounded backtracking. `2606.18790`.
4. **Memory-MoE router** — expert-choice routing + score-shape adaptation (LASER `2510.03293`) + load-balance
   loss turns many gated memories into one arbitration layer; the support mask becomes a routing outcome.
   Expert-choice `2202.09368`.
5. **SAE feature-basis gates** — each multiplicand a labeled monosemantic coordinate → auditable ("this
   memory writes only features X,Y"). Scaling Monosemanticity (transformer-circuits 2024); SAS `2503.00177`.

**Walls (cannot be passed):**
1. **Separability / mutual-information ceiling** — a label-free gate's best accuracy = `I(signal; correctness)`;
   where that ≈0 it is at chance. Probe-MI bound `2312.10019`.
2. **Confident-wrong ≡ confident-right without the label** — high-certainty hallucinations share the
   low-entropy signature of correct answers; dose/agreement gates are structurally blind. `2502.12964`,
   `2510.24222`. (This is §3.23's corruption seen information-theoretically.)
3. **Hallucination detection/control is formally impossible in general** (diagonalisation) — feasible only
   under assumptions or with labeled examples. `2401.11817`, `2504.17004`, `2506.06382`.
4. **Calibration/conformal guarantees void under distribution shift / first-encounter OOD** — exactly the
   scope-veto regime. `Podkopaev & Ramdas 2021`.
5. **Every gate signal is spoofable; a memory-router is a new attack surface** (prompt-stealing, forced
   mis-injection, DoS-by-imbalance). Calibration attacks `2401.02718`; MoE prompt-stealing `2410.22884`.
6. **Intrinsic closed-loop self-correction has no free lunch** — a feedback gate helps only when its error
   signal carries information the base didn't already use. Huang et al. ICLR'24 `2310.01798`.

**One-line ceiling:** the multigate can go all the way to a single calibrated, conformal, value-function-driven,
closed-loop, feature-interpretable memory-router — a large, real gain — but it can NEVER separate
confidently-wrong from confidently-right without an external label-bearing signal, because that separation
is information-theoretically absent from the signals the gate observes. The store's own assertion (it was
WRITTEN because the fact needed changing) is that external signal — which is why gating on *store presence*
(Gate A) is load-bearing in a way no base-side signal can be.
