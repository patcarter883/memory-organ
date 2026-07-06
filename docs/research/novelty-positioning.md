# Novelty & positioning — adversarial prior-art assessment (Track 5, #99, 2026-07-06)

An adversarial prior-art search (brief: try to KILL each claim) was run over the 2023–2026 literature.
Verdicts are deliberately harsh. **Load-bearing caveat: several 2026 (2601/2603/2604/2606) arXiv IDs came
from search-summariser output; only the five most load-bearing were verified against arXiv abstracts
(2506.00653, 2506.06609, 2606.27786, 2509.12527, 2601.15324). Re-confirm ANY 2026 ID before a filing.**

## Per-claim verdicts

| # | Claim | Verdict | Closest prior art | What survives |
|---|---|---|---|---|
| 1 | Affine translator ports a frozen memory across model families | **PARTIAL** | LRT Hypothesis `2506.00653` (affine map between hidden states transfers steering vectors); Model Stitching `2506.06609` (affine residual maps port SAEs/probes/steering); vec2vec `2505.12540` (cross-family embedding translation); Prometheus Mind `2601.15324` (memory on frozen LLM, single-model) | only the *object ported* (a whole trained memory, not a vector) × *cross-family* scope |
| 2 | Graded LEAN scaled by `1−p_base(target)`, not a replace | **PARTIAL** | DoLa `2309.03883` (amplify latent knowledge, unconditioned); ITI `2306.03341` (global strength); Memory Injections `2309.05605` (target-specific, fixed strength); TLVS `2606.07647` (per-token entropy-scaled) | the exact retrieval-conditioned `1−p(target)` weld — one obvious step from Memory-Injections + TLVS |
| 3 | 5-factor multiplicative multigate, each factor a distinct failure mode | **NOVEL-as-found** (incremental) | CAST `2409.05907` (1 gate); SHIFT `2606.27786` (learnable mult. gate in RAG); MAT-STEER `2502.12446` (per-attribute gates); Temporal Attractor Steering `2606.20959` (scope × confidence = 2 of 5) | the full 5-factor, failure-mode-disentangled, retrieval-keyed decomposition; field is at 1–2 gates |
| 4 | **Store PRESENCE beats the separability wall** | **PARTIAL — but strongest** | Kalai & Vempala `2311.14648` (calibration ⇒ hallucination lower bound); Information-Lift `2509.12527` (external reference lifts past internal ceiling); Too-Consistent-to-Detect `2505.17656`; impossibility `2506.06382` | **the escape framed as write-PROVENANCE, not retrieved CONTENT** — un-joined in the literature |
| 5 | Learned, OUTCOME-supervised, calibrated router replacing hand gates | **PARTIAL** | Guiding Giants `2505.20309` (learned continuous scale, label-supervised); DSAS `2512.03661` (learned when-vs-how gate); TAG `2604.18206` (heuristic memory gate); Self-RAG / Adaptive-RAG `2403.14403` | only the *outcome/uplift* supervision target + multi-signal *calibration* for injection |

## Honest synthesis
This is **mostly recombination** — every individual mechanism (affine transfer, uncertainty-gated strength,
multiplicative gates, calibrated routers) has priority, which is why 4 of 5 claims are PARTIAL. The system's
defensible identity is **not a single kernel but a coherent thesis:** *deliver a deliberately-written external
memory into a frozen model as a graded, gated LEAN rather than an edit, and make that whole delivery apparatus
base-agnostic.* Claim everything else as **composition/system**, not primitive novelty.

**The one genuinely least-trodden idea — and the sentence to defend in review:**
> The novelty is treating externally-authored **memory presence** as a ground-truth label that gates a graded,
> cross-model-portable **lean** on a frozen base — not the affine map, not the gates, not the calibrated
> router, each of which has priority.

Every "escape from the confidence wall" paper conditions on retrieved **content** (`I(Y;E)` over evidence text).
No paper found frames the escape as the **provenance/presence of a deliberate write** — the gate keying on
"a memory was written here on purpose" as a label-bearing *event*, prior to and independent of reading its
content or any base-side confidence. That specific weld of "internal confidence can't separate confident-wrong
from confident-right" + "the write-event IS the external label" is un-joined.

## Strategic implication — the experiment that turns the framing into a RESULT
Claim 4 is currently a *framing* of two proven halves. To make it a *contribution*, demonstrate empirically that
**provenance-gating rescues a confidently-WRONG base where no label-free confidence gate can:**
- Construct facts where the base is *confidently wrong* (low entropy, high `p_base(wrong)`, `p_base(true)≈0`).
- Show a base-side confidence/entropy gate is at chance on these (it cannot fire — the base looks confident).
- Show the store-presence gate (Gate A) fires and rescues them, because the memory was *written* for exactly
  these — the write-event carries the information the base-side signals provably lack.
- Pair with the §3.23 novel-relation *corruption* case (scope veto) to show the same signal also knows when
  NOT to fire. This is the money experiment; it is the only one that elevates Claim 4 from reframing to theorem-
  backed result, and it directly exercises the separability wall the search confirmed is real.

## Paper shape (if the router generalises and the rescue experiment lands)
Contribution = the **thesis + provenance framing + the empirical rescue** (Claim 4 as the spine), with the
multigate (Claim 3), the learned outcome-supervised calibrated router (Claim 5), and cross-family memory
portability (Claim 1) as the *system* that instantiates it. Positioned against LRT/Model-Stitching (transfer),
CAST/SHIFT/TAS (gating), Kalai–Vempala/Information-Lift (the wall). NOT claimed as primitive novelty on any of
those axes — claimed as the first system to make external-memory *presence* the gate that beats the wall.
