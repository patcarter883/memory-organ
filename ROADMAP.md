# Roadmap

The repository today is a **reproducible research preview** of a mechanism. The ambition behind it is
larger, and we want to be explicit about the gap between the two so nobody mistakes aspiration for
result.

## The north star: "Titans for everyone"

A single **canonical long-term memory** that *any* frozen LLM can attach to through a small, cheap,
learned translator — so that adding durable memory to a model is a lightweight bolt-on, not a retrain.
You'd carry one memory across models, model versions, and even model families, the way you carry a file
across applications.

## What is proven today (see [RESULTS.md](RESULTS.md))

- ✅ A memory can be **delivered** into a frozen base via a zero-init gated tap (MAG).
- ✅ The *same frozen memory* **transfers** to a different frozen base — including **cross-family
  Llama-3.2-3B** (foreign tokenizer + architecture) — through a tiny affine translator.
- ✅ Capacity scales: the product-key store holds flat through M=128 where the naive store walls at ~M=8.
- ✅ The single-token pipeline holds together end-to-end at honest difficulty (M=64), with **3-seed error
  bars** (±0.003–0.020).
- ✅ **Multi-token, store-side**: with disjoint per-position codebooks, a 2-token answer is addressed
  (0.964) and same-base-delivered (0.883) at single-token parity.

...all on a **synthetic recall probe**, on AMD ROCm.

## What is aspirational (not yet shown)

- 🟡 **Multi-token *cross-base transfer* (largely closed)**: a per-position + non-linear translator lifts
  it from 0.393 to **0.812** (~84% of ceiling) — a strong pass, but short of single-token parity (~0.94).
  Closing the last gap to parity is open (see [RESULTS.md §4](RESULTS.md)).
- ❔ Does it survive **real knowledge in real documents** rather than random name→word pairs?
- ❔ Does it hold at the **N-scale** of a useful memory (thousands–millions of facts)?
- ❔ Is the translator **trainable once and reused**, or does each base/task need a fresh fit?
- ❔ Does a **canonical** memory (trained against many models at once) beat a per-donor one? (Our first
  attempt at a canonical hub was falsified and dropped — see [DIARY.md](DIARY.md) Phase 3 — so this is
  genuinely open.)
- ❔ Does any of it run **off ROCm** (CPU/CUDA)? The code is pure PyTorch, so it should — untested.

## Staged plan

**Stage 0 — research preview (this repo).** Reproducible artifact: the harness, the results, the honest
record. *Current.*

**Stage 1 — realism.** Multi-token real-word answers; cross-tokenizer/cross-family transfer bases; the
first real-knowledge probe; multi-seed variance. Turns "mechanism" into "mechanism that survives contact
with real data," or finds where it breaks.

**Stage 2 — scale & reuse.** N-scaling the store toward useful sizes; testing whether a translator
generalizes across tasks once fit; backend portability (CPU/CUDA) verified.

**Stage 3 — the library.** A clean API — `attach_memory(frozen_model, memory)` — with the store,
delivery, and translator behind a small surface; packaged, documented, examples. This is the
"for everyone" artifact. The current flat harness gets reorganized into `store / delivery / transfer`
subpackages on the way here.

## Tracking

Live work is tracked in **[GitHub Issues](../../issues)**, grouped by
**[Milestones](../../milestones)** (`v0.1 — real data & parity`, `v0.2 — scale & reuse`). The stages
above are the narrative; the issues are the actual, current plan. Changes land via pull requests that
reference the issue they close.

## Where help is wanted

The most valuable contributions right now are **adversarial**: reproduce a number and tell us if it
doesn't hold; run it on CUDA (see the portability issue); design a real-knowledge probe that would
actually break the mechanism if it's going to break. The corrections in [RESULTS.md](RESULTS.md) exist
because controls caught our own errors — more controls, from more people, is exactly what this needs.
