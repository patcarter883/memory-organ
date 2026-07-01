# Disclosures

We would rather you judge this work on its evidence than discover its caveats yourself. So here is
everything we think a careful reader should know before trusting a number in this repository.

## How this was built

The research direction is **Pat Carter's**: the hypotheses, the design decisions, the judgment about
what was worth testing, and the originating idea — grafting a Titans-style test-time memory onto a
frozen LLM using a translator concept borrowed from [RecursiveMAS](ACKNOWLEDGMENTS.md).

The implementation is **Claude's** (Anthropic's Opus 4.x), working as an agent under Pat's direction:
the kernels, the math, the experiment harness, the falsification methodology, and this writeup.

We state this plainly because both contributions are real and different in kind — the ideas and the
direction on one side, the execution on the other — and because hiding it would be the dishonest move.
The thing that distinguishes this from unreviewed AI output is not who wrote it, but that every claim
here carries a chance baseline and an ablation, and where we got it wrong, we left the record in (see
[Corrections](#corrections-we-were-wrong-three-times)).

## Scope: what this is and is not

- It is a demonstration of a **mechanism** — that a long-term associative memory can be bound, delivered
  through a frozen base model, and transferred to a *different* frozen base via a tiny translator.
- The task is a **synthetic** name→cargo dictionary-recall probe (single- and multi-token answers,
  randomly assigned so the base cannot guess them — `no_memory` accuracy is pinned near 0).
- It is **not** real-world knowledge, **not** a product, and **not** a benchmark result. "Does this hold
  on real documents and real facts?" is the next question, and it is unanswered.

## Maturity

- This is a **research preview**. The results are days old and have **not been peer-reviewed**.
- They may not survive real data, independent reproduction, or harder adversarial tests.

## Statistics

- The single-token pipeline (carry / delivery / Gemma transfer) has **3-seed error bars** (±0.003–0.020)
  — near-deterministic. The multi-token and cross-family (Llama) numbers are **single-seed**; treat
  small differences there as noise, not signal.

## Hardware and portability

- Developed on an **AMD gfx1201 (RX 9070 XT/9070), ROCm**, inside a specific container.
- The memory/delivery/transfer code is **pure PyTorch** (it never touches the vendored HIP kernels),
  so CPU/CUDA *should* work — but **we have not verified that yet.** Treat non-ROCm as untested.

## Corrections (we were wrong three times)

We left these in `RESULTS.md` on purpose, because the correction record is the credibility:

1. We first reported the memory's capacity "ceiling at M≈3" as architectural. It was **under-training** —
   the bind shows a phase transition that moves right with difficulty; more steps fixed it.
2. We then reported that the product-key store only bought "~1 rung" of capacity. That was an **artifact
   of our own port dropping the store's addressing-supervision loss.** With the loss restored, the store
   binds flat through M=128.
3. We briefly reported multi-token recall as "solved end-to-end" before running the cross-base transfer
   number — it wasn't (0.393). We corrected that, then closed most of the gap with a per-position + MLP
   translator (**0.812**). We are deliberately **not** re-upgrading "0.812" to "solved": it's a strong
   pass short of single-token parity (~0.94), so multi-token is *largely closed*, not solved.

The first two reversals were caught by controls run *before* building on them; the third by insisting on
running the transfer number instead of stopping at the encouraging store-side result — and then on not
re-inflating the fix. We consider the record the point, not an embarrassment.

## Prior art and models

This work builds directly on RecursiveMAS, Titans, product-key memory, and relative representations,
and uses third-party open-weight models (Qwen, Gemma, Llama) as frozen bases. Full credit and citations
are in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). Model weights are not redistributed here.
