# memory-organ

**Attach a long-term associative memory to a frozen LLM — and carry that same memory to a *different*
frozen LLM through a tiny learned translator.**

A research preview of a mechanism for *base-agnostic* memory: bind facts into a compact memory module,
deliver them into a frozen base model via a zero-initialized gated tap, and transfer the *same frozen
memory* to a second frozen base (different size, different tokenizer, different model family) with a
small affine translator. No base-model weights are trained at any point.

> ### ⚠️ Status & caveats — read first
> - **Research preview, days old, not peer-reviewed.** It may not survive real data or independent
>   reproduction.
> - **The task is synthetic** — a name→cargo dictionary-recall probe, not real-world knowledge. This
>   demonstrates a *mechanism*, not a product.
> - Results are **3-seed** (tight error bars); **AMD ROCm**-developed, CPU/CUDA portability untested.
> - **An AI wrote the implementation** under a human's direction — see
>   [How this was built](#how-this-was-built). What separates this from unreviewed AI output is that
>   every number below has a chance baseline and an ablation, and the three times we were wrong are
>   [in the record](DISCLOSURES.md#corrections-we-were-wrong-three-times).
>
> The full caveat list is in **[DISCLOSURES.md](DISCLOSURES.md)**.

## Origin

This started in conversation, while sorting some Gemini hype from reality. Claude was explaining why you
*can't* just strip layers out of different models and sandwich them into a new base — a model's **hidden
states aren't a shared interface**; each model's residual stream is its own private coordinate system.
Mid-way through outlining where you'd inject a memory ([Memory-as-Context vs Memory-as-Gate](ACKNOWLEDGMENTS.md)),
the words "hidden states" landed, and Pat — meaning only to nudge — said: *"hang on, I remember a project
that was doing something with translating between hidden states."* That project was
[RecursiveMAS](ACKNOWLEDGMENTS.md), and the nudge reoriented everything: the same fact that makes
layer-sandwiching impossible (hidden states don't transfer) means you can't drop a memory built on one
model into another either — **you'd need a translator**, and someone had already shown you could learn
one. Combine that with a [Titans](ACKNOWLEDGMENTS.md)-style memory and you get the question this repo
chases. The full story is in [DIARY.md](DIARY.md#entry-0--origin).

## The result

Two things are demonstrated end-to-end, on one memory store, at honest difficulty. *M* is the
discrimination difficulty (pick the right item out of *M* candidates; chance = 1/*M*).

**1. Capacity.** The naive recurrent memory walls out past M≈8–12. A product-key store *with its
addressing-supervision loss* holds flat through M=128 (held-out recall carry, chance shrinking from
0.125 to 0.008):

| M     | 8 | 16 | 32 | 64 | 128 |
|-------|------|------|------|------|------|
| carry | 0.948 | 0.926 | 0.894 | 0.921 | 0.929 |

**2. Delivery + transfer.** That same store, bound, delivered into a frozen Qwen base, then transferred
to a *different* frozen base via a tiny affine translator (single-token answers, M=8, **3-seed mean**,
`no_memory` = 0.000 throughout):

| stage | result | chance |
|---|---|---|
| Stage-1 carry | 0.950 ±0.003 | 0.125 |
| delivery into frozen Qwen | 0.944 ±0.009 | 0.125 |
| transfer → frozen **Gemma** | 0.942 ±0.020 | 0.125 |
| transfer → frozen **Llama-3.2-3B** (foreign tokenizer + arch) | **0.656** | 0.125 |

Near-deterministic across seeds, and the **cross-family Llama** transfer (a genuinely foreign model —
tiktoken vocab, different architecture and width) passes with a clean 0.000 floor: the translator isn't
riding Qwen-family similarity. The naive store, for contrast, cannot even *bind* the harder M=64 case
that this store delivers + transfers at ~0.92.

> **The honest edge — largely closed.** The table above uses *single-token* answers. **2-token real-word
> answers** were the hard case, and both halves now work: the *store side* — addressing 0.000 → **0.964**,
> same-base delivery 0.486 → **0.883** (disjoint per-position codebooks) — and *cross-base transfer*,
> which a plain affine translator stalled on (0.393) but a **per-position + non-linear translator** lifts
> to **0.812** (~84% of ceiling, no_memory 0.000). Both fixes are the same idea — *go per-position* — at
> opposite ends of the pipeline. It's a strong pass, **not** single-token parity (~0.94), so we call it
> *largely closed*, not solved. The four-experiment hunt is in [RESULTS.md §4](RESULTS.md). Real-knowledge
> data is still untested.

See **[RESULTS.md](RESULTS.md)** for every number with its baseline and the full story including the
[three corrections](DISCLOSURES.md#corrections-we-were-wrong-three-times), and **[METHOD.md](METHOD.md)** for
how the mechanism works.

## How this was built

The research direction — the hypotheses, what to test, the originating idea — is **Pat Carter's**. The
implementation — kernels, math, experiment harness, falsification methodology, and this writeup — is
**Claude's** (Anthropic's Opus 4.x), working as an agent under Pat's direction. We say so plainly; the
full statement and reasoning are in [DISCLOSURES.md](DISCLOSURES.md#how-this-was-built).

## Where this is going

This is the reproducible artifact. The larger ambition — *one canonical memory that any frozen model
can attach to* ("Titans for everyone") — and what is proven versus aspirational is laid out in
**[ROADMAP.md](ROADMAP.md)**.

## Reproduce

> Pure PyTorch (`torch`, `transformers`, `numpy`); models download from their original sources. Run
> from the repo root, either as a module (`python -m cam.<driver>`) or as a file (`python cam/<driver>.py`).

```bash
# capacity ladder (product-key store + addressing supervision)
python -m cam.bind_msweep --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --Ms 8,16,32,64,128

# end-to-end: bind -> deliver into frozen base-1 -> transfer to frozen base-2
python -m cam.recall_mag --store pk --addr-sup-weight 1.0 --M 8 --save-ckpt ckpt/m8.pt
python -m cam.recall_v1  --load-ckpt ckpt/m8.pt --M 8 --base2 unsloth/gemma-3-4b-pt

# multi-token cargo: disjoint per-position store + higher-capacity per-position translator
python -m cam.recall_mag --store pk --readout perpos --perpos-key disjoint --cargo-tokens 2 \
    --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --save-ckpt ckpt/mt.pt
python -m cam.recall_v1  --load-ckpt ckpt/mt.pt --M 8 --cargo-tokens 2 --xlator perpos-mlp \
    --base2 unsloth/gemma-3-4b-pt
```

## License & credit

Apache-2.0 (see `LICENSE`). This work descends directly from RecursiveMAS, Titans, product-key memory,
and relative representations — full credit in **[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)**.
