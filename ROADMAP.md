# Roadmap

The repository today is a **reproducible research preview** of a mechanism. The ambition behind it is
larger, and we want to be explicit about the gap between the two so nobody mistakes aspiration for
result.

**The focus (settled by the scorecard below).** memory-organ is a **routed bank of cheap, editable,
non-interfering test-time memories** for frozen models — the one axis where a bolt-on *structurally beats* a
monolithic model — and it is now **deployed in a serving engine**. Above-in-context reasoning-*integration*
is a **measured wall** (a co-trained model's job, not a bolt-on's) and is explicitly not the target. The
north star below is the long-horizon version of the *portability* leg of that system.

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
- ✅ **Real-shaped knowledge**: the mechanism holds when facts are natural-language sentences, varied
  relations mixed per document, and multi-token answers — not just a terse `name: cargo` dict.
- ✅ **Knowledge editing**: the memory *overwrites* what a frozen model already believes — it makes the
  base emit a counterfactual (France→Tokyo, 0.996) while suppressing the true prior (1.000 → 0.004),
  validated by a prior-probe gate — **same-base (Qwen) *and* cross-family (Gemma)**.
- ✅ **Soft steering — a graded LEAN, not only a replace (Track 5, [#99](https://github.com/patcarter883/memory-organ/issues/99))**:
  the delivery primitive is an additive gated bias, so the memory can *reinforce* a fact the base already
  weakly holds instead of overwriting it. Bound to the true object it lifts P(true) monotonically in the
  base's uncertainty (+0.53 where unsure → inert where confident); the gentle operating point is a low
  injection gain (dose-response peaks α≈0.25–0.5), and a **learned per-token gate router** over label-free
  signals reaches **98 % of the per-fact oracle ceiling on held-out facts** and rediscovers the dosing law
  (corr +0.94). See [docs/research/multi-gate-steering.md](docs/research/multi-gate-steering.md) and
  RESEARCH §3.24–3.25.
- ✅ **Provenance beats the confidence-separability wall (Track 5)**: on facts the base is *confidently
  wrong* about, any base-side confidence gate is structurally blind (ΔP +0.000 — confident-wrong is
  indistinguishable from confident-right), while a **store-presence gate rescues them (+0.464)** — the
  deliberate write-event is an external label no base-side signal carries. Novelty/prior-art map in
  [docs/research/novelty-positioning.md](docs/research/novelty-positioning.md).
- ✅ **Serving state — decide what to remember + serve it (Track 5, `--serve`)**: a warm in-process memory
  service closing the read/write symmetry — the calibrated router decides *how much* to deliver, a
  **base-uncertainty write gate** decides *what to remember* (store iff `p_base < τ`). Streaming 24 facts it
  stored the base-unknowable ones and the router-gated decode served them back fluently **8/8** (RESULTS §8).
  The write-side is now demonstrated too: a scale-free **rank write gate** (skip iff the object is in the
  base's top-R — skipped 30/30 base-known facts vs 1/24 for the probability gate) and **capacity/eviction**
  (at cap=8/stream=30, evicts 22, survivors still deliver 6/8). Still open: the single-bank interference wall
  (M≈130, [#17](https://github.com/patcarter883/memory-organ/issues/17)) that eviction *bounds* but doesn't
  eliminate, and cross-fact conflict (RESULTS §8).

...all on a **synthetic recall probe** (curated/random facts, single- and 2-token answers), on AMD ROCm.

- ✅ **Where the frozen bolt-on stands vs Titans — measured** ([RESULTS.md §10](RESULTS.md),
  [docs/research/frozen-base-titans-scorecard.md](docs/research/frozen-base-titans-scorecard.md)). On a
  four-axis scorecard against Titans/HOPE's own headlines, a frozen base + distilled cartridge **reaches
  in-context quality** on integration (edit-ripple ≈ RAG) and capacity (NIAH ≈ ICL) but **cannot exceed
  it** — *injection cannot integrate an edit into multi-hop reasoning at all* (KV-append 0.000; the ceiling
  is perturb-vs-recompute), and only *distilling* the base's in-context behavior clears it. It falls
  **below** in-context on distractor-heavy multi-fact context (BABILong).
- ✅ **The one structural win: continual / no-forget by *routing*.** A bank of isolated per-fact cartridges
  + a cheap router preserves unrelated knowledge at **1.00** (vs 0.17 for a naively-composed prefix) with
  **zero cross-fact interference by construction** — a property a single shared-weight model (Titans/HOPE)
  cannot have. This sharpens the north star: the bolt-on's edge is not *better reasoning* but a **routed,
  editable, non-interfering memory bank** alongside a reasoning model.
- ✅ **Deployed in a serving engine — no longer only a harness.** The delivery mechanism runs in a
  tensor-parallel serving stack: a **namespaced, multi-tenant pointer store** delivers exact stored facts
  into a frozen **35B MoE (Qwen3.6-35B-A3B) under TP=2**, over an HTTP memory API (`/cam/*`) with live
  metrics. It is **concurrent-write-safe** — delivery is *rank-authoritative* (rank 0 broadcasts the forced
  tokens so per-replica store drift can't desync the tensor-parallel ranks) — and hardened against runaway
  generations, request-record leaks, and control-op latency. The routed non-interfering memory is an
  operational capability, not only a benchmark row.
- ✅ **The deployed architecture was *selected*, and coexists at scale.** An A/B/C bake-off on real
  CounterFact (N=300) chose the serving store: a **whitened-GTE pointer** (base-agnostic, training-free key
  with false-fire 0.05 at a 0.90 gate; raw GTE is dead at 0.986 nearest-other cosine / 100% false-fire) over
  a lossless **exact-object-token** value — beating the per-base *trained, lossy* incumbent tap. And that
  store holds **48 multi-token objects coexisting** in one read (counterfactual delivery 0.83→0.92 at
  BANKS=1024+bf16, out-of-store false-fire 0.00), a ~4× step past the demo (RESULTS §4, §7).

## What is aspirational (not yet shown)

- 🟡 **Titans' *above-in-context* headline (BABILong-beats-GPT-4) without training the base.** The
  scorecard shows this is a wall for a frozen bolt-on: exceeding in-context reasoning needs the memory
  *co-trained with attention* (from-scratch, à la Titans/HOPE). Open question we are now pushing: how close
  can we get to the Titans headline *overall* — even via distinct bolt-on / side-by-side modules (a small
  co-trained reasoning-memory, a LongMem-style side-net for long context, the routed bank for no-forget)
  rather than one monolith.

- 🟡 **Multi-token *cross-base transfer* (largely closed)**: a per-position + non-linear translator lifts
  it from 0.393 to **0.812** (~84% of ceiling) — a strong pass, but short of single-token parity (~0.94).
  Closing the last gap to parity is open (see [RESULTS.md §4](RESULTS.md)).
- ✅/🟡 **Real *datasets* of facts — VALID editing on ROME CounterFact (Track 1, [RESULTS.md §7](RESULTS.md)).**
  The first attempt invalidated itself (validity gate 0.164 — the eval hard-coded a capital prompt for every
  relation); the gate caught it. Fixed by editing one relation under its *true* prompt: validity gate
  **0.969 (VALID)**, edit-success **1.000**, prior fully suppressed — genuine valid editing on real data.
  Retrieval-conditioned banking + a tap gated on an explicit **store-confidence scalar** (the store's own
  retrieval strength, replacing the null slot's prompt-novelty proxy) make it **local AND generalizing at
  once** — locality drop **−0.008**, generalization **0.91–0.93** (up from 0.61 with the learned sink;
  generalization was never dead — 0.889 edit-only, the old 0.074 was a measurement artifact). **Multiple
  relations in one memory now works** (faithful prefix, `--phrasing counterfactual_multi`): 4
  relation-templates, and a **per-relation confidence-gate EMA** makes it VALID (0.99), delivered (0.92),
  LOCAL (−0.047), generalizing (0.74) — the leak is closed. And **relation DIVERSITY is unlocked**: the
  apparent ceiling was a *tractability* artifact, not base size (a 2.25× bigger base didn't help) — dropping
  the single-token-**subject** filter (allow multi-token subjects; store keys on the last token) edits **6
  semantically distinct relations** in one memory (VALID 1.00, delivered ~0.90, local −0.016, generalizing
  0.65). Requires `CAM_NATIVE_GDN=1` (fla segfaults in stage-2 on RDNA4)
  ([#16](https://github.com/patcarter883/memory-organ/issues/16)).
- ❔ Does it hold at the **N-scale** of a useful memory (thousands–millions of facts)?
- ❌ **Translator reuse — answered NO (fundamental at affine capacity).** A translator fit for one memory
  gives 0.000 on a different memory, and *joint* training on multiple memories still gives 0.002 on a
  held-out one (vs 0.898 for a fresh fit). The affine residual-stitch encodes the specific bank geometry;
  each (base, memory) pair needs its own small fit. A *reusable* translator would need a higher-capacity /
  memory-conditioned architecture — that's the open direction.
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
**[Milestones](../../milestones)**:

- `v0.1 — real data & parity`, `v0.2 — scale & reuse` — the first-cut realism and scale work.
- `v0.3 — real editing at scale` — take knowledge editing from curated facts to a real editing
  benchmark (CounterFact/zsRE) with **locality**, generalization, N-scale, and cross-edit
  interference (Tracks 1–2).
- `v0.4 — reuse & test-time` — the two hardest open directions: a memory-conditioned translator that
  might beat the affine reuse wall (#5 answered NO), and online/test-time binding (Tracks 3–4). The
  test-time direction now has a written plan — the sliding-window experiment ladder in
  [SLIDING_WINDOW.md](SLIDING_WINDOW.md).

The stages above are the narrative; the issues are the actual, current plan. Changes land via pull
requests that reference the issue they close.

## Where help is wanted

The most valuable contributions right now are **adversarial**: reproduce a number and tell us if it
doesn't hold ([REPRODUCING.md](REPRODUCING.md) has the exact commands, [CONTRIBUTING.md](CONTRIBUTING.md)
the report format); run it on CUDA (see the portability issue); design a real-knowledge probe that would
actually break the mechanism if it's going to break. The corrections in [RESULTS.md](RESULTS.md) exist
because controls caught our own errors — more controls, from more people, is exactly what this needs.
