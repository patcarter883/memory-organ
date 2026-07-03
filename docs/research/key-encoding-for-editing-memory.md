# Research notes — key encoding for a quantized associative editing-memory

**Status:** active · **Tracks:** issue #19 (Track 4, persistent/online memory) · **Started:** 2026-07-03

Living research log. Newest findings at the top of each section; the running Log at the bottom is
chronological. Numbers are 3-rep means unless noted (the delivery metric noise is ±0.10 at cohort=10, so
single runs mislead — see Methodology).

---

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

## 4. Theory connections *(to be filled by the literature pass — placeholder)*

- **Optimized Product Quantization (OPQ, Ge et al. 2013) & learned rotations** — align keys to the PQ
  sub-codebook axes to minimize quantization error. Likely the principled version of our whitening win.
- **Isotropy for quantized/hashing retrieval** — does isotropy help PQ/LSH specifically, or only cosine?
- **Online/streaming whitening** (shrinkage / Ledoit-Wolf / online PCA) — does a whitening fit on an
  initial population generalize to new items in a growing memory?
- **Learned hashing / LSH** — could a learned router beat our stable-random hash bucketing?
- **Editing-memory key encoding** (GRACE ε-ball, MEMOIR disjoint sparse masks, WISE) — what's portable.

## 5. Open questions & hypotheses

- **H1 (whitening × banks interaction).** Complementary (whitening lifts every B), substitutive (whitening
  at B=1 recovers most of B=32 → *drop the banks*, simpler store), or redundant (banks already saturate
  addressing). Each is decision-relevant.
- **H2 (rotation > whitening).** A learned rotation aligned to the product-key sub-codebooks (OPQ-style)
  beats generic ZCA whitening for *quantized* addressing.
- **H3 (generalization).** Whitening fit on an initial subject set holds for held-out/new subjects (a real
  memory grows). If not → need streaming/shrinkage covariance.
- **H4 (objective).** A quantization-aware separation objective (minimize expected top-k slot overlap)
  beats optimizing raw cosine separability.

## 6. Experimental design — factorial, not one-off A/Bs

Primary grid (delivery @ N=137, 3 reps each), key-transform × disjoint-banks:

| transform \ B | 1 | 8 | 32 |
|---|---|---|---|
| raw keys | 0.26 | 0.42 | 0.66 |  ← known
| whitened (ZCA) | ? | ? | ? |
| learned rotation / OPQ | ? | ? | ? |  ← pending H2

Distinguishes H1 (read the whitened row vs raw row) and, with row 3, H2. Protocol: **screen cheap →
confirm expensive**:
1. Proxy screen (CPU): NN-confusability **+ a quantization-aware metric** (expected top-k slot overlap
   under the actual product-key codebook — closer to the store than raw cosine). Rank transforms first.
2. Delivery confirmation (GPU): run proxy survivors on the grid, 3 reps, N=137 + retention curve.
3. Generalization: fit the transform on a train split, evaluate on held-out subjects (H3).

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
  reframed as product-quantization addressing (OPQ). Launched literature pass on OPQ / isotropy-for-PQ /
  online whitening / learned hashing / editing-memory key encoding. Next: fill §4 from lit, then run the
  §6 factorial (whitening-delivery grid) starting with the proxy screen + the whitened-input-embed row.
