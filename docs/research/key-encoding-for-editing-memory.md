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
