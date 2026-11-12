# M2 pre-registration, part 1: comparative positioning bars against LoRA and RAG (E5m, E6c, E6(k), E6s)

**Date: 2026-11-12. This file is fixed before any M2 measurement.** No M2 arm has been trained or evaluated, and no
M2 dev or test number exists. M2-0 (`experiments/m2_0/`) is the feasibility track; it produces timings and parity
checks, not E-numbers. The future full `experiments/m2/PREREG.md` must incorporate this file **unchanged**. Any
change to it, before or after launch, is listed in that file as a deviation, with its date, its reason, and whether
any M2 number had been seen when it was made. The post-review changes of 2026-11-12 (§10) were made before any M2
measurement.

Design references: DESIGN.md §6.A.3 (R1–R3), §6.A.4 (baselines), §6.A.6 (cross-host), §6.A.7 (E-criteria, the
operating point and its zones), §7.2 (costs), §8 (M2 row). Bar machinery: `experiments/m1c/PREREG.md` §4 and
`ngram_memory/eval/stage1/bars.py`.

## 0. Why these bars

The owner's position (2026-11-12): head-to-head raw gain against LoRA and RAG is the wrong yardstick.

- LoRA buys its gain by moving shared weights, so it can forget catastrophically.
- RAG buys its gain with context-window tokens on every request.
- The cartridge is meant to do neither: it is bit-exact when off, and it adds no context.

So the useful questions are these:

- **E5m.** Does the cartridge beat LoRA **at equal retention drift** (the measurable part of "equal forgetting")?
- **E6c.** How much retrieved context does the cartridge **replace**?
- **E6(k) and E6s.** Does it still **add** on top of retrieval at every budget, and on top of retrieval plus counts?

The proxy numbers that motivate the grids below come from M1b and M1c (Qwen2.5-0.5B, Split A). They are not M2
numbers.

| Proxy fact | Value | Endpoint | Where it is used |
|---|---|---|---|
| LoRA all-linear r16, equal tokens, test | E1 +8.16%; the M1b table reached 0.089 of it | all-target E1 (M1b had no non-dup primary) | old E5, now the report-only raw-gain ratio |
| LoRA's retention cost, dev | R1 +2.69%, R2 +11.96%, so share ≈ 24 (R2 binds; KL not recorded) | retention | α_post grid (§3.1) |
| M1c OP's retention share | 0.027 (R1 binds) | retention | the scale of the matched point |
| RAG-4k, test | +8.17% [3.93, 12.85] | all-target E1 | budget grid (§4.1) |
| RAG-4k + table, test | +8.78%, i.e. +0.61 points | all-target E1 | E6 at k = 4096 |

Neither LoRA nor RAG was re-run at M1c's operating point.

**Scale of the matched point.** The ΔNLL terms of the share (R1, R2) are first-order in α_post: the base is not
stationary along the adapter direction on WikiText or stdlib, so their directional derivative is generically
non-zero. Only the KL term is second-order. So the α_post needed to bring LoRA's share from about 24 (α_post = 1)
down to t ranges from about t / 24 (linear) to its square root (quadratic): for t = 0.027 from about 0.0011 to
0.034, and at the floor t = 0.01 from about 0.0004 to 0.02 (est., unverified). The α_post grid (§3.1) therefore
reaches 0.01, and a pre-registered per-adapter downward extension reaches 1e-4, followed by a per-adapter
refinement. M1b's RAG-4k gain is about 11× the table's (both all-target), so the cartridge's context equivalent is
expected near the bottom of the budget grid. Two resolution budgets, 64 and 128, are therefore added (§4.1).

## 1. Scope, and what these bars decide

- **They are comparative positioning bars.** They decide **how the cartridge is described**: "beats LoRA at equal
  retention drift", "saves retrieved context", "additive to retrieval" and "adds to retrieval and counts".
- **They do not override the operating-point decision.** That decision is the zone table of DESIGN §6.A.7 (E1, R1,
  R2 at the one operating point), with the gates, C1, NGI and REG as the M2 PREREG fixes them. A KILL here never
  stops the host, and a PASS here never rescues a host the zone table stops.
- **OP is taken as given.** OP is M2's operating point, as the full M2 PREREG selects it on dev. Nothing in this
  file changes how OP is selected, and OP gets no extra tuning for these bars. In particular, the cartridge gets no
  post-hoc scale, so its "frontier" here is the single point OP. That is conservative toward the cartridge.
- **Primary endpoint.** Every bar is judged on E1, Split A by file (DESIGN §6.A.1–2), at the fixed context
  C = 2,048 in-file tokens. Where Split-B (E2) scoring of the same conditions exists, it is reported, with no bar.
- **What "forgetting" means here.** The matching is on R1/R2/KL drift (near-zero, mostly in- and near-distribution
  NLL/KL), with an R3 capability guard and R4 out-of-distribution retention reported. Generation-level forgetting
  (instruction following, chat/tool format, HumanEval/MBPP) is not measured in M2 for either arm (§9, disclosure 7).
  The licensed sentence therefore says "equal retention drift (R1/R2/KL)", not "equal forgetting".

## 2. Shared definitions (M1c's names, used unchanged)

- **E1(X)**: the relative NLL reduction (%) of condition X against the base, at C = 2,048.
  - It is read on the targets whose ending 16-gram does not occur in the train files (`paired_nondup`, the
    **primary**).
  - The all-target E1 (`paired`) is reported next to every primary number.
  - "Points" are percentage points of E1. DESIGN's "0.5% E1" means 0.5 points.
- **R1, KL, R2** as in M1c PREREG §4.
  - R1 = relative ΔNLL (%) on WikiText-2 test.
  - KL = mean per-token KL(base ‖ cond) on R1.
  - R2 = the token-weighted relative ΔNLL (%) of R2 (stdlib) and R2-ML.
  - Both arms are scored on the identical retention sets and token counts that the M2 PREREG fixes for the OP.
- **Retention budget share (the ruler, used by E5m)**: `share(X) = max(R1⁺/0.2, R2⁺/0.5, KL⁺/0.005)`, with
  t⁺ = max(t, 0). This is `bars.share` with `bars.BARS`.
  - The budgets are M1c's (0.2%, 0.5%, 0.005 nats), even though DESIGN's M2 R2 bar is +1.0%. Reason: shares stay
    comparable with M1c's OP share (0.027) and with the proxy pilot (§3.6). The share is a ruler, not a bar.
  - Consequence: M1c's identity "share ≤ 1 ⇔ within the bars" does not hold for M2. An M2 OP with R2 up to +1.0% is
    within its own bars at a share of up to 2.
- **Share floor**: `s_floor = 0.01` (`bars.S_FLOOR`). A matched threshold is `t = max(share(OP), s_floor)`, as in
  `bars.matched`.
- **Within the M2 OP bars (the bar, used by E6s)**: the retention bars the M2 PREREG fixes for the OP; per DESIGN
  §6.A.7 these are R1 ≤ +0.2%, KL ≤ 0.005, R2 ≤ +1.0%. Implemented as `bars.within(p, M2_BARS)` with
  `M2_BARS = dict(r1_max_pct=0.2, r2_max_pct=1.0, kl_max=0.005)` (or the M2 PREREG's values, if it fixes others
  before any M2 measurement). M1c's `bars.BARS` (R2 ≤ +0.5%) is not used as a bar anywhere in this file.
- **Paired-difference CI**: `E1(A) − c·E1(B) = −100 · Σ[(a − base) − c(b − base)] / Σ base` over the primary
  targets, with a file-cluster bootstrap of 10,000 resamples, seed 0, and 95% percentile intervals (`bars.diff`,
  `bars.gain`).
  - When A and B come from different evaluation stages, the runner checks that the target keys, the primary mask,
    the base NLL and the file groups are identical, and fails the stage otherwise (M1c §4).
  - Retrieved context is extra prefill before the window and does not change the scored targets, so RAG conditions
    share the base's targets. This identity is checked, not assumed.
- **Joint resampling (implementation fixed here)**: every quantity built from several conditions (k_eq, k′,
  tokens saved, s_eq, interpolated values, frontier summaries) uses the **same** 10k file resamples for all of them.
  - `metrics.cluster_bootstrap` returns only percentiles, so a new helper draws the indices once, exactly as
    `cluster_bootstrap` does: over the shared file order from `per_group_sums` (k files), `default_rng(0)`, and
    `integers(0, k, size=(m, k))` in the same blocks of `m = max(1, min(10_000, 4_000_000 // k))` resamples.
  - It returns the per-draw (D, B) sums, and so the per-draw E1, of each condition on those draws.
  - Check: each condition's percentile CI from these draws must equal `bars.gain`'s CI for it (to float
    tolerance), else the stage fails.
- **"Frontier"** of a family = its points in (share, dev E1). The **upper hull** is the subset of points that no
  other point beats on both coordinates (lower or equal share, and higher dev E1).
- **Intervals and multiplicity.** Every CI is a 95% percentile interval, i.e. 2.5% one-sided per directional claim,
  with no family-wise correction (§8 lists the comparisons).

## 3. E5m: LoRA at matched retention drift

E5m replaces E5 as the primary LoRA comparison. Old E5 (cartridge E1 gain / LoRA all-linear r16 gain at equal
tokens) is kept as the report-only **raw-gain ratio** (§3.5).

### 3.1 The LoRA frontier (trained on the same data as the cartridge)

**Data.** Every LoRA run uses the cartridge's training data:

- the same Split-A train files at the same commit, in the same document format (`# file: <path>` header,
  EOS-separated);
- the same training windows, visited in the same seed-0 order, with the same token budget as OP's training (equal
  tokens).
- If LoRA cannot fit OP's window length on the host, the same tokens are re-cut into shorter windows in the same
  order (M1b precedent: 1,024-token windows). This is disclosed.
- LoRA never sees more tokens than OP. It never uses dev, test, near-duplicate files or any retention set.

**Objective.** Next-token NLL, AdamW, linear warm-up then cosine decay (`baselines/lora.py`), B zero-initialised.
The KL-anchored variant (tier T3) adds the cartridge's own anchor term: the same anchor sets, the same λ_near and
λ_chat, and the same top-64 targets.

**Numerics.** The LoRA branch is computed in fp32 and added to an fp32 upcast of the base projection's output:
`y = (base(x).float() + α_post · (lora_alpha/r) · (x.float() @ A.T @ B.T)).to(bf16)`, one cast. α_post is an
eval-time multiplier on the scale; adapters are never merged into weights (the routed experts are packed int4 and
cannot be merged). This keeps the small-α_post region from being rounded away inside the branch; the final bf16
cast remains, which the noise-floor check below guards.

**Grid.**

| Factor | Values |
|---|---|
| Target-module set | **attn+GDN**: attention and GDN projections, all layers (DESIGN §6.A.4 "LoRA r=16").<br>**dense-linear**: attn+GDN plus every dense MLP / shared-expert projection, router and `shared_expert_gate` frozen, no routed experts. This is what `baselines/lora.py` wraps today. On a dense host, dense-linear is all-linear.<br>**all-linear**: dense-linear plus the routed experts' projections, router frozen (DESIGN's "LoRA all-linear r=16" on the MoE host). |
| Rank r | {1, 2, 4, 8, 16}, with lora_alpha = 2r as in M1b |
| lr | a 3-point log grid {0.5, 1, 2} × 2e-4 (M1b's lr). **In T0 all three lrs are trained to full length**, so all three enter the frontier. T1 and T2 use one lr: the lr of T0's dev-best point in Q with α_post > 0 (§3.3; selected at matched share, not at α_post = 1); if Q holds no such T0 point, the lowest lr. T3 uses the lr of the configuration it anchors. The M2 PREREG may re-centre the grid before any M2 measurement. There is no 100k-token lr pilot for LoRA. |
| Training-length snapshots | adapters saved at 1/4, 1/2 and 1 of OP's token count. An **adapter** below means one (module set, rank, lr, snapshot, anchors) combination. |
| Post-hoc scale α_post | W + α_post · (lora_alpha/r) · BA, with α_post ∈ {0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}; α_post = 1 is the trained adapter. This is the continuous analogue of the count weight λ. It is eval-only: one evaluation per (adapter, α_post) on dev, R1, R2, R2-ML. α_post = 0 is the base, with share 0 and E1 0, and always belongs to the frontier. |
| KL anchors | off (T0–T2); as in the cartridge recipe (T3) |

**Downward extension (pre-registered, dev only, per adapter).** For each adapter separately: if none of its grid
points with α_post > 0 has share ≤ t, evaluate it at α_post ∈ {0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001}, in
that order, stopping at the first point with share ≤ t. The floor is α_min = 1e-4. Adapters that reach the floor
without a point with share ≤ t are recorded with their smallest share s_min and their dev E1 at α_min.

**Refinement (pre-registered, dev only, per adapter, eval-only).** For each adapter that brackets t (it has a
point with share ≤ t, and the next larger evaluated α_post has share > t; if share is not monotone in α_post, the
bracket is taken at the largest α_post with share ≤ t): bisect in log α_post between the two bracketing values
until a point has share ∈ [0.9 t, t], or for at most 3 extra points. Refinement points join the frontier.

**Noise-floor check (pre-registered, dev only).** Before any LoRA point is scored, base is scored twice on dev, R1,
R2 and R2-ML (two independent runs, same code path as the LoRA points with the adapters disabled). The repeat noise
of each retention quantity (R1, R2 parts, KL) is the absolute base-vs-base difference.

- A LoRA point all of whose retention differences from base (R1, each R2 part, KL) are within that repeat noise is
  flagged "below the noise floor" and excluded from Q. The exclusion is recorded with the point.
- Over α_post ≤ 0.1, E1(α)/α and the leading share term per α must vary smoothly per adapter (no sign flips or
  jumps > 3× between neighbouring grid values). Violations are recorded and the affected adapter's points below the
  violation are flagged the same way.
- If base's repeat noise is exactly 0 (deterministic kernels), no point is flagged by the first rule.

**Tiers.** The full grid is large, so it runs in this order. A tier counts as "run" only when every adapter in it
has its grid, extension and refinement points with dev E1 and a share.

| Tier | Contents | Status |
|---|---|---|
| **T0 (minimum)** | dense-linear, r ∈ {4, 16}, all 3 lrs at full length, 3 snapshots, full α_post grid with extension and refinement, no anchors. It contains **LoRA@raw** (§3.3). | **required for E5m to be decidable** |
| T1 | dense-linear at r ∈ {1, 2, 8}; attn+GDN at r ∈ {1, 2, 4, 8, 16} | widens the frontier |
| T2 | all-linear (routed experts, router frozen), r ∈ {4, 16}; fallback r ∈ {1, 4} if r = 16 does not fit (§3.6), which counts as "T2 run", labelled with its ranks | needed for an unqualified "beats LoRA" (§3.4) |
| T3 | KL-anchored variant of the T0–T2 (module set, rank, lr) whose points include the highest dev E1 among points in Q with α_post > 0 (§3.3), with the same snapshots and α_post grid, extension and refinement | needed for an unqualified "beats LoRA" (§3.4) |

### 3.2 Retention measured on LoRA points and on OP

- **R1, KL, R2 → share**, as in §2. These are the only inputs to the matching. Measured on every point.
- **R3, capability (guard).**
  - R3(X) = mean accuracy (points) over MMLU, ARC-Challenge and HellaSwag, 1k items each, forward-only
    log-likelihood (`eval/stage1/capability.py`), with paired McNemar against base.
  - It is measured on OP, on base, on the LoRA candidates of §3.3 in the order given there, on LoRA@equal-gain and
    on LoRA@raw. It is not measured on every grid point (about 300k tokens per point).
- **R4, out-of-distribution retention (new; report-only).** Two components, each the same token count as R1, fixed
  in the M2 PREREG:
  - **R4-code**: non-Python source (C/C++/CUDA, Rust, JavaScript/TypeScript, equal thirds) from permissively
    licensed repositories unrelated to the project.
  - **R4-chat**: chat-template conversations, multi-turn, with some non-English turns, scored on assistant turns.
    The assistant turns must be human-written or generated by the base model itself (never by another model, whose
    outputs would measure fit to that model rather than forgetting). The cartridge reads rows on these too: its
    history resets only at EOS, not at `<|im_end|>`.
  - Each component is disjoint from every regulariser, anchor, replay and R1–R3 set.
  - **Primary R4 statistics**: mean KL(base ‖ cond) and top-1 agreement with base, per component. ΔNLL % is
    reported beside them.
  - **Measured on**: base, OP, LoRA@matched, LoRA@equal-gain, LoRA@raw, and every α_post = 1 full-length adapter of
    every tier run.
  - This file fixes the composition rule only. The M2 PREREG names the sources and their sha256 before any M2
    measurement.

**Why R3 is a guard and not part of the share:**

1. Adding a term would change M1c's share definition, so OP's share would stop meaning what it meant at M1c.
2. MC accuracy on 3k items is coarse. The paired SE of a difference is about 0.4 points (est., from about 5%
   discordant items), so a share term would be noise-dominated.
3. Forgetting of capability is exactly the owner's concern. A guard keeps "equal retention drift" from being
   satisfied on perplexity while capability is lost.

The 1-point tolerance is about 2.4 SE (est.). It removes a LoRA point only for a clear capability loss beyond the
lower of OP's and base's R3, and it lets LoRA be up to 1 point worse than that before it does. At the small α_post
where LoRA@matched is expected, the guard is likely near-vacuous; its real work is at LoRA@raw and LoRA@equal-gain,
where R3 is reported (§3.5).

**Why R4 is reported only:** no budget for it has been calibrated, and adding it to the share would also redefine
OP's share. R4 does constrain the wording, though (§3.4).

### 3.3 Selection (dev only; frozen before the test read)

- **Qualified set** Q = all LoRA points (every tier run, every snapshot, grid, extension and refinement α_post,
  plus base), not flagged below the noise floor, with share ≤ t = max(share(OP), s_floor).
- **LoRA@matched**:
  - Take Q's points in descending dev E1 (ties: smaller share, then smaller r, then smaller α_post, then the earlier
    snapshot).
  - Measure R3 point by point, and take the first point with R3 ≥ min(R3(OP), R3(base)) − 1.0 point (one-sided).
  - Base always passes this test, so the search ends.
  - Record every candidate tried, with its R3, and why the search ended (a LoRA point passed; or base was reached
    because every LoRA point in Q failed the guard, or had dev E1 ≤ 0, or Q held no LoRA point).
- **LoRA@equal-gain** (report-only): the grid point with the smallest share among the points with dev E1 ≥ dev
  E1(OP). Ties: higher dev E1, then smaller r. Record its R3 and R4. If no point reaches E1(OP), record "LoRA does
  not reach OP's dev gain within the grid".
- **LoRA@raw** (report-only, old E5): r = 16, α_post = 1, full equal-token snapshot, middle lr (2e-4, M1b's), in the
  largest module set run: all-linear if T2 ran at r = 16, else dense-linear. Its module set is named in every
  report line. On a dense host, dense-linear is all-linear, so old E5 is comparable with M1b. On the MoE host without
  T2 at r = 16, the ratio is labelled "raw-gain ratio vs dense-linear r16 (old E5's all-linear arm not run)", and
  the same label carries over to DESIGN §6.A.6's cross-host fraction and M3's cross-host check.

### 3.4 Test bar (one read of each selected configuration)

D5 = E1(OP) − E1(LoRA@matched) on test, a paired difference with its CI. The rows are checked in this order; the
first that applies is the outcome.

| Outcome | Rule | Description it licenses |
|---|---|---|
| **KILL** | LoRA@matched is not the base, and −D5 (LoRA@matched minus OP) has CI lower bound > 0 | "LoRA dominates even at equal retention drift". The cartridge case then rests only on OFF-exactness, reversibility, per-commit cost and servability (the old E5 KILL consequence). It does not stop the host. |
| **no LoRA gain** | LoRA@matched is the base, or its test E1 ≤ 0 | "LoRA yields no gain at OP's retention drift (<reason>: no adapter reached share ≤ t above α_post = 1e-4, smallest LoRA share s_min; or the R3 guard; or no qualified point with positive E1)", followed by OP's E1 with its CI. It does **not** license "beats LoRA". |
| **PASS** | E1(OP) > 0, D5's CI lower bound > 0 | "beats LoRA at equal retention drift (R1/R2/KL)". Labelled **"clearly"** only when the CI lower bound of E1(OP) − 1.5 · E1(LoRA@matched) is > 0. The ratio E1(OP) / E1(LoRA@matched) is reported (it is defined here, since E1(LoRA@matched) > 0 on test). |
| neither | otherwise | "not resolved against LoRA at equal retention drift" |

**Why 1.0× and not Bar B's 1.5×.** Bar B's 1.5× was a complexity premium over a count-LM that is trivially cheap and
servable. At equal retention drift, the cartridge's other properties (OFF-exact, reversible, servable, cheap per
commit) are already in its favour, so a resolved win is enough for the description. Because E1(LoRA@matched) > 0 in
the PASS row, a CI lower bound > 0 already implies a point ratio > 1. The 1.5× premium earns the word "clearly",
through its own CI.

**Mandatory qualifiers on the description:**

- **Families run.** An unqualified "beats LoRA at equal retention drift" needs T0–T3 run. Otherwise the sentence
  names the families that ran, for example "beats dense-linear LoRA (r ≤ 16, no anchors) at equal retention drift".
  If T2 is infeasible (§3.6), every PASS says that routed-expert LoRA, the family most associated with knowledge
  storage on an MoE host, was not compared. The qualifier applies to PASS and to "no LoRA gain". A frontier from a
  subset of families lies under the full frontier, so a KILL from any subset stands.
- **R4 (exploratory).** Per R4 component and per statistic (KL, ΔNLL): the paired difference
  stat(LoRA@matched) − stat(OP), with a document-cluster bootstrap (R4 documents resampled with replacement),
  10,000 resamples, seed 0, 95% percentile interval. If the interval lies wholly below 0 (LoRA@matched drifts less),
  the sentence adds "on R1/R2/KL; on <component> LoRA@matched drifts less". If it lies wholly above 0, it adds "on
  <component> LoRA@matched drifts more". These qualifiers are uncorrected (§8).
- **R3.** R3 is reported for OP, LoRA@matched, LoRA@equal-gain and LoRA@raw.
- **No R3 guard.** If R3 could not be measured, the selection is made without the guard, test-read and judged, and
  every outcome is labelled "(no R3 guard)", for example "KILL (no R3 guard)". The missing guard can only admit
  more LoRA points, so it is conservative for PASS and lenient for KILL.

### 3.5 Reports (no bar)

- **Inverse, drift at equal gain.**
  - s_eq = share(LoRA@equal-gain), on the dev grid.
  - Also s_eq interpolated along the upper hull, between the two hull points that bracket dev E1(OP): linear in
    log(share), except when the lower point has share 0 (base, or a point whose retention terms are all ≤ 0), where
    it is linear in share.
  - The ratio s_eq / share(OP), i.e. how many times more of the retention budget LoRA spends for OP's gain.
  - On test: E1(LoRA@equal-gain) − E1(OP), with its CI.
  - **The catastrophic-forgetting evidence**: R3 and R4 deltas against base (and against OP) of LoRA@equal-gain and
    LoRA@raw, next to OP's. These are where LoRA's forgetting would show; LoRA@matched's will be near zero by
    construction.
- **Raw-gain ratio (old E5).** E1(OP) / E1(LoRA@raw) on test (ratio only when E1(LoRA@raw) > 0), with the CI of
  E1(OP) − 0.5 · E1(LoRA@raw) and of E1(OP) − 0.25 · E1(LoRA@raw) (the old 50% / 25% thresholds). Also LoRA@raw's
  share, R3 and R4, and its module set (§3.3). No verdict.
- **Frontier data** (`E5m_frontier.json`), one record per point: tier, module set, r, lr, snapshot tokens,
  α_post, grid/extension/refinement flag, noise-floor flag, dev E1 (primary and all-target), R1, R2 parts, KL,
  share, upper-hull flag; R3 and R4 where measured. Also the adapters that hit α_min without bracketing (with
  s_min), base's repeat noise, OP, and the counts-only frontier if M2 runs it.
- The bootstrap CI of share(LoRA@matched) − share(OP), resampling retention documents (report-only; see §9,
  disclosure 3).

### 3.6 Feasibility, cost, INCOMPLETE, and an optional pilot

**Feasibility.** LoRA on the 35B-A3B MoE host on 2 × 16 GB is **unmeasured**:

- minisgl has no LoRA path, so LoRA runs offline only.
- **T0 prerequisites** (none exists today):
  - LoRA training on the M2 split runner (mixed 4k/8–16k chunking, JIT-dequant autograd, fixed GDN backward;
    M2-0's planned split). `train_lora` (`lora.py:157-219`) runs a whole-model HF forward with HF gradient
    checkpointing and the full lm_head on-device, so it has to be ported.
  - The 35B's dense projections are F16 in the checkpoint and stay `nn.Linear` under `load_layer`
    (`host_quant.py:602-614`), so `add_lora` can wrap them.
  - Snapshot saving at 1/4, 1/2 and 1; α_post as an eval-time multiplier; the fp32 branch (§3.1); the training
    order shared with the cartridge runner.
  - A per-card peak check with adapters, gradients and optimizer state on top of M2-0.3 (≤ 15.5 GB at the planned
    split).
- **T2 is new engineering and memory-bound.** With the host shapes (E = 256, D = 2,048, I = 512, 40 layers;
  `experiments/m2_0/grouped_mm_probe.py:32`), routed-expert adapters are 5,632·r parameters per expert (fused
  gate/up plus down), 57.7 M·r in total: 0.92 B at r = 16, 0.23 B at r = 4, 58 M at r = 1. At about 12 bytes per
  parameter (bf16 parameter and gradient, fp32 AdamW moments) that is about 11 GB at r = 16, 2.8 GB at r = 4 and
  0.7 GB at r = 1. The cards already sit at 15.0 / 13.8 GB at 2k microbatches against a 16.82 GB cap (DESIGN §4.6,
  §7.2). T2 also needs a grouped per-expert LoRA path inside `_MoEFn` / `QuantMoEExperts`
  (`host_quant.py:194-238`), which today returns only dX.
  - Pre-registered venue: AdamW state for T2 lives in host RAM with the serve down; the H2D/D2H transfer per step
    is measured and reported.
  - Pre-registered fallback: if r = 16 does not fit (per-card peak > 15.5 GB with the adapter parameters and
    gradients on the cards), T2 runs r ∈ {1, 4}, which counts as "T2 run", labelled with its ranks.
  - If T2 is infeasible altogether, every PASS is qualified (§3.4) and that is stated in the summary sentence.
- The cartridge also back-propagates through every layer above decoder index 1, so the activation memory is
  comparable (est.). The adapter weight gradients and AdamW state come on top.

**Cost (est.).** Training rates from DESIGN §7.2 (35B: 0.1–0.45 h per 1M tokens with the fixed GDN backward,
[0.54–1.34 h] with the reference backward; unreviewed). OP's tokens taken as 3 epochs of the 1.22M-token Split-A
train set = 3.66M (fewer if OP uses fewer epochs); T3 adds 0.5× anchor KL tokens (5.5M). One LoRA eval point is dev +
R1 + R2 + R2-ML, about 0.3M tokens (est.), so about 0.5–1.5 min, scaled from §7.2's 1–3 min per 0.65–0.7M-token
point; unmerged adapters add overhead not counted here, more for T2's routed experts. Each adapter has 10 grid
points plus up to 6 extension and 3 refinement points (10–19).

| Tier | Full runs | Training tokens | Adapters | Eval points | Training h, fixed [reference] GDN bwd | Eval h |
|---|---|---|---|---|---|---|
| T0 | 6 | 22.0 M | 18 | 180–342 | 2.2–9.9 [11.9–29.5] | 1.5–8.6 |
| T1 | 8 | 29.3 M | 24 | 240–456 | 2.9–13.2 [15.8–39.3] | 2.0–11.4 |
| T2 | 2 | 7.3 M | 6 | 60–114 | ≥ 0.7–3.3 [3.9–9.8] + host-RAM optimizer traffic (unmeasured) | ≥ 0.5–2.9 |
| T3 | 1 | 5.5 M | 3 | 30–57 | 0.55–2.5 [3.0–7.4] | 0.25–1.4 |
| **Total** | **17** | **64 M** | **51** | **510–969** | **6.4–29 [35–86]** | **4.3–24** |

Plus R3 at about 300k tokens per candidate (OP, base, the LoRA@matched candidates, LoRA@equal-gain, LoRA@raw), R4 on
about 22 points at 2 × R1's token count, base's repeat scoring, and 2 × 300 s serve stop/boot per session. T0 alone
is about 4–18 h (est.), one to four evenings; the full tier list is several days of card time. **Planned inside
M2: T0 (required) and T1. T2 and T3 are optional, and a PASS without them is qualified.**

**Test-read gating.** M2's single test read (§6) happens only after T0 has finished on dev and LoRA@matched,
LoRA@equal-gain and LoRA@raw are frozen and recorded. The owner may waive this in a dated note, recorded as a
deviation; E5m is then INCOMPLETE ("test read before T0").

**INCOMPLETE** (never read as PASS, KILL or "no LoRA gain"):

- **E5m** is INCOMPLETE if any of these holds:
  - T0 did not finish on dev before the test read (including a waiver);
  - LoRA@matched has no test result.
- A failed bracket is **not** INCOMPLETE: after the per-adapter extension to α_min = 1e-4 it is the "no LoRA gain"
  outcome (§3.4), with s_min recorded.
- **Old E5** is INCOMPLETE if LoRA@raw has no test result.
- An INCOMPLETE E5m keeps M2's LoRA comparison open: it is listed among M2's outstanding items in the M2 report
  (DESIGN §8 M2 row).

**Late reads and late tiers.**

- A configuration frozen before the test read whose test read failed for a technical reason may be read once
  later, with its dev selection unchanged. Its outcome is reported whatever it is, labelled "read after the other
  test results were known".
- Tiers that finish after the M2 test read are **post-test exploratory frontier data**. They cannot change E5m's
  verdict and cannot remove a family qualifier. If such a tier would change LoRA@matched on dev, that new point may
  be test-read once and is reported, labelled "post-test". If it beats OP with CI lower bound > 0, the summary
  sentence must add "a LoRA family run after the test read (<family>) beats OP at equal retention drift". Every
  family run after the test read is listed.

**Optional pilot on the 0.5B proxy (named only; its bars are not pre-registered here).**

- It runs on the CPU runner, against M1c's OP (share 0.027), on **dev only**, so it is not a read of the Split-A
  test files.
- Its numbers are report-only and change nothing in this file.
- If the pilot shows the α_post grid with its extension would not bracket the threshold, or that the noise-floor
  rule would exclude the bracketing region, a wider grid or a changed rule may be adopted before M2. That is logged
  as a deviation.

## 4. E6c: RAG at matched context (tokens saved)

### 4.1 Conditions

- **Recipe.** RAG-BM25 exactly as DESIGN §6.A.4 and `baselines/bm25_rag.py` define it:
  - index = the Split-A train files, 40-line chunks with stride 20, Okapi BM25 with k1 1.2 and b 0.75;
  - query = the last 20 lines of the file prefix before the window's first scored target;
  - each chunk is rendered as `# retrieved from <path>\n<chunk>`, and **the header counts toward the budget k**;
  - chunks are selected best-first until the budget, the last selected chunk truncated to its leading tokens, and
    laid out with the best chunk adjacent to the in-file context (`bm25_rag.py:120-140`), so the truncated chunk is
    farthest from the code;
  - the in-file context C = 2,048 is unchanged, and the retrieved text is extra prefill.
- **Budgets** k ∈ {0, 256, 512, 1024, 2048, 4096} tokens (the main grid), plus **64 and 128** (resolution points;
  §0 gives the reason).
  - The selected chunk set at budget k is a rank-order prefix of the set at any larger budget (up to the truncated
    chunk). The token sequences themselves are not prefixes of each other, because of the layout.
  - At k ≤ 256 the context is the head of the best chunk, header included (a 40-line chunk is usually longer), so
    E1_RAG(64) and E1_RAG(128) depend mostly on the cut rule.
  - k = 0 is the base, or OP alone.
- **Curves.** E1_RAG(k) = E1(RAG@k) and E1_RAG+cart(k) = E1(RAG@k + OP), each on dev and on test.

### 4.2 Quantities

- **Interpolation rule.** Grid points k_a < k_b are adjacent. Take the first k_b (scanning up from the smallest
  k > 0) with f(k_b) ≥ y and f(k_a) < y. The crossing is
  `k = 2^(log2 k_a + (y − f(k_a)) / (f(k_b) − f(k_a)) · (log2 k_b − log2 k_a))`, linear in log2 k. Between k = 0
  and k = 64 nothing is interpolated: the value is reported as "≤ 64".
- **k_eq, the context equivalent of the cartridge alone**: the crossing of f = E1_RAG with y = E1(OP).
  - k_eq = 0 ("no context equivalent; OP gain ≤ 0") when E1(OP) ≤ 0.
  - "≤ 64" when E1_RAG(64) ≥ E1(OP) > 0.
  - "> 4096" when E1_RAG(k) < E1(OP) at every grid k.
  - Also reported: k_eq on the main grid only (k ∈ {0, 256, …, 4096}; "≤ 256" below 256), beside the version that
    uses 64 and 128.
- **S, the primary "saves context" quantity**: S = E1(RAG@1024 + OP) − E1(RAG@4096) on test, a paired difference
  with its CI.
- **k′**: the smallest budget at which RAG + OP matches RAG@4096. It is the crossing of f = E1_RAG+cart with
  y = E1(RAG@4096).
  - k′ = 0 if OP alone reaches it (E1(OP) ≥ y).
  - "≤ 64" if the crossing lies between 0 and 64.
  - "not reached" if f(k) < y at every grid k (no crossing anywhere).
  - Tokens saved = 4096 − k′ ("≥ 4032" for "≤ 64"; "none" for "not reached").
  - k′ and tokens saved are reported on test (primary) and on dev.
- **CIs of k_eq, k′, tokens saved**: the 2.5/97.5 percentiles over the joint 10k resamples (§2), with censored draws
  sorted as k_eq: 0 < "≤ 64" < numeric < "> 4096" (= +∞); k′: 0 < "≤ 64" < numeric < "not reached" (= +∞); tokens
  saved follows k′ in reverse. A CI end that lands on a censored value is reported by its label, and the share of
  draws in each censored class is reported.
- **Real cost of context, per budget k, on the host where it can be measured.** Each row of the result is labelled
  with that host. RAG is prompt text only, so it can be measured on the minisgl serve now, without M4. Measured:
  - extra prefill tokens per scored window: the mean and p90 of the tokens actually retrieved, which can be below k;
  - **cold** prefill time at C + k vs C: 20 distinct scored windows per budget (so no repetition can hit the radix
    prefix cache), with prefix caching disabled or the radix cache flushed between requests; the median, plus
    prefill throughput (tokens/s);
  - decode latency per token at a KV length of C + k vs C, at batch 1, median over the same 20 windows;
  - the serve configuration is recorded: EP vs TP arm, chunked-prefill size, which replica;
  - the context-window share k / (the serve's configured maximum context) and k / (C + k).
  - Next to these: the cartridge's own overhead. This is M4's measured decode/prefill overhead once it exists;
    until then, the offline graft overhead, labelled "offline", plus DESIGN's est. 0.5–0.7% decode.

### 4.3 Verdict (no KILL)

| Outcome | Rule | Description |
|---|---|---|
| **PASS** | S's CI lower bound ≥ 0 | "the cartridge replaces ≥ 3k tokens of retrieved context" (RAG@1024 + OP is at least RAG@4096) |
| not shown | otherwise | "does not save context at the 1k/4k comparison". k′ and tokens saved are still reported. |

**Strictness.** Because the rule asks for the CI lower bound, this is in effect a superiority test with zero
margin, which is strict. The graded information is in k′.

**INCOMPLETE** if RAG@1024 + OP or RAG@4096 has no test result. A censored k_eq or k′ is a result, not
INCOMPLETE. E6c feeds the summary sentence (§7) only.

## 5. E6, extended: additivity at every budget, and stacked on counts

### 5.1 E6(k)

- E6(k) = E1(RAG@k + OP) − E1(RAG@k), for every k in the grid, on dev and test, each with its paired CI.
- **E6 = E6(4096).** Its PASS and KILL are unchanged (DESIGN §6.A.7):
  - "additive" if E6(4096) ≥ 0.5 points with CI lower bound > 0;
  - KILL if E6(4096) < 0.25 points or its CI includes 0, which drops the additivity claim (DESIGN §1).
- **INCOMPLETE** if RAG@4096 or RAG@4096 + OP has no test result. A missing E6(k) at another budget is reported as
  missing.
- The other budgets are report-only. The report adds the k range over which the CI lower bound of E6(k) stays above
  0; this is a scan over 8 budgets, exploratory and uncorrected (§8). E6(0) = E1(OP).

### 5.2 E6s: stacked on retrieval and counts (frontier to frontier, as M1c Bar A)

- **Counts.** The count-LM is the Kneser–Ney trigram over the Split-A train files, mixed at the output. Its weight
  grid W is the one the M2 PREREG fixes for its counts frontier.
- **Retention.** Retention sets are scored **without** retrieval: §6.A.4 counts RAG's retention cost as zero,
  because retrieval runs only on project text. So the retention constraint is that of the counts (and of
  counts + OP), as in M1c.
- **The constraint is "within the M2 OP bars" (§2), for both weights**, so the two frontiers stay symmetric and OP,
  which M2's zone allows up to R2 +1.0%, can meet it:
  - **λ\*_RAG**: the counts weight w with the largest dev E1(RAG@4096 + counts@w), among the weights where counts@w
    alone is within the M2 OP bars (ties: the smaller weight). Weight 0 always qualifies.
  - **λ_X,RAG**: the weight w with the largest dev E1(RAG@4096 + counts@w + OP), among the weights where
    counts@w + OP is within the M2 OP bars (ties: the smaller weight). Weight 0 is RAG@4096 + OP; it qualifies when
    OP alone is within the M2 OP bars.
  - Implementation: `bars.lambda_star` and `bars.lambda_x` with `bars=M2_BARS`.
- **E6s** = E1(RAG@4096 + counts@λ_X,RAG + OP) − E1(RAG@4096 + counts@λ\*_RAG) on test, with its paired CI.
  - **PASS** ("adds to retrieval and counts") if E6s ≥ 0.5 points with CI lower bound > 0. No KILL.
  - The same-weight difference E1(RAG@4096 + counts@λ\*_RAG + OP) − E1(RAG@4096 + counts@λ\*_RAG) is reported next
    to it.
- **Dev conditions (required for E6s; listed in DESIGN §8's M2 row).** For every w in W: counts@w,
  RAG@4096 + counts@w and RAG@4096 + counts@w + OP on Split-A dev; counts@w + OP on R1, R2 and R2-ML. The count-LM
  is mixed at the output, so each family is one forward pass with a grid of mixtures (`countgrid`): about 7 Split-A
  dev pass equivalents of prefill (1 + 3 + 3, the RAG@4096 passes shareable with E6c's dev conditions) plus one OP
  retention pass.
- **INCOMPLETE** if:
  - OP alone is not within the M2 OP bars (then no weight of the stacked arm can qualify, since counts do not lower
    retention cost; reason "OP outside its own retention bars");
  - the dev conditions above were not run;
  - a required test condition (§6) is missing.

## 6. The test read

- **Single evaluation.** All conditions below join M2's single test evaluation of the Split-A test files, once each,
  after every dev selection in this file and in the M2 PREREG is frozen and recorded (for E5m, see the gating rule
  in §3.6). OP and base are the main PREREG's own conditions.

| Bar | Test conditions |
|---|---|
| E5m | LoRA@matched; LoRA@equal-gain; LoRA@raw (the all-linear r16 point too if T2 ran at r = 16) |
| E6c, E6(k) | RAG@k for k ∈ {64, 128, 256, 512, 1024, 2048, 4096}; RAG@k + OP for the same k |
| E6s | RAG@4096 + counts@λ\*_RAG; RAG@4096 + counts@λ_X,RAG + OP; RAG@4096 + counts@λ\*_RAG + OP (only when λ_X,RAG ≠ λ\*_RAG) |

- **Read count (disclosure).** The Split-A test files have been read twice, by M1b's T_test and M1c's T_test, both
  on the 0.5B proxy. M2's test read would be the **third read of these files**. It is on a different host, but M2's
  recipe was chosen partly on what the earlier reads showed, so a third read carries adaptive-selection risk at the
  level of recipe choices.
- **Recommendation (the owner decides; not decided here).**
  - If the owner supplies a fresh **by-file** held-out set before the M2 test read, these bars are judged on it. The
    candidates are corpus-v1 files (DESIGN §6.A.1) that no run has scored and whose path hash falls in the test
    split. The Split-A test files are then reported as a secondary stratum, labelled "third read".
  - A by-time set (lines added by commits after `df29117`; M1c PREREG §4 and §6 record that none existed then) would
    make these bars E2-type. If the owner chooses one, the M2 PREREG must restate the endpoint before the read, and
    the Split-A by-file third read stays the primary, for E1 comparability.
  - Otherwise the third read is disclosed in every bar's report line.
  - Either way, any unread files added to the test set (corpus v1) are also reported as their own stratum.

**Cost of the RAG conditions (est.).** Prefill scales as (C + k)/C, so the 7 non-zero budgets cost about 11 Split-A
pass equivalents per arm per split, about 44 for the 28 dev/test conditions, plus the E6s dev grid (§5.2). On the
35B this is a few hours at most. The timing measurements of §4.2 take minutes.

## 7. The summary sentence (Z_report composes it; no model is run)

Z_report emits exactly one clause per bar. A bar without a verdict or an INCOMPLETE reason is an error, not an
omission.

| Verdicts | Clause |
|---|---|
| E5m KILL / no LoRA gain / PASS / neither / INCOMPLETE | "LoRA dominates even at equal retention drift" / "LoRA yields no gain at OP's retention drift (<reason>); the cartridge gains …" / "[clearly] beats LoRA at equal retention drift (R1/R2/KL) [qualifiers §3.4]" / "not resolved against LoRA at equal retention drift" / "LoRA comparison incomplete: <reason>; open in M2" |
| E6c PASS / not / INCOMPLETE | "replaces ≥ 3k tokens of retrieved context (k′ = …, tokens saved …)" / "does not save context at 1k vs 4k (k′ = …)" / "context comparison incomplete: <reason>" |
| E6 PASS / KILL / gap / INCOMPLETE | "additive to RAG-4k" / "additivity claim dropped" / "not shown additive" (DESIGN §6.A.7) / "additivity incomplete: <reason>" |
| E6s PASS / not / INCOMPLETE | "adds to retrieval and counts" / "not shown to add to retrieval plus counts" / "stacked comparison incomplete: <reason>" |

The sentence always carries the raw-gain ratio (old E5, with its module set), k_eq and the context costs as numbers,
any post-test LoRA clause (§3.6), and, if T2 did not run, that routed-expert LoRA was not compared. It always states
that these bars position the cartridge, and that they do not set the M2 zone.

## 8. Deliberately NOT decided here

- Whether a fresh held-out set exists, and which set it is (the owner's decision; §6).
- How OP is selected, the zone thresholds, the M2 retention-set sizes, and every other part of M2. The full M2
  PREREG decides these, and this file does not constrain them beyond §1 (and §3.6's test-read gating).
- The exact R4 sources and the exact lr grid centre (rules fixed here, values fixed in the M2 PREREG before any M2
  measurement).
- Serving LoRA (minisgl has no path). E5m is a quality-at-equal-retention-drift comparison only.
- Matched-drift versions of the other comparators (Engram-Adapter-style, kNM-LM, the param-matched LoRA). Their
  §6.A.7 rows stand as they are.
- The bars of the optional proxy pilot (§3.6).
- **Any multiplicity correction.** Every interval is a 95% percentile interval (2.5% one-sided per directional
  claim), with no family-wise correction. The comparisons that carry a verdict or change the wording are:
  - verdict-bearing (8): E5m KILL and E5m PASS (two directional tests of D5), the "clearly" CI, the "no LoRA gain"
    clause's CI of E1(OP), E6(4096) PASS and KILL, E6c's S, and E6s;
  - wording-bearing, exploratory (8): the R4 qualifiers, 2 components × {KL, ΔNLL} × 2 directions;
  - exploratory scan (8 budgets): the k range over which E6(k)'s CI lower bound is > 0.
  - About 24 directional comparisons in all. If every null held, about 0.6 of them would come out "significant" by
    chance (24 × 2.5%); the exploratory ones are labelled as such in the report.

## 9. Disclosures and known biases

1. **Dev rests on few files.** On the proxy, one dev file held 62% of the dev targets (n_eff ≈ 2.5 of 13; M1c
   PREREG §8). LoRA@matched and λ\*_RAG are close to point selections on that file. Selection among many LoRA points
   on dev gives LoRA a winner's curse on dev E1. The test read of the selected point is unbiased.
2. **The tiers favour the cartridge until T2/T3 run.** A smaller LoRA family has a lower frontier. That is why a
   PASS names its families (§3.4) and a KILL from any subset stands.
3. **Matching biases, in three directions.**
   - (a) *Selection on the same retention data* favours LoRA: LoRA@matched is selected and reported on the same
     retention sets, so its measured share sits at or below t by construction, and its true share may be higher.
   - (b) *Grid discreteness* favours the cartridge: the selected point's share can sit below t, so LoRA is compared
     at less drift than OP. Without refinement it could be up to about 2× below t (first-order scaling, 1.5–2× grid
     steps); the refinement (§3.1) bounds this at [0.9 t, t] for adapters that reach it within 3 points.
   - (c) *OP's own share is biased low* and that favours the cartridge: OP is selected on dev partly on retention, so
     its measured share, and therefore t, tends to understate its true drift, and LoRA is held to that lower t.
   - The share-difference CI (§3.5) and the refinement outcome per adapter are reported. No correction is applied.
4. **α_post is a LoRA-only knob.** The cartridge gets no post-hoc scale here (§1). This favours LoRA.
5. **RAG retention.** Retention sets are scored without retrieval (§6.A.4 convention). A deployment that retrieves
   on every request, project or not, would see costs that this does not measure.
6. **The R3 guard uses a 1-point tolerance set from an estimated SE (§3.2).** On the 35B, its actual SE is reported
   with the first R3 measurement. At LoRA@matched's α_post it is likely near-vacuous.
7. **Generation-level forgetting is not measured** in M2 for either arm (instruction following, chat/tool format,
   HumanEval/MBPP; deferred to M4, and LoRA cannot be served). E5m matches near-zero NLL/KL drift; it answers "who
   gains more at negligible drift", not "does LoRA forget catastrophically". The latter evidence is R3/R4 at
   LoRA@raw and LoRA@equal-gain (§3.5).
8. **The lr of T1–T2 is transferred from T0** (§3.1): chosen at matched share on dense-linear r ∈ {4, 16}, then used
   for other ranks and module sets. A different lr could do better for attn+GDN or routed experts at matched share;
   this favours the cartridge for those tiers.
9. **bf16 at small α_post.** The fp32 branch (§3.1) removes rounding inside the LoRA branch but not the final bf16
   cast of the projection output. Points whose retention drift is within base's repeat noise are excluded from Q
   (§3.1); if that excludes the bracketing region, the outcome tends toward "no LoRA gain", which is reported with
   the exclusions.

## 10. Review changes (2026-11-12, before any M2 measurement)

Three reviews (stats, consistency, feasibility) of the first draft. Each finding was checked against the files and
code before it was applied. Changes to this file:

- **E6s retention constraint** (stats D1, consistency D1): both weights now use "within the M2 OP bars" (R1 ≤ 0.2%,
  KL ≤ 0.005, R2 ≤ 1.0%), not M1c's 0.5% R2 bar; OP outside its own bars makes E6s INCOMPLETE, not "fails" (§2,
  §5.2).
- **E5m base fallback** (stats D2, feasibility D4): new "no LoRA gain" outcome when LoRA@matched is the base or has
  test E1 ≤ 0; it never licenses "beats LoRA". The ratio is defined only for E1(LoRA@matched) > 0, and "clearly"
  needs the CI of E1(OP) − 1.5·E1(LoRA@matched) > 0. A failed bracket is this outcome, not INCOMPLETE (§3.4, §3.6).
- **Per-adapter downward extension** to α_min = 1e-4, with non-bracketing adapters recorded (stats D3, D4;
  feasibility D4). Not applied: stats D3's "INCOMPLETE if fewer than half of T0's adapters bracket". After a
  per-adapter extension to 1e-4, an adapter that still does not bracket could add at most about α_min times its
  α_post = 1 gain (first-order est.), and an INCOMPLETE would discard the case most favourable to the cartridge.
- **Scaling sentence corrected** (ΔNLL first-order, KL second-order), with a per-adapter log-α refinement to
  [0.9 t, t] and disclosure 3 rewritten to name all three bias directions (stats D4).
- **lr** (stats D5, feasibility D13): all three lrs trained to full length in T0; T1–T2 inherit T0's matched-share
  lr; the LoRA lr pilot is dropped; disclosure 8 added.
- **fp32 LoRA branch, noise-floor check and smoothness check** (feasibility D3); disclosure 9 added.
- **Licensed wording** "equal retention drift (R1/R2/KL)"; R4 measurement set, primary statistics and R4-chat source
  rule fixed; R3/R4 at LoRA@raw and LoRA@equal-gain reported as the catastrophic-forgetting evidence; disclosure 7
  (feasibility D8). R4 qualifier statistic and bootstrap specified (stats D11).
- **LoRA@raw comparator** labelled by module set; on the MoE host without T2 at r = 16 the old E5 ratio, §6.A.6's
  fraction and M3's check carry the "dense-linear" label (stats D14, consistency D2, feasibility D10). Making
  all-linear r16 mandatory (consistency D2 option a) was not chosen, because T2 at r = 16 does not fit the cards
  (feasibility D2).
- **T2 footprint, venue and fallback ranks**; T0 prerequisites; cost table per tier; planned tiers (feasibility
  D1, D2, D14).
- **Test-read gating, late reads and post-test tiers** (feasibility D5, stats D15): T0 must finish before the test
  read; post-test tiers cannot change the verdict or a qualifier but a post-test LoRA win must be disclosed in the
  summary; the no-R3-guard case is a labelled, judged verdict; an INCOMPLETE E5m stays open in M2.
- **INCOMPLETE clauses for every bar** in §7 and an INCOMPLETE rule for E6; Z_report must emit one clause per bar
  (feasibility D6).
- **E6s dev conditions listed** (feasibility D7) and the missing test condition added to §6 (stats D9).
- **Cold prefill timing** and recorded serve configuration (feasibility D9).
- **k′ / k_eq rules**: "not reached" = no crossing at any grid k; explicit sort order of censored draws; k_eq = 0
  when E1(OP) ≤ 0 (stats D7); header counts toward k, head-of-chunk cut, main-grid-only k_eq (feasibility D15).
- **Joint-resampling implementation and check** (stats D8); **s_eq** interpolation at share 0 (stats D10).
- **Multiplicity** listed, uncorrected, exploratory items labelled (stats D6).
- **Wording**: RAG layout and nesting (consistency D7); endpoint labels in §0 (consistency D8); fresh held-out set
  limited to by-file candidates (consistency D9); "(DESIGN §1)" (consistency D11); the share-budget reason in §2
  (consistency D6).

DESIGN.md changes made with this review: the status line (consistency D10); the LoRA-frontier row (consistency D5,
stats D5); the E5, E5m and E6s rows (stats D12, consistency D1, D4, D6, feasibility D10, D12); the stale sentence
under the zone table and the Engram-Adapter "E5 bar" remark (stats D13, consistency D3, feasibility D11); §6.A.6's
cross-host comparator label (consistency D2); the M2 row's conditions, cost and outstanding-item rule (feasibility
D1, D5, D7, consistency D2); the 2026-11-12 revision note.
