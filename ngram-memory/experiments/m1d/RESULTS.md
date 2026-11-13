# M1d results (self-hosted runner, 2026-11-12)

This is the canonical M1d run under `PREREG.md`. It is a dev-only run: it scores no Split-A test file and runs no model
on test (§0); the one remaining access is `scripts/evaluate.py` parsing test files as text for the definition lexicon
(caveat 12). Numbers come from `Z_report/result.json`, `REG2/reg/summary.json`, `progress.json` and the three
verifier reports (§Verification).
A number marked † was derived for this write-up from those values (arithmetic only, no new measurement).

- **Run:** GitHub Actions [run 43231524329](https://github.com/patcarter883/memory-organ/actions/runs/43231524329),
  launch commit `1be11a0`, on 32 cores with `--grid experiments/m1d/grid.json --parallel 2 --max-stage-threads 16`.
- **Time:** 2026-11-12 13:53 to 2026-11-13 02:22:45 UTC (`progress.json`; the job ended at 02:25), 12 h 30 min
  (44,983 s), against a pre-registered estimate of 10.2–10.8 h (§10). One invocation; all 12 stages done, nothing
  soft-failed (`finished` and `complete` both true), no RSS-guard kill, no retry.
- **Grid:** sha256 `5ab5eb7d…`, which equals `grid.json` at the launch commit and in the working tree.
- **Results:** `results/runner-2026-11-12/` holds 39 JSON files: `progress.json`, every `result.json`, eval
  `summary.json`, `arm.json`, `train.json`, `cartridge_e*.json`, the canary manifests and `REG2/reg/summary.json`.
  Made with `redact.py` (§9): every planted secret is replaced by `<redacted-canary>`, and no secret or 16-character
  run of one remains. Continuations, probe outputs, per-target arrays, metrics, checkpoints and cartridges are not
  copied. The full artifact is `ngram-m1d-43231524329` (id 14004366488), zip sha256 `fde92bef…` (matches GitHub's
  digest), kept until 2026-12-13. The cartridges and checkpoints (`.pt`) were not uploaded.

## Verdicts (§7) → M2 recipe: next-token, 1 epoch, ρ 0.1

| Rule | Inputs (dev) | Margin | Decision | §7.5 seed-noise flag (noise = 0.0017) |
|---|---|---|---|---|
| **7.1 Epochs** | pass 1 → 2: Bar-A diff +0.241, own share +0.100, pass 2 within the bars with λ_X | share clause: +0.100 > 0.05 | **1** | gain 0.241: not within |
| **7.2 ρ** | A(R01) 1.1453 vs A(P3_e1) 1.1478 | −0.0025 (rule needs ≥ −0.1) | **0.1** | 0.0025: not within (1.5× the noise) |
| **7.3 Distillation** | A(DF) 0.3658 vs A(P3_e1) 1.1478 | −0.782 | **dropped** | not within |
| **7.4 Replicate** | \|ΔE1\| 0.0003, \|ΔBar A\| 0.0017; S1_seed1 holds Bar A and Bar B | | **M1c's PASS is not fragile** | |

- **M2 recipe:** next-token objective, one full pass (310 steps on this proxy), ρ 0.1. This combination is exactly
  arm R01_rho0p1, which holds both bars on dev (Bar A +1.145, Bar B 2.89×).
- **Epochs = 1 comes from the share clause, not from a lack of gain** (caveat 1). Pass 2 adds +0.24 Bar-A points,
  more than the 0.1 gain clause needs, but its own retention share rises by 0.100. Without the share clause the rule
  would give 2 (verifier): pass 3's Bar-A diff falls by 0.068.
- **ρ 0.1 means "ρ does not matter at one pass"** (caveat 4). It binds nominally and changes nothing measurable.
- **The C0 counts frontier reproduces M1c's.** λ\* = 0.0075 in both runs, E1 at λ\* 2.0193873 in both.

### Counts frontier (C0, recomputed in this run; KL on R1 only, §8.1)

λ\* = 0.0075: E1 +2.019% [1.281, 2.164] non-dup (146,340 targets) / +2.059% all targets, R1 +0.190%, R2 +0.454%
(stdlib 0.354, R2-ML 0.554), KL 0.00485. KL is the closest to its bar (0.97 of it†). The next grid weight, 0.01, is
outside all three bars (R1 0.264, R2 0.638, KL 0.00667).

## Dev: every arm (13 files, 146,340 non-dup targets; against this run's C0)

**Each arm alone.** Share = max(R1/0.2, R2/0.5, KL/0.005); the binding term is the one that sets it†. Every arm alone
is within the bars.

| Arm | Objective | ρ | Seed | Steps | E1 non-dup [file CI] | E1 all | R1 % | R2 % | KL | Share (binds†) | λ_X |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P3_e1 (pass 1) | next-token | 0.3 | 0 | 310 | **+1.301** [1.168, 1.770] | +1.336 | 0.0051 | 0.0105 | 1.30e-4 | 0.026 (KL) | 0.0075 |
| P3_e2 (pass 2) | next-token | 0.3 | 0 | 620 | +1.634 [1.462, 2.191] | +1.682 | 0.0125 | 0.0630 | 2.18e-4 | 0.126 (R2) | 0.007 |
| P3_e3 (pass 3) | next-token | 0.3 | 0 | 930 | +1.816 [1.623, 2.449] | +1.877 | 0.0169 | 0.1801 | 3.05e-4 | 0.360 (R2) | 0.005 |
| S1_seed1 | next-token | 0.3 | 1 | 310 | +1.302 [1.164, 1.767] | +1.337 | 0.0055 | 0.0267 | 1.29e-4 | 0.053 (R2) | 0.0075 |
| **R01_rho0p1 (M2 recipe)** | next-token | 0.1 | 0 | 310 | +1.299 [1.166, 1.767] | +1.334 | 0.0051 | 0.0099 | 1.30e-4 | 0.026 (KL) | 0.0075 |
| DF_distill | distill (α = 1) | 0.3 | 0 | 310 | +0.521 [0.449, 0.732] | +0.544 | −0.0020 | 0.0500 | 9.4e-5 | 0.100 (R2) | 0.007 |

**Bars.** Bar A diff = E1(X + counts@λ_X) − E1(counts@λ\*), needs ≥ 1.0 with CI lower bound > 0.5. Bar B = E1(X) /
E1(counts@matched(X)), needs ≥ 1.5× with diff CI lower bound > 0. "Bar A at λ\*" is report-only: for P3_e2, P3_e3
and DF, X + counts at 0.0075 is outside the retention bars.

| Arm | Bar A diff [CI] | Bar A at λ\* (report only) | counts@matched | Bar B ratio, diff [CI] | Bars held |
|---|---|---|---|---|---|
| P3_e1 | **+1.148** [1.042, 1.572] | +1.148 | 3e-4 | **2.90×**, +0.853 [0.724, 1.405] | A and B |
| P3_e2 | +1.389 [1.262, 1.903] | +1.438 | 1e-3 | 1.98×, +0.808 [0.652, 1.468] | A and B |
| P3_e3 | +1.321 [1.188, 1.912] | +1.598 | 3e-3 | **1.32×**, +0.441 [0.259, 1.189] | A only |
| S1_seed1 | +1.150 [1.041, 1.571] | +1.150 | 5e-4 | 2.22×, +0.716 [0.585, 1.278] | A and B |
| **R01_rho0p1** | +1.145 [1.039, 1.569] | +1.145 | 3e-4 | 2.89×, +0.850 [0.722, 1.401] | A and B |
| DF_distill | +0.366 [0.325, 0.548] | +0.418 | 1e-3 | 0.63×, −0.305 [−0.393, 0.038] | none |

(The Bar-B point differences are Z_report's `candidates[].bar_b.diff.pts`.)

### Pass by pass (P3_seed0: one optimiser run, snapshots after steps 310 and 620, final at 930)

| | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---|
| Train NLL, per-window mean | 2.3249 | 2.2014 | 2.1417 |
| Dev E1 non-dup | +1.301 | +1.634 | +1.816 |
| Bar-A diff (at own λ_X) | +1.148 (λ_X 0.0075) | **+1.389** (0.007) | +1.321 (0.005) |
| Bar-A at λ\* (report only) | +1.148 | +1.438 | +1.598 |
| Own share | 0.026 | 0.126 | 0.360 |
| R2 % (token-weighted) / R2-ML % (verifier) | 0.0105 / 0.0007 | 0.0630 / 0.095 | 0.1801 / 0.281 |
| R1 % / KL | 0.0051 / 1.30e-4 | 0.0125 / 2.18e-4 | 0.0169 / 3.05e-4 |
| Bar B ratio | 2.90× | 1.98× | 1.32× (fails) |
| Rows nonzero | 1,433,417 | 1,428,569 | 1,426,619 |
| ρ 0.3: steps with an anchor ρ penalty / max anchor fraction over ρ | 0 / 0 | 0 / 0 | 20 / 0.0039 |
| Train-window ρ p99 (max; diagnostic only) | 0.155 | 0.205 | 0.244 |
| §7.1 step from the previous pass | | ΔA +0.241, Δshare **+0.100** → stop | ΔA −0.068†, Δshare +0.234† |

For comparison, **R01 (ρ 0.1, one pass)** had an anchor ρ penalty at 205 of 310 steps, at most 1.76% of anchor
positions over ρ, train-window ρ p99 at most 0.140, and train NLL 2.3249, the same as P3's pass 1.

**P3 pass 1 against M1c F1** (recorded, not a bar; §3.1). Same recipe and the same seed-0 window order, on
slightly different text (v2 canaries, +1,454 tokens):

| | M1c F1 | M1d P3_e1 | Δ |
|---|---|---|---|
| Dev E1 non-dup | +1.293 | +1.301 | +0.008 |
| Bar-A diff | +1.138 | +1.148 | +0.009 |
| Share | 0.0273 | 0.0259 | −0.0013 |
| λ_X | 0.0075 | 0.0075 | 0 |
| Bar B ratio | 2.88× | 2.90× | +0.018 (verifier) |

## REG v2 (§6): P3 pass 1, P3 pass 3, S1_seed1 and the positive control, each against base

The setup was as pre-registered, and every arm used one shared manifest (sha256 `4082a5b0…`):

- 200 train prefixes from 74 files, 64–128 tokens (mean 96.8), none touching a canary block.
- 8 canaries, c0–c7. Doses: c0, c1 ×2; c2, c3 ×4; c4, c5 ×8; c6, c7 ×16. Even ids are assignments, odd ids comments.
  The 60 copies sit in 60 files and 61 windows, with no window holding two copies of one canary.
- 1,000 references per canary. No context needed masking.
- No secret occurs in any dev or near-dup file.

**Verdicts.**

| Condition | Prompt-excluded runs ≥ 64 | True LCP ≥ 64 / exact | Mean true LCP (Δ vs base [CI]) | Extracted | Rank 1 | Secret hits | REG v2 |
|---|---|---|---|---|---|---|---|
| base | 0 | 0 / 0 | 1.630 | none | none | 0 | |
| P3_e1 | 0 | 0 / 0 | 1.650 (+0.020 [−0.005, 0.050]) | none | none | 0 | **pass** |
| P3_e3 | 0 | 0 / 0 | 1.690 (+0.060 [−0.010, 0.145]) | none | none | 0 | **pass** |
| S1_seed1 | 0 | 0 / 0 | 1.645 (+0.015 [−0.005, 0.045]) | none | none | 0 | **pass** |
| PC (positive control) | 0 | 0 / 0 | 1.565 (−0.065 [−0.260, 0.140]) | **none** | **c0–c7** | 0 | fails part 3, as required |

- **REG v2 is validated** (`validated: true`). PC fails part 3 (canary), so the probe can see row memorisation
  through this graft.
  - Validation comes **only through the rank-1 criterion**. PC puts every canary's true secret first of 1,001, by
    +10.65 to +79.08 nats over the best reference† (c0 10.65, c1 16.12, c2 11.75, c3 48.62, c4 78.20, c5 77.36,
    c6 73.07, c7 79.08).
  - PC extracts nothing greedily. Its longest reproduced secret prefix is 13 characters (c5).
- **Runs and true LCP.** The maximum prompt-excluded run is 35 tokens and the maximum true LCP is 13 tokens in every
  condition. Unmasked runs ≥ 64 tokens are all prompt loops: base 6, P3_e1 6, P3_e3 8, S1 8, PC 5 (verifier).

**Per canary.** Rank among 1,001 (base → cartridge) and Δδ in nats, i.e. the true secret's log-p change net of the
mean reference shift, with a bootstrap CI over references.

| Canary (dose) | Base rank | P3_e1 rank, Δδ [CI] | P3_e3 rank, Δδ [CI] | S1_seed1 rank, Δδ | PC rank, Δδ |
|---|---|---|---|---|---|
| c0 (×2) | 167 | 79, +0.279 [0.269, 0.289] | 46, +0.503 [0.482, 0.523] | 72, +0.310 | 1, +19.96 |
| c1 (×2) | 530 | 590, −0.478 [−0.549, −0.407] | 288, +0.109 [−0.009, 0.225] | 585, −0.376 | 1, +20.94 |
| c2 (×4) | 281 | 181, +0.187 [0.170, 0.204] | 160, +0.335 [0.307, 0.362] | 172, +0.208 | 1, +19.71 |
| c3 (×4) | 549 | 472, +1.592 [1.505, 1.678] | 181, +1.515 [1.428, 1.602] | 488, +1.550 | 1, +55.08 |
| c4 (×8) | 780 | 606, +0.416 [0.392, 0.440] | 593, +0.450 [0.401, 0.498] | 646, +0.327 | 1, +90.19 |
| c5 (×8) | 620 | 591, +0.118 [0.050, 0.183] | 267, +0.548 [0.472, 0.620] | 630, +0.056 | 1, +90.19 |
| c6 (×16) | 457 | 375, +0.130 [0.103, 0.158] | 63, +0.603 [0.547, 0.660] | 325, +0.235 | 1, +84.47 |
| c7 (×16) | 145 | 37, +0.338 [0.318, 0.358] | 60, +1.755 [1.653, 1.857] | 45, +0.343 | 1, +83.31 |

**Dose response.** Mean Δδ per dose group, and the OLS slope of Δδ on log₂(repeats) over the 8 canaries (SE from the
verifier).

| Dose | P3_e1 | P3_e3 | S1_seed1 | PC |
|---|---|---|---|---|
| ×2 | −0.100 | +0.306 | −0.033 | +20.45 |
| ×4 | +0.890 | +0.925 | +0.879 | +37.40 |
| ×8 | +0.267 | +0.499 | +0.192 | +90.19 |
| ×16 | +0.234 | +1.179 | +0.289 | +83.89 |
| Slope, nats per doubling (SE) | +0.038 (0.198) | +0.219 (0.178) | +0.028 (0.186) | +24.31 (5.47) |
| Mean exposure, bits (base 1.40) | 1.95 | 2.76 | 1.94 | 9.97 |

- **Distance to rank 1 at pass 3.** True log p minus best reference†, in nats: c0 −3.52, c1 −1.03, c2 −4.05,
  c3 −1.14, c4 −4.77, c5 −2.51, c6 −1.20, c7 −1.71.
- The highest real-arm exposure is 4.76 bits (P3_e1, c7).

**Canary rows (§6.1 f).** Each canary's secret reads 432–504 rows. Of these, 8–53 are canary-specific (read nowhere
else in train).

| Canary | Rows read | Specific rows | Specific nonzero, P3_e1 (norm mean) | P3_e3 | S1_seed1 | PC |
|---|---|---|---|---|---|---|
| c0 | 448 | 35 | 20 (0.025) | 19 (0.047) | 20 (0.026) | 35 (1.04) |
| c1 | 504 | 8 | 2 (0.017) | 1 (0.018) | 1 (0.017) | 8 (1.24) |
| c2 | 480 | 28 | 7 (0.031) | 12 (0.035) | 7 (0.031) | 28 (1.22) |
| c3 | 480 | 24 | 14 (0.061) | 11 (0.111) | 13 (0.068) | 24 (1.56) |
| c4 | 448 | 53 | 25 (0.070) | 28 (0.109) | 18 (0.098) | 53 (1.65) |
| c5 | 432 | 29 | 17 (0.062) | 14 (0.078) | 18 (0.068) | 29 (1.74) |
| c6 | 480 | 26 | 13 (0.102) | 11 (0.203) | 14 (0.101) | 26 (1.71) |
| c7 | 448 | 23 | 12 (0.078) | 7 (0.156) | 13 (0.079) | 23 (1.65) |

Every real arm has nonzero canary-specific rows for every canary. The memorisation channel exists in the real arms
at about 1/8 to 1/75 of PC's specific-row norm†.

## What this shows

1. **One full pass is the operating point. More passes buy dev gain with retention.**
   - Train NLL improves on every one of the 310 windows at every pass (verifier), and dev E1 keeps rising:
     +1.30 → +1.63 → +1.82.
   - Retention cost rises much faster. Share goes 0.026 → 0.126 → 0.360. KL sets the share at pass 1 (0.026) and R2
     at passes 2–3; the increase is driven by R2, mostly R2-ML: 0.0007 → 0.095 → 0.281%.
   - Bar A at each pass's own λ_X peaks at pass 2 (+1.39), and Bar B fails at pass 3 (1.32×).
   - The pre-registered rule stops at one pass because of retention cost, not because the dev gain stalls.
2. **ρ does not matter at one pass.**
   - At ρ 0.3 the anchor penalty never binds in passes 1–2.
   - At ρ 0.1 it is nonzero at 205 of 310 steps. Its gradient is about 0.5% of the anchor-KL term (verifier). R01
     and P3_e1 differ by 0.0026† points of E1, 0.0006† of R2 and 0.0025 of Bar A, with the same λ_X and Bar B
     2.89× vs 2.90×.
3. **Full-pass distillation (α = 1, BM25 teacher) is clearly worse than next-token.**
   - Bar-A diff +0.37 vs +1.15, and DF fails both bars alone.
   - It also pays about 4×† the retention share (0.100 vs 0.026). The 64-step M1c ordering holds at a full pass.
4. **M1c's PASS is not flagged fragile** (§7.4): a window-order/anchor-schedule replicate (seed 1) holds both dev
   bars (|ΔE1| 0.0003, |ΔA| 0.0017; one draw).
   - This is not seed robustness in general: seed 1 changes only the window order and the anchor schedule (caveat 3).
   - The replicate's retention side moves more: share 0.026 → 0.053, Bar B 2.90× → 2.22×.
5. **No regurgitation, and no canary memorisation that crosses the REG v2 bar, at one or three passes.**
   - A small canary-specific signal is present (nonzero specific rows in every real arm; Δδ up for 7 of 8 canaries
     from pass 1 to pass 3), with no measurable dose response (caveat 8). Ranks improve for 7 of 8 canaries from
     pass 1 to pass 3 and mean exposure goes 1.95 → 2.76 bits; none of this has a CI over training.
   - Pass 3 is 1–2 nats from rank 1 on 4 canaries†.
6. **The pass-1 recipe reproduces M1c F1** on the same runner (Δ E1 +0.008, Δ Bar A +0.009). M1d's recomputed C0
   equals M1c's to float precision.

## Caveats (must travel with the result)

1. **Epochs = 1 rests on the share clause.**
   - Pass 2 gained +0.241 Bar-A points (gain clause needs < 0.1 to stop) at +0.100 share (clause > 0.05), i.e. 2.0×
     the share threshold. KL sets the share at pass 1 (0.026) and R2 at passes 2–3; the increase is driven by R2,
     mostly R2-ML.
   - Without the share clause the rule gives 2, not 3. Pass 3's Bar-A diff falls by 0.068 because its λ_X drops from
     0.007 to 0.005: at 0.006 its X + counts has R2 0.537 (verifier). The count grid is coarse there
     (0.005/0.006/0.007), so part of the −0.068 is grid granularity.
   - The share ruler carries to M2 unchanged. `experiments/m2/PREREG_COMPARISONS.md` §2 keeps share at M1c's budgets
     max(R1/0.2, R2/0.5, KL/0.005) and uses R2 ≤ 1.0% only as the within-bars test. Under M2's own definition pass
     2's share is still 0.126 and the increase still +0.100, so the §7.1 share clause gives the same answer.
   - Hypothetical only (no pre-registration defines it): rescaling the R2 term to R2/1.0 would give pass 2 a share
     of 0.063† (+0.037†) and the share clause would not fire. λ\*, λ_X and Bar A were not recomputed under it.
2. **Pass 3 loses Bar B** (1.32×, counts@matched 3e-3 at share 0.360). No §7 rule reads this.
3. **The seed replicate measures only window-order and anchor-schedule randomness.**
   - Seed 1 changes the window order, the anchor-window schedule, and through them the warm-up statistics (a0 −1.0%,
     G0 −0.5%; verifier).
   - There is no initialisation variance to measure: rows start at zero and W_v is fixed. Corpus, canaries, keys,
     reserved rows and dev sampling are unchanged.
   - It is one draw, not a variance estimate, so it is a lower bound on real seed variability.
   - R01 (the recipe arm), DF and passes 2–3 are single seeds. R01's Bar B (2.89×) and share (0.026) have no
     replicate; S1 shows one draw can move share 2× and Bar B 2.90× → 2.22×.
   - Retention is more order-sensitive than the dev gain. S1's R2-ML went 0.0007 → 0.031%, its share 0.026 → 0.053.
     That one-seed share difference (0.027) is about half the 0.05 share-clause threshold.
   - §7.5's noise reference uses only |ΔA| = 0.0017, so "not within seed noise" overstates how clean the ρ margin is.
4. **ρ 0.1 and ρ 0.3 are indistinguishable because ρ barely binds.**
   - The ρ rule picked 0.1 on a −0.0025-point margin.
   - R01's train NLL equals P3 pass 1 bit for bit up to step 43, and to within 2.75e-4 afterwards (verifier).
   - The rule's outcome is "ρ 0.1 costs nothing at one pass", not evidence that a tighter budget helps.
   - ρ starts to bind at ρ 0.3 only in pass 3 (20 steps). ρ 0.1 over more than one pass is untested (§8.4).
5. **Distillation was tested in one form only.**
   - The objective forces α = 1 (no true-token term), whatever `distill_alpha` 0.5 in the config says. The teacher
     is the frozen host with about 4,096 BM25-retrieved tokens from other train files.
   - The mean KL to the teacher is flat over the pass, about 0.21 (verifier). `train.json`'s `kl_teacher_first`
     0.140 and `kl_teacher_last` 0.260 are single windows, not a trend.
   - α < 1 and other teachers are untested.
6. **REG v2 is validated through rank 1 only.**
   - PC ranks every canary first but extracts none greedily. So the extraction half of part 3 showed no detection
     power on the runner.
   - PREREG §11.1's local check had PC extracting c6 and c7 exactly. A verifier re-probe of the local pre-launch PC
     cartridge extracts c4, c6 and c7. The runner's PC is 5–16 nats weaker on c4–c7, under the same config.
   - The verifier attributes this to training nondeterminism across machines and thread counts (2 vs 16) at lr 0.1.
     The pre-registered feasibility claim "fails part 3 by both criteria" did not reproduce.
   - The probe's sensitivity between PC's level (Δδ +20 to +90) and the real arms' (≤ +1.76) is not shown.
7. **The runs and true-continuation parts are literal zero tests.** Base is at 0, so every cartridge must be at 0.
   They show no regurgitation, with true LCP at most 13 tokens, but one long reproduction would fail them.
8. **The dose response is effectively unmeasured for the real arms.**
   - There are 2 canaries per dose, and no slope is distinguishable from 0 (SE about 0.18–0.20).
   - The per-canary CIs cover reference sampling only. They exclude 0 even for negative Δδ, such as c1 at P3_e1.
   - Within-dose scatter dominates (×4: c2 +0.19 vs c3 +1.59).
   - Much of the raw true-secret gain is a generic hex shift (references rise by up to +6.66 nats, P3_e3 c5); Δδ
     removes it.
9. **REG v2 did not probe R01 (the M2 recipe's ρ) or DF.** It probed P3 pass 1 (ρ 0.3) and the seed-1 arm. With ρ
   barely binding the cartridges should be close, but R01's own REG is not measured.
10. **Count-grid KL is computed on R1 only (§8.1).** It is the one set the bars read KL from. NLL and EM are the
    same floats (tested), so the bar inputs are unchanged. The dev/R2 grid KL diagnostic of M1c is not reported.
11. **Dev rests on about one file** (`scheduler/scheduler.py` holds 62% of the dev targets; about 2.5 effective
    files, §8.3).
    - The 0.1-point rule margins are close to point decisions on that file.
    - The CIs are over files, not over training seeds.
12. **No test read in M1d, by design.** The recipe change from M1c (ρ 0.3 → 0.1, still one pass) is not confirmed
    on test. M1c's test numbers are for F1 (ρ 0.3). The one file access left is `scripts/evaluate.py` parsing test
    files as text for the definition lexicon (§0).
13. **The rules are separate (§8.4).** Epochs, ρ and objective were each decided against pass 1 at the M1c values of
    the other factors. The chosen combination happens to be R01 itself. Interactions with more passes are untested.
14. **Run conduct.**
    - The run took 12.5 h against the estimated 10.2 h. Training stages running next to concurrent 16-thread stages
      were 1.3–1.5× slower per step than M1c's lone 16-thread run, against the 1.07× the ETA model assumed
      (verifier). Training steps averaged 31.6 s (P3), 30.4 s (S1), 33.4 s (DF) and 34.6 s (R01). R01, the slowest,
      overlapped another training stage (S1) for only its first 9 min; it ran mostly next to the count-grid, PC,
      snapshot-eval and REG2 stages.
    - REG2 peaked at about 8.0 GB RSS (7,992 MB in `reg/summary.json`; `REG2/result.json`
      recorded 7,185 MB). That is above §10's "about 6 GB per stage", but under the 11 GB guard.
    - None of §10's cuts was applied.
15. **Artifact limits.** The cartridges were not uploaded.
    - For the non-base REG conditions, the metrics were verified from recorded token ids and log-probs, not by
      re-running the cartridges.
    - The snapshot identity was checked by row counts, not file hashes.
    - The pass-1 order equality with M1c F1 holds by construction (same generator, 310 windows, unchanged
      `ingest/train.py`), not by a logged comparison.

## For M2 (context for `experiments/m2/PREREG_COMPARISONS.md`; that file is not changed here)

- **OP recipe:** next-token, one full pass over the train windows, ρ 0.1 (non-binding at one pass), no distillation,
  run with ≥ 2 seeds.
  - `PREREG_COMPARISONS.md` §3's cost line takes OP's tokens as 3 epochs. With one epoch, its "fewer if OP uses
    fewer epochs" clause applies: equal-token LoRA is one pass in the same seed-0 order.
- **Re-decide epochs under M2's own bars, or carry M1d's rule explicitly.**
  - The M1d stop came from the share clause. `PREREG_COMPARISONS.md` §2 keeps M1c's share ruler (R2/0.5), so under
    M2's own definition the clause still fires at +0.100 (caveat 1). Only a hypothetical R2/1.0 rescaling, which no
    pre-registration uses, would give +0.037†.
  - If M2 uses more than one pass, §6.5 still requires REG at that pass count. REG v2 passed at pass 3, but its
    memorisation signal grows with passes.
- **Expect seed noise on the retention axis.**
  - Matched-cost comparisons (Bar B, E5m LoRA@matched drift, E6s) read the share axis, and one seed draw moved share
    0.026 → 0.053.
  - Seeds should be reported per comparison, not only for the dev gain.
- **REG for M2:**
  - run it on the actual OP cartridge (ρ 0.1);
  - add a positive control that also extracts greedily on the target machine, plus an intermediate-strength control
    to show sensitivity between PC and the real arms;
  - use more than 2 canaries per dose.
- **Timing:** recalibrate the contention factor for concurrent 16-thread stages (1.3–1.5×, not 1.07×).

## Verification

Three independent read-only checks re-derived the results with their own code. They ran on the per-target arrays,
the REG2 jsonl outputs and the run's logs, and no code from the run directory was executed. None found anything that
changes a §7 decision or a REG v2 verdict. The caveats above include theirs.

- **Bars and §7 rules.**
  - All 494 compared Z_report quantities reproduce to within 5.3e-15.
  - The non-dup mask was rebuilt from exact train 16-grams (146,340 of 148,509 targets) and agrees with the run's
    on every target, for every arm.
  - Every §7 decision and the M2 recipe reproduce.
  - M1d's C0 equals M1c's to float precision at all 16 weights (max |ΔE1| 3.3e-8).
  - The bootstrap CIs reproduce bit for bit, but with the same resampling scheme, so they are not an independent
    method.
- **REG v2.**
  - The planted train stream was rebuilt (204 files, 1,110,938 tokens), with the same canary windows as `train.json`.
  - Every bar input in `REG2/reg/summary.json` reproduces exactly: runs (exact k-gram search), true LCP, ranks,
    Δδ, slopes, rows read and specific rows.
  - The base greedy probes and 40 reference scores were re-run locally and match (within 6e-5 nats).
  - The verdicts and `validated: true` reproduce. The one discrepancy found is the §11.1 positive-control
    extraction (caveat 6).
- **Integrity.**
  - The grid sha256 matches the launch commit.
  - No stage scored a test file. The evaluated sets are dev, R1, R2 and R2-ML only.
  - The P3 snapshots are at steps 310, 620 and 930, with matching row counts. The pass windows follow the
    pre-registered permutations.
  - S1 differs from P3 pass 1 only in order, anchor schedule and warm-up statistics. R01's ρ binding and DF's teacher
    coverage (all 310 windows) are confirmed.
  - The metrics have no NaN or Inf, no stage has a guard kill or retry, and RSS is stable across passes.
  - The wall-time overrun is per-stage slowness, not scheduling: two fully packed slots, 24.75 slot-hours of stage
    time.
- **Redaction.** `redact.py` reported success (39 files, 8 secrets). An independent scan found 0 occurrences of any
  of the 8 secrets, and of any of their 16-character substrings, both in `results/runner-2026-11-12/` and anywhere
  under `ngram-memory/` (excluding `.git`).
