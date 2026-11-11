# M1c results (self-hosted runner, 2026-11-11)

This is the canonical M1c run under `PREREG.md`.

- **Run:** GitHub Actions [run 42971180317](https://github.com/patcarter883/memory-organ/actions/runs/42971180317), launch commit `1eec009`, on 32 cores with `--grid experiments/m1c/grid.json --parallel 4 --max-stage-threads 16`.
- **Time:** 06:15 to 15:22 UTC, 9 h 06 min. One invocation; every stage done, nothing soft-failed (`finished` and `complete` both true).
- **Grid:** sha256 `5d94254f…`, which equals `grid.json` at the launch commit.
- **Results:** `results/runner-2026-11-11/` holds every `result.json`, `progress.json`, eval `summary.json`, `arm.json`, `train.json` and canary manifest. The planted canary secrets are replaced by `<redacted-canary>`; the manifest hashes are kept. The full artifact (per-target npz, logs) is `ngram-m1c-42971180317`, sha256 `a6c9e1af…`, kept until 2026-12-11.

## Verdict: PASS (§7) → M2 with the operating point's recipe

**Operating point (OP) = F1_fullpass:**
- **Recipe:** next-token training, G-lin at layer 1, one full pass of 310 steps over the Split-A train windows.
- **ρ:** 0.3, but not binding (note 2).
- **S1:** N3_rho0p3 (64 steps). F1 was preferred because it holds both bars on dev and S1 holds neither.

### Test (the single M1c read of the Split-A test files, the second read overall)

19 files, 57,399 non-duplicated targets. This is the primary metric. All-target numbers are in brackets after the "/".

| Condition | E1 gain % [file-cluster CI] |
|---|---|
| OP alone | **+1.62** [1.27, 1.86] / all targets +1.71 |
| counts@λ\* (KN only, weight 0.0075) | +2.28 [1.66, 2.68] |
| counts@λ\* + OP | +3.67 [2.83, 4.23] |
| counts@matched(OP) (weight 3e-4, the same retention share as OP) | +0.54 [0.38, 0.63] |
| perm:OP (permuted keys, content control) | −0.006 [−0.013, 0.000] |

| Bar | Value | Needed | Result |
|---|---|---|---|
| **Bar A**, adds to counts: E1(OP + counts@λ_X) − E1(counts@λ\*), with λ_X = λ\* = 0.0075 | **+1.39 points** [1.13, 1.55] | ≥ 1.0, CI lower bound > 0.5 | **holds** |
| **Bar B**, beats counts at equal cost: E1(OP) / E1(counts@matched) | **3.02×**, diff +1.08 [0.84, 1.25] | ≥ 1.5×, diff CI lower bound > 0 | **holds** |
| C1: perm gain / OP gain | −0.004 | ≤ 1/3 | holds |
| OP retention on dev | R1 +0.0055%, R2 −0.005%, KL 1.3e-4 (2.7% of the budget) | R1 ≤ 0.2%, R2 ≤ 0.5%, KL ≤ 0.005 | holds |
| REG (OP vs base) | runs ≥ 64 tokens 6/200 vs 6/200; no canary reproduced; 0 secret hits | as §6 | holds (caveat 7) |
| Gates | host hash unchanged; dedupe met by the non-dup primary (test 16-gram-in-train fraction 3.83%) | | holds |

The R0/I2/I4 test suites (`test_hashing.py`, `test_graft_tiny.py`, `test_qwen25_real.py`) are not run by the runner. Run locally at code identical to the launch commit, they pass (73 tests, none skipped).

### Dev (selection reads these only; 13 files, 146,340 non-dup targets)

C0 counts frontier: λ\* = 0.0075 gives E1 +2.02%, R1 +0.19%, R2 +0.45%, KL 0.00485. KL binds.

| Arm | Objective | ρ | Steps | E1 | Retention share | Bar A diff [CI] | counts@matched | Bar B ratio | Bars |
|---|---|---|---|---|---|---|---|---|---|
| N1_rho0p03 | next-token | 0.03 | 64 | +0.57% | 0.080 | +0.45 [0.40, 0.67] | 5e-4 | 0.98 | none |
| N2_rho0p1 | next-token | 0.1 | 64 | +0.60% | 0.095 | +0.47 [0.42, 0.70] | 5e-4 | 1.03 | none |
| N3_rho0p3 (S1) | next-token | 0.3 | 64 | +0.60% | 0.095 | +0.47 [0.42, 0.70] | 5e-4 | 1.03 | none |
| D1_distill_rho0p03 | distill | 0.03 | 64 | +0.23% | 0.178 | +0.02 [−0.01, 0.14] | 1e-3 | 0.28 | none |
| D2_distill_rho0p1 | distill | 0.1 | 64 | +0.25% | 0.193 | +0.03 [0.00, 0.16] | 1e-3 | 0.30 | none |
| **F1_fullpass (OP)** | next-token | 0.3 | 310 | **+1.29%** [1.16, 1.72] | 0.027 | **+1.14** [1.04, 1.53] | 3e-4 | **2.88** | A and B |

The table's own retention share is max(R1/0.2, R2/0.5, KL/0.005). Every arm alone is within the bars.

## What this shows

1. **A full pass doubles the table's gain and changes the M1b verdict.**
   - M1b's 64-step table sat on the count-LM's gain/cost curve; the 64-step N1 here reproduces that (Bar B ratio 0.98).
   - One full pass gives +1.29% dev and +1.62% test (non-dup), at an even lower retention cost.
   - Against counts at the same cost (weight 3e-4), that is 3× the gain. On top of counts at the full retention budget, it adds +1.39 points.
2. **Most of F1's gain is more training on keys already seen, not wider coverage.**
   - The first 64 steps of F1 reproduce N3 exactly. On dev, 69% of the F1 − N3 gain is on targets whose key N3 had already trained, 12.5% on keys first seen after step 64, and 19% on unseen or count < 2 keys.
   - The F1/N3 ratio is about 2.0–2.4 across every token class.
   - So **every 64-step arm was undertrained**, and the 64-step comparisons say little about a full pass: next-token vs distill, and the ρ ordering.
3. **Distillation (Stage-1b) at 64 steps was clearly worse than next-token** (+0.23–0.25% vs +0.57–0.60%), at about twice the retention share. It was never run for a full pass, so this does not rule it out at full training.
4. **The table complements the counts.** Counts + OP beats either alone at every cost level, and OP repairs the counts' retention cost (OP alone improves R2-ML).

## Caveats (must travel with the PASS)

1. **"Beats counts" means at equal retention cost, and against this comparator.**
   - At full budget, counts alone beat OP alone on test (2.28 vs 1.62). OP's case rests on being cheap (2.7% of the budget) and on adding to counts (Bar A), not on being the better stand-alone memory.
   - The comparator is the pre-registered always-on KN trigram without the in-file cache. On the same test targets it is weaker than M1b's count-LM (+2.39% vs +2.95% all-target). Counts + OP, +3.85% all-target, still exceeds M1b's count-LM.
2. **ρ was not selected.**
   - The ρ penalty was zero at every step of N3 and F1 (`rho_train_frac_over` = 0), and N3 beat N2 by 3.6e-5 points.
   - The recipe is "ρ not binding at ≥ 0.1". F1's ρ p99 reached about 0.15, so a full pass at ρ 0.1 might have bound, and it is untested.
3. **One seed and one full-pass arm.** OP's dev Bar-A margin was +0.14. The CIs are over files, not over training seeds; seed variance is unmeasured. The test margins are large: Bar A is above 1.0 under every leave-one-file-out cut (1.26–1.42), and every test file is positive for both bars.
4. **λ_X = λ\* = 0.0075 sits on the dev KL edge.**
   - OP + counts there has KL 0.00498 against 0.005; resampling R1 puts it over in about 19% of draws.
   - Dev Bar A still holds at 0.007 (+1.09), and Bar B on test does not depend on it.
5. **Dedupe sensitivity.**
   - With stricter dup masks, Bar B holds at every threshold down to 4-grams (ratio 3.0–3.5). Bar A holds down to 6-grams (+1.17) and falls just under 1.0 at 4-grams (+0.97, with a third of targets dropped).
   - The duplicated slice is where both methods gain most (OP +5.6%, counts +6.7%), so all-target numbers overstate the gain.
6. **The test read is the second read of these files**, after M1b's.
   - M1c was designed after M1b's test numbers were known. No threshold was tuned on them: the bar values are unchanged from the first draft, and the §12 changes were checked on M1b dev arrays.
   - Both choices M1b's numbers informed make the test harder: the non-dup metric and the counts comparator.
7. **REG says "no detectable regurgitation or canary memorisation". It is weaker than a plain pass:**
   - **The run metric is mostly a loop count.** All 12 runs of ≥ 64 tokens (6 for base, 6 for OP) are the model looping over its own prompt. Excluding the prompt window gives 0/200 vs 0/200 (max 54 vs 32 tokens).
   - **The canary dose was minimal.** Each canary was planted twice, both copies in the same training window, so the canary-only rows got about one update. Per-canary Δlog p of the true secret (OP − base, net of the reference shift) averages +0.09 nats on an about 89-nat secret. The largest move is c4, rank 40 → 16 of 100.
   - **No positive control** showed the probe can detect row memorisation. An in-context control shows the probe's full range.
   - **Probe contexts contain other canaries' secrets.** Greedy probes copy them in 5 of 8 cases; the verdict code only scores each canary's own secret.
8. **Not re-run for OP:**
   - capability R3, Split B, RAG and kNM (M1b's numbers were for a different table);
   - out-of-distribution retention.

   R1 comes from the same distribution as one KL anchor (WikiText train vs test), and R1 is what binds OP's share. The R2 stdlib set is disjoint from the stdlib anchors.

## For M2 (from this run)

- **Recipe:** OP's, next-token with a full pass. M2 should treat ρ as a free parameter (try ρ 0.1 at a full pass) and run ≥ 2 seeds.
- **Training length:** epochs > 1 are untested, and the 64-step arms show training length is the main lever. Distillation deserves one full-pass arm before it is dropped.
- **REG:**
  - a verbatim-run metric that excludes the prompt;
  - true-continuation LCP as the primary statistic;
  - a canary dose-response (repeats 2/4/8/16 in different files and windows, after each epoch);
  - a positive control;
  - canary contexts with other canaries' secrets masked;
  - ≥ 1,000 reference secrets;
  - whether the canary rows survive into the shipped cartridge.
- **Count-LM in the serve:** this stays the owner's separate concept. M1c shows counts + table beats either alone, which supports testing the two together there.

## Verification

Four independent read-only checks re-derived every number above from the raw per-target arrays, using their own code (key = doc << 32 | pos, file-cluster bootstrap, 10k, seed 0). Every test and dev number in `Z_report` reproduces to all printed digits.

- **Non-dup mask:** rebuilt from the 204 train files alone; it agrees 100%.
- **Test read:** only `T_test` read the test set, once.
- **F1 vs N3:** the configs are identical except `steps`, and all 310 window starts reproduce.
- **REG:** ran on F1's cartridge, and the canary-free check covered all 46 held-out files.

None of the checks found anything that changes the verdict. The caveats above are theirs.
