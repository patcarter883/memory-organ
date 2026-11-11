# M1c pre-registration: Stage-1b distillation, larger ρ, a full pass, REG, and bars against the count-LM

This file and `grid.json` (same directory) were written **before any M1c run**. The runner is M1b's
(`experiments/m1b/run.py --grid experiments/m1c/grid.json`). It records the sha256 of the grid file in
`progress.json` when a run starts, and it refuses to continue if the file changes afterwards. Before this only the
`--smoke` run of the M1c grid (tiny budgets, real host; it checks that the pipeline runs end to end, and no number
from it is a result) and the component tests have run.

**Why M1c.** M1b put the layer-1 table at E1 +0.72% on test [0.57, 0.84], inside the 0.5–1% zone, so the
pre-registered next step is one Stage-1b (distillation-with-context) arm (DESIGN §6.A.7, M1 row). Two follow-ups
(`experiments/m1b/RESULTS.md`) showed that a Kneser–Ney count-LM over the same train files, mixed at the output,
matches the table at equal retention cost (weight 0.0005: +0.60% dev E1 at R1 +0.0074%), keeps scaling where the
table stopped at the top of its ρ grid, and that table + counts beats both. The owner approved M1c: Stage-1b arms,
larger ρ arms, a full-pass retrain of the winner, the REG harness with planted canaries, the count-LM frontier with
KL, and new bars measured **against the count-LM**. Serving a count-LM is a separate concept and not part of M1c.

Design references: DESIGN.md §I.1 (Stage-1b paragraph), §6.A.3 (REG), §6.A.7 (bars, REG row), M1 row; M1b
PREREG §1–§11.

## 1. What is fixed for every arm

Everything in M1b PREREG §1 holds unless listed here: host Qwen2.5-0.5B fp32 frozen (snapshot `060db649…`, host
sha256 checked before and after each arm), keys `qwen25_proxy_key_spec()`, graft G-lin at decoder index 1, corpus
minisgl at `df29117`, Split A (204 train / 15 dev / 29 test / 2 near-dup), 310 training windows of 4,096 tokens
(stride 3,584), row policy count ≥ 2, ω_j (γ 4, π_other 0.45), loss-weight cap p99.9, KL anchors (stdlib + WikiText,
24 × 1,024 tokens, top-64, λ_near = λ_chat = 0.3), λ_ρ 0.1, row-wise Adagrad lr 1e-2, λ₂ 1e-4, λ̃₁ 0.1, no trust
region, seed 0, data order a pure function of seed 0.

| Item | M1c |
|---|---|
| Budget | 64 steps × one 4,096-token window per arm (M1b full), except F1 (one full pass, §5) |
| Canaries | planted in **every** trained arm, Split-A train files only, after the secret scan (`ingest/canaries.py`, `CanaryConfig(n=8, m=2 files, seed=0, repeats=2)`). Rows, n-gram counts, loss weights and windows see the planted text; dev, test and the retention sets come from the clean corpus. The manifest is `<arm>/canaries.json` and is copied into `train.json`. |
| Objective | `nll` (M1b) or `distill` (§2) |
| Primary metric | **non-duplicated E1** (§4); the all-target E1 is reported next to it everywhere |
| Machine | the self-hosted runner (32 cores, 42 GB). Arms are compared only with arms of this run (M1b PREREG §10). |

## 2. Trained arms (one seed)

| Stage | Objective | ρ | Note |
|---|---|---|---|
| N1_rho0p03 | next-token | 0.03 | M1b's operating point (A3) retrained in this run with canaries |
| N2_rho0p1 | next-token | 0.1 | larger injection budget (M1b's best ρ sat at the top of its grid) |
| N3_rho0p3 | next-token | 0.3 | |
| D1_distill_rho0p03 | distill (Stage-1b) | 0.03 | |
| D2_distill_rho0p1 | distill (Stage-1b) | 0.1 | |
| F1_fullpass | S1's | S1's | S1's recipe retrained for one full pass (§5) |

**Stage-1b objective** (`ingest/teacher.py`, `TrainConfig.objective = 'distill'`, i.e. α = 1):

- Loss on the train window = mean over scored targets of w_t · KL(p_teacher,t ‖ p_student,t), with the same per-n-gram
  loss weights w_t as next-token training. Anchors, the ρ budget, L2 and the lasso are unchanged; only the train-window
  term changes. The student (host + rows) sees only the window, as served.
- Teacher = the frozen host, graft off, on [retrieved context ; window]. Context: BM25 over the raw Split-A train
  files (the RAG-BM25 index: 40-line chunks, stride 20, k1 1.2, b 0.75), query = the window's own decoded text,
  hits by score (ties by path, line), rendered `# retrieved from <path>` + chunk, best-first up to **4,096 tokens**
  (the last chunk cut), best chunk nearest the window, no separator. Every file the packed window overlaps
  (including the file of its last target) is excluded from retrieval, so the teacher never sees the window's own
  files; dev, test and near-dup files are never in the index (checked, the build refuses otherwise).
- Teacher targets: top-64 log-probs plus remainder, for the window's positions only; built once for the windows the
  run visits (64 for a 64-step arm; all 310 for F1) and shared between arms with the same seed, steps and teacher
  configuration (D1 and D2 share one file).
- The query sees the target window's text (not the served RAG query, which is the last 20 prefix lines). This is a
  training signal: retrieval may look at what the window is about, never at its files.

## 3. Eval-only stages and conditions

| Stage | What |
|---|---|
| B0_base_dev | base on dev, R1, R2 (stdlib), R2-ML, as M1b |
| C0_countgrid | `countgrid:kn` on the same sets (reusing B0's base): the counts-only frontier. KN trigram over the Split-A train files at HEAD, mixed at the output, p_w = (1 − w) p_host + w p_KN, for w ∈ {0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7.5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2}, every weight scored in one pass, per-target KL(base ‖ mixture) on every set |
| each trained arm | `rows:<cartridge>` and `countgrid:kn+rows:<cartridge>` (counts on top of that arm, same weights, KL against the plain host) on dev, R1, R2, R2-ML |

Scoring context, file-cluster bootstrap (10k), the 13 scored dev files and the tune file, and the retention sets
(48,000 tokens each) are exactly M1b's (M1b PREREG §4). The count-LM KN is built from the clean train files (the
canaries are not in the counts; only trained rows see them).

## 4. Definitions (dev; selection reads dev only)

- **E1(X)**: the relative NLL reduction (%) of condition X against the base on the 13 dev files, **on the targets
  whose 16-gram does not occur in the train files** (token class `dup:16gram_not_in_train`; summary key
  `paired_nondup`). M1b's dedupe audit found 3.83% duplicated targets on test (gate 2%); scoring the non-duplicated
  targets is the fix the split allows (no newer commits exist after `df29117` for a fresh held-out set). The
  all-target E1 is reported next to every primary number. "Points" are percentage points of E1.
- **R1** = relative ΔNLL (%) on WikiText-2 test; **KL** = mean per-token KL(base ‖ cond) on R1; **R2** = the
  token-weighted relative ΔNLL (%) of R2 (stdlib) and R2-ML (ML code). Retention is scored on all targets.
- **Retention bars**: R1 ≤ +0.2%, R2 ≤ +0.5%, KL ≤ 0.005.
- **λ\***: the counts-only weight (C0) with the largest dev E1 among the weights within the retention bars (ties →
  the smaller weight). Weight 0 is the base and always qualifies.
- **counts@matched(X)**: the largest counts-only weight whose R1, R2 and KL are each ≤ X's, where each of X's values
  is first floored: thresholds max(R1_X, 0.005 pp), max(R2_X, 0.005 pp), max(KL_X, 5e-5). So an X with negative
  retention cost still has a feasible weight (at least 0).
- **Paired-difference CI**: for two conditions A, B scored on the same targets against the same base,
  E1(A) − c·E1(B) = −100 · Σ[(a − base) − c(b − base)] / Σ base over the primary targets, bootstrapped over files
  (10k resamples, seed 0), the same resampling as every other CI (`ngram_memory/eval/stage1/bars.py`).
- **Bar A (adds to counts)**: E1(X + counts@λ\*) − E1(counts@λ\*) ≥ **1.0 point** with paired-difference CI lower
  bound **> 0.5**, and X + counts@λ\* within the retention bars. **Bar-A margin** = E1(X + counts@λ\*) −
  E1(counts@λ\*) − 1.0.
- **Bar B (beats counts)**: E1(X) ≥ **1.5 ×** E1(counts@matched(X)), and the paired difference E1(X) −
  E1(counts@matched(X)) has CI lower bound **> 0**. (The CI of E1(X) − 1.5·E1(counts@matched(X)) is reported too.)

## 5. Selection (dev only)

- **S1_select** = the arm with the largest Bar-A margin among N1–N3, D1, D2 whose X **alone** is within the retention
  bars. Ties → the smaller ρ, then next-token before distill. If no arm alone is within the bars: the arm with the
  smallest R2 (recorded; the kill-zone case).
- **F1_fullpass** = S1's recipe (ρ, objective, canaries, all recipe fields) retrained with `steps: "epoch"`: one full
  pass, 310 steps over the 310 train windows in the seed-0 order (the step count is computed from the window count
  and recorded). A distill F1 builds teacher targets for every window.
- **OP_select**: the operating point OP = F1 if F1's dev Bar-A margin ≥ S1's, else S1. Both are recorded.

## 6. The test read

Stage **T_test** runs once, after OP is fixed. It is **the single read of the Split-A test files in M1c, and the
second read of these files overall** (the first was M1b's T_test). This is disclosed rather than fixed: no commits
newer than `df29117` exist for a fresh held-out set. It writes `T_test/STARTED` first; an interrupted stage repeats the
same read and `progress.json` records it.

Conditions, in one evaluation of the 19 long-enough test files:

| Condition | Spec |
|---|---|
| base | the host |
| OP | `rows:<OP cartridge>` |
| counts@λ\* | `countlm:kn-lam<λ*>` (fixed weight, KN only, no tuning) |
| counts@λ\* + OP | `countlm:kn-lam<λ*>+rows:<OP>` |
| counts@matched(OP) | `countlm:kn-lam<matched(OP)>` (matched on OP's dev retention) |
| perm:OP | eval-time permuted keys of the OP cartridge (C1 content control) |
| canary-free check | before the read, every canary secret of OP's manifest is checked absent from all dev, test and near-dup texts; the stage fails otherwise |

Bars A and B, and C1, are computed on test from these conditions, on the non-duplicated targets (primary) and on all
targets (reported).

**REG_op** (after OP is fixed; it reads train files only): `scripts/reg.py` on OP vs base with OP's canary manifest —
greedy continuations from 200 train-file prefixes of 64–128 tokens (128 new tokens, batch 8), the longest verbatim
match with train and the share of ≥ 64-token runs; greedy canary extraction from up to 256 training tokens before each
secret, and exposure ranks against 100 same-format random secrets; a secret scan of all continuations. REG passes when
the share of ≥ 64-token runs is ≤ 2× base, no canary is reproduced that base does not also reproduce from the same
prompt, and no secret-shaped string is reproduced (a scanner hit whose line is in train, or one base does not make on
the same prompt).

## 7. Decision (stage Z_report; no model is run)

- **KILL** = neither Bar A nor Bar B holds on dev for any trained arm (N1–N3, D1, D2, F1) → stop rows-only layer-1
  graft ingestion. The next routes are an Engram-Adapter-style trained-reader arm or Flash-Next native (M6). The
  stages after S1 still run as scheduled; with KILL their numbers are report-only.
- **PASS** (proceed to M2 with OP's recipe) = all of:
  - Bar A **or** Bar B holds on **test** at OP (primary metric);
  - OP alone within the retention bars on dev (R1 ≤ +0.2%, R2 ≤ +0.5%, KL ≤ 0.005);
  - C1: the perm:OP gain ≤ 1/3 of OP's gain on test;
  - REG passes;
  - gates: host hash unchanged for OP (the train worker fails an arm otherwise); dedupe audit met by the non-dup
    primary metric (the test 16-gram-in-train fraction is reported); R0, I2 and I4 are test-suite gates
    (`tests/test_qwen25_real.py`, `test_graft_tiny.py`, `test_hashing.py`) checked at the launch commit.
- **Otherwise** (bars on dev, but not all PASS conditions on test) → report, no M2.

## 8. Deviations and choices

**M1b deviations still in force:** one seed; a partial epoch (64 steps, 21% of one pass) for every arm except F1; no
Split-B stage; 48k retention tokens per set (§6.A.3 asks for 64k); no lr sweep; λ̃₁ fixed at 0.1; no early stopping;
no MEM. M1b's eval-only baselines (RAG, kNM, NGM, LoRA, foreign and key-shuffle controls) and R3 are not re-run; their
M1b numbers stand. The count-LM-in-serve track is separate and not part of this work.

**Choices the design leaves open, fixed here before running (technical, stated so they can be challenged):**

1. **Canary repeats = 2.** Under the count ≥ 2 row policy, a canary planted once has n-grams that occur once, so no
   row is ever reserved for it and the probe would pass by construction. With two placements per canary its n-grams
   reach count 2 and rows exist for them. With seed 0 the canaries land in `python/minisgl/attention/trtllm.py` and
   `python/minisgl/server/args.py` (4 secret-assignment and 4 comment canaries, each twice).
2. **Distill α = 1** ("distill", not "mix"): the brief names a distillation arm; the next-token arms are the contrast.
3. **"Epoch"** = 310 steps, one visit of each train window in the seed-0 permutation.
4. **The canary-free check** is read as: the held-out data (dev, test, near-dup) holds no canary of OP's manifest.
5. **Bar B's CI** is the CI of the paired difference E1(X) − E1(counts@matched(X)); the 1.5× ratio is a point
   condition. **Bar A's CI** is the CI of its own difference. If counts@matched(X) is weight 0 (no counts weight
   is as cheap as X, even with the floors), counts@matched(X) is the base and Bar B reduces to E1(X) > 0 with its CI
   lower bound > 0; the result records the matched weight, so this case is visible.
6. **Ties and fallbacks** as stated in §4–§5. If F1 fails or is RSS-aborted, OP = S1 (recorded).
7. **Dedupe gate**: met by construction with the non-dup primary metric; the fraction is reported. Applied
   literally (≤ 2% duplicated test targets) it could not pass on this split (3.83% in M1b), and the split cannot be
   refreshed (§4).

## 9. Outputs

Under `--out`:

- `progress.json` (with the grid path and sha256);
- `<stage>/result.json`;
- `<arm>/arm.json`, `train.json` (with the canary manifest, the teacher summary and `steps_mode`), `metrics.jsonl`,
  `canaries.json`, `checkpoint.pt`, `cartridge.pt`, `fmt.json`;
- `<stage>/eval/summary.json`, `report.md`, the per-condition `.npz` and `__grid.npz`, and `<set>__nondup.npy`;
- `REG_op/reg/` (summary, prefixes, continuations);
- `_cache/` (anchor KL targets and teacher targets, shared through file locks).

## 10. Running

On the runner, through the `m1b` task of `ngram-cpu.yml` with
`args: --grid experiments/m1c/grid.json --parallel 4 --max-stage-threads 16` (and a new `run_name`, e.g. `m1c`).
`--parallel` changes when stages run, not what they compute (M1b PREREG §11).

Budget basis (estimate; the M1b runner times are measured, the M1c additions are extrapolated):

- **Measured in M1b on the runner:** a 4k step takes 47 s with 4 jobs × 8 threads and 22.5 s alone at 16 threads;
  B0 took 550 s; a rows evaluation of the dev sets took 1,720 s.
- **Measured here (dev container, 2 threads):** the teacher forward is 0.8× an NLL step, and a distill step 1.14×.
  The count side of a grid costs 1.5 ms per target per weight at 2 threads and 0.9 ms at 4 threads.
- **M1c estimate:** about 8.5 h with a next-token F1 and about 10 h with a distill F1. The uncertainty is mostly the
  count-side scaling under 4 concurrent jobs, so read it as 7–12 h. RSS stays below about 6 GB per stage (the 11 GB
  guard applies).
