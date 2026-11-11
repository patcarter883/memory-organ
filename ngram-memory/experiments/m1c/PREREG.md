# M1c pre-registration: Stage-1b distillation, larger ρ, a full pass, REG, and bars against the count-LM

This file and `grid.json` (same directory) were written **before any M1c run**. A pre-launch review changed both
before anything ran on the runner; every change is listed with its reason in §12. The runner is M1b's
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
| C0_countgrid | `countgrid:kn` on the same sets (reusing B0's base): the counts-only frontier. KN trigram over the Split-A train files at HEAD, mixed at the output, p_w = (1 − w) p_host + w p_KN, for w ∈ {0, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 7e-3, 7.5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2} (16 weights; 3e-4, 6e-3 and 7e-3 added before launch, §12), every weight scored in one pass, per-target KL(base ‖ mixture) on every set |
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
- **Budget share**: share(X) = max(R1⁺/0.2, R2⁺/0.5, KL⁺/0.005) with t⁺ = max(t, 0): the largest fraction of a
  retention bar that X uses (a negative cost uses none). share ≤ 1 is the same as "within the retention bars".
- **λ\***: the counts-only weight (C0) with the largest dev E1 among the weights within the retention bars (ties →
  the smaller weight). Weight 0 is the base and always qualifies.
- **λ_X**: for a trained arm X, the weight of X's own `countgrid:kn+rows` grid (§3) with the largest dev E1 among
  the weights where X + counts is within the retention bars (ties → the smaller weight). Weight 0 is X alone and
  qualifies only if X alone is within the bars. If no weight qualifies, X has no λ_X and Bar A fails for X; the
  result records why.
- **counts@matched(X)**: the largest counts-only grid weight w with share(counts@w) ≤ max(share(X), s_floor),
  **s_floor = 0.01** (1% of the budget). Weight 0 (share 0) always qualifies. The floor only matters when X's cost
  is about 0; it keeps a non-zero weight feasible then (on M1b's numbers counts@1e-4 has share ≈ 0.008, §12).
- **Paired-difference CI**: for two conditions A, B scored on the same targets against the same base,
  E1(A) − c·E1(B) = −100 · Σ[(a − base) − c(b − base)] / Σ base over the primary targets, bootstrapped over files
  (10k resamples, seed 0), the same resampling as every other CI (`ngram_memory/eval/stage1/bars.py`). When A and B
  come from different stages (Bar A: X's evaluation and C0's), the runner checks that the target keys, the primary
  mask, the base NLL and the file groups are identical, and fails the stage otherwise.
- **Bar A (adds to counts; frontier to frontier)**: E1(X + counts@λ_X) − E1(counts@λ\*) ≥ **1.0 point** with
  paired-difference CI lower bound **> 0.5**. Both sides are within the retention bars by construction.
  **Bar-A margin** = E1(X + counts@λ_X) − E1(counts@λ\*) − 1.0 (none when X has no λ_X). The same-weight difference
  E1(X + counts@λ\*) − E1(counts@λ\*) is reported next to it (not a bar).
- **Bar B (beats counts)**: E1(X) **> 0**, E1(X) ≥ **1.5 ×** max(E1(counts@matched(X)), 0), and the paired
  difference E1(X) − E1(counts@matched(X)) has CI lower bound **> 0**. (The CI of E1(X) − 1.5·E1(counts@matched(X))
  is reported too.)

## 5. Selection (dev only)

- **Selection order** (S1 and OP): the tuple (any bar holds on dev [Bar A or Bar B], Bar A holds, Bar-A margin [no
  λ_X ranks last]), larger first; ties → the smaller ρ, then next-token before distill.
- **S1_select** = the best arm by the selection order among N1–N3, D1, D2 whose X **alone** is within the retention
  bars. If no arm alone is within the bars: the arm with the smallest R2 (recorded; the kill-zone case).
- **F1_fullpass** = S1's recipe (ρ, objective, canaries, all recipe fields) retrained with `steps: "epoch"`: one full
  pass, 310 steps over the 310 train windows in the seed-0 order (the step count is computed from the window count
  and recorded). A distill F1 builds teacher targets for every window.
- **OP_select**: the operating point OP = F1 if F1's tuple (any dev bar, Bar A holds, Bar-A margin) ≥ S1's, else
  S1. So OP is an arm that holds a bar on dev whenever one exists among S1/F1. Both are recorded. If F1 failed or
  was RSS-aborted, OP = S1 and the rule records `F1 failed: <error>` (F1 is an optional dependency of OP_select,
  §12).

## 6. The test read

Stage **T_test** runs once, after OP is fixed. It is **the single read of the Split-A test files in M1c, and the
second read of these files overall** (the first was M1b's T_test). This is disclosed rather than fixed: no commits
newer than `df29117` exist for a fresh held-out set. It writes `T_test/STARTED` immediately before the test
evaluation, after OP is resolved and the canary-free check passed; an interrupted stage repeats the same read and
`progress.json` records it. (T_test is a soft stage, §12: a failure is retried at most once, on a resume.)

Conditions, in one evaluation of the 19 long-enough test files:

| Condition | Spec |
|---|---|
| base | the host |
| OP | `rows:<OP cartridge>` |
| counts@λ\* | `countlm:kn-lam<λ*>` (fixed weight, KN only, no tuning) |
| counts@λ\* + OP | `countlm:kn-lam<λ*>+rows:<OP>` |
| counts@λ_X(OP) + OP | `countlm:kn-lam<λ_X(OP)>+rows:<OP>`, λ_X(OP) fixed on dev (§4); only when λ_X(OP) ≠ λ\* (else the row above; λ_X = 0 is OP) |
| counts@matched(OP) | `countlm:kn-lam<matched(OP)>` (matched on OP's dev retention) |
| perm:OP | eval-time permuted keys of the OP cartridge (C1 content control) |
| canary-free check | before the read, every canary secret of OP's manifest is checked absent from all dev, test and near-dup texts; the stage fails otherwise |

Bars A and B, and C1, are computed on test from these conditions, on the non-duplicated targets (primary) and on all
targets (reported). Test Bar A = E1(counts@λ_X(OP) + OP) − E1(counts@λ\*) with its CI (no λ_X(OP): Bar A fails on
test too); test Bar B uses counts@matched(OP).

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
- **INCOMPLETE**: bars on dev, but T_test or REG_op has no result (they are optional dependencies of Z_report,
  §12). Z_report still runs and records the reason. KILL is a dev-only verdict and is written whatever T_test and
  REG_op did.

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
5. **Bar B's CI** is the CI of the paired difference E1(X) − E1(counts@matched(X)); the 1.5× ratio and E1(X) > 0
   are point conditions. **Bar A's CI** is the CI of its own difference. If counts@matched(X) is weight 0 (no
   non-zero counts weight within max(share(X), 0.01)), counts@matched(X) is the base and Bar B reduces to E1(X) > 0
   with its CI lower bound > 0; the result records the matched weight and both shares, so this case is visible.
6. **Ties and fallbacks** as stated in §4–§5. If F1 fails or is RSS-aborted, OP = S1 (recorded).

**Disclosures (added before launch, §12):**

1. **Dev selection rests on about one file.** `minisgl/scheduler/scheduler.py` holds 62% of the dev targets
   (M1b's dev arrays, all targets); the effective number of dev files is n_eff = 1/Σ share² ≈ 2.5 of 13. Dev
   decisions are close to point decisions on that file. Z_report lists the per-file Bar-A differences on dev and
   test with each file's share of the targets. On test (19 files) the CI conditions are not expected to be the
   binding ones.
2. **What KILL covers.** KILL is judged on S1's recipe at one full pass (F1) and on every other recipe at 64 steps
   (21% of a pass), one seed, with no tie band. A recipe that loses at 64 steps but would win at a full pass is not
   tested.
3. **Teacher retrieval near-duplicates.** The teacher's own-file exclusion is by path. Near-duplicate siblings in
   the MinHash cluster that was moved to train (`llama`, `mistral`, `qwen2`, `qwen2_moe`, `qwen3_moe` model files,
   all train) can still be retrieved for each other's windows. This is train-to-train; no dev, test or near-dup file
   is in the index.
4. **Token-class definitions read every split.** `scripts/evaluate.py`'s project-definition token classes
   (`id:proj` etc.) are defined from the texts of all splits. They are definitions only and diagnostics; no bar or
   selection reads them (as in M1b).
5. **λ\* is bound by the KL bar.** With output mixing, KL grows about linearly in the weight (on WikiText KL/w ≈
   0.33 at 1e-4, 0.45 at 5e-4, 0.55 at 2e-3 on the 48k R1 tokens, 0.69 at 1.5e-2 in M1b's ablation), so the
   KL ≤ 0.005 edge is expected near w ≈ 7e-3, about where R1 (+0.19% at 7.5e-3) and R2 (+0.45%) also reach their
   bars. The weights 6e-3 and 7e-3 were added to resolve λ\* there. C0's numbers decide λ\*.
6. **Teacher cache.** The teacher-target cache is all-or-nothing per configuration: a distill F1 (310 windows)
   cannot reuse D1/D2's 64-window file and rebuilds all 310 windows (about 1.3 h).

7. **Dedupe gate**: met by construction with the non-dup primary metric; the fraction is reported. Applied
   literally (≤ 2% duplicated test targets) it could not pass on this split (3.83% in M1b), and the split cannot be
   refreshed (§4).

## 9. Outputs

Under `--out`:

- `progress.json` (with the grid path and sha256);
- `<stage>/result.json`;
- `<arm>/arm.json`, `train.json` (with the canary manifest, the teacher summary and `steps_mode`), `metrics.jsonl`
  (one record per step; after a resume from a checkpoint at step s the records with step ≥ s are dropped first),
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
- **M1c estimate:** about 9.5 h with a next-token F1 and about 12 h with a distill F1 (restated before launch: the
  three added grid weights, the counts@λ_X(OP) + OP test condition, and a distill F1's full teacher build of about
  1.3 h). The uncertainty is mostly the count-side scaling under 4 concurrent jobs, so read it as 8–13 h. RSS stays
  below about 6 GB per stage (the 11 GB guard applies).
- **Exit code:** 1 after a code error or a failed stage that something depends on; 0 otherwise. A failed optional
  dependency (F1, T_test, REG_op; §12) does not set it: the dependants run without it, and `progress.json` lists it
  under `soft_failed` (the decision then reads `INCOMPLETE` or names the F1 fallback).

## 12. Pre-launch review changes

An adversarial review of the runner, the bar arithmetic and this file found the problems below before anything ran
on the runner. Every change to the pre-registration it led to is listed here with its reason. (§11 is not used; the
numbering keeps M1b's sections where they coincide.) Nothing in M1b's grid or PREREG changed, and M1b's runner
behaviour is unchanged by default (checked by an M1b smoke of B0 and A1 against the previous commit).

1. **Bar B matching (§4; was a blocker).** The old counts@matched(X) matched R1, R2 and KL each separately. With
   output mixing the KL is relatively expensive (KL/w ≈ 0.3–0.5 at w ≤ 1e-3), so KL bound and pushed the matched counts to weight
   1e-4, while X's own budget use sat on R2. On M1b's real arrays, M1b's own A3 table would then **pass** Bar B,
   although A3 sits on the counts' gain/cost curve (`experiments/m1b/RESULTS.md`). New rule: one budget share,
   share = max(R1⁺/0.2, R2⁺/0.5, KL⁺/0.005), counts@matched(X) = the largest counts weight with
   share ≤ max(share(X), 0.01); and Bar B also needs E1(X) > 0 (and counts that hurt count as 0 in the 1.5× ratio).
   **Check on M1b's arrays** (dev, all targets; counts-only mixtures rebuilt from the always-on ablation's KN
   arrays; R1 KL of the counts measured with `countgrid:kn` on the same 48k WikiText tokens, whose R1 reproduces the
   sweep's to 4 decimals): A3 has E1 0.600, R1 +0.0075%, R2 +0.026%, KL 9.2e-5, so share 0.052 (R2 binds). Counts:
   1e-4 → E1 0.252, share 0.0067 (KL 3.3e-5); 2e-4 → 0.370, share 0.015 (KL 7.6e-5); 3e-4 → 0.459, share 0.025;
   5e-4 → 0.598, share 0.045 (KL 2.2e-4); 1e-3 → 0.844, share 0.100. Old rule: counts@matched = 2e-4 (KL binds),
   0.600 ≥ 1.5 × 0.370 and diff CI [+0.14, +0.61] → Bar B **PASS**. New rule: counts@matched = 5e-4, 0.600 <
   1.5 × 0.598 = 0.897 (diff +0.002, CI [−0.09, +0.39]) → Bar B **FAIL**, as it should for a point on the counts'
   curve. The floor: share(counts@1e-4) = 0.0067 ≤ 0.01, so an X with about zero cost is matched by weight 1e-4
   (C0's own numbers will be recorded with every match).
2. **Bar A frontier to frontier (§4, §6; major).** Bar A compared X + counts and counts alone at the same weight λ\*.
   λ\* is where counts alone are best within the bars, so X + counts@λ\* is usually outside them (X's cost adds on
   top), and the old rule then failed Bar A for the retention, not the gain. New rule: E1(X + counts@λ_X) −
   E1(counts@λ\*), λ_X = X's own best weight within the bars; thresholds unchanged. The two columns come from
   different stages, so the runner now checks keys, mask, base NLL and file groups. At test, a fixed-weight
   condition `countlm:kn-lam<λ_X(OP)>+rows:<OP>` is added when λ_X(OP) ≠ λ\*.
3. **Selection consistent with the decision (§5; major).** S1 and OP were ranked by the Bar-A difference alone,
   so an arm holding Bar B (or Bar A) could lose to one holding no bar. Now both rank by (any bar on dev, Bar A
   holds, Bar-A margin), then the old tie rules; OP is an arm that holds a bar on dev whenever S1 or F1 does.
4. **F1 failure fallback (§5, §7; major).** §8.6 said OP = S1 if F1 fails, but the scheduler only let dependants run
   past an RSS-aborted stage; a failed F1 would have stopped the run before OP, T_test and REG. New grid field
   `optional_after`: OP_select lists F1, Z_report lists T_test and REG_op. An optional dependency counts as settled
   in any terminal state; a stage whose every dependant lists it ("soft": F1, T_test, REG_op) does not stop the run
   or set the exit code when it fails (§10). On resume a failed soft stage is retried once (two attempts in all),
   and is final at once if a dependant already finished without it (`failed_final` in `progress.json`). Z_report
   writes the dev-only KILL verdict whatever T_test and REG did, and otherwise `INCOMPLETE` with the reason.
5. **T_test/STARTED (§6; minor).** It was written before OP was read and before the canary-free check, so a failure
   there looked like a started test read. Now it is written immediately before the test evaluation.
6. **λ\* resolution (§3; minor).** The KL edge (§8 disclosure 5) falls between 5e-3 and 7.5e-3 in the old grid;
   3e-4, 6e-3 and 7e-3 were added (16 weights). The fixed-weight `countlm:kn-lam<w>` tags round-trip for every grid
   weight (tested); the count-LM's own M1b tuning grids are separate and unchanged.
7. **metrics.jsonl after a resume (§9; minor).** Steps after the last checkpoint were logged twice after a resume,
   inflating `train.json`'s step count and timing statistics. Records with step ≥ the checkpoint's step are now
   dropped on resume (the no-resume path is unchanged).
8. **Upload excludes (§10; minor).** `ngram-cpu.yml` also excludes `**/*.tmp` (partial writes) from the uploaded
   artifact, next to the `.pt`, `.safetensors` and `_cache` excludes.
9. **Z_report extras (§7).** Per-file Bar-A differences on dev and test, each file's share of the targets, and
   n_eff (dev is effectively one file, §8 disclosure 1).
10. **Disclosures (§8)** 1–6 added, and the ETA restated (§10): ~9.5 h (next-token F1) / ~12 h (distill F1), range
    8–13 h.
11. **A falsy-result bug.** OP_select tested F1's result with a truth test, so an empty result would have been read
    as missing; it now tests for presence.
