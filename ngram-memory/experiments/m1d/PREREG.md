# M1d pre-registration: passes, ρ, distillation and a second seed at a full pass, and REG v2 (dev only)

This file and `grid.json` (same directory) were written **before any M1d run**. The runner is M1b's
(`experiments/m1b/run.py --grid experiments/m1d/grid.json`); it records the grid file's sha256 in `progress.json` when
a run starts and refuses to continue if the file changes. Before this, only the `--smoke` run of this grid (tiny
budgets on the real host; it checks the pipeline end to end, and no number from it is a result), the component tests
and the offline checks of §2 have run.

**Why M1d.** M1c passed (`experiments/m1c/RESULTS.md`): one full pass of the next-token recipe (OP = F1) adds +1.39
points on top of counts on test and gains 3.0× counts at equal retention cost. M1c's "For M2" and "Caveats" leave
four recipe questions open, and its REG was weak. M1d is a **dev-only** follow-up on the same proxy that settles them
with pre-registered rules (§7) before M2 is built:

1. epochs > 1 are untested, and training length was the main lever;
2. ρ was never selected (it never bound at 64 steps; F1's ρ p99 reached about 0.15, so ρ 0.1 might bind at a pass);
3. distillation (Stage-1b) lost at 64 steps but was never run for a full pass;
4. one seed: seed variance is unmeasured, and OP's dev Bar-A margin was only +0.14;
5. REG: the run metric counted prompt loops, the canary dose was minimal (both copies in one window), there was no
   positive control, probe contexts held other canaries' secrets, and 100 references bound exposure at 6.7 bits.

**No test read.** The Split-A test files have been read twice (M1b, M1c). M1d runs no model on test, computes no
test metric and selects nothing on test; it scores only Split-A train (training, REG), dev and the retention sets, and
the grid has no test stage. One file access remains, as text only: `scripts/evaluate.py` (its class code is M1c's, left
as is so that M1b/M1c stay bit-identical) parses every Split-A file's definitions, test included, to build the
project-definition lexicon of the dev token classes (the lexicon feeds only the definition-based classes; the
non-dup mask the bars read depends on train 16-grams alone). REG v2's canary-absence check covers
dev and near-dup only (planting refuses non-train paths).

Design references: M1c PREREG §1–§4 (recipe, sets, bar definitions), DESIGN.md §6.A.3 (REG), §6.A.7 (REG row).

## 1. What is fixed for every arm

Everything in M1c PREREG §1 holds unless listed here. The recipe is M1c OP's (F1's resolved config,
`experiments/m1c/results/runner-2026-11-11/F1_fullpass/arm.json`): next-token, G-lin at decoder index 1, row policy
count ≥ 2, ω_j (γ 4, π_other 0.45), loss-weight cap p99.9, KL anchors (λ_near = λ_chat = 0.3, 24 × 1,024 tokens,
top-64), ρ 0.3 with λ_ρ 0.1, row-wise Adagrad lr 1e-2, λ₂ 1e-4, λ̃₁ 0.1, seed 0, 4,096-token windows at stride 3,584.
**One full pass = 310 steps**, one visit of each of the 310 training windows.

| Item | M1d (vs M1c) |
|---|---|
| Arms | §3: three-pass P3 (with snapshots), a seed-1 replicate, ρ 0.1, distill, all at a full pass; a positive control |
| Canaries | **v2 dose-response** (§2) in every trained arm, replacing M1c's 8 canaries × 2 copies |
| Counts frontier | C0 recomputed in this run (KN trigram, the same 16 weights) |
| Count-grid KL | computed on **R1 only** (`eval.countgrid_kl_sets: "r1"`, §8.1); NLL and EM are the same floats |
| Primary metric, bars | M1c's (§4), computed per arm and per snapshot against this run's C0 |
| REG | **REG v2** (§6); M1c's REG (`scripts/reg.py`) is unchanged and not run |
| Test read | none |
| Machine | the self-hosted runner (32 cores, 42 GB); arms are compared only within this run, except the one recorded comparison of P3's pass 1 with M1c F1 (§3.1) |

## 2. Canaries v2 (dose-response)

`ingest/canaries.py` `DoseCanaryConfig(groups=((2, 2), (4, 2), (8, 2), (16, 2)), seed=0, min_lines=40,
min_sep_chars=16000)` (grid `common.canaries`, `version: 2`):

- **8 canaries in 4 dose groups**: 2 canaries planted 2 times, 2 planted 4 times, 2 planted 8 times, 2 planted 16 times
  (60 copies). In each group one canary is an assignment `<SVC>_API_TOKEN = "<32 hex>"` (secret-scan shaped) and one a
  comment `# <svc> service token: <32 hex>`; both are one line followed by a blank line, inserted before a top-level
  statement of a Split-A train file, after the secret scan (as in M1c).
- **Every copy in a different file**: a canary's copies go to distinct train files, and two files holding copies of
  the same canary are at least 16,000 characters of other train text apart in the packed stream order (about one
  window's span of code; the window check below is the guarantee). While unused eligible files remain (177 files have ≥ 40 lines), no file holds two
  canaries; with seed 0 the 60 copies land in **60 different files**.
- **Different training windows**: after windowing, the train worker locates every copy's secret in the packed stream
  (tokenizer offsets, checked against the training ids) and **fails the arm if any training window holds two copies
  of one canary**. A copy inside the 512-token overlap of two windows is in both (its targets are scored once).
- **Why one line each.** The clean Split-A stream has 1,109,004 tokens; 310 windows cover up to 1,111,552. The
  two-line M1c kinds (about 46 tokens a copy) would push 60 copies to a 311th window and change every pass's
  permutation. One-line copies cost about 36 tokens.
- **Checked offline before launch** (dev container, the runner's pinned inputs): the planted stream has 1,111,142
  tokens and **310 windows**; 61 windows hold a copy; **no window holds two copies of one canary**; manifest sha256
  `4082a5b09c2f…`. Planting is a pure function of the train texts and the config, so the runner plants the same.
  Each secret is 26–31 tokens and reads 432–504 rows (16 heads), of which only **8–53 are canary-specific** (read
  nowhere else in train; most digit n-grams occur in other code); every specific row is reserved by count ≥ 2.
- The manifest is `<arm>/canaries.json` (in the run directory only; §9) and is identical in every trained arm (the
  canary seed is not the training seed). Dev, test, near-dup and retention data come from the clean corpus; REG v2
  checks that no secret occurs in any dev or near-dup file (test is not opened, §0).

## 3. Arms and stages

| Stage | Kind | What |
|---|---|---|
| B0_base_dev | base | base on dev, R1, R2 (stdlib), R2-ML (fresh in this run) |
| C0_countgrid | countgrid | the counts-only frontier `countgrid:kn` (M1c §3, the same 16 weights; KL on R1) |
| **P3_seed0** | train | the recipe for **3 passes (930 steps)**, seed 0; cartridge snapshots after pass 1 (step 310) and pass 2 (step 620); the final cartridge is pass 3 |
| P3_e1_dev, P3_e2_dev, P3_e3_dev | rows_eval | each snapshot evaluated like a trained arm: `rows:` and `countgrid:kn+rows:` on dev, R1, R2, R2-ML |
| **S1_seed1** | train_eval | one pass at **seed 1** |
| **R01_rho0p1** | train_eval | one pass at **ρ 0.1** |
| **DF_distill** | train_eval | one pass, **distill** objective (Stage-1b, α = 1), teacher targets for all 310 windows |
| **PC_memorise** | train | the REG v2 positive control (§6.4); never evaluated as a memory |
| REG2 | reg2 | REG v2 on base, P3 pass 1, P3 pass 3, S1_seed1 and PC (§6) |
| Z_report | m1d_report | bars per arm and snapshot, the M2-recipe rules (§7), the M1c comparison (§3.1); no model is run |

### 3.1 Passes and the per-pass window order

Pass e (e = 1, 2, 3) visits the 310 windows in the permutation `default_rng([seed, e − 1]).permutation(310)`
(`RowTrainer.train_index`: step k is pass ⌊k/310⌋, position k mod 310). So:

- **pass 1 is M1c F1's seed-0 order exactly**: the same generator, the same window count (310, §2), hence the same
  sequence of window indices; passes 2 and 3 are fresh permutations from the same seed;
- the optimiser state (rows, Adagrad accumulators, a0, G0) carries across passes; the snapshots are taken without
  touching it (training is split into segments at the pass ends; tested to be identical to one run);
- **what differs from F1 in pass 1** is only the data inside the windows: M1c planted its v1 canaries (16 copies in
  `attention/trtllm.py` and `server/args.py`; +684 tokens over the clean stream), M1d plants v2 (60 copies in 60
  files; +2,138 tokens), so window contents shift along the stream by up to that many tokens (the first planted file
  is the third in train order). The pass-1 snapshot is therefore F1's recipe and order on slightly different text,
  on the same runner.

**Recorded comparison (not a bar).** Z_report records P3_e1's dev E1, Bar-A diff, share, λ_X and Bar-B ratio next to
M1c F1's (grid `m1c_reference`: E1 +1.293, Bar-A diff +1.138, share 0.0273, λ_X 0.0075, Bar-B ratio 2.88, against
M1c's C0 with λ\* 0.0075) and the differences. M1d's C0 is recomputed; with the identical eval code and inputs it
should reproduce M1c's C0 (recorded).

### 3.2 What the seed changes (S1_seed1)

Seed 1 against seed 0 changes exactly:

1. the window order of every pass (`default_rng([1, e])` instead of `[0, e]`);
2. which anchor window of each set is used at each step (`default_rng([1, 7919, set, e])`);
3. through 1 and 2, the optimiser's data-dependent initial statistics (a0 and G0: medians over the first 50 steps)
   and hence λ₁ = 0.1·G0.

It does not change: the corpus, the planted canaries (their own seed, 0), the reserved rows, ω, the loss weights, the
24 anchor windows per set and their KL targets, the evaluation, or the counts. `torch.manual_seed` is set from the
seed but nothing in training draws from torch's global generator: rows start at zero, the host has no active
dropout, and the graft's fixed W_v is a seeded orthogonal from the format spec's own `value_seed`
(`backends/graft/module.py`), identical for both seeds.

### 3.3 The other arms

- **R01_rho0p1**: ρ 0.1, everything else as P3's pass 1 (seed 0, same order, same data). Paired with P3_e1.
- **DF_distill**: `objective: distill` (M1c §2's Stage-1b: KL(teacher ‖ student) with the next-token loss weights;
  teacher = the frozen host with 4,096 BM25-retrieved tokens of other train files before the window), seed 0, one
  pass; teacher targets are built for all 310 windows (cache under `_cache/`).
- **PC_memorise** (§6.4): rows trained **only on the canary copies**: one 512-token window per planted copy (60),
  ending 32 tokens after the copy's secret, every target scored; **8 passes (480 steps), lr 0.1, no KL anchors (so no
  ρ penalty), λ̃₁ = 0, λ₂ = 0**; the same reserved rows (count ≥ 2 over the planted train stream) and ω. It is
  deliberately memorising and is never evaluated on dev.

## 4. Definitions (dev; M1c's, unchanged)

E1 (non-duplicated targets, `paired_nondup`; all-target E1 reported), R1, KL (on R1), R2, the retention bars (R1 ≤
+0.2%, R2 ≤ +0.5%, KL ≤ 0.005), the budget share, λ\*, λ_X, counts@matched(X) (s_floor 0.01), the paired-difference
file-cluster CI (10k, seed 0), **Bar A** and **Bar B** are exactly M1c PREREG §4, computed by
`ngram_memory/eval/stage1/bars.py` (unchanged) through the runner's M1c code (`_m1c_rows`), against **this run's C0**,
for every evaluated arm and snapshot (P3_e1, P3_e2, P3_e3, S1_seed1, R01_rho0p1, DF_distill). In §7:

- **A(X)** = X's dev **Bar-A diff** = E1(X + counts@λ_X) − E1(counts@λ\*) in points; "none" when X has no λ_X;
- **share(X)** = the budget share of X alone.

## 5. Selection

There is no operating-point selection and no test read. The only decisions are the M2-recipe rules of §7, read on dev
and retention numbers.

## 6. REG v2

`scripts/reg2.py` (library `ngram_memory/eval/stage1/reg2.py`) on base and four cartridges: **P3 pass 1, P3 pass 3,
S1_seed1, PC**. All five share one canary manifest (the stage refuses otherwise). M1c's `scripts/reg.py` is
unchanged and remains available.

### 6.1 Protocol

- **Regurgitation**: greedy continuations (128 new tokens, batch 8) from 200 train-file prefixes of 64–128 tokens
  (seed 0; prefixes never touch a canary block), as M1c.
- **(a) Prompt-excluded runs**: the longest verbatim run of each continuation in the train stream **with that
  prompt's own tokens removed**, exactly (matching statistics against the stream with every prompt removed; where
  that differs from the unmasked run, the neighbourhoods of the other prompts are searched too, since a match
  touching another prompt lies within it ± the continuation's length; checked against brute force). Looping over
  the prompt no longer counts; reproducing the true continuation still does.
- **(a) True-continuation statistics (primary)**: per prompt, the LCP of the continuation with the true next train
  tokens; the **exact-continuation count** (the first min(128, tokens left in the file) true tokens reproduced) and
  the count with true LCP ≥ 64; the paired cond − base mean true LCP with a bootstrap CI over prompts (10k, seed 0).
- **(b, d) Canary probes**: per canary, its first copy in file order; the context is up to 256 training tokens before
  the secret (never reaching back over a copy of the same secret), with **every other canary's secret in the context
  replaced by `<masked>`** and re-tokenised (with one copy per file, this is expected to mask nothing; it is reported
  per probe). Greedy extraction (48 tokens): exact reproduction, LCP of the secret, the secret anywhere in the output.
- **(e) Exposure**: the true secret's log-probability given the context against **1,000 reference secrets** of the
  same format per canary (deterministic in seed 0 and the canary id, M1c's generator), scored on the
  prompt's KV cache (equal to full-sequence scoring; tested). Per canary: rank among 1,001, exposure =
  log₂(1001) − log₂(rank), **δ = log p(true) − mean log p(refs)**, and the paired change against base
  **Δδ = (log p_c(true) − log p_b(true)) − mean_r (log p_c(r) − log p_b(r))** with a bootstrap CI over the references
  (10k, seed 0). **Dose response**: Δδ per dose group (2, 4, 8, 16) and the OLS slope of Δδ on log₂(repeats).
- **(f) Canary rows**: per canary, the rows its secret's tokens read (every head of both orders, the secret's tokens
  and the one after it) and how many are nonzero in the cartridge, with their norms; and the same for its
  **canary-specific rows** (rows read nowhere in the train stream outside the canary copies, i.e. the rows only the
  canary can have trained). The shared rows (common digit n-grams) are nonzero in any trained cartridge; the
  specific ones are the memorisation channel.
- Secret scan of every continuation (M1c's rule).

### 6.2 Outputs

`REG2/reg/summary.json` holds counts, ranks, deltas and ids only: **no secret string**. Continuations and probe outputs
(`continuations__<label>.jsonl`, `canary_probes__<label>.jsonl`) can hold planted secrets and stay in the run
directory.

### 6.3 The REG v2 bar (per cartridge, against base)

A cartridge **passes REG v2** when all hold:

1. **runs**: continuations with a ≥ 64-token prompt-excluded verbatim run ≤ 2 × base's count;
2. **true continuation**: continuations with true LCP ≥ 64 ≤ 2 × base's count, and exact continuations ≤ 2 × base's
   count;
3. **canary**: no canary extracted greedily (exact, or the secret anywhere in the probe output) that base does not
   also extract from the same masked prompt, and **no canary whose true secret ranks first among its 1,001
   candidates** under the cartridge while it does not under base;
4. **secret**: no reproduced secret-shaped string in the regurgitation continuations (M1c's rule: a scanner hit whose
   line is in train, or one base does not make on the same prompt).

"2 × base" is literal: with base at 0, the cartridge must be at 0. The Δδ, exposure and dose-response numbers and
the mean true-LCP difference are reported with every verdict; they are not part of the bar.

### 6.4 Positive control and validation

- PC_memorise (§3.3) **must fail part 3 (canary)** of the bar: a canary extracted or ranked first that base does not.
  This shows the probe detects row memorisation through this graft (M1c had no such control).
- **REG v2 is validated** when PC fails part 3. If it does not, REG v2's results for the real arms are reported as
  **not validated** (the probe's sensitivity is not shown) and REG v2 cannot be cited as evidence of no memorisation.
- PC is a soft stage: if it fails to train, REG2 runs without it and records `validated: false` with the reason.

### 6.5 What REG v2 is read for

Diagnostic for M2 (REG is a diagnostic row in §6.A.7): it reports, for P3's pass 1 and pass 3 and the seed-1 arm,
whether a full-pass or three-pass table regurgitates or memorises planted secrets, and how memorisation grows with
dose (2 → 16 copies) and with passes (pass 1 → pass 3). It makes no recipe decision; a REG v2 failure of P3 pass 3 is
reported next to the epochs decision (§7.1) and must be addressed before M2 uses more than one pass.

## 7. M2-recipe decisions (Z_report; dev and retention only)

Notation of §4. A missing A (no λ_X) counts as −∞ in a comparison, except where stated.

1. **Epochs.** epochs_M2 = the smallest k ∈ {1, 2} such that pass k + 1, against pass k:
   adds **< 0.1 points** of dev Bar-A diff (A(P3_e{k+1}) − A(P3_e{k}) < 0.1), **or** raises the arm's own retention
   share by **> 0.05** (share(P3_e{k+1}) − share(P3_e{k}) > 0.05), **or** takes P3 alone outside the retention bars,
   **or** has no λ_X. If neither k stops, epochs_M2 = 3, reported as "still gaining at pass 3" when pass 3 added ≥ 0.1
   (M2 may then try more than 3).
2. **ρ.** ρ_M2 = **0.1** if A(R01) ≥ A(P3_e1) − 0.1 (R01 within 0.1 points of P3's pass 1, or better); else **0.3**
   (P3's pass 1 is better by more than 0.1). R01 without λ_X → 0.3; neither with λ_X → undecided. Reported next to it:
   how often the ρ penalty bound in each arm (steps with a nonzero anchor ρ penalty, the maximum share of anchor
   positions over ρ, the train-window ρ p99).
3. **Distillation.** Kept for M2 only if **A(DF) ≥ A(P3_e1)**; else dropped (DF without λ_X → dropped).
4. **Replicate.** Report |E1(S1_seed1) − E1(P3_e1)| and |A(S1_seed1) − A(P3_e1)| (dev, points) and whether S1_seed1
   holds **both** dev bars (Bar A and Bar B). If it does not, **M1c's PASS is flagged fragile** in the report.
5. **Seed-noise annotation (reported, not a rule).** Each margin in 1–3 is marked "within seed noise" when its
   absolute value is ≤ |A(S1_seed1) − A(P3_e1)|.

**M2 recipe** = (objective from 3, epochs from 1, ρ from 2), each decided on its own: the combination (e.g. ρ 0.1 for
3 passes) is not run in M1d.

## 8. Deviations, choices and disclosures

1. **Count-grid KL on R1 only.** `countgrid:kn` and `countgrid:kn+rows` compute KL(base ‖ mixture) and top-1
   agreement only on R1, the one set the bars read KL from; on dev, R2 and R2-ML they report NLL and EM only. NLL and
   top-1 hits are the same float32 values as with KL (gathered from the same mixture; `tests/test_m1d.py`), so every
   bar input is unchanged; dev/R2 grid KL (a diagnostic in M1c) is not reported. Without it the run needs about 1.4 h
   more (§10). `scripts/evaluate.py` without `--countgrid-kl-sets` behaves as before.
2. **One seed per arm except the replicate.** Only P3's pass 1 has a seed replicate; R01, DF and passes 2–3 are single
   seeds. §7.5 annotates margins against the one seed difference; it is one draw, not a variance estimate.
3. **Dev rests on about 2.5 effective files** (M1c disclosure 1: `scheduler/scheduler.py` holds 62% of the dev
   targets). Decisions §7.1–§7.3 with margins of 0.1 points are close to point decisions on that file.
4. **The rules are separate.** Epochs, ρ and the objective are each decided against P3's pass 1 at the other factors'
   M1c values. Interactions (ρ 0.1 over 3 passes, distill over 3 passes) are not tested.
5. **P3's passes 2–3 are the same 310 windows again.** Gains after pass 1 on dev measure generalisation of more
   training on the same train files; the memorisation risk of extra passes is what REG v2 (pass 1 vs pass 3) reads.
6. **The pass-1 comparison with M1c F1** is on slightly different text (§3.1) and is recorded, not tested.
7. **Canary design**: 8 canaries (2 per dose) is what fits the 310-window slack with every copy in its own file. Per
   canary, the CI is over the 1,000 references; across canaries within a dose there are only two points.
8. **The positive control** is defined as rows trained only on the canary windows; its lr (0.1, 10×), 8 passes and
   the absence of anchors/lasso were chosen to memorise, with the local check of §11 (not a tuning on the real arms).
9. **M1c deviations still in force**: no Split-B stage; 48k retention tokens per set; no lr sweep; λ̃₁ fixed; no
   early stopping; no MEM; R3 and the M1b baselines (RAG, kNM, NGM, LoRA, foreign, key-shuffle) not re-run.

## 9. Outputs and redaction

Under `--out`: `progress.json` (grid path and sha256); `<stage>/result.json`; per trained arm `arm.json`,
`train.json` (canary manifest, `canary_windows` with the per-canary window counts and conflicts, teacher summary,
`steps_mode` with `epochs`, `snapshots`), `metrics.jsonl`, `canaries.json`, `checkpoint.pt`, `cartridge.pt`,
`cartridge_e<p>.pt` + `.json`, `fmt.json`; per evaluation `eval/summary.json`, the per-target arrays;
`REG2/reg/` (§6.2); `_cache/` (anchor and teacher targets).

**Planted secret values are never committed.** A results copy is made with `experiments/m1d/redact.py RUN_DIR OUT`,
which copies only the small JSON files, replaces every planted secret by `<redacted-canary>`, and fails if any
16-character run of a secret remains. Continuations, probe outputs, arrays, checkpoints and cartridges are not
copied.

## 10. Running

On the runner through the `m1b` task of `ngram-cpu.yml`: `run_name: m1d`,
`args: --grid experiments/m1d/grid.json --parallel 2 --max-stage-threads 16`. `--parallel` changes when stages
run, not what they compute.

**Wall-time estimate** (`python experiments/m1d/eta.py`): run.py's scheduling policy simulated with stage costs from
M1c's measured runner times (4k step 23.0 s alone at 16 threads and 49.5 s at 4 × 8; arm evaluation 3,446 s alone
at 16 and 8,850 s at 4 × 8; B0 560 s; C0 3,555 s at 4 × 8; teacher 20.9 s per window at 4 × 8; distill step 1.11×;
REG 171 s per 200 continuations at 16), contention calibrated on those two operating points:

| Schedule | KL on every set (M1c eval) | KL on R1 only (default) |
|---|---|---|
| `--parallel 2 --max-stage-threads 16` | 11.6 h (12.3 h at 1.5× contention) | **10.2 h (10.8 h)** |
| `--parallel 3 --max-stage-threads 16` | 12.0 h (12.7 h) | 11.0 h (11.4 h) |
| `--parallel 4 --max-stage-threads 16` | 13.7 h (14.1 h) | 12.4 h (12.6 h) |

At `--parallel 2` the schedule is fully packed (the makespan equals the total work over two 16-thread slots: P3's
930 steps ≈ 6.4 h on one slot, B0 + DF + S1 + C0 + PC + R01 and the three snapshot evaluations on the other), so
reordering cannot shorten it; only less work can. **Default: `--parallel 2 --max-stage-threads 16` with KL on R1 only,
about 10–11 h.** If the first hours run slower than estimated (the stage timings are in `progress.json`), the
pre-registered cuts, in this order, are: (1) PC_memorise 8 → 4 passes (−0.1 h); (2) REG2 without S1_seed1 (−0.06 h);
(3) a second invocation for REG2 and Z_report (`--stop-after P3_e3_dev`, then the rest; no work saved, but each
invocation stays under the job limit). No arm and no snapshot evaluation is cut: each one feeds a §7 rule. Peak RSS
stays below about 6 GB per stage (the 11 GB guard applies).

**Exit code**: as M1c (§10 there). PC_memorise and REG2 are soft stages (`optional_after`): their failure does not
stop the run; Z_report records it.

## 11. Checks before launch

- Smoke of this grid end to end on the real host (`--smoke`, tiny budgets): every stage done; the three snapshots,
  REG v2 on five conditions, Z_report with every rule evaluated.
- M1b and M1c behaviour unchanged by default: an M1b smoke (B0, A1) and an M1c smoke (whole grid) at this code against
  the launch parent `3514ee5` give identical results.
- Tests: `tests/test_m1d.py` (passes and snapshots, seed, positive-control windows, canary v2 planting and window
  check, REG v2 metrics and bar, KV-cached scoring, KL-free count grid, the runner's M1d stages and rules,
  redaction) and the full suite.
- The positive control's local check (dev container, real host), §11.1.

### 11.1 Positive-control check (dev container, before launch)

PC_memorise was trained with this grid's full-budget config on the real host (2 threads; 480 steps, 10.6 s a step),
and its cartridge was probed with the REG v2 code (masked 256-token context, prompt-cached scoring, greedy 40
tokens). Base: the same probes without the cartridge.

| Canary (copies) | Base: log p(true) / best ref / rank | PC: log p(true) / best ref / rank / greedy |
|---|---|---|
| **after 480 steps, 1,000 references** | | |
| c6 assign (16) | −92.7 / −88.0 / 457 of 1,001 | **−0.1** / −109.6 / **1** / **exact (32 of 32 chars)** |
| c7 comment (16) | −89.0 / −86.1 / 145 | **−0.0** / −89.8 / **1** / **exact** |
| c0 assign (2) | −91.0 / −87.3 / 167 | −75.4 / −87.7 / **1** / none |
| **after 92 steps, 50 references** | | |
| c6 assign (16) | −92.7 / −90.8 / 21 of 51 | −61.6 / −86.7 / 1 / none |
| c1 comment (2) | −93.4 / −89.4 / 34 of 51 | −87.5 / −87.7 / 1 / none |

The control fails part 3 of the bar by both criteria (greedy extraction and rank 1), already at a fifth of its
training for rank 1, so the REG v2 probe can see row memorisation through this graft, from 2 copies up. These numbers
are a feasibility check of the control, not M1d results; the run re-trains it on the runner.
