# M1b pre-registration: proxy ingestion on a Qwen2.5-0.5B graft

This file and `grid.json` (same directory) were written **before any M1b training**. The runner
(`experiments/m1b/run.py`) records the sha256 of `grid.json` in `progress.json` when a run starts, and it refuses to
continue if the file changes afterwards. The only training before this was:

- a 4-step timing probe of the base recipe (`scratchpad/nm/m1b_probe/timing/`), used to set the token budget and not
  used for any result;
- the `--smoke` run of the runner with tiny budgets (`scratchpad/nm/m1b_smoke/`).

Design references: DESIGN.md §8 row M1, §I.1, §I.2, §6.A.1–§6.A.7 and §D.3.

## 1. What is fixed for every arm

| Item | Value |
|---|---|
| Host | Qwen2.5-0.5B, fp32, frozen. Snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987`. The host's sha256 is checked before and after each arm; any change fails the arm. |
| Keys | `qwen25_proxy_key_spec()`: the Flash-Next-style KeySpec over the Qwen2.5 vocabulary. Orders (2,3), 16 heads, base 200,000, 3,201,408 padded rows, width 56. |
| Graft | `graft_format(896, key)`. Injected at the input of decoder index 1, `value_scale` 1.0, editor init (§2.4). The arm is G-lin unless stated. |
| Corpus | minisgl `python/minisgl/**/*.py` at `df29117`, document format `# file: <path>\n` + source, EOS-separated (§6.A.1). |
| Split A | Frozen in `ingest_splits/split_a.json`: 204 train, 15 dev, 29 test, 2 near-dup diagnostic files. |
| Training data | Split-A train only (1,109,004 tokens, 310 windows of 4,096 tokens, stride 3,584, true n-gram history). Data order is a pure function of seed 0 and is identical across arms. |
| Token budget | **64 steps × one 4,096-token window = 262,144 training tokens per row arm** (about 21% of one pass over the 310 train windows). Each step also adds one 1,024-token near anchor window and one 1,024-token chat anchor window. |
| Budget basis | The timing probe ran at 82–84 s per step on 4 idle threads, so one arm takes about 1.5 h. With 8 trained row arms, LoRA and the evaluations (§7), M1b fits in about one day, which is the M1 row's "about one day". |
| Row policy (default) | n-gram count ≥ 2 on Split-A train (`RowPolicy('count', k=2)`), 1,464,015 reserved rows. |
| Reuse weight ω_j | γ = 4, π_other = 0.45. Reference = 500k tokens of stdlib (excluding the 16 R2 evaluation files) plus 500k tokens of WikiText-2 train. |
| Per-n-gram loss weight cap | 99.9th percentile of counts. |
| KL anchors | D_near = stdlib, D_chat = WikiText-2 train, 24 windows of 1,024 tokens each. Top-64 targets plus a remainder bucket. λ_near = λ_chat = 0.3. Positions with project-specific exact n-grams are masked. |
| Injected-residual budget | ρ (per arm), λ_ρ = 0.1, penalty on anchor batches (§I.1). |
| Optimiser | Row-wise Adagrad, lr 1e-2, eps 1e-30, accumulator init from the median of the first 50 steps. |
| Regularisers | λ₂ = 1e-4; λ̃₁ = 0.1 (λ₁ = 0.1·G₀); no trust region (κ = None). |
| Precision and memory | fp32 rows. Activation checkpointing is non-reentrant, and the output head runs in 128-token chunks. |

## 2. Trained arms (one seed, one factor at a time), in dependency order

| Stage | Arm | Factor | Depends on |
|---|---|---|---|
| A1_base | base | count ≥ 2 + ω_j + KL anchors at **ρ = 0.01** | B0 |
| A2_rho003 | ρ | ρ = **0.003** | B0 |
| A3_rho03 | ρ | ρ = **0.03** | B0 |
| S1_rho | *selection* | best ρ among A1–A3 (rule §4) | A1–A3 |
| A4_enrich | row policy | **enrich ≥ 4** against stdlib + WikiText (`RowPolicy('enrich', r=4)`) at the best ρ | S1 |
| S2_setting | *selection* | best of A1–A4 (rule §4) | S1, A4 |
| A5_gdir | arm | **G-dir** at the best setting: current-position tap (0,0,0,1), α = 0.16 | S2 |
| S3_op | *selection* | the **operating point**: best of A1–A5 (rule §4) | S2, A5 |
| A6_foreign | C1 control | **foreign-repo cartridge**: same recipe as the operating point (ρ, policy, λ's), G-lin, trained on networkx 3.4.2 (284 files, 958,171 tokens; sdist sha256 `307c3669…`) | S3 |
| A7_keyshuffle | C1 control | **train-time key shuffle**: same recipe as the operating point, G-lin. Positions of n-gram A train the slot of σ(A), a fixed per-head permutation of the reserved rows (seed 1). The cartridge is served through the **true** keys (VI-1 fix). | S3 |
| A8_lora | E5 reference | **LoRA all-linear r = 16** (attention + MLP, 24 layers): α = 32, AdamW lr 2e-4, 5% warm-up and cosine decay. Same 262,144 tokens, as 256 windows of 1,024 tokens sampled from the Split-A train stream (seed 0). | B0 |

Notes:

- A5 copies the recipe fields of S2's selection (ρ, policy, λ's) and changes only the arm.
- A6 and A7 copy those of S3's selection, but they use G-lin. If the operating point is G-dir, C1 is computed against the best G-lin configuration as well as against the operating point (both reported).
- The G-dir α = 0.16 is the §2.4 rule ("α set from S so that ‖c‖/‖h‖ matches the measured per-stream ratio"). It is the saturated calibration for Qwen2.5-0.5B in `P/dtrack/S/s_recommend.json` (`gdir_alpha_saturated_stream_median`).
- **Known equivalence (pre-registered):** with a bijective σ, A7's served function equals the eval-time permuted-keys control on a G-lin cartridge with σ⁻¹, up to float noise. A7 is kept because the DESIGN lists it. Its result is expected to track "perm_op" when the operating point is G-lin.

## 3. Eval-only conditions

Each condition is scored **once on Split-A test** in stage T_test, together with the operating point:

| Condition | Definition |
|---|---|
| B0 base | the host with the graft off |
| op | the operating point's cartridge |
| countlm | modified Kneser–Ney 2/3-gram on the Split-A train files, plus an in-file cache. λ is tuned on the tune dev file. |
| rag1k, rag4k | BM25 over the train files: 40-line chunks with stride 20, query = last 20 prefix lines, budget 1k or 4k tokens |
| rag4k+op | RAG plus the cartridge (E6) |
| knm | kNM-lite datastore over 50k train tokens, tuned on the tune dev file |
| ngm | NGM untrained 2/3-gram embedding average, ratio tuned on the tune dev file |
| perm_op | eval-time permuted keys of the op cartridge (seed 0) |
| random_op | random rows at matched norms (sanity) |
| constbias_op | constant mean graft output of op, calibrated on the tune dev file |
| controls | A6_foreign, A7_keyshuffle |
| LoRA | A8's saved adapter |

The **headroom** number (§I.3) is `tuning.countlm_headroom_tune_docs` in the test summary. It is computed on the tune dev file.

## 4. Scoring and selection (dev only)

**Scoring context (§6.A.1, fixed for every condition):**

- Sliding window with left context C = 2,048 in-file tokens and stride 1,024.
- Only targets with ≥ 1,024 in-file tokens of context are scored, so files of ≤ 1,024 tokens have no target.
- File-clustered paired bootstrap with 10k resamples.
- One consequence: 19 of the 29 test files and 14 of the 15 dev files are long enough.

**Dev (selection) set:**

- The first **13** long-enough Split-A dev files, in sorted path order, are scored for every candidate.
- The 14th is the **tune file**, used by countlm, kNM, NGM and constbias, and never scored.

**Retention sets for selection, scored ON with KL against the base:**

- R1 = WikiText-2 test.
- R2 = stdlib (16 files no regulariser uses) together with R2-ML (an unrelated ML codebase, transformers models).
- **48,000 tokens each** before the context rule.
- R2 is the token-weighted mean of the two R2 sets.

**Candidate metrics on dev:**

- E1_dev = −relative ΔNLL (%) on the dev files.
- R1_dev = relative ΔNLL (%) and mean KL on WikiText.
- R2_dev = relative ΔNLL (%) on R2 (the token-weighted mean above).

**Selection rule (S1_rho, S2_setting, S3_op; the §6.A.7 operating-point rule applied to dev):**

1. Among the candidates with **R2_dev ≤ +1.0% and R1_dev ≤ +0.2% and R1 mean KL ≤ 0.005**, take the **largest E1_dev**. Ties go to the smaller ρ.
2. If no candidate qualifies, take the candidate with the smallest R2_dev. This is recorded as the kill-zone case.

How "best ρ" is chosen before the dependent arms run:

- S1_rho applies this rule to A1–A3 only, as soon as all three have finished their dev evaluation.
- A4 then reads S1's ρ from `S1_rho/result.json`.
- A5 reads S2's full recipe, and A6/A7 read S3's.

Nothing on Split-A test is read by any selection.

**Pruning frontier (F_frontier, dev):**

1. Take A3 (ρ = 0.03, the least retention-regularised arm).
2. Make two pruned copies of it: zero the rows of the top-100 2-grams + top-100 3-grams by train count, and of the top-1,000 + 1,000 (all 8 heads of each order).
3. Score both like a candidate.
4. The frontier is the piecewise-linear curve E1(R2) through the points (0, 0) [base], A3 minus top-1,000, A3 minus top-100, and A3.
5. The operating point is **above the frontier** if its E1_dev exceeds the frontier's E1 at the operating point's R2_dev (clamped at the ends).

## 5. The single test read

- Stage **T_test** runs **once**, after S3_op, F_frontier, A6, A7 and A8 have finished.
- It scores Split-A test (19 long-enough files) for the conditions in §3.
- It writes `T_test/STARTED` first. If the stage is interrupted and resumed, the resume repeats the same pre-registered read, and `progress.json` records that.

**After T_test:**

- **R3_capability** is report-only. It scores the base and the operating point: 300 items each of MMLU, ARC-Challenge and HellaSwag, log-likelihood multiple choice, paired.
- **SB_splitb**:
  - The operating-point recipe is retrained on the Split-B train tree at T_cut = `a4a5543` (245 files).
  - Split B (lines added in `(a4a5543, df29117]`, 52 files) is then scored with base, that cartridge, countlm and rag4k.
  - The data-dependent baselines are built from the tree at T_cut (VE-1 fix).

## 6. Gates and bars (quoted verbatim from DESIGN.md)

§6.A.7 gates:

> **Gates (any failure is a bug):** R0 bit-exact OFF; I2 at the host's repeat-run noise floor with a broken-state control ≥ 100× larger; host hash unchanged; I4 row-id parity on ≥ 10k positions; dedupe audit (held-out 16-gram-in-train ≤ 2%, near-duplicate files excluded).

M1 row, Pass:

> Gates; operating-point decision of §6.A.7 with these bars: E1 ≥ 1.5% (lower bound > 0.5%) **with** R2 ≤ +0.5% and R1 ≤ +0.2%, and the operating point lies **above the pruning frontier** (more E1 than the frontier at equal R2); C1 and NGI pass; REG; report E2, E5, MEM

M1 row, Kill / stop:

> no configuration with R2 ≤ +1% reaches E1 ≥ 1% (lower bound > 0.5%) at the fixed context → rows-only layer-1 ingestion does not transfer from the proxy; stop before M2 and report. In the 0.5–1% zone: one Stage-1b (distillation-with-context) arm, then decide

How each one is evaluated here:

**Gates:**
- **R0 and I2:** `tests/test_qwen25_real.py` on the same host. Gate R0 is the no-op, bitwise. Gate I2 is decode parity, with a broken-state power check in `tests/test_graft_tiny.py`.
- **Host hash:** checked per arm by the train worker.
- **I4:** `tests/test_hashing.py`.
- **Dedupe audit:** the `dup:16gram_in_train` class fraction on dev and test, from the evaluate summary.

**E1 and the bars:**
- **E1:** the operating point's test relative NLL reduction and its file-cluster CI.
- **R1 and R2:** the operating point's dev-stage R1 and R2. These sets are not project files, so the dev-stage numbers are the final ones.
- **Frontier:** F_frontier's `op_above_frontier`.

**NGI:** E1(op) ≥ 1.5 × E1(countlm), both on test.

**C1:**
- best control / op ≤ 1/3 (pass); best control ≥ 50% of the op gain (kill).
- The control set is: foreign (on project-specific identifier tokens, i.e. the class breakdown), key-shuffle, perm_op and NGM (on all test tokens).

**REG: not implemented in this package** (no regurgitation or canary harness yet). M1 PASS therefore cannot be declared from this run. The run can still reach the KILL or the 0.5–1% zone, which REG does not affect. Recorded as an open issue.

**Report-only:**
- E2 is Split B.
- E5 is op / LoRA E1 on test.
- E6 is (rag4k+op) − rag4k.
- **MEM is not measured.** There is no train-file scoring pass. Only the train-window NLL in `metrics.jsonl` is available.

## 7. What D-track S changes in the grid (pre-registered decision)

S measured Flash-Next's own layer-1 ratio ‖PLE out‖/‖X‖:

- median 0.48 / 2.20 / 0.37 / 0.29 per stream, 0.55 over all streams;
- that is 30–220× the ρ grid (`P/dtrack/S/s_recommend.json`).

This is the co-trained model's operating regime, not a retention-safe budget for a frozen host, so **the ρ grid {0.003, 0.01, 0.03} is unchanged**.

S's other outputs used here:

- **G-dir α = 0.16.** S confirmed conv ≥ 2× direct on 4/4 streams, so the G-dir arm is required (§D.3 rule fired: YES).
- **The fixed 0.5 gate.** The calibrated-gate rule did not fire: the ANOVA within-n-gram share of gate-logit variance is 0.0003 / 0.032 / 0.010 / 0.008 per stream, against a 50% bar (`P/dtrack/S/s_within_anova.out`, VD-1 fix).
- **G-lin value_scale stays 1.0.** The trainer normalises rows by s_j = 2‖h̄‖/value_scale, so the scale only relabels the row units. S's derived `glin_value_scale` is recorded but not applied.

D-track G (f* = 22.9 / Mtok for 2-grams, 1.07 / Mtok for 3-grams) feeds the M2 row-policy rule. That rule needs the M2 cartridge's E1, so it does not change M1b.

## 8. Deviations from DESIGN (decided before running)

**Smaller than the DESIGN asks for:**
- **Training tokens:** 262,144 per arm, a partial epoch (§1 budget basis).
- **No lr sweep for LoRA:** single lr 2e-4.
- **λ̃₁ fixed at 0.1:** the §I.1 sweep {0, 0.1, 0.3} is deferred.
- **Retention:** 48k tokens per retention set (§6.A.3 asks for 64k).

**Not done in this run:**
- **No early stopping:** a single dev evaluation at the end of the budget. A 2-window dev NLL is logged before and after training (monitor only).
- **No Engram-Adapter, RAG-oracle or param-matched LoRA arms.**
- **No REG or MEM** (§6).
- **Corpus v1 (tools/, tests/) is not added to dev/test:** MDE ≈ 0.35–0.43% on 19–29 files.

**Choices the DESIGN leaves open:**
- **LoRA windows:** 1,024 tokens. A 4,096-token LoRA window needs [4096, 151936] fp32 logits plus their gradient, about 5 GB.
- **Frontier source:** A3 (ρ = 0.03), not an unregularised arm.
- **Eval-only baselines** are scored on test only. They are not selection candidates, and their λ's are tuned on the tune dev file.

## 9. Outputs

Under `--out` (default for the full run: `scratchpad/nm/m1b/`):

- `progress.json`
- `<stage>/result.json`
- `<arm>/metrics.jsonl`, `train.log`, `checkpoint.pt`, `cartridge.pt`, `fmt.json`, `train.json`, `arm.json`
- `<stage>/eval/summary.json` and `report.md`, plus the per-condition `.npz`
- `run.log` (top level)

The runner aborts a stage cleanly if the child's RSS exceeds 11 GB, and dependants then select among the remaining
candidates.

## 10. Which run is canonical (amendment 2026-11-10, written before any arm result was read)

The full run started in the dev container on 2026-11-10 08:46 UTC. A self-hosted CPU runner was added the same day
(`runner/README.md`), with every input pinned by content (`runner/inputs/inputs.lock.json`) and verified before a
run starts. Code changes made for the runner leave the dev container's paths and texts unchanged.

- If the dev-container run completes, it is the M1b result. A runner run of the same grid, if any, is reported
  only as a reproducibility check and never replaces or is pooled with it.
- If the dev container is reclaimed before the run completes, the M1b result is a fresh runner run of the same
  `grid.json` (same sha256) from stage B0. Stages finished in the container are not combined with runner stages,
  because training is not bit-identical across machines.
- Any further CPU experiments (the λ̃₁ sweep, an L2 fix) get their own pre-registration and are compared only with
  arms trained on the same machine.
