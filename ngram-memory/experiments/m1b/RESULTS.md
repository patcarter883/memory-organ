# M1b results (self-hosted runner, 2026-11-10)

This is the canonical M1b run under PREREG §10. The dev-container run was lost at stage A1, before any arm result.

- **Run:** GitHub Actions [run 42815401534](https://github.com/patcarter883/memory-organ/actions/runs/42815401534), on 32 cores with `--parallel 4 --max-stage-threads 16`.
- **Time:** 12:24 to 18:09 UTC, 5 h 45 min.
- **Grid:** sha256 `f7bacf56…`, unchanged.
- **Results:** `results/runner-2026-11-10/`, holding every `result.json`, `progress.json` and eval `summary.json`. The full artifact (npz, logs) is `ngram-m1b-42815401534`, sha256 `c9a6346d…`, kept until 2026-12-10.

## Outcome against the pre-registered bars

**E1 on test is +0.72%, with file-cluster CI 0.57–0.84%. That puts it in the 0.5–1% zone.**

The M1 kill rule (§6) reads: *no configuration with R2 ≤ +1% reaches E1 ≥ 1% (lower bound > 0.5%) … stop before M2. In the 0.5–1% zone: one Stage-1b (distillation-with-context) arm, then decide.*

The kill condition is met, but the result falls inside the zone clause. The runner's own verdict is `zone_6A7 = "do not scale: one Stage-1b arm on this host, then decide"`. Its `M1_kill = true` flag only records that E1 < 1%. **The pre-registered next step is one Stage-1b arm, not M2.**

| Bar | Value | Result |
|---|---|---|
| E1 test (op = A3, ρ = 0.03) | +0.72% [0.57, 0.84] | below the 1.5% pass bar and the 1% kill threshold |
| R2 (dev, stdlib + ML code) | +0.026% | pass (≤ +0.5%) |
| R1 (dev, WikiText-2) | +0.0075%, mean KL 9.2e-5 | pass (≤ +0.2%) |
| C1 content controls: best control / op | 0.0014 (key-shuffle, permuted keys, NGM, foreign all ≈ 0) | pass |
| NGI: E1(op) ≥ 1.5 × E1(count-LM interpolation) | count-LM +2.95% [1.95, 3.56] | **fail**: count-LM gains 4× as much |
| Above the pruning frontier | false | see note 2 |
| REG | not measured | open (no harness) |

Reported only:

| Measure | Value |
|---|---|
| E2, Split B (later commits) | +0.82% [0.66, 1.02] |
| E3, project-specific tokens | +1.89% [0.98, 2.65] |
| E5, LoRA r16 all-linear | +8.16% on test. The op reaches 0.089 of LoRA's gain. LoRA costs R1 +2.69% and R2 +11.96% on dev |
| E6, RAG-4k | +8.17% [3.93, 12.85]. RAG-4k plus the op: +8.78%, so the op adds +0.61 points on top of RAG |

## Dev results per arm (selection reads these only)

| Arm | ρ | E1 dev | R1 | R2 | Non-zero rows |
|---|---|---|---|---|---|
| A2 | 0.003 | +0.46% | +0.002% | +0.011% | 1.00M |
| A1 | 0.01 | +0.52% | +0.003% | +0.017% | 0.99M |
| **A3 (selected, all three selections)** | 0.03 | **+0.60%** | +0.008% | +0.026% | 1.00M |
| A4, enrich ≥ 4 | 0.03 | +0.46% | −0.002% | +0.021% | 0.34M |
| A5, G-dir α 0.16 | 0.03 | **−1.53%** | **+1.64%** | **+12.2%** | 1.05M |
| A6, foreign repo (control) | 0.03 | −0.01% | +0.006% | +0.068% | 0.81M |
| A7, key-shuffle (control) | 0.03 | −0.00% | −0.004% | +0.013% | 1.00M |
| A8, LoRA | — | +7.22% | +2.69% | +11.96% | — |

## Notes

1. **ρ was best at the top of its grid.** The largest injection budget tested, ρ = 0.03, gave the most E1, and retention cost stayed tiny (R2 +0.03% against a +0.5% bar). The grid is bounded on that side. This run cannot say how much more a larger ρ would give.
2. **The frontier bar cannot pass by construction.** The frontier source is A3, which is also the operating point. The pruned copies zero 1,598 and 15,914 of 1M rows. So the frontier E1 "at the op's R2" is the op itself (0.6003 = 0.6003), and "strictly above" is impossible. This is a design flaw in the bar, not a measured shortfall. It does not change the zone.
3. **G-dir at α 0.16 breaks the 0.5B host** (dev E1 −1.5%, R2 +12%). D-track rule G ("G-dir mandatory", taken from Flash-Next's own PLE) does not carry over to this graft at this α. Revisit it before M2 uses G-dir.
4. **Dedupe gate: test has 3.83% of targets whose 16-gram appears in train**, above the 2% gate. Dev has 1.46%. On the non-duplicated test tokens the op gain is +0.71%, against +1.25% on the duplicated ones. The leak does not explain the result, but by §6.A.7 it is a gate failure to fix in the split.
5. **kNM-lite scored exactly 0.000% on every class.** Its tuned weight is probably 0, or the baseline is broken. Check it before citing it.
6. **The count-LM baseline is the main finding.** A plain count-based n-gram LM over the same train files, interpolated into the output, gains 4× as much as the trained table at layer 1. The count-LM's retention cost is not measured: it is evaluated on project text only.

## Follow-up: what the count-LM lead is made of (2026-11-11, diagnostic)

This is not pre-registered. It ran in GitHub Actions [run 42918990623](https://github.com/patcarter883/memory-organ/actions/runs/42918990623) (artifact sha256 `a9be36df…`) via `experiments/m1b/countlm_ablation.py`; the outputs are in `results/countlm-ablation-2026-11-10/`.

- **What was scored:** Split-A dev (the 13 scored files, M1b's windows and base) and the three retention sets.
- **How:** every condition was left on for every token. The count-LM weights were tuned on M1b's single unscored dev file.
- **Test set:** not read.

| Condition | Dev E1 gain [CI] | R1 WikiText (mean KL) | R2 stdlib / ML code | λ (counts, cache) |
|---|---|---|---|---|
| count-LM as in M1b | +3.05% [1.74, 3.30] | −0.03% (1.1e-2) | +0.57% / +1.13% | 0.01, 0.01 |
| count-LM, fine grid | +3.17% [1.92, 3.42] | +0.17% (1.3e-2) | +0.71% / +1.31% | 0.015, 0.005 |
| **training-file counts only** (KN) | **+2.67%** [1.70, 2.88] | +0.42% (1.0e-2) | +0.75% / +1.29% | 0.015, 0 |
| in-file cache only | +1.03% [0.31, 1.18] | −0.28% (1.9e-3) | −0.06% / −0.04% | 0, 0.005 |
| **the table (A3, M1b operating point)** | **+0.60%** [0.52, 0.92] | +0.008% (9.2e-5) | +0.004% / +0.048% | — |
| count-LM (fine) + table | +3.67% [2.41, 3.94] | +0.18% (1.3e-2) | +0.72% / +1.38% | 0.015, 0.005 |

### What this shows

1. **Most of the lead is storable knowledge, not the in-file cache.** The training-file counts alone gain 2.67%, which is 84% of the full count-LM and 4.4× the table. The cache alone gains 1.03%. The two overlap: 2.67 + 1.03 > 3.17.
2. **Left on, the count-LM fails the retention bars and the table does not.**
   - The count-LM's R2 on ML code is +1.1–1.3%. That is above the +0.5% pass bar and the +1% kill-zone limit.
   - The counts-only version also costs +0.42% on WikiText (R1 bar +0.2%).
   - The table costs at most +0.05%, and its mean KL is about 100× lower.
   - M1b's +2.95% test figure for the count-LM was "gated", applied to project text only.
3. **The in-file cache is a gain for any text.** It helps the retention sets too (R1 −0.28%). It is a general cache effect, not project memory.
4. **The table adds on top of the counts.** Count-LM + table gains +3.67%, against +3.17% for the count-LM alone. That is +0.50 points, or 83% of the table's own gain. The table also repairs the count-LM's damage on operators: lex:op goes from +0.94% to −0.05%.
5. **Open:** the count-LM's gain at the table's retention cost. That needs λ swept down to about 0.0005, scored on R2. The stored NLLs cannot be re-mixed, so it needs a re-run.
