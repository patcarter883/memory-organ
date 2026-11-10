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
