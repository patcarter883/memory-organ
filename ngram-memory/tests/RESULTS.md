# Test results (M0 integration)

All runs are on the M0 container: 4 CPUs, no GPU, torch 2.15.0+cpu, transformers 5.22.2, Python 3.13. Torch runs
one thread per pytest worker (`tests/conftest.py`).

## Default suite (the M0 bar: < 10 min on 4 CPUs)

```
cd ngram-memory
OMP_NUM_THREADS=1 python -m pytest -q -m 'not slow' -n 4 -p no:cacheprovider
```

| Result | Value |
|---|---|
| Counts | **284 passed, 1 skipped, 0 failed, 0 xfailed** (2 slow tests deselected) |
| Wall-clock | 40.7 s in pytest, 42.5 s `real` |

The one skip is `tests/test_beside_native.py:118`, a crop on the tiny qwen4_exp host. HF's qwen4_exp cache cannot
crop: `TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'`. The graft's own crop is covered on the
qwen2 / qwen3_5 / gemma4 hosts.

Before integration, the same command gave 258 passed, 1 skipped, 22 xfailed in 47 s. The 22 strict xfails were the
verifiers' defect tests. All of them now pass as normal tests (see `M0_STATUS.md`, "Verifier defects").

Log: `scratchpad/nm/m0/integrate/suite_final.log`.

## Slow suite (real weights, run once)

```
OMP_NUM_THREADS=1 NGRAM_MEMORY_LOG_DIR=<dir> python -m pytest -q -m slow -p no:cacheprovider -s
```

| Result | Value |
|---|---|
| Counts | **2 passed** (`test_qwen25_real.py`, the real-corpus census test in `test_ingest_rowpolicy.py`), 285 deselected |
| Wall-clock | 31.1 s in pytest, 33.8 s `real` |

The Qwen2.5-0.5B real-weights test gave:

- **No-op:** max|Δlogit| = 0.0, bitwise.
- **Decode parity:** host alone 2.67e-5, with the graft 3.17e-5.
- **Graft effect:** 2.85.
- **Removal and host:** removing the graft restores the base bitwise, and the host weights are unchanged (sha256 over every state_dict tensor).
- **Peak RSS:** 3,557 MB.

Logs: `scratchpad/nm/m0/integrate/suite_slow_final.log`, `scratchpad/nm/m0/integrate/slow/qwen25_real.json`.

## Census acceptance (not a pytest test; M0 "census exact")

```
OMP_NUM_THREADS=1 python scripts/count_rows.py --out X.json --reference --framings --commits
```

**38 MATCH, 0 MISMATCH.** It reproduces:

- 7,555,967 distinct rows on all files;
- 2,385,780 row-level count ≥ 2;
- 2,040,221 / 6,721,239 / 550,891 on the legacy 85/15 split.

It ran in 194 s with peak RSS 2,086 MB. Log: `scratchpad/nm/m0/integrate/count_rows.log`.

The HF stub parity (11,405 / 11,405 positions) is a default-suite test: `tests/test_hf_stub_parity.py`. It skips
when the Flash-Next tokenizer is not available.
