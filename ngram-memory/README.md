# ngram-memory

ngram-memory is a project of its own, separate from memory-organ's `cam/` concept. It is a portable editor that
writes rows into a model's **native n-gram table** (Engram-style; Qwen3.8-Flash-Next's per-layer embedding module).
It also grafts a table in the same format onto models that don't ship one, and ingests a codebase into such a table
as a "project cartridge" (Stage 1).

**Status: M0 is implemented (package, tests, ingest pipeline, Stage-1 eval harness, M2-0 kit, D-track G and S).**
See [M0_STATUS.md](M0_STATUS.md) and [tests/RESULTS.md](tests/RESULTS.md). M1b, the proxy-ingestion experiment on
Qwen2.5-0.5B, is pre-registered in [experiments/m1b/PREREG.md](experiments/m1b/PREREG.md).

Documents:

- [DESIGN.md](DESIGN.md) is the design. It covers:
  - the format spec (§2);
  - the package layout (§3.1);
  - backends (§4);
  - ingestion (§I);
  - the solver (§5);
  - evaluation (§6);
  - the dissection program (§D);
  - the compute plan (§7) and milestones (§8).
- [serve/INTEGRATION.md](serve/INTEGRATION.md) covers the live-serve integration plan for the owner's minisgl fork.
- [proofs/](proofs/) holds the proof scripts and outputs that DESIGN.md cites. `proofs/dtrack/` holds the M0 D-track runs (G, S).

## Install

CPU is enough for everything in M0. Python ≥ 3.11.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt        # torch (CPU wheel is fine), transformers 5.22.2, datasets, pytest...
pip install -e .                           # or just put this directory on sys.path
```

- `ngram_memory.spec` and `ngram_memory.hashing` are numpy-only and import without torch or transformers. They are the part a torch-free engine can vendor.
- Everything else is the training side and needs torch + transformers.
- The tests and scripts expect the Qwen2.5-0.5B weights in the local HF cache. Only the slow tests and the real-host scripts use them.
- The minisgl clone at `/home/user/patcarter883/minisglang-rdna4` is read-only. It provides the corpus (git blobs at pinned commits) and the reference implementations some tests compare against.

## Test

```bash
cd ngram-memory
OMP_NUM_THREADS=1 python -m pytest -q -m 'not slow' -n 4     # default suite (~1 min on 4 CPUs)
OMP_NUM_THREADS=1 python -m pytest -q -m slow -s             # real-weight tests (Qwen2.5-0.5B, ~3.6 GB RSS)
```

`conftest.py` pins torch to one thread per worker. The recorded counts and wall-clock are in
[tests/RESULTS.md](tests/RESULTS.md).

## Run

| What | Command |
|---|---|
| Row census (M0 acceptance, 38 checks against the proof JSONs) | `python scripts/count_rows.py --out OUT.json --reference --framings --commits` |
| Check / rebuild the frozen splits | `python scripts/ingest.py splits [--rebuild]` |
| Train a cartridge (proxy host) | `python scripts/ingest.py train --out DIR --steps 20 --threads 4 [--arm G-lin\|G-dir\|rows+Wv\|key-shuffle] [--rho 0.01] [--lambda-l1-rel 0.1]` |
| Stage-1 evaluation | `python scripts/evaluate.py --splits SPLITS.json --out DIR --sets dev,r1,r2 --conditions base,countlm,rows:CART.pt --fmt FMT.json` (Split B: run `--sets splitb` on its own) |
| Reproduce the §I.8 / rt-science numbers | `python scripts/repro_ctx_len.py`, `python scripts/repro_tokdelta.py` |
| GDN backward benchmark (M2-0) | `python scripts/gdn_bench.py --cases 35b:4:2048 --impls ours,hf,fork --out X.json` |
| M2-0 one-layer feasibility (owner's GPUs; CPU smoke here) | `python scripts/m2_0_feasibility.py --help` |
| 4-bit host loaders check | `python scripts/host_quant_validate.py --help` |
| D-track S slice fetch (HTTP range reads) | `python scripts/fetch_slice.py --help` |
| Inspect / verify / compose `ngram-edit/v1` patches and cartridges (numpy only) | `python scripts/apply.py --help` |
| **M1b** (pre-registered grid) | `python experiments/m1b/run.py --out DIR [--smoke]` (resumable, see the PREREG) |

## Package layout (DESIGN §3.1)

```
ngram_memory/
  spec.py hashing.py tables.py codecs.py      # format contract, torch-free hashing, row stores, row codecs
  ledger.py patch.py cartridge.py             # edit ledger, ngram-edit/v1 patches, project cartridges
  backends/  base.py qwen4exp_native.py gemma4_ple.py alias.py
             graft/ module.py qwen35.py gemma4.py beside_native.py gdn.py host_quant.py
  serve/     hf_adapter.py                     # HF row overlay (the serve-side rule: cast(base + delta))
  ingest/    corpus.py rowpolicy.py kltargets.py train.py
  eval/      metrics.py report.py stage1/ (splits, scoring, conditions, tokclass, datasets, qa, capability,
             foreign, baselines/{countlm, bm25_rag, knm, ngm, controls, lora})
  dissect/   hf_range.py row_stats.py slice_probe.py      # D-track G and S
```
