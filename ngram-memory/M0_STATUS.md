# M0 status (integration)

M0 is mostly done. Every M0 invariant listed below holds, the default suite runs in under a minute against the
10-minute bar, and the row census matches its 38 reference counts exactly. The open items are the binary fixtures,
`scripts/fetch_fixtures.py` and the 1B-token row-load count (table below).

This file tracks M0 against the M0 row of DESIGN §8. It also covers the pieces built for M1b, M2-0 and D-track G and S.

- **Run here:** every number below comes from a command run in this container.
- **Paths:** `S/` = `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/`, and `P/` = `proofs/`.
- **Test counts:** see [tests/RESULTS.md](tests/RESULTS.md).

## M0 deliverables (DESIGN §8, row M0)

| Deliverable | Status | Evidence |
|---|---|---|
| Package skeleton (§3.1) + `requirements-dev.txt` | **done** | `ngram_memory/` layout as in README; `requirements-dev.txt` |
| Hashing parity (I4) against real buffers | **done** | `tests/test_hashing.py` covers both checkpoint buffer fixtures, q4 parity on 4 keys, minisgl chunked prefill (19,200 rows) and the Eiffel real-tokenizer rows. The verifier re-ran about 137k row ids with 0 mismatches (`S/m0/verify-core/probe4.json`). |
| Native locality / decode / gradient on tiny qwen4_exp | **done** | `tests/test_native_qwen4exp.py` (B/C/D locality, decode, gradient; apply/revert bit-exact; shard store) |
| Graft no-op, decode, padding, gradient on tiny qwen3_5, qwen3_5_moe | **done** | `tests/test_graft_tiny.py`: no-op `torch.equal`, cached decode ≤ 2e-6 against a > 1e-2 effect, left padding, sparse COO gradient |
| … and gemma4 | **done** | `tests/test_gemma4.py::test_gemma_graft_noop_decode_padding_gradient` |
| Graft built without `Qwen4ExpTextConfig` | **done** | `NGramGraft` builds the PLE from a SimpleNamespace (`backends/graft/module.py`) |
| G-dir arm: no-op, decode-state, gradient | **done** | `tests/test_gdir_unigram.py`; dilated-tap decode parity |
| Unigram-head subclass | **done** | `tests/test_gdir_unigram.py`; heads 0–15 keep Flash-Next row ids at **every** `ple_layer_index` (VC-4 fix, `tests/test_verify_core_defects.py`) |
| Graft beside native on tiny qwen4_exp | **done, crop skipped** | `tests/test_beside_native.py` covers no-op, decode state, padding and gradient. The crop test skips because HF's qwen4_exp cache cannot crop (see RESULTS.md). |
| Graft `crop()` | **done (qwen2, gemma4); partial (qwen3_5)** | qwen2 runs a 5-token speculative rollback plus HF assisted decoding. qwen3_5 only crops the graft's own slots, because HF cannot crop the GDN recurrent state. |
| Gemma PLE editor-init | **done** | `tests/test_gemma4.py::test_gemma_ple_editor_init_graft`, `::test_gemma_ple_backend_locality_decode_gradient_saturation` |
| Ledger rollback | **done** | `tests/test_ledger.py::test_v5_naive_rollback_residue_vs_ledger_exact`. The ledger view now equals the served bits bit-for-bit (VF-1). |
| `ngram-edit/v1` + cartridge manifest round-trip, HF adapter ↔ overlay prototype bitwise | **done** | `tests/test_cartridge_roundtrip.py` (graft and native, `torch.equal` logits) |
| `ingest/` pipeline: blob reader, secret scan, dedup, Splits A/B, framings, windows with true history | **done** | `tests/test_ingest_*.py`. The secret scan was widened (VI-3) and still has 0 hits on the 877 `.py` files of the repo. Frozen Split A gates new files (VI-2). `scripts/ingest.py splits` reports that the stored splits match a rebuild. |
| Row census reproduces `count_rows.out` (7,555,967 rows) | **done** | `scripts/count_rows.py`: **38/38 MATCH** (`S/m0/integrate/count_rows.log`) |
| 11,405 / 11,405 HF stub parity | **done** | `tests/test_hf_stub_parity.py`: 21 files, 11,405 positions, 0 mismatches (Flash-Next tokenizer sha 0997f410…) |
| Fixtures + `fetch_fixtures.py` | **partial** | The JSON fixtures were copied byte-identical from `P/` (`tests/fixtures/`). **`scripts/fetch_fixtures.py` is not written**, the binary `.pt` fixtures are absent, and the tests that need them skip. |
| Vendored `R/ple/hashing.py` | **done** | `tests/fixtures/vendor/minisgl_ple_hashing.py` (df29117 plus a 2-line header) |
| Start the 1B-token row-load count | **not done** | D-track G counted a public proxy instead: 52.1M code + 34.3M prose tokens (`P/dtrack/G/g1_counts.out`). The 1B-token count on the owner's traffic counters is not started. |

**Pass bars:**

- **All invariants bitwise as in P/:** met. The no-op, removal and rollback checks are `torch.equal` in the tiny-host tests, and the real Qwen2.5-0.5B no-op is max|Δlogit| = 0.0.
- **Census exact:** met.
- **Default suite < 10 min on 4 CPUs:** met, at 42 s.
- **Slow proofs marked:** met. The 2 real-weight tests are `@pytest.mark.slow`. There is no solver yet: it is M7.

## Verifier defects (all 21 fixed; 0 rejected)

Each defect's strict-xfail test is now a normal passing test in `tests/test_verify_*_defects.py` (and
`test_verify_gdnkit_*.py`, `test_verify_dissect_vd1.py`). All of them were re-run as failing before the fix.

| ID | Piece | Fix |
|---|---|---|
| VC-1 (major) | core | Under gradient checkpointing, the graft's layer pre-hook now uses the ids, mask and context bound at forward time, keyed by the layer input's storage. Before, it used the latest top-level call. Two forwards then one backward now give the reference gradient. |
| VC-2 | core | `SparseRowEmbedding` gathers only hit positions. A miss adds no COO entry, so SparseAdam leaves unread rows alone. |
| VC-3 | core | A miss reads an exact 0, even next to a non-finite stored row. |
| VC-4 | core | Unigram-extension primes: the order ≥ 2 heads skip (#order≥2 heads)·ple_layer_index primes, and the unigram heads follow them. |
| VC-5 | core | `test_qwen25_real.py` hashes **every** state_dict tensor, and its RSS note is corrected to about 3.2 GB. |
| VI-1 (blocker) | ingest | The key-shuffle cartridge is the trained slots served through the **true** keys. ω and s_j follow the n-gram that trains each slot. `relabelled_store()` is kept as a diagnostic. |
| VI-2 | ingest | With a frozen Split A, new files still go through the near-dup cluster rule and the verbatim-line gate. The moves are recorded. |
| VI-3 | ingest | The secret name may be the suffix of an identifier. New rules cover env-var defaults, Stripe keys and AWS secret keys; dotted identifiers are excluded. Census and splits are unchanged. |
| VE-1 | eval | Split B builds countlm, RAG, kNM and LoRA from the tree at T_cut, and is scored in its own run. A mixed run with data-dependent arms is refused. |
| VF-1 | formats | Ledger `materialise` = base + Σδ, the same association as the served compile. |
| VF-2 | formats | Zero-row patches and EditRecords work (`patch.rows2d`). |
| VF-3 | formats | Chains compose in §I.7 layer order: cartridge, then increments, then facts. Fact patches may not reset rows. |
| VF-4 | formats | The native backend on a bf16/fp16 table defaults to the one-cast serve rule. |
| VF-5 | formats | The graft host-config pin drops runtime-only fields (`RUNTIME_CONFIG_KEYS`). |
| VF-6 | formats | The served file authenticates `row_ids` (sha256) and the row count. |
| VF-7 | formats | `verify_patch` requires `module_fingerprint` and checks `module_spec`. |
| VF-8 | formats | `RowLedger.save` always rewrites the edit files, atomically. |
| VF-9 | formats | ModuleSpec canonicalises numeric types before fingerprinting. |
| VGK-1 | gdnkit | A fresh-cache prefill (use_cache=True, no previous state) goes through our kernel. The conv and recurrent cache state is filled the way HF fills it, and greedy generation matches HF (new test). |
| VGK-3 | gdnkit | The native `gdn_prefill` receives fp32 `A_log` / `dt_bias`, as the fork does. |
| VD-1 | dissect | The within-n-gram variance share is the one-way random-effects ANOVA component estimate, not SS_w/SS_t. |

Counting: 5 core (the core verifier gave no IDs; VC-1..5 are the integration's labels in the core verifier's
order), 3 ingest, 1 eval, 9 formats, 2 gdnkit and 1 dissect. VC-5 is a test-quality issue without an xfail test;
the other 20 had strict-xfail tests (22 test items, since VC-1 and VGK-1 are parametrised over two hosts). New
regression tests were added for VI-3 (no false positives on code) and VGK-1 (fresh-cache prefill fills the cache
like HF). No VGK-2 test or report was handed over, so there was nothing to integrate under that ID.

API changes between pieces:

- `scripts/evaluate.py` gained `--reuse-base`, `--fmt-map`, `--max-dev-files` and split-B window pruning (same targets, less compute).
- The tune files no longer depend on which sets are scored.
- `datasets.train_stream(..., at=)` was added.
- `gated_delta_rule_from_raw(..., return_final_state=)` was added.

## M2-0 kit (written in M0, runs on the owner's cards)

| Piece | Status | Evidence |
|---|---|---|
| `backends/graft/gdn.py`: chunked GDN forward plus the fixed backward (`RawGDN`); saves only raw tensors and boundary states | **done** | dX/dq/dk/dv/dg/dβ cosine vs HF's torch delta rule ≥ 0.999999999997 on 35B shapes; forward relative error 5.1e-8 (`S/m0/integrate/gdn_bench_35b_T2048.log`) |
| Native-forward plumbing (fork `gdn_hip.gdn_prefill`) behind our backward | **done (CPU stand-in only)** | `tests/test_gdn.py::test_native_forward_plumbing_matches_ours`; fp32 gate parameters (VGK-3). Not run on a GPU. |
| `host_quant.py`: AWQ-4bit / MXFP4 loaders | **done** | 768/768 matrices bit-exact vs the reference unpack and the fork (`S/m0/gdnkit/hq_validate_A.json`); real-layer GDN fp32 dX cosine 0.99999999999947 (`hq_validate_C.log`) |
| `scripts/m2_0_feasibility.py` (one real 35B layer of each kind, timings, peak VRAM) | **CPU smoke only** | `S/m0/gdnkit/m2_0_cpu_small.log`. GPU bars are N/A here: per-card peak VRAM and the native-forward ratio need the owner's cards. |

**Measured GDN backward speed-up** (CPU, 4 threads, idle box, 35B-A3B GDN shapes, B=4, T=2048;
`S/m0/integrate/gdn_bench_35b_T2048.log`):

| Implementation | Backward (ms/token) | Forward + backward (ms/token) |
|---|---|---|
| ours | 0.737 | 1.109 |
| fork reference | 2.196 | 2.425 |
| HF torch | 1.096 | 1.364 |

- **Against the fork's reference backward:** 2.98× faster on the backward, 2.19× on forward + backward.
- **Against HF's torch delta rule:** 1.49× and 1.23×.
- **At T=4096** (gdnkit run, `S/m0/gdnkit/bench_core_T4096_{ours,hf}.log`): ours 0.839 / 1.287 ms/tok, HF 2.309 / 2.691, so 2.75× on the backward and 2.09× on forward + backward.
- **What these numbers are not:** they are CPU proxies. The M2-0 bars (fixed backward ≤ 1.5× the native forward per token; < 0.2 h per 1M tokens on the 35B) must be measured on the GPUs.

## D-track G and S (§D.3)

**G** (row coherence vs frequency; 5,000 stratified n-grams per order + 300 project-only + 2,000 null, Flash-Next
rows via HTTP range reads, frequency from a public proxy of 52.1M code + 34.3M prose tokens; `P/dtrack/G/g2_rows.out`):

- **Crossing frequency f\* (coherence = 0.05):**
  - 2-grams: **22.9 per Mtok**, 95% bootstrap CI [18.7, 28.2]. That is 1,977 occurrences in the proxy.
  - 3-grams: **1.07 per Mtok**, CI [0.88, 1.36].
- **Coherence by frequency:**
  - Top-frequency bins: 0.22–0.25 (2-gram) and 0.21–0.23 (3-gram).
  - Project-only n-grams: 0.0011 (2-gram) and 0.0058 (3-gram).
  - Random null: 0.0001 and 0.0002.
- **minisgl Split-A train count ≥ 2 n-grams below f\*:** 98.4% of distinct n-grams and 70.1% of occurrences (2-gram); 92.7% and 70.1% (3-gram).
- **W_v Frobenius:** 2-gram half 34.92, 3-gram half 17.72. The mean 3-gram value share is 0.249.

**S** (2-layer Flash-Next slice, 30,000 tokens: 20k code + 5k md + 5k wiki; `P/dtrack/S/s_probe.out`,
`s_recommend.json`, `s_within_anova.out`):

- Per stream:
  - gate sigmoid mean 0.056 / 0.144 / 0.122 / 0.015;
  - conv/direct median **3.81 / 34.8 / 14.4 / 11.1**;
  - ‖PLE out‖/‖X‖ median **0.48 / 2.20 / 0.37 / 0.29** (all streams: median 0.55, p95 1.18).
- Gate-logit within-3-gram-context variance share (ANOVA, VD-1 fix): **0.0003 / 0.032 / 0.010 / 0.008**.

| §D.3 decision rule | Fired? |
|---|---|
| G-dir mandatory in M2/M3 if conv ≥ 2× direct on the median token in ≥ 2 of 4 streams | **YES** (4/4 streams, all text and code) |
| Calibrated gate arm if ≥ 50% of gate-logit variance is within n-grams | **NO** (0/4 streams; max 3.2%) |
| Row policy "count ≥ 2 in project, below f\* in general" if project rows below f\* carry ≥ 70% of the M2 E1 gain | **not decidable yet**: f\* is measured; the rule needs M2's cartridge |
| Transplant F dropped if < 10% of the trained cartridge's gain | **not decidable** (F follows M3) |

The S scale recommendation (Flash-Next injects about 0.55‖X‖) does **not** change the M1b ρ grid. The
pre-registered reason is in `experiments/m1b/PREREG.md` §7. S does set the M1b G-dir α = 0.16.

## Open issues

- `scripts/fetch_fixtures.py` and the binary `.pt` fixtures, and the 1B-token row-load count (table above).
- **REG** (regurgitation, canary and secret harness) is not implemented, so M1 PASS cannot be declared. **MEM** is not measured.
- 10 of the 29 Split-A test files (and 1 of 15 dev files) have ≤ 1,024 tokens, so they have no target at the fixed context. The test set scores 19 files, about 70% of the test targets (`S/m0/verify-eval/cov.out`). Corpus v1 (tools/, tests/) is still not added.
- Every scoring window starts the graft with an EOS-seeded n-gram history. Mid-file windows therefore hash their first two positions differently from the serve. These are context positions, identical for every arm.
- LoRA covers attention and the dense MLP only. Fused 3-D MoE experts are not adapted, which is irrelevant for the dense 0.5B proxy.
- On qwen3_5 hosts, HF cannot crop the GDN recurrent state, so full speculative-decode crop is tested only on qwen2.
- `ForConditionalGeneration` (multimodal) host classes are untested. inputs_embeds-only calls raise.
- The ingest piece's regulariser defaults:
  - λ̃₁ is now relative to G₀ (`lambda_l1_rel`), and M1b fixes λ̃₁ = 0.1.
  - The §I.1 sweep {0, 0.1, 0.3} is deferred.
  - The ρ budget is still a soft penalty on anchor batches only.
- Speed on this box:
  - A 4,096-token proxy training step takes about 82 s on 4 idle threads.
  - Hence M1b's partial-epoch budget (PREREG §1, §8).
  - A bf16 host would be 1.8–2.8× faster per forward/backward (AMX is present). It is not used, to stay comparable with the fp32 reference numbers.
