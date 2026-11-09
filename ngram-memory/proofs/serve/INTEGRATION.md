# Applying n-gram edits to the live Flash-Next minisgl serve: integration and solve plan (v2)

Repo read: `/home/user/patcarter883/minisglang-rdna4`, branch rdna4, HEAD `df29117` ("fix(muse): let the DFlash drafter boot"). Read-only; nothing in it was modified. v1 of this plan was written against `b98eb99`.

Abbreviations:
- `R/` = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/`.
- `T/` = `/home/user/patcarter883/minisglang-rdna4/tools/`.
- `DC/` = `/home/user/patcarter883/minisglang-rdna4/docker-compose.yml`.
- `HF/` = `.../site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py` (transformers 5.22 in the scratchpad venv; unchanged since v1).
- "same" = the cited range is byte-identical at the same lines in b98eb99 and df29117 (mapped with `git diff -U0` hunks, then read by eye).

Sources: v1, plus three v2 re-checks under `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/`: `v2-editpath/` (claim-by-claim line map, torch-free overlay prototype, hook cost, requantisation), `v2-solve-local/` (checkpoint inventory, CPU solve benchmarks, cost model), `v2-deltaverify/` (independent requantisation measurement), and `v2-solve-venue/` (autograd coverage on HF modules).

**Engine design constraint (owner).** The minisgl engine is deliberately as torch-free as possible. Every engine-side change in this plan (overlay, control op, invalidation) is numpy/mmap/ctypes on the row path and adds no torch dependency. The edit **solve** runs in ngram-memory's own separate process (torch is fine there) on transformers' pure-torch qwen4_exp modules plus our own loader. It reuses only the fork's torch-free numpy row decoder (`R/weights/row_table.py`, stdlib+numpy imports only, `:37-47`) and, for P0, the numpy-only `R/ple/hashing.py`. No engine forward or backward path is used by the solve.

---

## 0. Changes since v1 (b98eb99 -> df29117)

**What changed in the code that matters here.** 293 commits; `R/ple/*` byte-identical; `R/weights/row_table.py` and `R/weights/uring.py` also byte-identical (`git diff b98eb99 df29117` empty; both last touched in d59884d, older than b98eb99). Also unchanged: `R/kvcache/radix_cache.py`, `R/scheduler/cache.py`, `R/scheduler/io.py`, `R/kvcache/composite_state.py`, `R/ple/graph_capture.py`. Changed: `R/scheduler/scheduler.py` (+293/-45; almost every v1 line moved), `R/server/api_server.py` (+127/-32), `R/models/qwen4exp.py` (+1 line shift; only MTP changed), `R/models/config.py`, `R/engine/config.py`, `R/core.py`, the MXFP4/quant code, `T/serve.sh`, and the new `T/ple_split.py`. Full map: `v2-editpath/claims_map.txt`.

**Corrections to the brief this v2 was commissioned with.**
1. **The deployed n-gram rows are not MXFP4. They are int6 group-32: 120 B of 6-bit codes + 5 fp16 scales = 130 B per row.** MXFP4 is the expert format only.
   - The opener picks the codec from the scale dtype: F16 scales -> `INT6_G32` (`R/weights/row_table.py:1001-1003`); U8 (E8M0) scales, i.e. an MXFP4 PLE table, raise `NotImplementedError` (`:1004-1010`, again at `:828-834`). The serve boots, so it is decoding int6.
   - `T/serve.sh:1369-1372, 1398` ("int6 group-32 rows as `shard_N.weight_packed` + `shard_N.weight_scale`"; "row_table decodes int6 group-32") and `DC/:903-905` say the same.
   - Sizes only fit int6: 320,001,536 rows x 130 B = 38.74 GiB (the owner's 38.7 GiB). MXFP4 would be 85 B/row, 25.3 GiB. The "~120 B per row" in the brief and in `T/ple_split.py:4`, `DC/:907` is the code bytes only.
   - Checkpoint headers (fetched by range request, `v2-solve-local/ckpt_inventory.txt`): `ngram_embedding.shard_S.weight_packed` U8 [2,500,012, 120] + `weight_scale` F16 [2,500,012, 5]; 128 row tensors in 8 safetensors files that carry nothing else (`T/ple_split.py:6-7, 56-61`).
2. **`row_table.py` and `uring.py` did not change.** The int6 fused table (`FusedScaleRowTable`, `row_table.py:796-909`) already existed at b98eb99; v1 treated it as a side option. It is the served codec.

**What this changes in the plan.**
- **Codec (§2.2, §3.5, §5 P1).** Every fp8-specific v1 statement is obsolete: the ~2e-4 global scale, clipping at ~0.09, E4M3 rounding, 160 B rows read once. The global fp8 scale path (`row_table.py:1088-1102`) is never reached: the packed branch returns at `:1077-1079`. A row is two reads (codes, scales), each through its own 16-worker pool (`:860-878`).
- **Storage (§2.2).** Rows are in the **split layout**: 8 row-only shards relocated to Optane XFS (`Q4E_PLE_DIR`) by `T/ple_split.py`; the heads metadata stays in the model dir and is named via `MINISGL_PLE_META_FILES` from the heads-only shards (`T/serve.sh:1308-1368`). Residency is the XFS page cache, not the ZFS ARC. An inline copy remains in the model dir (`T/ple_split.py:153-159`), and `serve.sh` falls back to it if the split is not found (`:1369-1398`).
- **Overlay hook (§3.1).** Same place (`R/ple/source.py:243`, after `decode_raw`), now proven torch-free with the fork's own int6 opener. Revised representation: **compiled bf16 replacement rows** (bit-identical to fp32 deltas after the serve's bf16 cast, half the memory), with per-edit fp32 deltas kept on disk for composition and rollback.
- **In-file requantisation (§3.5, new measurement).** Writing edits into the int6 files only holds for large deltas (>= ~30% of the row norm, scales rewritten); small solver steps vanish. Plus read-only mounts, torn two-tensor rows, and the inline-fallback trap. Not recommended.
- **Control path (§3.2).** Every scheduler line moved. The deployed loop is `ep_loop` (EP=1, EP-over-TP, TP=2), not `normal_loop`; the control op must be rank-symmetric and never raise on one rank only, because of the EP lockstep drift check (`R/scheduler/ep.py:60-79`). New opt-in `split_loop` (EP=0 only) needs a deferred flush.
- **Prefix cache (§3.4).** Recurrent radix with the combined GDN+PLE snapshot is the plan for q4e (`R/engine/config.py:95-123`; `gdn_radix` default True, `R/scheduler/config.py:23`), and `ep_loop` commits through `_process_last_data` (`ep.py:164, 188`), which stashes and captures snapshots (`scheduler.py:1638, 1744`). The "naive" comments (`T/serve.sh:1015-1017`, `_stage_ple` docstring) and the "inert under EP" help text (`R/server/args.py:386-388`) look stale; see open question 1. v1's invalidation work is still required.
- **Solve venue (§4).** v1 rejected a local process concurrent with the serve. **v2 reverses that:** with ~20 GB free and the solve built on the compact MXFP4 checkpoint read by O_DIRECT, a local CPU solve (option A) fits in 11-15 GiB, hard-capped with no swap, and is the recommended first experiment. Not interactive: ~20-40 min per single edit, ~2-5 min per edit in batches of 32 x 4 (only with the prefix state spilled to NVMe; §4.2).
- **Memory figures.** v1 planned on MemAvailable 11.1 GiB with 33.9 GiB swap used (`T/serve.sh:1116-1122`). Owner's figure now: **~20 GB free while serving; ample swap, on the same Optane drive as the n-gram table**. Consequence: neither the overlay nor the solve may use swap (it would contend with the 16-rows-per-token gather).
- **Boot time.** ~300 s (`R/weights/ckpt_read.py:5-10`), not v1's 618 s.

---

## 1. TL;DR

**How an edit is applied, in both the first experiment and steady state.**
- An edit is a set of fp32 deltas keyed by global row id, persisted per edit on disk.
- The serve holds a numpy-only overlay inside `PLEEmbeddingSource.finish_rows`, directly after the int6 decode and before the H2D copy (`R/ple/source.py:243`; copy `:244`, bf16 cast `:245`). It overwrites the edited rows with precompiled bf16-exact values.
- Every forward reads every row from disk with no row cache, so the overlay covers prefill, decode and spec-verify.
- Graph capture is unaffected: the static device buffers keep their pointers (`R/ple/graph_capture.py:132-147`).
- Cost per forward (sandbox, 200k edited rows): ~11 µs per decode call, 1.1-1.8 ms per 2048-token chunk; the int6 dequant it follows costs 61-67 µs and ~45 ms respectively (`v2-editpath/bench_hook.txt`).
- Do **not** write into the int6 files (§3.5).

**The hard part of application is still the prefix cache, not the table.**
- Radix KV pages and recurrent snapshots both contain table contents, and recurrent radix is on.
- Nulling `rec_state` alone is not enough: stale KV gets re-adopted.
- **First experiment:** stop traffic, apply, flush the whole tree.
- **Steady state:** targeted detach of stale radix nodes plus an epoch guard.

**Where the solve runs.**
- **First experiment: (A) local CPU process, concurrent with the serve.** Our own torch process on transformers' qwen4_exp modules; our loader reads the served MXFP4 experts, FP8-block projections and int6 rows. Prefix-cached suffix-only iterations; 4 pinned cores; `MemoryMax=15G`, `MemorySwapMax=0`; experts re-read from NVMe with O_DIRECT, never cached, never via swap. ~1-2 min per iteration, ~20-40 min per single edit. Serve impact estimated 0-10% decode, to be measured.
- Faster alternatives with the same code: **(B)** the serve's own GPUs during a ~10-15 min downtime (3-10 s per iteration); **(C)** one cloud H200 holding the whole MXFP4 body (1-5 s per iteration).

**Steady state.**
- Overlay plus targeted invalidation in the serve.
- (A) as nightly batches. Under `MemoryMax=15G` at most ~64 sequences fit in RAM (§4.2): 16 edits x 4 prompts = 3.2-6.9 min per edit. 32 x 4 (2.2-4.8 min per edit) needs its prefix state spilled to an NVMe file, not swap. Pair it with forward-only acceptance checks through the serve's overlay (e3). (B) for urgent large batches.
- An in-serve backward (v1 option c) is not planned: it would put torch autograd into the engine, against the design constraint.

---

## 2. The serve's PLE path

### 2.1 Hashing parity: bit-identical (checked on CPU; code unchanged)

| Piece | minisgl at df29117 | HF | Same? |
|---|---|---|---|
| splitmix64, `_PRIME_1=10007` | `R/ple/hashing.py:56-68` | `HF/:1025-1037` | verbatim |
| seed 1234 (absent from config.json) | `hashing.py:61` (`DEFAULT_NGRAM_SEED`) | config default | yes |
| multipliers | `hashing.py:71-92` | `HF/:1040-1049` | yes |
| prime head sizes | `hashing.py:95-131` | `HF/:1091-1095` | yes |
| shift right, ignoring EOS | `hashing.py:134-160` | `HF/:1107-1121` | yes |
| head layout: heads 0-7 bigram, 8-15 trigram; each order's key repeated 8 times with different prime moduli | `hashing.py:254-272` | `HF/:1155-1167` | yes |
| `row = off[h] + key mod size[h]`, offsets = prefix sum | `R/weights/row_table.py:936-954` | `HF/:1165-1166` | yes |
| EOS = `text_config.eos_token_id[0]`; history seeded with EOS | `R/models/config.py:969-975` (text_config unwrap; now also accepts `llm_config`, same result for qwen4_exp), `:1428-1438` (EOS); `R/ple/state.py:56-58` | `HF/:1134-1148` | yes |
| history = last 2 tokens across chunks | `state.py:136-150` | `HF/:1137-1148` | yes |
| 128 row tensors, row r in tensor `r // 2,500,012` | `row_table.py:361-378`; deployed packed regex `:1059-1062` and fused-table height check `:823-827` (`:1048-1053` is the fp8 `.weight$` regex, not used) | `Concatenate` over shards in natural numeric order (`transformers/conversion_mapping.py:2132-2141`) | same row order |
| `ple_layer_index=0` (ordinal among PLE layers; HF `ple_layer_ids=[2]` is 1-based = decoder index 1) | `R/ple/runtime.py:407-419` | `HF/:1262` | yes |

**CPU check.** `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/ple_path/hash_parity.py` (v1) compares HF's full-sequence row ids with minisgl's chunked-prefill-then-decode ids, including EOS at the start, middle and repeated, plus token 0 and V-1.
- Result: all equal (tiny config 1,904 ids; real parameters V=248320, base 20M: 8,112 ids). `hashing.py` is unchanged, so the result stands.
- Multipliers match the checkpoint values at `hashing.py:11-13`. Primes sum to 320,001,446, padded to 320,001,536 rows = 2,500,012 per tensor.

**Padding: the only place the two sides can disagree.** With a `conv_mask`, HF replaces padded ids with EOS (`HF/:1466-1470`) and zeroes gated values at padded positions (`HF/:1245-1247`). The serve never pads inside a sequence. In the solve, one prompt per sequence or **right**-pad; never left-pad.

### 2.2 Table storage and streaming (rewritten for int6 + split layout)

- **What is served.** `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` (named at `row_table.py:1056`, `R/models/weight.py:1406-1416`), deployed as "q4e-mxfp4", 108.5 GiB payload (`R/weights/ckpt_read.py:531-536`).
  - **Rows:** 128 tensors `...ngram_embedding.shard_S.weight_packed` U8 [2,500,012, 120] + `weight_scale` F16 [2,500,012, 5]; int6 group-32 (`row_table.py:101-151`, dequant `dequant_int6_g32` `:109-148`, opener `:983-1023`). 38.74 GiB in 8 row-only files.
  - **Split layout** (`T/serve.sh:1342-1368`): the 8 row files are detected by tensor header (`_q4e_inline_ple`, `:1308-1322`) in `Q4E_PLE_DIR` (Optane XFS) and become `MINISGL_PLE_FILES`. Only taken when `Q4E_PLE_DIR != model_id` (`:1353`).
  - **Heads metadata** (`ngram_heads_offsets/vocab_sizes`) stays in the model dir; `MINISGL_PLE_META_FILES` names **only** the heads-bearing, row-free shards (`_q4e_ple_heads`, `:1326-1341`). `R/ple/source.py:49-52` reads it; missing heads raise `KeyError` (`:85-91`).
  - **Trap, last-writer-wins:** `index_safetensors` indexes `ple_files + meta_files` and the later name wins (`serve.sh:1356-1363`). Naming a row-bearing shard in META silently sends row reads back to the model-dir copy. Any harness (P0/P1, the solve) must name files exactly as `serve.sh` does.
  - **Inline fallback:** if the split is not detected, `serve.sh` serves the inline copy from the model dir (`:1369-1398`). `DC/:903-905` still calls `Q4E_PLE_DIR` "VESTIGIAL ... never reads /ple"; that comment predates the split branch and is stale.
  - `scale_override` (`source.py:74`; v1 cited `:75`) must be None for a packed table, otherwise `ValueError` (`row_table.py:1064-1070`).
  - The plefp8 sidecar text (`row_table.py:19-22, 236-238`) describes the RadixArk fp8 layout, not this deployment.
- **What one row is.** One head's 160-value slice. A token reads 16 rows and concatenates them into 2560 values (`source.py:96-100`).
- **I/O.**
  - Files are mmapped `PROT_READ`, `MAP_SHARED|MAP_NORESERVE`, `MADV_RANDOM` (`row_table.py:387-408`).
  - The gather goes `begin_rows` -> `gather_raw_into_async` (`source.py:213-229`, `row_table.py:675-722`). For the fused table that is **two** reads per row, codes and scales, each through its own persistent 16-worker `os.pread` pool (`row_table.py:860-878`; pool threshold 16 rows `:307`; workers=16 `source.py:73`).
  - io_uring (`R/weights/uring.py`, ctypes) is still never reached from the serve: it sits only in `gather_raw` at >= 256 rows (`row_table.py:570-576`), which the async path does not use.
- **No row cache in minisgl.** Residency is the OS page cache over Optane XFS (`T/ple_split.py:4-7`: 4.1 KiB fetched per 120 B row on XFS vs 1024 KiB on a 1M ZFS record). The ARC note at `row_table.py:290` applies to the inline layout only.
  - Reads are issued in `_finish_prepare` and awaited at `_finish_ple` (`R/scheduler/scheduler.py:2230-2259`).
  - Gather costs about 300 µs against a ~50 ms decode step (`source.py:197-211`).
- **Path to the GPU.**
  - Raw uint8 code and scale bytes land in `_raw`.
  - `FusedScaleRowTable.decode_raw` (`row_table.py:880-884`) -> `dequant_int6_g32` writes **fp32** (codes x fp16 scale, computed in fp32) into the pinned `_host` through its numpy view `_host_rows` (`source.py:114-117, 243`).
  - One non-blocking H2D copy into the static `_dev_f32` (`:244`), cast into the static bf16 `embeddings` (`:245`) (`source.py:110-134, 231-246`).
  - Every entry path, including `stage_rows`/`stage_batch` (`source.py:169-187, 248-252`), goes through `finish_rows`.
- **Where torch is on the PLE path** (everything else on the row path is numpy, mmap or ctypes): `source.py:41` (import), `:113-120` (pinned `_host`, `_dev_f32`), `:132-134` (`embeddings`), `:244-245` (H2D, cast); `runtime.py:33` (import), `:89` (`_slot_idx`), `:164-166` (slot H2D), `:343` (sizing signature), `:415` (multipliers `.cpu().numpy()`). Off the row path, `R/ple/state.py` also imports torch (`:28`) for the device `conv_state` (`:50-52`) and its clone/load/zero/verify-install (`:76, 84, 103, 131`); `token_history` is numpy (`:56-58`). `graph_capture.py` and `hashing.py` import no torch. Type annotations aside, nothing else.

### 2.3 Lookup, gate, conv, residual
- **Host side, every forward.**
  - `Scheduler._stage_ple` (`scheduler.py:3170-3251`) takes host tokens `input_ids[cached_len:device_len]` and resets the slot when `cached_len==0` (`~3226-3227`).
  - `PLERuntime.prepare_begin/finish` hashes `history ++ tokens`, gathers rows, fills `_slot_idx` (`runtime.py:128-177`).
  - **Loop selection** (`scheduler.py:1468-1515`): with EP enabled the scheduler enters `ep_loop` (`:1468-1474`, `R/scheduler/ep.py:99`) before any other branch. The deployment is EP=1, `EP_SPLIT=2:1` (`T/serve.sh:1155-1157`), which is EP-over-TP (`R/distributed/info.py:75, 94`; TP=2, DP=1). Otherwise PLE forces the synchronous loop unless `MINISGL_OVERLAP_SPLIT=1` selects the new `split_loop` (`scheduler.py:1212-1244, 1497-1501`; default 0, `DC/:251`).
- **GPU side** (`R/models/qwen4exp.py:421-443`):
  - `key = norm_key(key_proj(e))`, [2560->10240]; `value = value_proj(e)`; `query = norm_query(hidden_wide)`;
  - `gate = signed_sqrt(k·q/√H)`; `gated = sigmoid(gate)·value`;
  - `out = gated + silu(dilated depthwise conv(norm_conv(gated)))`;
  - added to the 10240-wide hyper-connection stream (`:543-544`) at decoder index 1 only (`:14, 522-528`).
  - Decode runs a static conv; prefill runs `F.conv1d` per sequence (`:374-419`). HF matches: `HF/:1236-1249`.

### 2.4 Decode state and spec decoding
- **State is indexed by the GDN slot; slot 0 is NULL** (`R/ple/state.py`): `conv_state` GPU [slots, 10240, 9] (`:50-52`); `token_history` host [slots, 2], EOS-seeded (`:56-58`).
- **History commit** in `commit_staged` after the forward (`runtime.py:220-260`, called at `scheduler.py:3402-3403`).
- **Spec verify.** Staged with `defer_commit` (`scheduler.py:5253-5261`); conv window stashed (`qwen4exp.py:410-411`) and rolled back (`runtime.py:262-299`, `state.py:106-131`). Verify goes through `_stage_ple`, so it sees the overlay.
- **Deployment:** the q4e arm defaults to no spec (`spec_default="none"`, `T/serve.sh:847`), so this is latent.
- **MTP** (if enabled): the head has no PLE (`qwen4exp.py:785-790`); per-slot draft KV (`R/spec/mtp.py:41-45, 147-151`) is reset per new request (`mtp.py:200-216`). Stale drafts cost acceptance, never correctness. The n-gram proposer is token-only (`R/spec/proposer.py:88-100`).

### 2.5 Graph capture (unchanged)
- **Capture** (`R/ple/graph_capture.py:60-101`) stages a synthetic NULL-slot batch and discards it.
- **Replay** (`:105-131`) stages nothing; `_check_addresses` (`:132-147`) asserts `embeddings`, `_slot_idx`, `conv_state` are at captured pointers.
- **Consequence:** anything written into `_host_rows` before the H2D in `finish_rows` is invisible to the graph and safe.
- **Inference mode.** Scheduler methods run under `@torch.inference_mode()` (`scheduler.py:1369, 3877, 4037`); capture uses it explicitly (`R/engine/graph.py:401-407`). The serve cannot produce gradients, and v2 does not try to make it.

---

## 3. Live edit application

### 3.1 Overlay design (revised: numpy-only, bf16 replacement rows)

**Hook.** `PLEEmbeddingSource.finish_rows`, directly after `R/ple/source.py:243` (`self._host_rows[:flat_n] = self.table.decode_raw(raw)`). The row ids come from `begin_rows` (`:227`: `flat = ids.reshape(-1)`), captured there.

**Torch-free, checked.** `v2-editpath/ple_overlay_np.py --block-torch`:
- imports the fork's `row_table` with `import torch` blocked (register the module in `sys.modules` before exec; its dataclasses need it);
- builds a small int6 checkpoint, opens it with the fork's `open_qwen4exp_ngram_table`, runs the same `begin/wait/decode` sequence as `source.py:229/241/243`;
- applies the overlay: "selftest ok: torch blocked=True; fused int6 table via fork opener; 7 hits; bf16-replacement == fp32-delta after device cast; untouched rows bit-identical" (`ple_overlay_np.txt`).

**Representation.**
- **On disk, per edit:** `edit_<id>.safetensors` with `ids` int64 [K] and `delta` f32 [K,160], plus `base_sha` of the base rows used. Edits compose (deltas sum) and `remove` is exact.
- **Compiled, in RAM:** one sorted `ids` int64 [U] + `rows` bf16-exact values [U,160], where `rows = bf16(fp32(base) + Σ delta)`, with `base` read through the fork's own `table.gather` (int6 decode, fp32). Stored as fp32 holding bf16-representable values, or as uint16 bf16 bits expanded on hit; the served value is `bf16(row)` either way. Published by one attribute assignment; `epoch` increments on every change.
- **Apply** = `np.searchsorted(ids, flat)` + masked assignment into `_host_rows`. Bit-identical to the fp32-delta path after the device cast (asserted in both scripts).
- **Why replacement rows instead of v1's fp32 deltas:** half the memory (328 B vs 648 B per row with ids) and the same served bits. The cost is that the compiled rows depend on the base file: if the row files are ever replaced, recompile (check `base_sha`).
- **Memory, per rank:** 63 MiB per 200k rows, 313 MiB per 1M rows. Each TP rank builds its own PLE runtime (`R/engine/engine.py:736-749`), so either keep one overlay per rank or put the compiled arrays in a read-only `np.memmap` on `/dev/shm` shared by both. Lock it with `mlock` via ctypes: swap is on the same Optane drive as the table.
- **Cost** (`bench_hook.txt`): 11 µs per decode call (128 rows), 1.1-1.8 ms per 2048-token chunk (32,768 rows). Realistic hit fractions are well under 1% (each edited row is one bucket of ~20M per head).

**Precision.** The served value is `bf16(base_int6_fp32 + delta)`. Deltas below ~2^-9 of an element vanish in that cast (a 0.3%-of-row-norm dense delta keeps ~98% but with error 0.52 in `v2-deltaverify/requant_survival.txt`; 0.69 in `v2-editpath/requant_delta.txt`, which the §3.5 table uses. These are two independent delta draws.). The solve must reproduce the cast (§5 P2). The base decode itself is changed by the cast on 92.9% of elements (median rel 1.4e-3), which the solve must also see.

**Lower-RAM fallback.** An overlay of raw int6 rows (138 B/row) that swaps bytes in `_raw` between `gather_wait` (`:241`) and `decode_raw` (`:243`): rescale-column precision (§3.5), no file writes. Only if RAM ever becomes the limit; it is not at ~20 GB free.

### 3.2 Patch plan (all diffs approximate; df29117 lines)
1. **New `R/ple/overlay.py`:** numpy + stdlib only (+ ctypes `mlock`). Loads `edit_*.safetensors` with a small header parser (or `row_table.index_safetensors`), compiles, applies.
2. **`R/ple/source.py`:**
   - `__init__` (~`:102`): `self.overlay = None; self._pending_ids = None`.
   - `begin_rows` (`:227-229`): after `flat = ids.reshape(-1)`, `self._pending_ids = flat`.
   - `finish_rows`, after `:243`: `if self.overlay is not None: self.overlay.apply(self._pending_ids[:flat_n], self._host_rows[:flat_n])`.
   - Nothing on the device changes; `_check_addresses` stays satisfied. No new torch.
3. **`R/ple/runtime.py`:**
   - In `build_ple_runtime` (`:420-428`), attach the overlay, loading persisted edits from `MINISGL_PLE_OVERLAY` (a directory).
   - `PLERuntime.apply_edit(edit_id, path)` / `remove_edit(edit_id)`: refuse while a forward is open (`self._begun` `:103`, `self.batch` `:90`, `self._pending` `:86`); return the affected row ids.
4. **Control plane, cloned from CAM (no new socket):**
   - `SamplingParams`: add `ple_op: str|None`, `ple_path: str|None` next to `mem_op` (`R/core.py:139`).
   - `Scheduler._process_one_msg` (`scheduler.py:1753-1770`): beside the CAM branch at `:1769`, `if self._ple is not None and getattr(sp, "ple_op", None): self._ple_direct_control(msg); return`.
   - `_ple_direct_control` modelled on `_cam_direct_control` (`scheduler.py:3109-3136`): answers with no forward; only `_tp_is_primary` replies.
   - **Both TP ranks get the op** via the rank-0 count broadcast (`R/scheduler/io.py:119-151`, unchanged). The deployment is EP-over-TP with DP=1, so this path applies.
   - **`ep_loop` rules (deployed).** Messages are handled synchronously at the top of each step with nothing in flight (`R/scheduler/ep.py:107-108`): safe. But the op must do identical work on both ranks and must never raise on one rank only (validate the file before touching state, catch and reply on error), and must be fast (load + compile ~ms-s): the lockstep check (`ep.py:60-79`) raises on step drift, and a slow op stalls the peer at that rendezvous.
   - **`split_loop` (EP=0, opt-in).** Messages are processed before the in-flight forward's `_process_last_data`, so defer the flush/invalidation until after that commit.
   - Ops: `put {edit_id, path}`, `remove {edit_id}`, `list` (ids, row counts, epoch, compiled hash per rank), `flush_prefix`, `invalidate {ids}`.
   - **Payload by file path, not over ZMQ.** `MINISGL_PLE_OVERLAY` needs its **own writable mount**: the model and table mounts are read-only (`DC/:910-911`).
   - **FastAPI router** `/ple/*` behind `MINISGL_PLE_EDIT=1`, mounted like CAM (`R/server/api_server.py:3615-3624`), sending a 1-token request carrying `ple_op` (`R/cam/runtime.py:206-220`). Neither env var exists yet in the fork.
   - With `--dp-size > 1` (not deployed), broadcast to every replica; CAM's pinning (`R/tokenizer/server.py:39-47`) is wrong for this.
5. **Prefix and stash invalidation** inside `_ple_direct_control`, after put/remove: §3.4.

### 3.3 Rollback
- `remove(edit_id)` recompiles without that edit. Rows return to `base + Σ(remaining)` bit-for-bit, because the base is re-read from disk every forward and replacement rows are recompiled from it.
- Invalidate the prefix cache for the removed edit's row ids exactly as for a put.
- `list` returns the per-rank compiled hash.
- Persisted edit files make "rollback to last known good" = delete files and restart (~300 s).

### 3.4 Prefix-cache and spec invalidation (unchanged logic, new lines)

**What goes stale.**
- Recurrent radix is the plan for q4e (`R/engine/config.py:95-123`, `gdn_radix=True` `R/scheduler/config.py:23`). A node holds KV pages for the full-attention layers plus optionally a `rec_state` snapshot: GDN conv+ssm for every GDN layer plus PLE `conv_state` and `token_history` (`scheduler.py:474-486`; `R/kvcache/composite_state.py:24-71`; `state.py:68-85`). Snapshots get 7 GiB of host frames (`T/serve.sh:862`), freed by refcount (`R/kvcache/host_arena.py:231, 268`).
- Under the deployed `ep_loop`, commits go through `_process_last_data` (`ep.py:164, 188`) -> `_stash_rec_state` / `_maybe_capture_rec_state` (`scheduler.py:1638, 1744`). The `--gdn-radix` help says "inert under spec-decode / EP" (`R/server/args.py:386-388`, likewise the field comment `R/scheduler/config.py:21-22`) but no EP check exists in the plan code (`R/engine/config.py:115-121` returns `recurrent_radix` on `gdn_radix` alone); confirm from the boot log (`scheduler.py:300-304`, open question 1).
- Every layer from index 1 on is downstream of the PLE output, so KV and GDN state both embed table contents.
- A node at boundary B is stale if any position p < B on its path reads an edited row id (row-id test; hash collisions and 8 moduli per order matter). Descendants of a stale node are stale.

**Why nulling `rec_state` is not enough** (code unchanged since v1).
- `match_prefix` caps reuse at the deepest node with a `rec_state` (`R/kvcache/radix_cache.py:199-219`).
- `insert_prefix` keeps the KV already in the tree and adds only the new tail (`:299-312`); the inserter's duplicate pages are freed (`R/scheduler/cache.py:116-120`).
- `_maybe_capture_rec_state` then attaches a post-edit snapshot to a node whose pre-edit KV remains (`scheduler.py:1928-1960`, `radix_cache.py:221-236`). The next match restores mixed state.
- `RadixPrefixCache.reset()` raises `NotImplementedError` (`radix_cache.py:362-363`).

**First experiment: drain and flush.**
1. Stop traffic, or have `/ple/put?drain=1` wait until nothing is running or queued.
2. Apply the overlay.
3. `flush_prefix`: with every `ref_count` 0, `evict(self.evictable_size)` frees every node (`radix_cache.py:314-360`; nulls `rec_state` at `:351`); prune `_rec_nodes`; clear `_pending_rec_snap` and `_rec_snap_ladder` (`scheduler.py:1072, 1081`; stash/ladder code `1898-1926`).
4. Assert the tree is empty (`evictable_size == 0`). Both ranks do the same.

**Steady state: targeted detach plus epoch guard.**
- **(a) Find stale nodes.** DFS from `root_node` carrying the EOS-seeded history; per node `ids = hasher.row_ids(heads, hist ++ node._key, len)` (numpy, `hashing.py:254-272`); `stale |= np.isin(ids, edited).any()`.
- **(b) Detach each stale subtree.** Remove it from `parent.children`; free every unreferenced node (pages via the cache manager, subtract from `evictable_size`, `_evicted=True`, `rec_state=None`). Referenced nodes stay as an orphan spine marked `_detached=True`; `unlock` still walks `node.parent` (`radix_cache.py:175-197`, `178-190`); guard `del parent.children[...]` at `:353` with `if not node._detached`. Prune `_rec_nodes`. ~40 lines plus `check_integrity` coverage.
- **(c) Epoch guard.** Stamp `req._ple_epoch` at admission. For old-epoch requests whose committed tokens hit edited ids: skip `_maybe_capture_rec_state` (`1928-1960`), ladder and `_stash_rec_state` (`1898-1926`), drop their stash (per-request clear `1870-1871`); in `free_and_cache` skip the non-finished insert and on finish take `cache_output=False` (`cache.py:108-121`); optionally abort or re-queue.
- **Spec decoding.** Nothing needed (§2.4); off by default for q4e anyway.

### 3.5 Hazards, and why not to write edits into the int6 files

**Do not:**
- apply while a forward is staged (`apply_edit` refuses);
- replace `embeddings`, `_dev_f32` or `_slot_idx` (pointer gate, `graph_capture.py:132-147`);
- raise on one EP rank only, or block for long inside the op (`ep.py:60-79`);
- let the overlay or the solve page out to swap (same Optane drive as the table).

**In-file requantisation: measured, not recommended.** `v2-editpath/requant_delta.py` / `.txt` (independently reproduced in `v2-deltaverify/requant_survival.txt`). 8,192 real bf16 rows; file base = RNE int6 with absmax/31 scales, decoded by the fork's `dequant_int6_g32` (pack/decode round trip asserted exact). The int6 step is ~0.59% of a row's L2 norm (median step 5.6e-4, median row norm 0.096). Everything ends with the serve's bf16 cast. Error = median |realised - intended| / |delta|.

| Dense delta / row norm | Overlay (bf16 rows) | int6, keep scales (rewrite 120 B) | int6, rescale (rewrite 130 B) | MXFP4, rescale (reference) |
|---|---|---|---|---|
| 0.3% | 0.69 (bf16 cast loss) | 1.00 (all lost) | 1.22 | 1.00 |
| 1% | 0.22 | 1.00 | 1.07 | 1.00 |
| 3% | 0.07 | 0.71 | 0.70 | 1.00 |
| 10% | 0.02 | 0.25 | 0.22 | 0.80 |
| 30% | 0.01 | 0.16 | 0.08 | 0.38 |
| 100% | 0.00 | 0.25 (clips at ±31 steps) | 0.03 | 0.16 |
| 300% | 0.00 | 0.54 | 0.02 | 0.12 |

- Sparse deltas (8 of 160 elements): rescale survives from ~1% (0.51 at 1%, 0.17 at 3%, ~0.05 beyond) but disturbs untouched elements in the same 32-group by 4-5% of |delta|.
- Earlier write experiments used ~5x row norm (`nm/t1/f_write.out`); that would survive at ~2% error, small solver steps would not.
- **Practical blockers too:** both mounts are read-only (`DC/:910-911`); codes and scales are two writes at different offsets, so a concurrent gather can read a torn row (new codes, old scales); the inline copy in the model dir (`T/ple_split.py:153-159`) would silently win if `serve.sh` falls back to INLINE (`:1369-1398`); the original 130 B must be saved for rollback; it still needs §3.4.
- **Verdict:** keep edits in the overlay. RAM is not the constraint (§3.1).

### 3.6 Overhead
- Hook (bf16 replacement rows, 200k edited rows; `v2-editpath/bench_hook.txt`): 11.2 µs per 128-row decode call, 1.8 ms per 32,768-row chunk; fp32-delta variant 10.9 µs / 1.1 ms. Versus the int6 dequant it follows: 61-67 µs / 44-46 ms on this sandbox.
- v1's hit-rate sweep (`integration/bench_overlay_hits.txt`: 7-24 µs decode, 1.5-17.5 ms per chunk from 0% to 100% hits) still bounds the worst case; the searchsorted cost is codec-independent.
- So the overlay is ~0.02% of a ~50 ms decode step and a few % of the dequant per prefill chunk.
- Control ops: recompile O(K log K) plus base-row gathers for touched rows (K x 130 B pread); the radix walk in §3.4(a) hashes every cached token once (tens of ms for a large tree). Keep them bounded because of the EP rendezvous.

---

## 4. Where the solve runs (rewritten)

### 4.1 Facts that constrain every option

**The checkpoint the solve must reproduce** (`v2-solve-local/ckpt_inventory.txt`; 23 headers, 110.6 GiB of files, 108.52 GiB payload).

| Tensors | On disk | Size | Fork reference |
|---|---|---|---|
| Routed experts | `experts.gate_up_proj_packed` U8 [512,1280,1280] + `_scale` U8 [512,1280,80]; `down_proj_packed` U8 [512,2560,320] + `_scale` [512,2560,20]. Pre-stacked, gate rows then up rows on N. MXFP4 (OCP microscaling): E2M1 codes, low nibble = even K index, one E8M0 scale per 32 values. 2,611,200 B per expert, 1.25 GiB per layer. | 59.77 GiB | `R/models/weight.py:1406-1424, 953-962`; `R/quant/mxfp4.py:1-26, 262-275` |
| PLE rows | int6 group-32 (§0), 130 B/row | 38.74 GiB | `row_table.py:101-151, 983-1023` |
| q/k/v/o, GDN `in_proj_qkv`/`in_proj_z`/`out_proj` | F8_E4M3, block [128,128], bf16 `weight_scale_inv` used as a multiplier; served W8A16 | 2.49 GiB | `R/quant/method.py:1060-1127` |
| Shared expert | MXFP4, dequantised to bf16 once at load | 0.11 GiB | `R/models/weight.py:2115-2130` |
| Hyper-connections, router, norms, A_log/dt_bias/conv, PLE key/value projections | BF16 | ~1.4 GiB | - |
| embed_tokens, lm_head | BF16 | 1.18 GiB each | - |
| Skipped | mtp, visual, q/k/v_scale, `expert_routing_counts` | - | `R/models/weight.py:1588-1619` |

**Serve numerics the solve differs from.** The MXFP4 experts run W4A8: per-token fp8 activations by default (`MINISGL_MOE_W4A16`, `MINISGL_MOE_MXFP4_REGDIRECT` default off, `R/quant/kernels.py:61, 72-77`; `mxfp4.py:1-10`, `method.py:729-733`). The q4e arm sets neither. A bf16-activation solve is therefore not bit-the-serve; the size of the gap is unmeasured for MXFP4 (v1's 4.09e-2 per layer, `R/weights/cpu_tier.py:38-40`, was NVFP4). Keep P3/P4 and optional fp8 fake-quant. The CPU expert tier is off (`T/serve.sh:708-711`).

**Memory.** Owner: the serve pins ~56 GB of 96 GB; **~20 GB free while serving**; swap plentiful but on the Optane drive with the table (and the serve has previously been pushed to 33.9 GiB of swap, `T/serve.sh:1116-1122`). So: the solve gets a hard cgroup cap below the free figure with swap 0, and the NVMe checkpoint (re-read, O_DIRECT) is the overflow, not swap.

**Disk.** Checkpoint pool reads (`R/weights/ckpt_read.py:60-63, 89-98, 546-550`): O_DIRECT 2,979-4,967 MiB/s; mmap 551-627 MiB/s and a page-cache page plus ARC buffer per 4 KiB; 3.3 MiB reads at QD1 1,225 MiB/s. The source gives these in MiB/s; v1/v2 had relabelled them GB/s, which understated them by ~5%. **The solve must not mmap the checkpoint**: O_DIRECT `preadv` into a reused aligned numpy buffer, whole per-layer scale tensors (78 MB) rather than 51-102 KB slices (each would pull a 1 MiB ZFS record). The serve does not read the checkpoint after boot; the solve's only Optane traffic is its few hundred base rows, read through `row_table` (a few KB, once per batch).

**Routing.** The checkpoint's own profile `expert_routing_counts` [48,512]: the top 64 experts per layer carry 43% of routing (median over layers), the top 128 carry 61%. Expected distinct experts per layer (independent tokens, an upper bound; `v2-solve-local/distinct_experts.txt`): 160 at 24 tokens, 285 at 64, 446 at 256, 500 at 1024.

**HF as-is still cannot be used** (`from_pretrained` concatenates the n-gram table on CPU, `conversion_mapping.py:2133-2139`; no MXFP4 qwen4_exp expert path). Every option uses the same "shared core": build on `meta`, our loader, an MXFP4 expert `autograd.Function`, a touched-rows module, the indexer bypass, a 2-D lm_head, and the Stage-1/Stage-2/verify loop.

**Autograd coverage** (`HF/`; re-run `v2-solve-venue/autograd_coverage.py`: on a tiny fp64 model the gradient flows only to the rows and matches finite differences at rel 4.7e-6; checkpointing gives an identical gradient; the MXFP4 module equals dense experts exactly):
- GDN: `torch_chunk_gated_delta_rule` (`HF/:254`) or `torch_recurrent_gated_delta_rule` (`HF/:391`) for suffixes <= 8 tokens; conv falls back to `F.conv1d` (`HF/:222`).
- Gated attention `HF/:811-895`. Its QSA indexer (`HF/:665-771`) selects every visible token when kv_len <= `indexer_budget` = 2048, so replace it with the plain visible mask: exact, and removes a per-query Python loop.
- MoE with sigmoid-gated shared expert `HF/:913-993`; hyper-connections `HF/:995-1031`; PLE `HF/:1072-1250`.
- The only custom backward is the MXFP4 expert linear (~30 lines): dequantise again in backward, compute only the input gradient, save nothing weight-sized.

**What the loader implements (ngram-memory code, not engine code):**
1. safetensors header index + O_DIRECT reader; expert e at `data_start + e*stride`.
2. Name remap `model.language_model.*` -> HF `model.*`.
3. MXFP4 dequant: 256-entry table of bf16 pairs + shift of the E8M0 byte into the exponent; **bit-exact against the fork's `mxfp4.dequant_reference`** (`mxfp4.py:262-275`; LUT `:37-38`, low nibble = even K `:44-52`, bias 127 `:41`). Assert scale exponents in range per tensor. The serve converts E8M0 to **fp16** per-group scales at load (`mxfp4.py:13-14, 24`), so "in range" means 2^(s-127) is exactly representable in fp16 (s-127 in [-24, 15]); outside that range the served value differs from the reference.
4. FP8-block dequant: e4m3 x scale per 128x128 tile (`method.py:1117-1121`).
5. bf16 tensors including `model-hc-bf16.safetensors`; embed_tokens row gather.
6. Rows via the fork's `row_table.open_qwen4exp_ngram_table` imported by path (register in `sys.modules`), with the **same file lists as `serve.sh`** (split row shards as PLE files, heads-only shards as META, `scale_override=None`); then `bf16(fp32(decode) + delta)` mirroring `source.py:243-245`.
7. Optional per-token fp8 fake-quant on expert inputs.

### 4.2 Options (iteration = forward + backward)

| Option | Feasible? | s / iteration | Per edit | Peak RAM | Effect on the serve | Engineering |
|---|---|---|---|---|---|---|
| **(A) Local CPU, concurrent with the serve** (v1's "b", redesigned) | **Yes**, batched and non-interactive | 1 edit x 8 prompts: 54-115; 8 edits: 147-315; 32 edits x 4 prompts: 205-439 | 19-40 min single; 6-14 min at K=8; **2.2-4.8 min at K=32** (1.2-2.6 h per batch) | 11-15 GiB up to ~64 sequences, `MemoryMax=15G`, `MemorySwapMax=0`. 32 x 4 (128 sequences) needs NVMe-spilled prefix state, see below | 4 of 8 physical cores pinned; DDR < 1 GB/s; NVMe only; no Optane, no GPU; est. 0-10% decode slowdown (to measure) | Shared core 4-6 d + 2-3 d (prefix-cache driver, prefetch); optional own AVX-512 MXFP4 GEMV 3-5 d |
| **(B) Local GPUs during downtime** (v1's "f") | Yes | 3-10 (2-6 s PCIe at 28.93 / 14.48 GB/s for card 0 / 1, `cpu_tier.py:6`, + 1-4 s compute) | ~1-2 min single; seconds at K=32; plus 10-15 min downtime per session | ~70 GiB host (serve down; 59.8 GiB compact experts fit once the 55.9 GiB pinned arena is gone); 5-9 GiB VRAM per card | Serve offline (~300 s reboot) | Shared core + 2-3 d |
| **(C) Cloud 1x H200, MXFP4 body resident** (v1's "d2", now without bf16 conversion) | Yes | 1-5 | ~0.5-1 min single; 1-3 s at K=32; plus ~10 min setup | ~66 GiB HBM resident (experts 59.8 + compact dense 4.0 + lm_head/embed 2.4). On an H100 80 GB (74.5 GiB) that leaves ~8 GiB for activations and context: fits, but tight | None; prompts leave the machine | Shared core + 1-2 d |
| (C') Cloud, bf16 base | No (~360 GB, and not the served weights) | - | - | - | - | - |
| (c) Backward inside minisgl | Rejected | - | - | - | - | Would add torch autograd to the engine (against the design constraint) and pause serving |
| (e2) Exact forward, skip routed-expert Jacobian | Optional | halves expert backward | - | - | - | 1 d; measure its cosine against the exact gradient first |
| (e3) 1-3 exact gradients + forward-only acceptance checks through the serve's overlay | **Always** | - | - | none | short requests | 2-3 d; log routing overlap between iterations |
| (e1) Truncated backward / (e4) zero-order | Controls only | - | - | - | - | - |

**How (A) gets to these numbers.**
- **Prefix caching (the main new trick).** Positions before the anchor cannot depend on the delta (causal; the delta enters at the anchor). Run the prefix once without gradients and cache KV, GDN conv/recurrent state and PLE history; each iteration runs only the anchor-to-target suffix (~3 tokens per prompt). HF supports multi-token continuation (`HF/:529-555, 1127-1147`) but updates caches in place, so the driver clones prefix state per iteration and does its own per-layer checkpointing (HF's checkpointing disables the cache).
- **Experts.** Per layer, read only the routed experts' compact bytes with 4-8 O_DIRECT reader threads; backward knows its expert set and prefetches.
- **Measured on this sandbox** (4-vCPU Sapphire Rapids with AMX; owner has Zen 4, projected x0.7-1.5, a guess): dequant 2.26 ms per expert on 4 threads (6.7 ms on 1; the fork's reference 24 ms); expert matmuls fwd+input-grad 0.65-1.2 ms; a real-width 4-layer HF model (3 GDN + 1 attention): 5.0-5.5 ms per touched expert, 0.30 / 0.41 / 0.90 s per layer for the rest at 32 / 64 / 256 tokens (`bench_dequant2.T*.txt`, `bench_solve_cpu.T*.txt`, `realdim_layers.T4.txt`, `cost_model.txt`). Dequant is ~90% of the expert cost; an own AVX-512 kernel would cut that ~10x (the fork's VNNI core shows 27-35 GB/s per core, `cpu_tier.py:14-20`), but it must be ngram-memory's own code, not `T/cpu_moe`.
- **Peak RAM 11-15 GiB (up to ~64 sequences):** dense weights kept compact with fp8 dequant per use 4.0 GiB (4.3 GB; `ckpt_inventory.txt` 4.035 GiB; 14 ms per 10240x2560 matrix, ~4 s/iteration); lm_head 1.2 GiB; double-buffered compact experts for one layer <= 2.7 GiB; checkpointed activations ~1-1.5 GiB; Python/torch ~1.5 GiB. That is a fixed ~10.4-10.9 GiB before the per-sequence state.
  - **GDN prefix state is 54 MiB per sequence only if stored as bf16** (36x48x128x128). HF keeps the recurrent state in **fp32**: values are cast to fp32 (`HF/:291-294`), the state is created with their dtype (`:360-362`) and cached via `zeros_like` (`transformers/cache_utils.py:1186`). That is 108 MiB per sequence. The driver must cast cached prefix states to bf16. This also matches the serve, which stores the GDN ssm state as bf16 by default (`R/engine/engine.py:652-659`; `MINISGL_SSM_BF16=1`, `DC/:315`).
  - At bf16: 0.4 GiB at 8 sequences (11 GiB peak), 3.4 GiB at 64 (13.8-14.3 GiB), 6.8 GiB at 128 (17.2-17.7 GiB, **over the 15G cap**). So **~64 sequences per batch is the in-RAM limit**, not 128.
  - **Larger batches.** Spill the prefix states to a file on the checkpoint NVMe (never swap, never Optane) and re-read them per iteration with O_DIRECT: 6.8 GiB at ~3 GB/s is ~2.4 s, ~1% of a 205-439 s iteration. Raising `MemoryMax` to ~18G would leave only ~2 GB of the ~20 GB free.
- **Trap found by OOM:** lm_head on a 3-D hidden tensor makes CPU autograd materialise a broadcast 1.2 GiB weight copy per sequence in backward (10 GiB at batch 8). Compute logits on 2-D target rows only (fixed in `realdim_layers.py`).
- **Serve impact (estimate):** the serve measured 1.88 cores + ~0.5 OS (`cpu_tier.py:313-335`); a dequantised expert (9.8 MB) stays in the 96 MB V-cache against a 57 GB/s DDR ceiling (`cpu_tier.py:420`); O_DIRECT bypasses the ARC; Optane and GPUs untouched. Expect 0-10% decode, mostly L3/DDR co-residency. Must be A/B'd.

**Iterations per edit.** Stage 1 in the audited EngramEdit code is 20 Adam steps at lr 0.08 (`memory-organ/tools/engram_audit/gemini/engram_graft.py:211-212, 233-235`); Stage 2 is a closed-form ridge solve (`:270+`); verification ~2 forward-only passes (forward ~55% of fwd+bwd). Total ~21-25 iteration-equivalents per batch. **Every per-edit figure in this plan uses 21** (`v2-solve-local/cost_model.py`, `STEPS = 20 + 1`). At 25, multiply by 1.19: a single edit is 22-48 min, not 19-40. The "A s/it" figures also exclude the 13-39 s/iteration of O_DIRECT expert reads (the `A IO s` column of `cost_model.txt`). They assume those reads are fully overlapped by the prefetch threads; at roughly 0.3-0.9 s of IO against 1-9 s of compute per layer, that is plausible but unmeasured. Twenty is that code's default, not a measured convergence count: log NLL per step in the first run.

**Batching shares iterations exactly:** per-edit deltas, summed losses, per-coordinate Adam, per-delta norm clamp, no cross-sequence interaction (right padding). K edits cost one iteration at K x tokens; the expert term saturates at ~500 per layer. Edits whose n-grams share rows need a joint Stage 2.

**Dropped from v1:** the NVFP4 figures (2,764,800 B per expert, `weight_scale_2`, 84 GB body, 34-68 GB per pass); "do not keep the dense core resident" (it is 4.0 GiB compact now); the bf16-conversion-on-H200 design; the 11.1 GiB headroom; 618 s boot; the fork's differentiable GDN/ROCm path and `T/cpu_moe` as solve dependencies (engine code paths; not needed).

### 4.3 Recommendation
1. **First native experiment: (A) on the serve host.**
   - One edit, 8 paraphrases plus controls, prefix-cached, per-layer checkpointing; `systemd-run --scope -p MemoryMax=15G -p MemorySwapMax=0`, 4 cores pinned away from the serve's.
   - Before the run: (i) our loader bit-for-bit against the fork's decoders (`mxfp4.dequant_reference`, `dequant_int6_g32`, fp8-block) on real shards; (ii) P0-P3.
   - ~1-2 min per iteration, ~20-40 min per run; no downtime, no cloud, the exact served weights.
   - Run the serve tokens/s A/B with and without the solve in the same session.
   - Ship the delta file to `MINISGL_PLE_OVERLAY`, drain-and-flush, apply, run §5 P4-P6.
   - For faster lr/step sweeps: (C) if cloud is acceptable, else one (B) downtime session; same code, different device placement.
2. **Steady state.** Overlay + targeted invalidation in the serve; (A) as nightly batches. In RAM: <= 64 sequences, e.g. 16 x 4 (N=192 suffix tokens, 147-315 s/iteration), 3.2-6.9 min per edit, ~70-150 edits in 8 h. With prefix states spilled to NVMe: 32 x 4, 2.2-4.8 min per edit, ~100-200 per night. (16 x 8 is also 128 sequences and costs the same per iteration as 32 x 4, so it comes to 4.5-9.6 min per edit, not 2-5.) Pair them with (e3) forward-only acceptance through the overlay; (B) for urgent large batches; the AVX-512 kernel only if throughput falls short.

---

## 5. Parity: solve-time numerics vs serve-time numerics

Run in order; each gates the next.

- **P0. Row ids.** Re-run `hash_parity.py` with the deployment's `config.json` and the heads-only META shards (seed, EOS id, `ngram_heads_offsets`, `ngram_heads_vocab_sizes`). Hash through the fork's numpy `hashing.py`/`row_table.py` imported by path. Pass: identical ids on >= 10k positions including EOS edge cases.
- **P1. Base row values.** The solve's base rows are `FusedScaleRowTable` decodes (`row_table.py:880-884`) from the **same split files** with the heads-only META list and `scale_override=None`. Check: fp32 bytes identical between the solve harness and a CPU instance of `PLEEmbeddingSource` on the same ids (no torch needed for the decode itself). Also confirm the split copy and the inline copy are byte-identical (or that the serve log says SPLIT, `serve.sh:1364-1366`), since the fallback would serve the other copy.
- **P2. Edit transport.** The solve emits `delta = row_solved - base` (fp32 [K,160]). The serve computes `bf16(fp32(base) + delta)` (`source.py:243-245`); with compiled replacement rows the served bits are the same (asserted). The solve forward must use exactly that, with a straight-through bf16 cast. Check bitwise on CPU: edit file -> compile -> `apply` on the batch's ids -> bf16 equals the solve's embedding tensor. Report how many delta elements fall below bf16 resolution.
- **P3. Pre-edit model agreement.** Per-target logprob, NLL and top-1 between the solve model and the serve on edit prompts, paraphrases, controls. The serve exposes logprobs for generated tokens only (top-N <= 20, plain decode path; `R/core.py:30-35`; `api_server.py:415-420, 1540-1544`), so probe with one `max_tokens=1`, temperature-0, `top_logprobs=20` request per prefix (spec is off by default for q4e). Expect a gap from the W4A8 fp8 activations (unmeasured for MXFP4). Report Pearson and mean |ΔNLL| over >= 200 positions and top-1 agreement; if top-1 agreement < ~95%, add fp8 fake-quant on expert inputs and repeat.
- **P4. Edit-effect transfer.** Predicted ΔNLL (solve) vs realised ΔNLL (serve + overlay, after flush) on targets, paraphrases, and controls sharing edited row ids. Pass: realised/predicted >= ~0.7 on targets; controls within P3 noise. Below ~0.5: fp8 emulation, or (e3) rescaling.
- **P5. Cache hygiene.** Warm the radix with the edit prompts before the edit; apply; query again; must equal a cold run. Repeat for `remove`. Run once with drain-and-flush and once with targeted detach.
- **P6. Rank consistency.** `/ple/list` returns the same compiled hash and epoch from both TP ranks; no EP drift error in the log across put/remove/flush.

---

## 6. Open questions for the owner

Answered since v1 and dropped: headroom (~20 GB free while serving); swap (plentiful, on the Optane drive with the table); expert checkpoint (tcclaviger MXFP4 repack); TP/DP/EP (TP=2 EP-over-TP, DP=1, from `serve.sh` defaults); CPU-MoE tier (off).

1. **Prefix cache under EP.** Does the boot log say "using recurrent radix prefix cache" (`scheduler.py:300-304`)? The plan code enables it for q4e and `ep_loop` captures snapshots, but `--gdn-radix` help says "inert under ... EP" (`R/server/args.py:386-388`). Either way the invalidation work is required if it is on; if it is off, drain-and-flush degenerates to a no-op.
2. **Row codec confirmation.** The brief says the PLE rows are MXFP4; the code can only serve int6 group-32 from this checkpoint (U8 scales raise). Please confirm the boot line "PLE n-gram: SPLIT layout (8 row shards from ..., heads from ...)" (`serve.sh:1364-1366`) and that no local, unpushed MXFP4 PLE codec exists.
3. **Local deployment vs df29117.** Is the running serve at df29117 or newer (anything touching `R/ple/source.py`, `graph_capture.py`, `radix_cache.py`/`scheduler/cache.py`, `_process_one_msg`, `ep.py`)?
4. **"~20 GB free":** is that `MemAvailable` (includes reclaimable page cache over the Optane rows and the ZFS ARC) or `MemFree`? It sets the solve's `MemoryMax` (planned 15G) and whether the overlay should be `mlock`ed.
5. **Serve-host CPU.** Exact core count and current pinning (the solve plan assumes 8 physical Zen 4 cores, 4 for the solve); is a 0-10% decode slowdown during overnight batches acceptable?
6. **Solve venue.** Is (A) on the serve host acceptable as the default? Is a cloud GPU session (C) ever acceptable (prompts leave the machine)? Or a scheduled ~10-15 min downtime (B)?
7. **Traffic.** Single-user, so drain-and-flush per edit is acceptable? Or targeted invalidation plus an abort/re-run policy for running requests?
8. **Persistence.** Which writable path for `MINISGL_PLE_OVERLAY` (both current mounts are read-only)? Persist edits across restarts by default? One overlay per rank, or a shared `/dev/shm` memmap?
9. **Prompt logprobs.** Any newer local scoring endpoint that would avoid one teacher-forced request per target token in P3/P4?
10. **Checkpoint integrity.** May we sha256 the local shards against the HF tree, so the solve's loader is known to read the same bytes the serve does?

### Files
- This plan: `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/INTEGRATION.md`
- v2 edit path (line map, torch-free overlay prototype, hook bench, requantisation): `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/v2-editpath/` (`claims_map.py/.txt`, `ple_overlay_np.py/.txt`, `bench_hook.py/.txt`, `requant_delta.py/.txt`)
- v2 independent requantisation check: `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/v2-deltaverify/requant_survival.py` and `.txt`
- v2 local solve (inventory, routing, CPU benches, cost model): `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/v2-solve-local/`
- Autograd coverage: `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/v2-solve-venue/autograd_coverage.py`
- v1 artifacts still valid: `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/ple_path/hash_parity.py`, `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/integration/bench_overlay_hits.py`

---

## Verification log (skeptic pass, df29117)

**Checked and found correct** (read at df29117 with `nl -ba`):
- Codec and layout:
  - `row_table.py`: `:37-47` (stdlib+numpy imports only); `:1001-1003` F16 -> INT6_G32; `:1004-1010` U8 raise; `:828-834`; `:1048-1053` fp8 regex; `:1059-1062` packed regex; `:1064-1070` `scale_override` ValueError; `:1077-1079` early return; `:1088-1102` global fp8 scale; `:860-878` two-tensor gather; `:880-884` `decode_raw`; `:307` threshold 16; `:570-576` io_uring only in `gather_raw`; `:936-954` offsets prefix sum and `off + mod`.
  - `hashing.py:44-49` (numpy only) and `:254-272`.
- `source.py`:
  - `:41`, `:49-52`, `:73-74`, `:85-91`, `:96-100`, `:113-120`, `:117` (`_host_rows` is a numpy view), `:132-134`, `:197-211`, `:227-229`, `:241-246`.
- `runtime.py`:
  - `:33`, `:86`, `:89`, `:90`, `:103`, `:128-177`, `:164-166`, `:220-260`, `:343`, `:407-428`, `:415`.
- Scheduler, EP and config:
  - `scheduler.py`: `:300-304`, `:1468-1474`, `:1497-1502`, `:1212`, `:1753-1771`, `:3109`, `:3170`, `:3226-3227`, `:3402-3403`, `:1638`, `:1744`, `:1072`, `:1081`, `:1870-1871`, `:1898`, `:1928-1960`, `:474-476`, `:5253-5261`, `@torch.inference_mode()` at `:1369, 3877, 4037`.
  - `ep.py`: `:60-79` (SUM drift check), `:99`, `:107-108`, `:164`, `:188`.
  - `scheduler/config.py:23`; `engine/config.py:95-123`; `args.py:386-388`; `core.py:30-35, 139`.
- Radix and cache:
  - `radix_cache.py`: `:175-197`, `:199-219`, `:221`, `:299`, `:314`, `:351`, `:362-363`.
  - `scheduler/cache.py:108-121`; `graph_capture.py:132-147`; `engine/engine.py:736-749`.
- Server, model and weights:
  - `api_server.py`: `:415-420`, `:1540-1544`, `:3615-3624`.
  - `qwen4exp.py`: `:408-412`, `:421`, `:543-544`, `:785-790`.
  - `models/config.py:969-975, 1428-1438`; `distributed/info.py:75`.
  - `quant/kernels.py:61, 72-77`; `quant/method.py:729-733, 1060, 1117-1121`; `mxfp4.py:262-275`.
  - `models/weight.py`: `:953-962`, `:1406-1424`, `:2115-2130`, `:1588-1619`.
  - `ckpt_read.py`: `:5-10`, `:60-63`, `:89-98`, `:531-536`, `:546-550`.
  - `cpu_tier.py`: `:5-6`, `:38-40`, `:420`.
- Tools, compose and HF:
  - `serve.sh`: `:708-711`, `:847`, `:862`, `:1015-1017`, `:1116-1122`, `:1155-1157`, `:1302-1368`, `:1376-1400`.
  - `ple_split.py:4-7, 153-159`; `DC/:251, 903-911`.
  - `HF/`: `:222`, `:665`, `:1262`, `:1466-1470`.
- `git diff b98eb99 df29117` is empty for `R/ple/*`, `row_table.py`, `uring.py`, `radix_cache.py`, `scheduler/cache.py`, `scheduler/io.py` and `composite_state.py`. The diff stats are scheduler +293/-45 and api_server +127/-32; there are 293 commits.
- Arithmetic re-done:
  - Rows: 320,001,536 x 130 B = 38.74 GiB; MXFP4 85 B = 25.33 GiB; 2,500,012 rows per tensor; 160 x 6 bit = 120 B.
  - Experts: 2,611,200 B per expert, 1.245 GiB per layer, 59.77 GiB in total; scale tensors 78.6 MB per layer, 51/102 KB slices.
  - Overlay: 328/648/138 B per row; 62.6 MiB per 200k rows, 312.8 MiB per 1M rows. Hook cost 11 us / 50 ms = 0.02%; 1.1-1.8 ms / 45 ms = 2-4%.
  - int6 step 5.63e-4 / 0.0961 = 0.59%. §3.5 table matches `requant_delta.txt` cell for cell.
  - Per-edit times reproduce from `cost_model.py` with 21 steps. Distinct-expert counts match. Dense fp8 dequant is ~102 matrix-equivalents x 14 ms x 3 passes, about 4.3 s.

**Changed:**
1. `R/weights/weight.py` -> `R/models/weight.py` (the file does not exist under `weights/`), 4 places. Skip range `:1590` -> `:1588`.
2. `T/serve.sh:1371-1373` -> `:1369-1372` (the "int6 group-32" comment).
3. `radix_cache.py` `del parent.children` is at `:353`, not `:352`.
4. Torch inventory (§2.2): added `R/ple/state.py` (`:28`, device `conv_state` `:50-52, 76, 84, 103, 131`). Noted that `graph_capture.py` and `hashing.py` are torch-free.
5. §3.4: the stale "auto-disabled under EP" text is also in `scheduler/config.py:21-22`. `engine/config.py:115-121` has no EP check.
6. **GDN prefix state and the batch limit (§4.2, real defect).**
   - HF caches the recurrent state in fp32 (108 MiB per sequence). The plan's 54 MiB holds only if the driver stores bf16, which also matches the serve's bf16 ssm state (`engine.py:652-659`, `DC/:315`).
   - Fixed RAM is ~10.4-10.9 GiB, so 128 sequences need 17.2-17.7 GiB, which exceeds `MemoryMax=15G`. The in-RAM limit is ~64 sequences, not 128.
   - The steady-state batches 32 x 4 and 16 x 8 now require NVMe-spilled prefix state (~2.4 s per iteration).
7. **16 x 8 per-edit cost.** It is 128 sequences with the same N=384 as 32 x 4, so 4.5-9.6 min per edit, not "~2-5". The in-RAM alternative 16 x 4 is 3.2-6.9 min per edit, ~70-150 edits per night. Updated in §0, §1, §4.2 and §4.3.
8. Iteration count: per-edit figures assume 21 iteration-equivalents, while the text says 21-25. Added the x1.19 note (a single edit is 22-48 min at 25). Stated that the A s/it figures exclude the 13-39 s of expert IO per iteration and assume it is overlapped.
9. Dense compact size is 4.0 GiB (4.035 GiB in the inventory), not 4.3 GiB (that was GB).
10. Disk rates: the source is in MiB/s, not GB/s.
11. (C) H100 sizing: replaced the unsourced "~62 GiB" with the arithmetic (66 GiB resident vs 74.5 GiB on an H100).
12. "The solve never touches Optane" -> its only Optane traffic is its few KB of base rows.
13. §3.1: reconciled 0.52 vs 0.69 (two independent delta draws).
14. Loader item 3: the serve turns E8M0 into fp16 group scales (`mxfp4.py:13-14, 24`), so the range assert must use fp16 range.

**Recommendation check.** The ordering still follows from the evidence and is unchanged: overlay plus drain-and-flush first, (A) local solve as the first experiment, no in-file requantisation, no engine-side autograd. The first experiment (1 edit x 8 prompts, ~11 GiB) is unaffected by change 6. Only the steady-state batch shape changes.
