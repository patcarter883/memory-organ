# ngram-memory — design

Status: founding design, 2026-11-09, revised the same day after three red-team reviews (correctness, science, feasibility; see the **Review log** at the end). Nothing here is implemented yet. Every number below cites the script that produced it or a code location. Anything not measured is marked **(unverified)** or **(est.)**.

Conventions used throughout:

- **`P/`** = `/home/user/memory-organ/ngram-memory/proofs/`. All proof scripts and their outputs are copied there from the session scratchpad, in the same sub-folders (`.py` plus their `.out`/`.txt`/`.log`/`.json` outputs, and `serve/INTEGRATION.md`).
  - The copies are kept for provenance. Most scripts still hard-code the scratchpad paths they ran from, and none of the binary fixtures (`t1/real/*.pt`, `t2-gemma4-ple/hf/real_small.pt`, tokenizer files) were copied. M0 makes them re-runnable (§8).
  - `P/redteam-correctness/` and `P/redteam-feasibility/` hold the reviewers' re-runs and new checks. `P/review-apply/` holds the budget script written for this revision.
- **Code citations** are relative to the installed transformers 5.22.2 (`…/site-packages/transformers/`):
  - `q4:` = `models/qwen4_exp/modeling_qwen4_exp.py`, `q4c:` = `models/qwen4_exp/configuration_qwen4_exp.py`;
  - `g4:` = `models/gemma4/modeling_gemma4.py`;
  - `q35:` / `q35moe:` = `models/qwen3_5/modeling_qwen3_5.py` / `models/qwen3_5_moe/modeling_qwen3_5_moe.py`;
  - `cu:` = `cache_utils.py`, `cm:` = `conversion_mapping.py`, `cml:` = `core_model_loading.py`, `tp:` = `distributed/tensor_parallel.py`.
- **`R/`** = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/`, and `R/../../tools/` for its tools.
  - Citations are against local **HEAD `df29117`** (2026-11-06). That commit is 293 commits after `b98eb99`, which is an ancestor.
  - `R/ple/`, `R/kvcache/radix_cache.py` and `R/weights/config.py` are unchanged between the two commits (`git diff --stat` is empty). `scheduler.py`, `weights/cpu_tier.py` and `tools/serve.sh` have changed, and are cited at HEAD.
  - This fork is our *best guess* for the owner's Flash-Next serve. The owner has not confirmed it.
- **Metric names** follow memory-organ's conventions: validity gate, efficacy, generality, locality. Nothing else from memory-organ (its `cam/` package or its framing) is reused.

---

## 1. Thesis and scope

### Thesis

Language models have started to ship **native n-gram lookup tables**: large embedding tables, keyed on the recent tokens, whose output is added to the residual stream near the bottom of the network.

- **Qwen3.8-Flash-Next** has an Engram-style "PLE" module. It is 51.2 G parameters, 102.4 GB in bf16, and is keyed by hashed 2- and 3-grams (`P/compute/count.out`).
- **Gemma 4 E4B** has Per-Layer Embeddings. It is 2.82 G parameters and is keyed by token id.

In such a model a fact can be added by **writing rows**. An edit is a small, auditable, reversible diff to a named tensor. It is not a weight-space edit spread through MLPs.

The durable asset is therefore a **portable table editor**: one key/row addressing scheme, one solver, one edit-record and patch format, and one evaluation harness. Each target model plugs in through a thin backend.

Models without a table get a **graft**: a module that matches the native format **in keys and module math**, zero-initialised so it is an exact no-op, after which the same editor writes into it.

- The graft differs from the native format in three places: it has its own config schema (§2.4), its row width depends on the host's hidden size, and it can add unigram heads.
- "The same editor gives the same results on native and grafted tables" is a hypothesis. M3 tests it directly with a controlled comparison on Flash-Next (§6, §8). It is not an assumption of the design.

### Targets

| Tier | Model | Table | Key | Source |
|---|---|---|---|---|
| Native | Qwen/Qwen3.8-Flash-Next (`qwen4_exp`; 48 layers; d=2560; 512 experts top-10; GDN hybrid; 360 GB) | `layers.1.ple`: 16 heads × ~20M prime-sized rows × 160 | hashed 2-grams (8 heads) and 3-grams (8 heads) | §2 |
| Native | google/gemma-4-E4B(-it) | `embed_tokens_per_layer` [262144, 42·256] | token id, per layer | §4.2 |
| Graft | Qwen/Qwen3.8-27B (`qwen3_5`, 64 layers, d=5120) | Flash-Next format, sparse, row width 320 | **same keys as Flash-Next** | §4.3 |
| Graft | Qwen/Qwen3.6-35B-A3B (`qwen3_5_moe`, 40 layers, d=2048) | Flash-Next format, sparse, row width 128 | **same keys as Flash-Next** | §4.3 |
| Graft | google/gemma-4-26B-A4B-it, gemma-4-31B-it (`gemma4`, PLE disabled) | Flash-Next format with a Gemma key instance (primary, row width 176 / 336), or Gemma-PLE format with alias ids (alternative) | own tokenizer | §4.4 |

### Out of scope

Training any weights other than table rows (a graft's fixed init is part of the format); retrieval or in-context memory; multimodal keys (text `input_ids` only); weight-space editors except as baselines; memory-organ's `cam/` code; building engines (we define a patch interface and write small overlay adapters).

---

## 2. The format spec — the contract shared by native tables, grafts and the editor

Derived from `q4` and checked byte-for-byte against the real checkpoint's hash buffers (`P/t1/fetch_buffers.py`: `MATCH`, `sizes match: True`, `offsets match: True`). minisgl's independent `R/ple/hashing.py` is bit-identical to HF on 8,112 real-parameter ids, including chunked prefill and EOS edge cases (`P/serve/ple_path/hash_parity.py`).

### 2.1 KeySpec (decides row ids)

Parameters: `(V, eos, seed, ple_layer_index, ngram_size, heads_per_ngram, base, orders)`. Flash-Next's values are `(248320, 248044, 1234, 0, 3, 8, 20_000_000, {2,3})`.

1. **Multipliers.** For `i = 0..ngram_size−1`:
   - `m_i = 2·(splitmix64(seed + 10007·ple_layer_index + GAMMA·(i+1)) mod ⌊⌊(2^63−1)/V⌋/2⌋) + 1` (q4:1033-1050).
   - Flash-Next's values are `[23703573157769, 20109073645365, 8052911324071]`.
   - The bound keeps `x·m < 2^63`, so int64 products never overflow.
2. **Heads.** There are `(ngram_size−1)·heads_per_ngram = 16` heads (q4:1079): heads 0–7 hash the 2-gram and heads 8–15 hash the 3-gram. **There is no unigram head**; the loop starts at n=2 (q4:1154).
3. **Head sizes.** Head h has `p_h` = the k-th prime after `base−1`, with `k = 16·ple_layer_index + h + 1` (q4:1091-1096). This gives 20000003 … 20000171. Offsets `off_h` are prefix sums. The total is 320,001,446 logical rows.
4. **History.** `x_k` is the token k positions back within the current EOS segment. Wherever that crosses an EOS or the start of the sequence, `x_k := eos` (`_shift_right_ignore_eos`, q4:1107-1123).
   - Only `eos_token_id = 248044` (`<|endoftext|>`) resets.
   - `<|im_end|>` (248046) does not reset, so n-grams span chat turns.
   - **Padding.** When the recurrent attention mask exists, positions it masks out are replaced by eos before hashing (q4:1465-1470). The token's value plays no part, so a pad-id token under mask=1 is hashed as it is.
5. **Mixing.**
   - `mix2 = (x0·m0) XOR (x1·m1)`, `mix3 = mix2 XOR (x2·m2)`.
   - `row_h = off_h + (mix_n mod p_h)` (q4:1154-1166).
6. **Second PLE layer.** A second PLE layer would use `ple_layer_index = 1`, which gives new multipliers and the next 16 primes.

The fingerprint of a KeySpec is `sha256(V, eos, seed, ple_layer_index, multipliers, head sizes, orders and head order)` plus the tokenizer.json sha256. Every edit record and every patch carries it.

### 2.1b TableSpec (decides physical layout, not row ids)

Fields: `tensor_name`, `row_width`, `pad_divisor`, `n_shards`, `shard_name_template`, `base_codec`.

- **Padding.** The logical row count is padded up to a multiple of `pad_divisor`, which comes from `make_ngram_vocab_size_divisible_by` (q4:1103-1104; default 128, q4c:154). Flash-Next: 320,001,536 rows.
- **Shards.** On disk the table is split into `n_shards` equal row shards, `n_shards = split_ngram_parts` (cm:2136 `num_shards_attribute="split_ngram_parts"`; config default **512**, q4c:156; Flash-Next ships 128). Each Flash-Next shard is `…ngram_embedding.shard_{s}.weight`, [2,500,012, 160] bf16, in files model-00005…00037.
  - The two fields happen to be equal on Flash-Next. They are kept separate because a model that ships the config defaults would get the wrong addresses otherwise.
- **Row width.** `row_width = ple_embed_dim / n_heads` (q4:1083). Flash-Next: 160. Grafts: d/16 per host (§2.4). Payload widths differ by host, so `row_width` is part of the TableSpec fingerprint.
- HF concatenates the shards in numeric order at load (cm:2131-2140; cml:1319-1337, 1701) and re-chunks them on save (cml:112-143).
- **Address of a row:** `rows_per_shard = padded_rows / n_shards`; `shard, local = divmod(row, rows_per_shard)`; byte offset `8 + header_len + data_offsets[0] + local·row_width·2` (`P/t1/a_rows_for.out`).

### 2.2 ModuleSpec (q4:1194-1249; placement q4:1262, 1277-1280)

Notation: d = hidden size; s = hyper-connection stream (`hc_count`, 4 on Flash-Next); `N_*` = RMSNorm with weight `(1+w)` and eps 1e-6 (q4:143-163).

```
e_t   = concat_h  T[row_h(t)]                        (16·row_dim)
v     = W_v e_t                                      one value shared by all streams
k_s   = N_key(W_k e_t)_s ;  q_s = N_query(X_t)_s
g_s   = ssqrt(<k_s,q_s>/sqrt d),  ssqrt(x)=sign(x)·sqrt(max(|x|,1e-6))     (q4:1240-1241)
u_s   = sigmoid(g_s)·v
c_t   = silu( Σ_{j=0..3} w_j ⊙ N_conv(u)_{t-3(3-j)} )    depthwise, kernel 4, dilation 3, causal
X_t  += u_t + c_t                                    into ALL streams, before attn hyper-connection
```

- **Placement.** `ple_layer_ids` is 1-based: `[2]` puts the module at decoder index 1 (q4:1262), which must be a linear-attention layer (q4c:239-253).
- **Conv init.** HF's `_init_weights` zeroes `conv1d` (q4:1354).
- **Schema.** ModuleSpec is ngram-memory's own dataclass, independent of `Qwen4ExpTextConfig`. A native ModuleSpec is read from the HF config. A graft's is not expressible as one (§2.4).

### 2.3 Decode-state spec

The layer-1 cache entry holds `number_of_conv_states = 3` slots (q4c:171):

- slot 2: the last 2 token ids, eos-seeded (q4:1127-1148);
- slot 1: the last 9 positions of `N_conv(u)`, each 4·d values (q4:1210-1226; cu:1192-1230).

An engine that caches only attention KV and the GDN state is wrong: wiping these two slots changed logits by up to 0.32 (`P/t1/bcd_proofs.out`). **Any cached prefix whose tokens read an edited row is stale after an edit**, because slot 1 and every downstream KV page embed row values.

### 2.4 Graft profile ("editor init")

A graft is a ModuleSpec instance with:

- `hc_count = 1`: one stream and one gate.
  - This is **outside Qwen4Exp's validated config domain.** `validate_architecture` raises "Qwen4-Exp requires hc_count > 1" (q4c:195-196; `P/redteam-correctness/checks.out [1]`).
  - So a graft cannot round-trip through `Qwen4ExpTextConfig`, and no qwen4_exp loader or engine will accept it as a Flash-Next checkpoint.
  - The prototype builds `Qwen4ExpTextPLELayer` from a `SimpleNamespace` config (`P/graft/graft_proto.py:175-176`). The package does the same through ModuleSpec. M0 includes a test that builds the graft without the HF config class.
- `ple_embed_dim = 16·w` with row width `w = d/16`, so each host has its own width: 27B 320, 35B-A3B 128, gemma-26B 176, gemma-31B 336 (`P/redteam-correctness/graft.rerun.out`, `"row_width_27B": 320`).
- table = 0, `W_k = 0` (gate ≡ 0.5), `conv1d = 0`, norm weights = 0 (scale 1), `W_v` = fixed seeded random orthogonal matrix.

Then `out_t = 0.5·W_v e_t`, which is **linear in the rows and independent of the query**. The minimum-norm rows for a target residual change δ are `e = 2·W_vᵀδ`.

- **No-op.** An empty table gives exactly zero output for *any* other weights. The reason is `value_proj` with no bias (q4:1197): `value_proj(0) = 0`, so u = 0, `N_conv(0) = 0`, conv = 0 and `silu(0) = 0`. (`key_proj(0) = 0` only makes the gate logit 0, so sigmoid = 0.5, not 0.) This was bit-identical in fp32 and bf16 on all four hosts (`P/graft/results_all.json`, `P/graft/bf16_noop.out`).
- **Format extension: unigram heads (graft-only, unproven).**
  - `Qwen4ExpTextNGramEmbedding` cannot hold unigram heads. It hard-codes `ngram_heads = (ngram_size−1)·heads_per_ngram` (q4:1079), `head_dim = embedding_dim // ngram_heads` (q4:1083) and `for ngram in range(2, …)` (q4:1154).
  - The extension is therefore an ngram-memory **subclass** with an explicit head order: 2-gram heads 0–7 and 3-gram heads 8–15, exactly as on Flash-Next so those row ids stay identical, then 1-gram heads 16–23 (`mix1 = x0·m0`) at primes k = 17…24.
  - Row width stays `w = d/16`, so `ple_embed_dim = 24·w` and `W_v ∈ R^{d×24w}`. A plain `d/24` does not divide on 27B, 35B-A3B or gemma-26B (`d % 24 = 8`; `P/redteam-correctness/checks.out [2]`).
  - It gets its own KeySpec fingerprint (`orders = {1,2,3}` plus the head order). It needs an M0 test for no-op, gradient coverage and row-id parity of heads 0–15 with Flash-Next.
  - The only evidence for it is synthetic. `P/solver/v2_allocation.py` is a 64-d linear model with random-id subjects, puts the unigram heads on the *first* primes, and gives them a different width. Its 1-token "synthetic linear reach" rose from 0.05–0.17 to 0.66–0.81 (`P/solver/v2.out`). That number is also exactly the rate at which the head fires on every other occurrence of the token (§5.3). It does not validate this layout or predict CounterFact generality.
  - Native tables cannot have it.
- **Format option.** `ple_embed_dim = 2·d` lets 2-gram-only writes reach the whole residual. A 2-gram-only write reached 0.46 of a random target at width d (`P/graft/results_0.json`). Reasoned, not tested end to end.

### 2.5 Gemma-PLE format (second table kind)

Key = token id t and layer l. Row = `W[t, 256l : 256(l+1)]`. The module math is in §4.2. The editor treats it as `KeySpec(kind="unigram", V=262144)` with `n_layers` value slots per key.

---

## 3. Architecture

### 3.1 Package layout

```
ngram-memory/
  DESIGN.md
  requirements-dev.txt   pytest, pytest-xdist (pytest is not in the venv today)
  proofs/                provenance copies of the founding proofs (P/)
  ngram_memory/
    spec.py          KeySpec / TableSpec / ModuleSpec dataclasses (own schema, not HF configs), fingerprints, JSON
    hashing.py       transformers-free rows_for(ids)->[T,H] rows, rows_for_ngram(), checkpoint_loc();
                     port of P/t1/src/ngram_format.py; cross-checked against q4 and a vendored copy of
                     R/ple/hashing.py at a recorded commit (test skips cleanly if absent)
    tables.py        RowStore implementations:
                       HFTensorStore      (a loaded nn.Embedding weight)
                       ShardMmapStore     (safetensors shard files, pread by byte offset; never loads the table)
                       SparseRowStore     (sorted row ids + fp32 values; graft storage; P/graft/graft_proto.py:52)
                       QuantRowCodec      (fp8-global-scale / int6-g32 decode, matching the serve)
    ledger.py        EditRecord, RowLedger, materialise(), rollback(), conflict sets
    patch.py         ngram-edit/v1 manifest + safetensors payload; apply/verify against any RowStore
    backends/
      base.py        Backend protocol (3.2)
      qwen4exp_native.py
      gemma4_ple.py   (+ alias.py: reserved-id alias manager and matcher)
      graft/
        module.py     editor-init PLE: q4.Qwen4ExpTextPLELayer built from a ModuleSpec namespace (not
                      Qwen4ExpTextConfig, which rejects hc_count=1); UnigramNGramEmbedding subclass for
                      orders {1,2,3}; SparseRowStore
        qwen35.py     hooks + cache slot extension for qwen3_5 / qwen3_5_moe
        gemma4.py     hooks + side LinearAttentionLayer for gemma4
        beside_native.py  graft installed next to Flash-Next's native PLE (native-vs-graft arm, §6)
    solver/
      prompts.py     X_rw / X_ess / X_col / X_ret builders, template train/held-out split, dedup
      anchors.py     candidate-row policy (tiers), corpus row-load counters
      penalties.py   per-row output-space ridge + reuse/collateral weights
      rowsolve.py    projected Adam in row space through the frozen model (all backends); GN warm start;
                     closed-form joint ridge as a warm start for linear grafts
      quantise.py    serve-codec straight-through materialisation (overlay semantics, §3.3)
    serve/
      hf_adapter.py      overlay hook on the embedding output (also the graft runtime)
      minisgl_adapter.py client for the /ple/* control plane (incl. /ple/score) + overlay patch for R/ple/source.py
      offline_patch.py   in-file shard patching (bf16 tables only; refused for quantised tables)
    eval/
      cohorts.py  metrics.py  parity.py  capability.py  matcher_eval.py  report.py
  tests/   tiny-config proofs (M0), one file per P/ proof; solver proofs marked slow
    fixtures/  small fixtures (headers, hash buffers, .out references); .pt blobs via scripts/fetch_fixtures.py
  scripts/ solve.py, apply.py, evaluate.py, fetch_rows.py (HTTP-range row fetch), fetch_fixtures.py,
           count_rows.py (corpus row-load counters)
```

### 3.2 Backend interface

```python
class Backend(Protocol):
    spec: FormatSpec                      # KeySpec + TableSpec + ModuleSpec (+fingerprint)
    def keys(self, ids, positions) -> list[Key]           # n-gram (or token,layer) keys at positions
    def rows_for(self, keys) -> RowAddr                   # (tensor, row, col0, ncols) per head/layer slot
    def read_rows(self, addr) -> Tensor[fp32]             # effective rows as served (base + active diffs)
    def forward(self, batch, delta: SparseDelta|None)     # frozen model; delta added after the gather;
                                                          # differentiable w.r.t. delta only
    def apply(self, diff: RowDiff) -> int                 # returns epoch
    def revert(self, edit_id) -> int
    def stale(self, token_seqs, row_ids) -> mask          # which cached prefixes read these rows
    def decode_state(self) -> StateSpec                   # what a cache must carry (2.3)
    def parity(self) -> ParityReport                      # rows_for vs model lookups; cached vs full decode
```

Design rule: **gradients reach only a sparse leaf added after the gather**. No backend ever makes the table trainable; on Flash-Next a full-table fp32 gradient would be about 205 GB (`P/compute/count.out`). This pattern was checked on a tiny Qwen4Exp with GDN layers, and per-layer checkpointing gave exactly the same gradient:

- 561 trainable rows out of 16,896;
- finite-difference relative error 4.7e-6;
- gradient-checkpointing difference 0.0 (`P/serve/v2-solve-venue/autograd_coverage.txt`).

### 3.3 Edit record and ledger

An `EditRecord` holds: `edit_id`, time, author; `spec_fingerprint`, model repo and commit; the fact `(s, r, o_true, o_new)` and prompt sets (with their train/held-out template split); `keys` (token-id n-grams with tier); `rows` (tensor, row, col0, ncols) with `base_bits_sha256` each; fp32 `delta`; solver config and solve-time metrics; `parent_epoch` and `conflict_set` (earlier edits sharing rows).

The `RowLedger` maps row → [(edit_id, fp32 delta)] and saves each row's original bits on first write.

- **Materialisation** is an overlay: `served[r] = engine_cast(decode(orig_bits[r]) + Σ active δ)`, summed in canonical edit-id order. The base bits are never rewritten.
- Re-encoding through the base codec, `codec(decode(orig) + Σ δ)`, is allowed only in the offline bf16 mode (§3.4 adapter 4). For fp8 or int6 tables it re-quantises and loses small deltas: int6 in-file kept 0.00 of a 0.003× delta and 0.88 of a 0.03× delta (`P/serve/v2-deltaverify/requant_survival.txt`).
- Naive in-place subtraction is not exact: after sequential bf16 edits it left residue on 50/64 rows (max 4.9e-4), against 0.0 for ledger re-materialisation (`P/solver/v5.out`).
- Removal is exact as state, but later edits were solved conditioned on it, so its conflict set is re-validated.

### 3.4 Engine-agnostic patch interface `ngram-edit/v1`

Manifest (JSON): model repo and commit; `table_spec` (§2.1b: tensor-name template, `pad_divisor`, `n_shards`, rows per shard, row width, base codec); `key_spec`; `mode` (`delta` for native, `absolute` for grafts); a fingerprint of the dequantised base rows; per-edit provenance. Payload (safetensors): `row_ids` int64 [R], `values` fp32 [R, W]. W is the host's row width, so graft payloads are not interchangeable across hosts.

Served row = `engine_cast(codec_decode(base[r]) + delta[r])`, the same overlay rule as §3.3. **Edits are never re-quantised into a quantised base table.** Measured on 8,192 real bf16 rows (`P/serve/v2-deltaverify/requant_survival.txt`):

| Delta size (× row norm) | bf16 fp32-overlay keep | int6 in-file keep |
|---|---|---|
| 0.003 | 0.98 | 0.00 |
| 0.03 | 1.00 | 0.88 |
| 0.1 | 1.00 | 0.98 |

The serve's fp8 global scale (about 2e-4) also clips values above about 0.09 (`P/serve/INTEGRATION.md` §3.5).

Operations: `apply(patch) → epoch`, `revert(edit_id)`, `list()` (per-rank compiled hash), `read_rows(ids)`, `invalidate(row_ids | all)`, `score(ids, positions) → log-softmax` (forward-only, §8 M1), plus an **epoch guard**. The guard stamps requests at admission, and a request admitted before an edit cannot insert prefix-cache state that read edited rows.

Adapters:

1. **HF:** an overlay on the embedding output. This is also the graft runtime.
2. **minisgl:** an fp32 overlay applied right after `decode_raw` in `PLEEmbeddingSource.finish_rows` (`R/ple/source.py:242-244`), before the H2D copy.
   - Graph-captured buffers keep their pointers (`R/ple/graph_capture.py:132-147`).
   - Measured overhead is 7–24 µs per decode call and 1.5–2.6 ms per 2048-token prefill chunk at realistic hit rates (`P/serve/integration/bench_overlay_hits.txt`).
3. **Any engine with an unquantised table on the device:** row `index_copy_` over RPC. The adapter reads the engine's **actual** shard layout for the table and slices accordingly: column-sharded, row (vocab-parallel) sharded, or replicated.
   - HF's own plan is `embedding_colwise` (q4c:99, `EmbeddingParallel(shard_dim=1)` at tp:650), but that is HF's choice. Other engines may differ; this is owner question 9.
4. **Offline:** patch bf16 shard bytes at the §2.1b byte offset, then restart. Allowed only for bf16 tables.

Cache invalidation is part of the contract. A radix node is stale if any position on its path reads an edited row id. Nulling the recurrent snapshot alone re-adopts stale KV (`R/kvcache/radix_cache.py:299-312`; `P/serve/INTEGRATION.md` §3.4). Version 1 therefore **drains traffic and flushes the whole tree**. Targeted detach plus the epoch guard is the steady-state design.

---

## 4. Backends

### 4.1 `qwen4exp_native` (T1)

- **Tensor.** `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight`.
- **Loading.** Never `from_pretrained` with the table. HF concatenates it on CPU (cm:2133-2139) and skips it from `device_map` (`_no_placement_params`, q4:1321-1324). Build the model on meta, load the body, and replace `ngram_embedding` with `ShardMmapStore` (or the serve's codec files) plus a sparse delta leaf. The lookup already moves results back to the model's device (q4:1169-1174).
- **Rows per anchor.** 16 (5 KB in bf16). Gradient reaches exactly the rows the prompt reads: 380 of 380 in the tiny model, with autograd at 1.27628e-3 against finite differences at 1.27670e-3 (`P/t1/bcd_proofs.out`, re-run `P/redteam-correctness/bcd.rerun.out`).
- **Locality.** The PLE output changes exactly at the hit positions plus +3, +6 and +9 through the dilated conv: the predicted and observed sets agreed in 100/100 random sequences (`P/t1/bcd_proofs.out`).
  - Logits are bitwise unchanged in sequences with no hit (47/47) and before the first hit.
  - At and after a hit, every later position changes through GDN and attention (min |Δ| 8.2e-4 after the anchor in the re-run).
  - Real-size per-head collision rate is about 1/2e7 per position.
  - The real collateral comes from *the same bigram in any other context*, which hits all 8 2-gram heads.
- **Decode.** Cached decode equals a full forward with an edited table to 2.7e-8 (`P/t1/bcd_proofs.out`). The HF cache handles state natively; the serve's slot state (`R/ple/state.py`) handles it there.
- **Real-weight facts that shape the solver** (`P/t1/real_stats.out`, `P/t1/e_express.out`):
  - RMS(v) ≈ 0.0058 vs 0.0042 for a token embedding: PLE is a first-order input, not an adapter. `W_v` is near full rank (2541/2560); 2-gram head slices are 2× the 3-gram ones (Frobenius 12.3 vs 6.3).
  - The conv's current-position tap on stream 1 (mean |w| ≈ 0.071) depends only on the direction of v and is about 10× embedding scale at the anchor (2.14 vs 0.21).
  - Row edits reach R² 0.29–0.36 of 0.2–1× targets with the conv on, 0.80–0.91 with it off, 0.89 only at 25× (proxy queries). Per-stream targets cap at about 1/4: the reachable change is `(σ1·v′, …, σ4·v′)`. So the solver optimises final NLL in row space, not a free residual target.
- **Prior positive evidence.** BadEngram (2609.13478, abstract) steers behaviour by modifying only the parameters of "Qwen3.8-Flash-Next's native Per-Layer Embedding subsystem", with the backbone weights and execution graph unchanged (47.8% HarmBench ASR). So this table can move outputs. Whether it can hold *facts* locally is the open question.

### 4.2 `gemma4_ple` (T2)

- **Tensor.** `model.language_model.embed_tokens_per_layer.weight` [262144, 10752] bf16, 5.64 GB.
- **Math** (g4 lines; E4B: H=2560, D=256, L=42):
  - token term `τ_l = 16·W[t, lD:(l+1)D]` (g4:1580-1585, 1443-1444, 1740-1744);
  - context term `c_l = RMSNorm((W_M·√H·E[t])/√H)_l` (g4:1587-1593, 1768-1774);
  - combined `π_l = (c_l+τ_l)/√2` (g4:1586, 1779);
  - per layer `h ← (h + N_{w_l}(P_l(gelu_tanh(G_l h) ⊙ π_l)))·layer_scalar_l` (g4:1358-1363, 1420-1429).
  - `Gemma4RMSNorm` multiplies by `weight` directly, **not** `1+w` (g4:212, 222). A norm weight of 0 therefore silences its branch.
  - `π_l` depends on t alone: a captured per-layer input matches `effective_ple(t)` within 8.3e-7.
- **Write limits.** The post-norm fixes the branch norm (11.33–11.37 while the row delta sweeps 0→1000, `P/t2-gemma4-ple/tiny_results.json`): edits rotate, never grow. Layers 24–41 share KV (`num_kv_shared_layers=18`; g4:1171-1175, 1225-1229, 1241-1244; `P/t2-gemma4-ple/e4b_meta.py`), so a non-final-position write reaches later tokens only through layers 0–22; the tiny analogue's shared layers had exactly zero downstream effect. Row 0 (pad and multimodal, g4:2316-2320) is never edited.
- **Decode.** `per_layer_inputs` are recomputed from each step's ids (g4:1637-1640, 2683-2685). No extra state is needed. Cached and full decode agree to 4.1e-5 (`P/t2-gemma4-ple/tiny_run.log`).
- **Unigram problem → alias ids.** A row fires on every occurrence of its token. Gemma 4 has 6,227 reserved `<unusedN>` ids with untrained E4B PLE rows (std exactly 1/16).
  - A serving-loop matcher rewrites the subject's last token in context to an alias whose main and PLE rows start as copies. Installing it is an exact no-op, and edits then change only aliased occurrences (proof f, `P/t2-gemma4-ple/tiny_results.json`). The tied output head masks alias columns.
  - "KL ≈ 0 on plain tokens" holds **by construction** here, so it is not a locality result. Locality and generality depend on the matcher's precision and recall, which are measured on the alias, surface-variant and lexical-neighbour cohorts (§6).
  - String matcher plus a token-keyed embedding is essentially SWEA (2401.17809) placed at the per-layer tables, so SWEA is the baseline it must beat.
- **Capacity (tiny, random weights; `P/t2-gemma4-ple/capacity_results.json`).** Row edits ranked between a free residual delta and a norm-clamped one. Optimised rows grew to norm 58–108 vs about 2.4 natively, destroying the token's own content, so row trust regions are mandatory. Real efficacy is **unverified**.

### 4.3 `graft/qwen35` (Qwen3.8-27B, Qwen3.6-35B-A3B)

- **Module.** `q4.Qwen4ExpTextPLELayer`, imported and built on meta from a ModuleSpec namespace, with the editor init from §2.4 and `SparseRowStore`. The sparse store matched a dense zero table with the same rows written to max |Δlogit| 0.0 (`P/graft/test_graft.py`).
- **KeySpec.** Exactly Flash-Next's: V=248320, eos=248044, `ple_layer_index` 0. The graft's buffers equal the real checkpoint's (`P/graft/fetch_fn_buffers.out`), so **row ids are identical to Flash-Next's** and do not depend on hidden size. Row *width* does (320 on the 27B, 128 on the 35B-A3B).
- **Tokenizers** (`P/graft/tok_check.out`):
  - Qwen3.8-27B's tokenizer files are byte-identical to Flash-Next's.
  - Qwen3.6-35B-A3B differs only by 7 extra Flash-Next special tokens (ids 248070–248076). 310 of 310 texts gave identical ids.
  - So Flash-Next keys are valid on both hosts. (Key transfer under a shared tokenizer is true of any token-id hash; it is a convenience, not a finding.)
- **Injection.** A forward-pre-hook on `layers[1]` (a GDN layer, as in Flash-Next) adds the output before the sublayer norm. A top-level pre-hook captures `input_ids` and the 2-D mask, because `ForConditionalGeneration` calls the language model with `input_ids=None` (q35:1644, q35moe:1754).
  - The padding mask reproduces `create_recurrent_attention_mask`; it was equal to the host's GDN mask.
  - **Unverified:** the multimodal class path itself. Only `ForCausalLM` was tested.
- **Cache.** Extend layer 1's `LinearAttentionLayer` from 1 to 3 state slots in place (`P/graft/graft_proto.py:96`). The PLE uses slots 1 and 2 as natively (q4:1128, 1146, 1216).
  - Tested: cached and full decode agree to 1.8e-7, against a host baseline of 2.1e-7; `reorder_cache` and `generate` parity.
  - **Unverified:** `crop()` and `activate_past_recording`. No proof calls them. M0 adds a speculative-decode crop test.
- **Value transfer from Flash-Next.** Keys and edit specs transfer; values do not (trained `W_v`, query gates and 4 streams vs d=5120/2048 with a fixed orthogonal `W_v`). A fitted map M (host layer-1 activations regressed on Flash-Next's, aligned position-for-position by the shared tokenizer) is an optional warm start only, since matching shallow states failed before (§6). Default: re-solve per host. **Unverified:** no map was fitted.
- **Gotchas.** `W_k=0` is a dead point: its gradient is exactly 0 through `sign`/`clamp_min` (q4:1241; `P/graft/calib_grad.out`), so v1 does not calibrate gates. MTP draft heads never see the graft, which only lowers acceptance (reasoned).

### 4.4 `graft/gemma4` (gemma-4-26B-A4B-it, gemma-4-31B-it)

- **Primary format.** The Flash-Next format with a **Gemma key instance**: V=262144, eos=1 (`<eos>`), and therefore different multipliers. Row width 176 (26B) / 336 (31B).
  - **To confirm:** the reset token could instead be 106 `<end_of_turn>`.
  - All checks passed on tiny 26B-like (MoE, `k_eq_v`) and 31B-like hosts: no-op bit-identical; cached/full decode 4.1e-6 vs a host baseline of 3.4e-6; left-padded batches match; gradient on 192 of 192 rows read (`P/graft/results_all.json`).
- **Gemma-specific handling.**
  - `Gemma4Model` passes a dict mask (g4:2422), so the 2-D mask is captured at the top level.
  - `layer_scalar` multiplies the whole residual (g4:1429) and `embed_scale=√d` makes the residual large (g4:1565); per-fact solving absorbs both. In the 26B the router reads the injected residual (g4:1404-1406), so edits also move routing.
  - Layer 1 is a sliding-window KV layer, so a side `LinearAttentionLayer` hangs off the Cache with `reorder_cache`/`reset`/`batch_*` forwarded (`P/graft/graft_proto.py:126`). `crop()` is not forwarded: speculative decoding must raise until it is.
- **Alternative format (A/B in M5): Gemma's own PLE** via `hidden_size_per_layer_input>0`. Stock `Gemma4ForCausalLM` builds it and reports only the PLE tensors missing.
  - **Editor-init:** `embed_tokens_per_layer = 0` and `per_layer_model_projection = 0` (optionally `per_layer_projection_norm.weight = 0`) give an exact no-op. Keep `post_per_layer_input_norm.weight` **nonzero** (1, or a calibrated w_l) and G_l, P_l nonzero.
  - Do **not** zero `post_per_layer_input_norm.weight`: because `Gemma4RMSNorm` scales by `weight` itself (g4:222), the branch then outputs 0 whatever the rows hold, so writes are no-ops and have zero gradient. With weight 0 a row write changed logits by 0.0 and the gradient was 0.0; with weight 1 the change was 2.9e-1 and the gradient 7.5e-2 (`P/redteam-correctness/checks.out [3]`).
  - This is the init proof g actually used (`P/t2-gemma4-ple/tiny_ple.py:339-343`: projection and projection-norm weights 0, post-norm weight 1.0; result in `P/t2-gemma4-ple/tiny_results.json`).
  - Pros: stock-loadable, maybe engine-supported (**unverified**), exact alias keys. Cons: needs the serving-loop alias matcher; dense table (4.0 GB 26B, 8.05 GB 31B at D=256); on/off write strength to calibrate via `w_l`.

### 4.5 Graft physical size

Each host keeps the full logical row space, so primes and row ids are unchanged, but stores only the rows that have been written. Each edited anchor writes 16 rows of the host's width `w = d/16`. Overlay = anchors · 16 · (4w + 8) bytes (fp32 values + int64 row ids; `P/review-apply/budget.out`). Collision rates are computed by formula, assuming uniform n-grams (`P/compute/collide.out`).

| Edited anchors | 27B (w 320) | 35B-A3B (w 128) | gemma-26B (w 176) | gemma-31B (w 336) | Flash-Next overlay (w 160, ref.) | Positions hitting an edited row by collision |
|---|---|---|---|---|---|---|
| 1e3 | 20.6 MB | 8.3 MB | 11.4 MB | 21.6 MB | 10.4 MB | 0.08% |
| 1e4 | 206 MB | 83 MB | 114 MB | 216 MB | 104 MB | 0.8% |
| 1e5 | 2.06 GB | 0.83 GB | 1.14 GB | 2.16 GB | 1.04 GB | 7.7% |

A dense graft at width 320 would be 204.8 GB in bf16 on the 27B. This is why grafts are sparse.

---

## 5. The solver ("RowSolve")

The abstraction: a key function κ(x, t) → {(slot, row)}; a module F gives the additive residual m_t; an edit is a sparse fp32 delta U = {u_j}.

- **W1** = hashed-n-gram tables (native Flash-Next and grafts).
- **W2** = Gemma (token, layer) rows.

### 5.0 Validity gate

- Base P(o_true) > P(o_new) on the rewrite prompt.
- The subject span is locatable in every prompt.
- The keys at the anchor are computable for every prompt, which rules out mid-token subject boundaries.
- The rejection rate is logged per model.

### 5.1 Prompt sets

- **Template split.** Every template family (paraphrases, essence templates, SSOR relations) is split **by template** into a train part, used in the objective, and a held-out part, used only for evaluation. Held-out paraphrases whose suffix from the subject onward duplicates a training prompt are dropped. (In the earlier experiment, 4 of 24 CounterFact paraphrases were such copies; gemini-engram-code-review.md:300.)
- **X_rw:** the rewrite prompt plus K=4 train paraphrases, each also under C=5 prefix contexts. In the synthetic allocation, context diversity raised synthetic linear reach from 0.65 to 0.82 (`P/solver/v2.out`; synthetic, not generality).
- **X_ess (train only):** `"{s} is"`, `"{s},"`, `"{s} was"`, plus at least 4 train SSOR templates. `"{s} is a"` is in X_ess only in the default arm. A leave-one-out arm excludes it, and the `"{s} is a"` bar and K-leak are read only from that arm.
- **X_col:** corpus windows that read a candidate row through a different n-gram, or through the same n-gram in another context.
- **X_ret:** rewrite prompts of earlier edits that share rows with this one. Retention is scored on their held-out paraphrases, which are never in X_ret.
- Prompts are right-padded or batched one per sequence, **never left-padded**. Left padding changes the PLE's padding semantics relative to the serve (`P/serve/INTEGRATION.md` §2.1).
- **Size per iteration** (`P/review-apply/budget.out`, three hand-written CounterFact-style facts on the Qwen tokenizer; illustrative, recount on real CounterFact in M0). There are 38 prompts per fact (30 X_rw + 8 X_ess).
  - Full prompts: 418–524 tokens per fact, or 674–1548 with 8–32 X_col windows of 32 tokens.
  - With prefix caching (`P/serve/INTEGRATION.md` §4.3: positions before the anchor run once without grad and are cached), each iteration recomputes only anchor→end: 187–241 tokens, or 251–497 with X_col.
  - The solver uses prefix caching. §7 budgets both N.

### 5.2 Anchor policy (n-gram selection)

Subject-token statistics from the real Qwen tokenizer (`P/t1/coverage.out`): of 30 subjects, 10 are 1 token, 8 are 2, 5 are 3, 5 are 4 and 2 are 5.

- **Tier 1 — subject-internal n-grams.** The window lies wholly inside the subject span, mainly at the last subject token q.
  - A 2-token subject has an internal 2-gram only.
  - A subject of 3 or more tokens has both an internal 2-gram and an internal 3-gram.
  - Tier-1 keys are relation-blind: every relation of one subject shares the same 16 rows. The same-subject multi-relation cohort (§6) tests this.
- **Tier 2 — subjects with no internal key at q** (all 1-token subjects on native tables).
  - (a) Right-crossing (s_last, r1) and (s_last, r1, r2). These are relation-wording-specific, which helps SSOR. But paraphrases that change the relation wording miss by construction, and they fail when r1 is " is".
  - (b) Left-crossing (c, s) for the top left contexts covering ≥90% of corpus P(c | s), plus the segment-start (eos, s) key. Example: Paris or Toyota in 9 contexts gave 7 distinct preceding tokens.
  - **Prior evidence against (b):** keys reaching into pre-subject context gave a realised δ* fraction on held-out prompts of 0.066 for 2-token and 0.000 for 1-token subjects (gemini-engram-code-review.md §8). Tier 2 is therefore reported per sub-tier and per subject length, with its own kill (K-native-short, §6).
- **Grafts only — tier 0.** Unigram heads (§2.4).
- **Gemma.** Prefer the alias of the last subject token. Prefer layers 0–22 with large rms(w_post); real values range 0.05–8.6 (`P/t2-gemma4-ple/real_stats.json`).
- **Tokenisation varies with position.** "Nokia" is 1 token mid-sentence but 2 at sentence start, so keys are enumerated for every surface form the prompt set produces.

### 5.3 Penalties and trust regions

- **Per-row ridge in output space.** The regulariser is `Σ_j λ_j ‖V_j u_j‖² / ‖h̄‖²`, with `λ_j = λ_ridge + λ_reuse·w_j`. Output space is used rather than row space because `W_v` head slices differ by 2×.
- **Reuse weight.** `w_j = min(w_len + w_freq − 1 + γ·occ_j, w_max)`.
  - `w_freq` is the percentile of row j's **total corpus load**: corpus counts of every n-gram that hashes to row j, *including the edited n-gram's own occurrences in other contexts*. Counts come from streaming a corpus through the real hash into 320M uint32 counters (1.28 GB; `P/redteam-feasibility/count_bench.out`). Venue and budget in §7.
  - `occ_j` counts earlier edits on the row.
  - On Gemma, unigram counts are used instead.
- **Starting values.** EngramEdit's defaults: λ_ridge 0.01, λ_reuse 0.05, γ 9, w_freq_max 10, w_max 64.
- **Length weight.** The n-gram length weight is flattened for crossing keys. In the synthetic allocation, EngramEdit's η = 8/2/1 pushed 74–81% of the edit into crossing 3-grams and cut 2-token synthetic linear reach from 0.75 to 0.59 (`P/solver/v2.out`).
- **What the synthetic allocation does and does not show.** `P/solver/v2_allocation.py` is a linear 64-d model with random-id subjects in a Zipf corpus.
  - Its "fit" is the projection of the realised output onto a random target (`:61-63`), called **synthetic linear reach** here. It is not CounterFact generality, and no accept bar is derived from it.
  - Its collateral is `load − own` (`:66`): only hash co-tenants are counted, and the edited n-gram's own other-context occurrences are left out. So the unigram keys show collateral 0.00e+00 with held-out = train = 0.660 / 0.809 (`v2.out` check4). That "reach" and the head firing on every occurrence of the token are the same number.
  - The per-row versus per-n-gram comparison ("collateral cut by 25–60%") is **not at matched fit**: train fit fell from 0.841 to 0.748 and from 0.935 to 0.879. M3 re-runs it at equal train fit, with full-load collateral.
- **Trust regions.**
  - Residual change: `‖Δm_t‖ ≤ ρ‖h_t‖`, ρ ∈ [0.5, 1], by projection after each step.
  - Native rows also: `‖u_j‖ ≤ κ‖e_j‖`, κ ≈ 10.
  - EngramEdit's clamp relative to the memory output is meaningless for a zero graft, because the memory output is 0 there.

### 5.4 Objective and optimisation

```
min_U  mean_{X_rw} NLL(o_new | x; U)                        all object tokens, teacher-forced, FINAL layer
     + λ_KL  · mean_{X_ess} mean_{t≥q} KL(p_0 ‖ p_U)         per-token mean, subject end + 3 greedy tokens
     + λ_col · per-token KL on X_col  + λ_ret · NLL of overlapping earlier edits
     + Σ_j λ_j ‖V_j u_j‖²/‖h̄‖²
```

- **All backends.** Projected Adam in row space through the real frozen model, with gradients only on the sparse leaf. Using the same path on grafts keeps the solver shared for the native-vs-graft comparison.
  - Optional warm start: Gauss–Newton matching of a stage-1 δ* restricted to the reachable family.
  - **Never** use the EngramEdit Eq. 28 one-shot linearisation as the answer. On the real `Qwen4ExpTextPLELayer` with hc=4 it left 0.12–0.23 residual on reachable targets, where GN and Adam reached about 0. On Gemma's gated, normed PLE it left 0.18–0.79 (`P/solver/v1.out`, `P/solver/v4.out`).
  - The frozen-gate least-squares solve gives negative R² with the conv on (`P/t1/e_express.out`).
- **W1 graft warm start.** The module is linear, so the batch closed form `U* = (AᵀA+Λ)⁻¹AᵀB` is exact; it matched L-BFGS to 1.5e-7 (`P/solver/v2.out`). Adam on the full objective follows. Graft results are reported for both the closed-form-only and the Adam path.
  - On a tiny model the direct solve beat the two-stage one: Δlog P of +0.29 against +0.16 (`P/solver/v3.out`). Directional only.
- **Quantisation-aware finish.** Materialise with overlay semantics (§3.3: fp32 delta on the decoded base, then the engine's cast), re-evaluate, and take straight-through steps if efficacy drops. Diffs below about 3% of the row norm partly vanish under a bf16 cast (`P/solver/v5.out`).

### 5.5 Multi-token objects

Subject-anchored rows carry all object tokens through the NLL. The fallback is object-internal keys, for example (o1, o2) → o3, or o1 unigrams in grafts.

On native Flash-Next the key for o2 is necessarily relation-crossing, (r_last, o1). These rows usually carry a large collateral load, so the penalty normally forbids them. When they are used, X_col must contain corpus windows of that n-gram.

### 5.6 Sequential edits and rollback

- Batches of about 100 edits, targets recomputed on the current model. The conflict set (earlier edits sharing candidate rows; mostly same subject, other relation) feeds X_ret. Conflict clusters are periodically re-solved from the original table; stored diffs make this order-free.
- **XK mode (exact-key overlay):** private rows behind a 64-bit n-gram fingerprint check. This is EngramEdit's B.4.3 side-vector scheme. Needs an engine hook rather than a tensor diff; only κ changes in the solver.
  - XK rows go through the same module (same W_v, gate and conv) as in-place rows. So XK removes hash co-tenancy and collisions, but **not** a layer-1 expressivity limit. M2's three-arm design (§6) separates the two.
- Rollback = ledger re-materialisation (§3.3).

---

## 6. Evaluation protocol

### Priors from earlier work

From `docs/research/gemini-engram-code-review.md`, on frozen Qwen2.5-0.5B (one model, one seed, n=12):

- **Free delta (reference, not a bound).** A free layer-2 delta at the last subject token only, optimised for **final-layer** NLL, reached:
  - efficacy 1.000;
  - paraphrase 0.875–0.917 (base 0.208);
  - same-subject other-relation leakage, top-1 = new, of 0.194–0.250, mostly on `"{s} is a"` (0.21 on the 5-fact re-run).
  - A table write is a *different* family, not a subset: it fires at every position whose key hits, plus the conv taps at +3, +6, +9. The source document corrected exactly this "strict subset" claim (gemini-engram-code-review.md:291-296). So these numbers are a reference point, not an upper bound.
- **In-context ceiling** in the same run: leak KL 0.512, leak top-1 = new 0.028 (gemini-engram-code-review.md:207).
- **Shallow targets fail.** Matching shallow teacher states fails: the fact arrives at blocks 20–23 of 24, and the block-6 oracle gave P(new) 0.012.
- **Short subjects.** Keys reaching into pre-subject context gave a realised δ* fraction on held-out prompts of 0.066 for 2-token and 0.000 for 1-token subjects (§8 there).
- **Cached decode** must carry the n-gram and conv state.

**Prior art on native n-gram memory editing exists.** EngramEdit (2610.10533) edits the natively pretrained n-gram memory of LongCat-Flash-Lite (68.5B total, ~3B active, over 30B in embeddings; 2601.21204). Its abstract says it "jointly updates the shared n-gram embeddings", and it reports that > 92% of new unrelated errors involve updated rows (engram-graft-assessment.md:101). This repo's earlier review reads its B.4.3 as storing edits in exact-n-gram side vectors with the pretrained tables fixed (gemini-engram-code-review.md:148-150). Which mode its headline numbers use is **unverified** here (full text not re-read).

### Oracles (defined)

- **O-site (primary):** one free δ ∈ R^d per fact, shared across positions, added at the PLE-output site (decoder layer 1 input on Flash-Next and Qwen grafts; the graft site on Gemma) at **the same hit positions as the anchor keys**, with the same ρ trust region and the same objective.
- **O-last (comparability):** one δ at the last subject token only, same site and ρ, for comparison with the earlier result.
- On Flash-Next, O-site is also the expressivity arm (c) of M2.

### Cohorts

- CounterFact: 2,000 sequential; 500 for ablations. zsRE: 2,000.
- **SSOR:** CounterFact subjects × other-relation templates, split by template into train (in X_ess) and held-out. Relations the new fact implies (ripple, e.g. country → language) are reported separately from non-implied ones.
- **Lexical neighbours (primary locality cohort):**
  - different entities that share the anchor n-gram or token (e.g. "Toyota Corolla" when editing "Toyota", other "X Jordan" names);
  - Wikipedia sentences that mention the subject in unrelated roles;
  - CounterFact+ prefixed neighbours, where the edited subject sits in the prefix, its rows fire, and the effect is carried by the conv (+3, +6, +9) and by attention/GDN.
- **CounterFact neighbourhood (sanity check only).** These prompts name other subjects, read no edited row, and by I1 are bitwise unchanged apart from collisions. The earlier proposal called this "near-vacuous for exact keys" (engram-relaunch-proposal.md:500).
- **Alias and surface variants:** case, retokenisation, sentence-start forms, and aliases from RippleEdits/UniEdit.
- **Short-subject cohort:** at least 150 subjects each of 1, 2 and ≥3 tokens per tokenizer, crossed with anchor tier.
- **Same-subject multi-relation:** k = 2..5 edits per subject, the conflict mode tier-1 keys are exposed to.

### Metrics

- **Efficacy:** P(new) > P(old) on the rewrite prompt; also top-1 and P(new). Reported as a mean with a Wilson 95% interval.
- **Generality:** held-out paraphrases never in X_rw (after dedup), **split by whether the subject token sequence is preserved**; novel wordings and the alias/surface cohort reported separately. EngramEdit's 97% is over paraphrases that keep the subject string (engram-graft-assessment.md:96-97), so only the preserved split is comparable to it.
- **Locality:**
  - lexical-neighbour drop and per-token KL; NS+ drop and per-token KL;
  - held-out SSOR: top-1 = new and per-token KL on non-implied relations, with ripple-implied relations reported separately;
  - generic-text KL per token, split by whether the window reads an edited row; collision-window KL;
  - CounterFact NS as a sanity check.
- Fluency and consistency as in ROME; zsRE token accuracy.
- **Matcher (Gemma aliases):** false-fire and miss rates on the alias, surface-variant and lexical-neighbour cohorts.
- **Capability retention:**
  - Unconditioned: MMLU/GSM8K/HellaSwag subsets and WikiText perplexity at 0/1k/2k/5k edits. These are nearly guaranteed by I1 for items that hit no edited row, so they are reported but not decisive.
  - **Hit-conditioned (decisive):** the same benchmarks and perplexity restricted to items or windows that read an edited row.
  - Subset sizes: ≥ 2,300 items per benchmark arm where available, so that a 2-point drop lies outside an unpaired 95% binomial CI at p ≈ 0.6; paired (McNemar) analysis on the same items is used where the item pool is smaller.
- **Sequential curves:** early-edit retention on held-out paraphrases, row occupancy, conflicts.
- Bootstrap 95% CIs throughout.

### Baselines

- No edit; in-context ceiling (same run, same prompts); O-site and O-last oracles.
- **EngramEdit**, verbatim configuration (two-stage δ* then ridge, its η and λ), on every W1 host.
- **SWEA** (token-id-keyed input-embedding edit) on every host. It needs no graft, so it is the null hypothesis for "is a graft needed".
- **NeuralDB** control, and **GRACE / WISE / MEMOIR** for the sequential runs.
- ROME/MEMIT/AlphaEdit on graft hosts, and MEMIT/AlphaEdit at small n on Flash-Next (MoE/GDN feasibility **unverified**).

### Native vs graft (controlled)

On Flash-Next itself, install a zero-init editor-init graft **beside** the native PLE at decoder layer 1, with the same KeySpec, orders {2,3}, and output added into all 4 streams (**unverified**: needs an M0 tiny test). Solve the same facts both ways with the same solver code (Adam path). Report the paired difference in efficacy, generality and leakage, normalised by the same-host O-site oracle. This is the only test of "same editor, same results" with the host held fixed.

### Hard invariants (a failure is a bug and stops the run)

- **I1.** Windows that read no edited row are unchanged.
  - On HF: **bitwise**. This held on all tiny proofs.
  - On the serve: bs=1, sequential requests, fixed chunk size, compared against a no-edit control under identical conditions. Bitwise only where that control is itself bitwise across repeat runs. Otherwise the tolerance is the control's measured repeat-run noise floor, and the report says which was used. Concurrent batched runs are not bitwise reproducible even without edits (`R/../../docs/journal/PERF_PUSH_CONTINUANCE.md:128-136`).
- **I2.** Cached decode equals a full forward at the host's own noise floor, over 50 prompts × 32 greedy tokens with identical tokens. This also holds on the live serve after a patch, including for a prefix cached before the edit.
- **I3.** Rollback restores pre-edit outputs: bitwise on HF; on the serve, under the same pinned conditions and tolerance rule as I1.
- **I4.** `rows_for` matches the model's lookups on at least 10k positions, including EOS edge cases.

### Serve-transfer checks (P0–P5, from `P/serve/INTEGRATION.md` §5)

- Row ids, base row bytes and edit transport are bitwise equal between the solve and the serve.
- Pre-edit top-1 agreement between the solve model and the serve is at least 95%. Otherwise add fp8 activation emulation: the serve's per-token fp8 expert activations have 4.09e-2 relative error per layer (`R/weights/cpu_tier.py:38-40`).
- Realised ΔNLL / predicted ΔNLL ≥ 0.7 on targets, read through `/ple/score` (§8 M1).
- Cache hygiene: warm, then edit, then query must equal a cold run, under the I1 pinned conditions.

### Proposed accept bars (unverified; to be revised after M2)

All locality bars are read on **held-out** templates.

| Metric | Bar |
|---|---|
| Efficacy | ≥ 0.95 |
| Generality, subject-preserved, overall | ≥ 0.80 |
| Generality, 1-token subjects, graft | ≥ 0.60 (a priori target, not derived from synthetic reach) |
| Generality, 1-token subjects, native | reported per tier; see K-native-short |
| Lexical-neighbour drop / NS+ drop | ≤ 0.05 each |
| Lexical-neighbour and NS+ per-token KL | ≤ the same-run in-context arm |
| `"{s} is a"` top-1 = new (leave-one-out arm) | ≤ 0.05 |
| Held-out SSOR (non-implied) top-1 = new | ≤ 0.10 |
| Held-out SSOR (non-implied) per-token KL | ≤ the same-run in-context arm |
| Capability retention, hit-conditioned | ≥ 96% at 5k edits (EngramEdit's stability tests run to 5,000) |
| Alias matcher (Gemma) | false-fire ≤ 0.01 on lexical neighbours; miss ≤ 0.10 on surface variants |

### Ablations

λ_KL ∈ {0, 0.0625, 0.25, 1}; anchor tier; per-row vs per-n-gram penalty at equal train fit; in-place vs exact-key overlay (XK mode); ρ and κ; grafts: orders {2,3} vs {1,2,3} (each paired with its lexical-neighbour and SSOR leakage), width d vs 2d, closed-form vs Adam; native vs graft on Flash-Next; Gemma: Flash-Next-format graft vs Gemma-PLE plus aliases vs SWEA.

### Kill criteria

- **K-native (with attribution).** On Flash-Next, with ρ ≤ 1 and κ ≤ 10, on 50 validity-gated facts, run three arms on the same facts: (a) in-place rows, (b) XK rows, (c) O-site free δ. Kill if the upper 95% Wilson bound of **mean** efficacy for (a) is below 0.5. Then:
  - (c) passes and (a) fails → the cause is hashing or co-tenants; move to XK.
  - (c) also fails → the cause is layer-1 expressivity. Stop native editing at this site; do **not** move to XK. Record the negative result.
  - Reported per tier and per subject length.
- **K-native-short.** 1-token native generality (subject-preserved) below 0.3 at the trust region, per tier → declare native short subjects unsupported.
- **K-leak.** Held-out non-implied SSOR top-1 = new above 0.10 at the best λ_KL, or `"{s} is a"` top-1 = new above 0.10 in the leave-one-out arm, or lexical-neighbour/NS+ drop above 0.10, on every tier. Conclusion: the keying is relation-blind. Restrict to right-crossing keys; if those fail efficacy, stop.
- **K-transfer.** Realised/predicted ΔNLL stays below 0.5 even with fp8 emulation. Conclusion: offline solves are not valid for the serve. Use forward-only acceptance or zero-order steps through the serve, or stop.
- **K-scale.** At 2k sequential edits with conflict re-solving: retention of early edits on held-out paraphrases below 0.8, or a same-subject multi-relation cohort (k=2..5) retention below 0.8. Hit-conditioned capability below 96% at 5k.
- **K-graft.** A graft fails I1 or I2 on real weights, or its efficacy is below 0.5 where the O-site oracle on the same host exceeds 0.9.

---

## 7. Compute plan

### Sizes and per-iteration cost

Sizes are exact, from safetensors headers (`P/compute/count.out`). FLOP is **6·active·N**: forward, recompute forward under per-layer checkpointing, and backward. N is the per-iteration token count of §5.1: 251–497 with prefix caching (used), 674–1548 for full prompts. The earlier figure used 4·active·N at N=320 (`P/compute/plan_calc.py:34`), which left out the recompute and did not match §5.1. Activations scale linearly from `P/compute/plan_calc.out`. All per-iteration figures come from `P/review-apply/budget.out` and are **est.**

| Target | Weights (bf16) | Table | Active params / token | TFLOP / iter, prefix-cached N | TFLOP / iter, full N | Ckpt activations (GB) |
|---|---|---|---|---|---|---|
| Flash-Next | 360 GB (body 251 GB bf16, 73 GB at ~4.5-bit) | 102.4 GB | 6.67 G | 10–20 | 27–62 | 0.46–2.86 |
| Gemma E4B | 16 GB | 5.64 GB | 4.70 G | 7–14 | 19–44 | 0.10–0.63 |
| Qwen3.8-27B | 55.6 GB | sparse graft | 25.6 G | 39–76 | 104–238 | 0.32–1.98 |
| Qwen3.6-35B-A3B | 71.9 GB | sparse graft | 2.95 G | 4.4–8.8 | 12–27 | 0.14–0.88 |
| gemma-4-26B-A4B | 51.6 GB | sparse graft | 3.82 G | 5.8–11 | 16–36 | 0.11–0.70 |
| gemma-4-31B | 62.6 GB | sparse graft | 30.7 G | 46–92 | 124–285 | 0.27–1.65 |

Weights, not activations, decide the hardware. The table never needs to be resident: a batch reads about 5 KB of rows per token.

### Per-venue plan

> **Open conflict, to be settled by the owner (2026-11-09).** [`serve/INTEGRATION.md`](serve/INTEGRATION.md) v2
> (verified against minisgl `df29117`) recommends the **local CPU solve beside the serve** as the venue for the *first*
> native experiment: ngram-memory's own MXFP4 loader, `MemoryMax=15G`, 4 cores, 54–115 s per iteration, 19–40 min for
> one edit, 3.2–6.9 min per edit when 16 facts × 4 prompts are batched in RAM. This document instead treats local CPU
> streaming as **steady-state only** (below, venue (b)) and puts the first native solve in the cloud (M2-lite, then (d2)).
> The two documents estimate different per-iteration rates (32–91 s here, 54–115 s there), because they make different
> kernel assumptions.
>
> What settles it:
> - the owner's answers on free memory (`MemAvailable` vs `MemFree`, about 20 GB stated);
> - core count and acceptable slowdown;
> - whether the EP arm leaves the cards to the serve;
> - an M0 measurement of one decoder layer's forward+backward on CPU.
>
> Until then, M2-lite in the cloud stays the cheapest experiment that can fire K-native, and the local CPU solve is
> its no-cloud alternative.

- **This container** (4 CPUs, 15 GB RAM, no GPU, about 27–30 GB free disk): M0 — all tiny proofs, the full pipeline, and patch round-trips through the HF adapter and the overlay prototype (`P/serve/ple_path/ple_overlay.py`).
  - Measured CPU rates: bf16 GEMM 2,235 GFLOP/s; bf16 gated-MLP forward plus input gradient 1,457 GFLOP/s (`P/redteam-feasibility/cpu_flops.out`, 4 threads). The end-to-end HF rate (GDN torch fallbacks, small ops) is **unmeasured**. M0 measures one E2B-shaped decoder layer forward+backward at the corrected N. Use `OMP_NUM_THREADS=1` when the CPUs are shared.
  - Optional: Gemma E2B real weights (9.3 GB text-only fetch, **needs the owner's go-ahead**), est. 8–47 s per iteration with prefix caching and 21–147 s with full prompts, assuming 10–30% of the measured MLP rate (`P/review-apply/budget.out`). This is the only real-weights native proof possible here.
- **Owner, 2×16 GB RDNA4 (ROCm).** Every job must go through `gpu-lease -n N -- <cmd>`, because the cards are shared across repos (fork `CLAUDE.md:3-30`).
  - Gemma E4B native fits one card (about 9.4 GB body + up to 0.6 GB activations at the corrected N, table on CPU; est. 0.3–1.2 s per iteration).
  - Grafted hosts at about 4.5 bits per weight (15–21 GB) need both cards and a differentiable dequant matmul on ROCm (**unverified**).
  - **Conflict:** the qwen4exp serve arm (`R/../../tools/serve.sh:823`) now defaults to `EP=1`, `EP_SPLIT=2:1` and a 40 GB host weight-offload clamp (`serve.sh:1145-1158`). Its pinned host arena is 53.75 GiB, and boots failed when other work held about 8 GiB (`serve.sh:1149-1153`). The serve very likely occupies both cards, so local GPU work needs serve downtime.
- **Owner's Flash-Next serve: forward-only.** It runs under `torch.inference_mode` with graph capture (`R/scheduler/scheduler.py:1369`).
  - It serves the validity gate, evaluation, P0–P5 and forward-only acceptance, and applies patches through the overlay.
  - The only measured headroom, 11.1 GiB available with 33.9 GiB swap in use, is a **TP=2** measurement (`serve.sh:1119-1123`). The EP arm's headroom is unmeasured (owner question 2). No concurrent local HF solves.
  - **Scoring path.** At HEAD the API returns logprobs only for generated tokens, top-N capped at 20, on non-streaming requests (`R/server/api_server.py:415-420`). `echo` (prompt logprobs) is rejected (`api_server.py:1637-1638`). So teacher-forced NLL, multi-token P(o_new) vs P(o_old), full-vocabulary KL and P4 need the `/ple/score` control (M1 deliverable).
  - **Eval time budget** (est.). Scoring is prefill-only: about 1,250–1,430 tok/s at 16k-token prompts on the EP arm (`serve.sh:1145-1148`), so 2,000 facts × ~50 prompts × ~25 tokens ≈ 2.5 M tokens ≈ 30 min per evaluation point. Generation is 12.9 tok/s at TP=2 (`serve.sh:692-694`) or about 20 tok/s with EP (`serve.sh:1145-1148`), so 2,000 × 100-token generations ≈ 2.8–4.3 h per evaluation point, each behind a drain-and-flush. Generation-heavy capability evals therefore move to the cloud solve model wherever P3 parity holds.
- **Flash-Next solve venues** (`P/serve/INTEGRATION.md` §4.2, `P/serve/v2-solve-venue/cost_model.txt`; all est.):
  - **(M2-lite) stock HF bf16, cloud.** The table sits in 31 table-only files plus 2 mixed ones; the non-table files total 255.7 GB (`P/redteam-feasibility/hub_quant.out`). 4× H100 SXM 80 GB or 2× H200 hold the bf16 body, with table rows read through `ShardMmapStore` or HTTP range. No NVFP4 expert module is needed.
  - **(d2)** 1× cloud H200, quantised body dequantised per layer: 1.1–3.2 s per iteration at 256 tokens, 1.5–4.0 s at 1024 (cost_model column C); the serve stays up. Needs the 2–4-day NVFP4 expert module plus loader (INTEGRATION.md §4.2). The body can come from the public `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135.2 GB including a 51.2 G-element F8_E4M3 table, so ~84 GB body). Table rows are range-fetched on the pod from the repo the owner serves (e.g. int6-g32 `tcclaviger/...`, 120.3 GB), if that repo is public (owner questions 3–4). Otherwise rows are shipped from home.
  - **(f)** Local downtime, both cards, experts in RAM: 3.6–30 s per iteration plus about 10 min reboot (`R/weights/config.py:46`).
  - **(b)** Local CPU streaming beside the serve: 32–50 s per iteration at 256 tokens and 57–91 s at 1024 with a native kernel (cost_model column A-native); overnight steady state only.
  - Expert codec: INTEGRATION.md says NVFP4; `P/serve/v2-solve-local/bench_dequant2.T4.txt` benchmarks MXFP4 bit-exact against the fork; confirm. A dequantised-expert module matched dense experts exactly (`P/serve/v2-solve-venue/autograd_coverage.txt`).
- **Corpus row-load pass (§5.3).** Measured at 0.21 M tok/s per process for the vectorised 16-head hash plus `np.add.at`: 1.3 h per 1B tokens per core, 13 h for 10B, before tokenisation (`P/redteam-feasibility/count_bench.out`). 10B tokens of raw text (~40 GB) exceeds the free disk here. The Gemma key instance needs a second counter set.
  - Plan (M0/M3 task): stream 1B tokens here on 4 processes with per-process counters (~5 GB RAM) and a sort-and-bincount kernel, est. under an hour. Run 10B on cloud CPU only if M3 shows the 1B percentiles are unstable.
- **Cloud, RunPod** ($/GPU-hour community/secure, read 2026-11-09 from a page summary; reconfirm): 1× A100 80 GB $1.39/$1.79; RTX PRO 6000 96 GB $1.69/$2.49; 1× H200 $3.59/$5.29; 4× H100 $10.76/$15.96. Volume storage $0.10/GB/month.

### Cost per milestone (est., `P/review-apply/budget.out`)

Cost = solves × iterations/solve (100–300, **assumed**) × s/iteration × $/h, plus setup hours. The s/iteration for graft hosts on A100 (0.3–2.0 s) and for 4× H100 bf16 is **assumed**.

| Milestone | Venue | Solves | GPU-h | $ |
|---|---|---|---|---|
| M2-lite (10 facts) | 4× H100 bf16 | 10 | 1.8–4.0 | 19–64 |
| M2 (3 arms × 2 λ × 60 facts) | H200 (d2) | 360 | 12–97 | 43–513 |
| M2g (2 hosts × 3 arms × 50) | A100 | 300 | 3–51 | 4–90 |
| M3 native (150/length × 3 lengths × 2 tiers) | H200 (d2) | 900 | 29–241 | 102–1,275 |
| M3 graft (450 × 4 λ × 2 orders) | A100 | 3,600 | 31–601 | 42–1,075 |
| M3 native-vs-graft on Flash-Next | H200 (d2) | 100 | 4–28 | 15–146 |
| M4 E4B (200 × 2) | A100 | 400 | 4–67 | 5–120 |
| M5 (2 formats × 2 hosts × 200) | A100 | 800 | 7–134 | 10–240 |
| M6 sequential 5k + re-solves (~1.5×) | H200 (d2) | 7,500 | 230–2,001 | 826–10,585 |
| **Total** | | | | **≈ $1.1k–14k** + storage (~$15/month for 150 GB) |

The earlier "$60–120 campaign" was wrong by more than an order of magnitude. M6 dominates the cost. The main levers are fewer iterations per solve (measured in M2-lite), batching facts per iteration, and running most of M6 on a graft host with only a 2k subset on Flash-Next.

### Questions for the owner about the Flash-Next serve

1. **Engine.** Is the serve `patcarter883/minisglang-rdna4`, and at which commit (local HEAD is `df29117`)? Has anything changed in `R/ple/source.py` (`finish_rows`), `graph_capture.py`, `radix_cache.py` or `_process_one_msg`?
2. **Hardware and topology.** Which arm runs: EP=1 with `EP_SPLIT=2:1` (the current default) or TP=2? What are RAM, swap and NVMe, and what do `free -g`, `swapon --show` and `arc_summary` show under load on that arm?
3. **Table codec.** fp8 E4M3 with a global scale, or int6-g32? What is the exact scale and any `scale_override`? Is `MINISGL_PLE_META_FILES` set? Where does the table live (mmap from NVMe, with the ZFS ARC)? Is the served table file a public HF repo (so rows can be range-fetched on a cloud pod)?
4. **Body.** Is the expert codec NVFP4 or MXFP4, and which checkpoint and HF commit are served? Which tokenizer and chat template?
5. **Caches.** Is `--gdn-radix` on, and what is `max_rec_snapshots`? Is MTP speculative decoding on? Is traffic single-user, so drain-and-flush on every edit is acceptable?
6. **Control plane.** Is it acceptable to add a `/ple/*` control plane, including the forward-only `/ple/score` endpoint, behind `MINISGL_PLE_EDIT=1` and an overlay directory `MINISGL_PLE_OVERLAY`? Should edits persist across restarts?
7. **Hidden states.** Can the serve export layer-1 hidden states (for an optional value map)?
8. **Solve venue.** Is a cloud solve (M2-lite on stock bf16; then (d2) with an ~84 GB body download) acceptable, or scheduled downtime of about 10 min plus the session?
9. **Table placement.** Under the running arm, is the PLE table replicated per rank, column-sharded or row-sharded? This decides how the overlay is sliced and what the per-rank `list()` hash covers.

---

## 8. Milestones

| M | What | Where | Pass | Kill / stop |
|---|---|---|---|---|
| **M0** | Package skeleton (§3.1) with `requirements-dev.txt` (pytest, pytest-xdist). Port the P/ proofs into `tests/`: hashing parity (I4) against real buffers; native locality, decode and gradient on tiny qwen4_exp; graft no-op, decode, padding and gradient on tiny qwen3_5, qwen3_5_moe and gemma4 (26B- and 31B-like); graft built **without** `Qwen4ExpTextConfig`; unigram-head subclass (no-op, gradient, heads 0–15 row-id parity); graft-beside-native on tiny qwen4_exp; graft `crop()` / speculative-decode test; Gemma PLE locality, alias, and the corrected editor-init (post-norm weight nonzero); ledger rollback with overlay materialisation; `ngram-edit/v1` round-trip HF adapter ↔ overlay prototype with bitwise equal logits. Copy small fixtures (headers, hash buffers, reference outputs) into `tests/fixtures/`; replace the `.pt` blobs with `scripts/fetch_fixtures.py` (pinned repo + commit + byte ranges); vendor `R/ple/hashing.py` at a recorded commit or skip cleanly. `scripts/fetch_rows.py` HTTP-range reader. Count CounterFact prompt tokens to replace the illustrative N_fact. Start the 1B-token row-load count. | this container | All invariants bitwise as in P/ outputs; default suite under 10 min on 4 CPUs (measured pieces sum to about 280 s serial, `P/redteam-feasibility/times.txt`); solver proofs (v3 took 273 s) marked `slow` and excluded | none (engineering) |
| **M1** | **First write into the REAL Flash-Next table, through the owner's serve, forward-only.** Overlay patch to `R/ple/source.py:242-244`, drain-and-flush, and a forward-only `/ple/score` (token ids in, full log-softmax at named positions out). Run P0–P2 against the deployment's files. Apply a *synthetic* diff (random direction, norms 0.1×/1×/10× row norm) on the rows of a chosen 2-gram, then measure: I1 on 200 prompts that do not read it; ΔNLL / KL on prompts that do, via `/ple/score` (the real conv-saturation curve); I3 after revert; P5 cache hygiene. All at bs=1, sequential, fixed chunk size, against a no-edit control under identical conditions. | owner's serve (forward-only); patch built here | I1, I3 and P5 under the §6 serve rule (bitwise where the control is bitwise, else within its measured noise floor; stated); `/ple/score` matches HF on a tiny or shared prompt set; dose-response curve recorded | If I1 or P5 fails: engine integration bug; fix before any solve |
| **M2-lite** (parallel with M1) | 10 validity-gated facts, tier 1, on stock HF bf16 Flash-Next in the cloud, with arms (a) in-place, (b) XK and (c) O-site. Measures iterations per solve. | 4× H100 or 2× H200, stock HF | Efficacy of (a) ≥ 0.9 | K-native can fire here |
| **M2** | **Native fact solve on Flash-Next.** 50 validity-gated CounterFact facts, tiers 1 and 2, λ_KL ∈ {0, 0.0625}, three arms (a)/(b)/(c). Run P3 and P4; ship deltas home; apply through M1's path; evaluate on the serve via `/ple/score`. | (d2) 1× H200 cloud, or (f) local downtime | Mean efficacy (a) ≥ 0.9; subject-preserved generality ≥ 0.6; realised/predicted ≥ 0.7; I1 holds | K-native (with attribution), K-transfer |
| **M2g** (parallel) | Real-weights solver validation on a cheap graft host: Qwen3.6-35B-A3B, then Qwen3.8-27B, with Flash-Next keys; Adam and closed-form paths. Includes the multimodal-class path check, and re-uses M2's edit specs (keys transfer). Optional: fit the value map M from paired layer-1 activations and measure the warm-start gain. | RunPod RTX PRO 6000 / A100 | I1 and I2 on real weights; efficacy ≥ 0.9; within 0.1 of the O-site oracle | K-graft |
| **M3** | Leakage, short subjects and native-vs-graft: held-out SSOR, lexical neighbours, NS+, alias/surface cohort, short-subject cohort per tier and length, the λ_KL sweep, tier ablation, unigram-order grafts (with their leakage), per-row vs per-n-gram penalty at equal fit, and the controlled native-vs-graft comparison on Flash-Next. | same venues as M2 / M2g | Locality bars of §6 on held-out templates; 1-token generality ≥ 0.6 (graft) | K-leak, K-native-short |
| **M4** | Gemma E4B native (W2) with aliases: 200 facts; layer selection 0–22; matcher false-fire/miss rates; SWEA baseline. | 1× RDNA4 card (during serve downtime, via `gpu-lease`) or cloud A100; E2B on CPU here as a fallback (needs the owner's go-ahead) | Efficacy ≥ 0.9; matcher bars of §6; locality bars on lexical neighbours; row-norm trust region respected; beats SWEA on generality at equal locality | Efficacy < 0.5 at the trust region: Gemma PLE is not a fact store |
| **M5** | Gemma 26B/31B grafts: Flash-Next-format (Gemma keys) vs Gemma-PLE plus aliases vs SWEA, A/B/C. | RunPod A100 | Pick the format with the better generality at equal locality (matcher errors counted against the alias format) | K-graft for both formats |
| **M6** | Scale: sequential 100 → 2k (→ 5k, 10k) edits with conflict re-solving; same-subject multi-relation cohort (k=2..5); a stress run at ≥ 1e4 anchors with real corpus loads; hit-conditioned capability retention; GRACE/WISE/NeuralDB baselines. | cloud (solve; mostly on a graft host) + serve (eval) | Early-edit retention on held-out paraphrases ≥ 0.8 at 2k; multi-relation retention ≥ 0.8; hit-conditioned capability ≥ 96% at 5k | K-scale: fall back to the exact-key overlay (XK mode) if M2 attributed failures to hashing |
| **M7** | Steady-state serve integration: targeted radix detach plus epoch guard, persistence, `/ple/list` per-rank hash agreement (P6), batched edits with forward-only acceptance. | owner's serve | P5 holds with targeted detach equal to drain-and-flush; no serve regression | — |

M1 deliberately comes before any serve-side solve. It edits the real native table with no gradient machinery, tests the engine contract, and measures how the real model's output responds to row-delta magnitude through the conv-dominated path. M1 cannot decide the thesis, so M2-lite runs in parallel on stock HF in the cloud: it is the cheapest experiment that can fire K-native.

---

## 9. Risks, open questions, prior art

### Risks

1. **Conv dominance on native Flash-Next.** Small edits cannot steer the direction-only conv term. Row edits reached only R² 0.29–0.36 with the conv on (`P/t1/e_express.out`), but with proxy queries, not real layer-1 states. XK does not remove this limit (§5.6). The real-weight answer comes from M1 (dose-response) and the M2-lite/M2 arm (c). *Mitigation:* larger κ, accepting the conv spill to t+3, t+6, t+9 (measured at 1–8% of the target size).
2. **Shared bigram rows leak.** A 2-gram edit fires in every context containing the bigram. The prior reference for same-subject leakage is 19–25%. *Mitigation:* KL term, right-crossing keys, per-row full-load penalty; measured on held-out templates and lexical neighbours.
3. **Short subjects have no subject-only native key.** One third of the subjects are 1 token. Left-crossing keys failed before (0.000 for 1-token subjects). *Mitigation:* the unigram extension (grafts only, unproven) and right-crossing keys; K-native-short decides.
4. **Collisions as edits accumulate.** About 0.8% of positions are touched at 1e4 anchors and 7.7% at 1e5, under a uniform assumption. *Mitigation:* the exact-key overlay (XK mode).
5. **Solve/serve numeric mismatch.** fp8 table, fp4 experts and fp8 activations. Gated by P3 and P4.
6. **Stale prefix caches.** Contract plus invalidation (§3.4).
7. **Hardware contention.** The owner's GPUs are probably busy serving, and are shared across repos through `gpu-lease`.
8. **Engine support for grafted serving** (vLLM, SGLang, llama.cpp) does not exist, and a graft is not a valid qwen4_exp config (hc_count=1). Grafts are HF-only until a model-class patch is written.
9. **Cost.** The corrected campaign estimate is $1.1k–14k (§7), dominated by sequential scale runs.
10. **Ephemeral evidence.** The binary fixtures still live only in the session scratchpad; M0's fetch script removes the dependency.
11. **Security.** A writable table is a backdoor surface (BadEngram). Patches carry provenance and fingerprints; the serve's control plane must be authenticated (**owner decision**).

### Open questions

- Real layer-1 query states and gate sparsity on Flash-Next.
- Whether the fitted Flash-Next → host value map M beats re-solving per host.
- Which reset token Gemma grafts should use (`<eos>` vs `<end_of_turn>`).
- Whether to calibrate the graft gate and conv (dead point at `W_k = 0`).
- Whether alias handling belongs in the tokenizer or the serving loop.
- How many iterations a solve needs (drives cost; M2-lite measures it).

### Prior art

From `docs/research/engram-relaunch-proposal.md` and the reviews; arXiv abstracts re-read for BadEngram, EngramEdit and LongCat-Flash-Lite only.

- **EngramEdit (2610.10533).** Edits the natively pretrained n-gram memory of LongCat-Flash-Lite (2601.21204). Two-stage: δ* with the model frozen, including an essence term and a norm clamp, then regularised least squares into n-gram rows with a reuse and frequency penalty. App. A.3.2 covers the gated case; B.4.3 describes exact-key side vectors. The XK mode here is B.4.3, and the per-row penalty here is its A.3.3 frequency penalty moved from per-n-gram to per-row.
- **BadEngram (2609.13478).** Modifies only the memory parameters of Flash-Next's native PLE, with the backbone unchanged, to plant a trigger (47.8% HarmBench ASR). Same table, a trigger-keyed write, optimised through the frozen model.
- **User-as-Engram (2606.19172).** Post-hoc per-user row overlays on an Engram-native base (as described in engram-relaunch-proposal.md:467; abstract not re-read).
- **Cross-Model Memory Transfer (2608.17050).** Moves a frozen Engram memory to a different model with a trained reader.
- **Engram Adapter (2608.29327):** a frozen-base Engram adapter whose residual is about 0.08% of the hidden norm.
- **SWEA (2401.17809):** token-id-keyed subject embeddings at the input. **NeuralDB (2507.18028).**
- **GRACE (2211.11031), WISE (2405.14768), MEMOIR (2506.07899):** discrete keyed edit codebooks on frozen models.

### What would be novel (narrowed)

1. **Hashed co-tenant in-place writes** into a shipped native table, measured against EngramEdit-style exact-key (XK) writes on the same table and the same facts, with the three-arm attribution of §6.
2. A **live-serve diff and cache-invalidation contract**: engine-agnostic, reversible row diffs with an epoch guard and radix invalidation, applied to a running Flash-Next serve.
3. A **controlled native-vs-graft comparison** on one host, with the same keys and solver code.

Not claimed as novel: key transfer across models that share a tokenizer (true of any token-id hash); optimising memory parameters through a frozen model (BadEngram, EngramEdit); frequency-priced penalties (EngramEdit A.3.3).

### Sources

Every script cited inline has a matching `.out`/`.txt`/`.json`/`.log` beside it under `P/`. The folders are:

- `P/t1/`: native Flash-Next.
- `P/t2-gemma4-ple/`: Gemma PLE.
- `P/graft/`: grafts.
- `P/solver/`: V1–V5.
- `P/compute/`: sizes and costs.
- `P/serve/`: serve integration, overlay, requantisation and venue benchmarks (and `INTEGRATION.md`).
- `P/redteam-correctness/`, `P/redteam-feasibility/`: review re-runs and checks.
- `P/review-apply/`: the corrected N, FLOP, overlay-size and cost budget.

Prior evidence: `/home/user/memory-organ/docs/research/gemini-engram-code-review.md` (§5, §8), `/home/user/memory-organ/docs/research/engram-relaunch-proposal.md`, `/home/user/memory-organ/docs/research/engram-graft-assessment.md`.

---

## Review log

Each finding was checked against the cited code, outputs or documents before it was applied. "C" = correctness review, "S" = science review, "F" = feasibility review.

| # | Finding | Verdict | Check / reason |
|---|---|---|---|
| C1 | Gemma-PLE graft init with post-norm weight 0 is dead | **applied** (§4.2, §4.4) | g4:212-222 multiplies by `weight`; `checks.out [3]` 0.0 vs 2.9e-1; `tiny_ple.py:339-343` used post-norm 1.0 |
| C2 | Unigram extension not buildable from the native module; widths don't divide | **applied** (§2.4) | q4:1079, 1083, 1154; `checks.out [2]`; v2_allocation uses first primes |
| C3 | `hc_count=1` outside validated configs | **applied** (§1, §2.2, §2.4, §3.1) | q4c:195-196; `graft_proto.py:175-176` SimpleNamespace |
| C4 / F11 | Overlay sizes use width 160 for all hosts | **applied** (§4.5) | q4:1083; recomputed in `P/review-apply/budget.out` |
| C5 | Locality evidence overstated | **applied** (§4.1) | `bcd.rerun.out`: 100/100 is set agreement; 47/47 no-hit bitwise; after-hit min 8.2e-4 |
| C6 / F12 | FD and autograd values swapped | **applied** (§4.1) | `bcd_proofs.out`: "autograd 1.27628e-03 vs FD 1.27670e-03" |
| C7 | §3.3 vs §3.4 materialisation conflict | **applied** (§3.3, §3.4, §5.4) | `requant_survival.txt` int6 0.00 / 0.88 |
| C8 | `divisor` conflates two config fields | **applied** (§2.1, new §2.1b) | q4:1103-1104, q4c:154, 156 (default 512), cm:2136 |
| C9 | Padding rule wording | **applied** (§2.1) | q4:1465-1470 uses `conv_mask`, not token value |
| C10 | No-op reasoning wrong | **applied** (§2.4) | q4:1197 `value_proj` bias=False |
| C11 | TP slicing rule is HF's | **applied** (§3.4, owner Q9) | q4c:99, tp:650 |
| C12 | Graft `crop()` untested | **applied** (§4.3, M0) | grep for `crop`/`activate_past_recording` over graft/t1 proofs finds nothing |
| C13 / F6 | minisgl pin stale | **applied** (header, §6, §7) | local HEAD `df29117`, 293 commits after `b98eb99`; ple/, radix_cache, config.py unchanged; `inference_mode` at scheduler.py:1369; headroom at serve.sh:1119-1123 (TP=2); EP default at serve.sh:1155-1158; gpu-lease in fork CLAUDE.md |
| S1 | K-native remedy (XK) cannot fix an expressivity failure | **applied** (§5.6, §6 K-native, M2-lite, M2) | XK uses the same module; three-arm attribution plus Wilson bound on the mean |
| S2 | Leakage metric trained on; K-leak threshold unfalsifiable | **applied** (§5.1, §6 bars, K-leak) | gemini review:207 (in-context 0.028), :200-202 (A1 0.194–0.250) |
| S3 | CounterFact NS vacuous; lexical neighbours missing; NS+ no bar | **applied** (§6 cohorts, bars, K-leak) | follows from I1; relaunch proposal:500 |
| S4 | Capability bar near-guaranteed and misquoted | **applied** (§6 metrics, bars, K-scale) | engram-graft-assessment.md:56 (5,000 edits) |
| S5 | Synthetic "fit" and collateral relabelled as generality | **applied** (§2.4, §5.1, §5.3) | v2_allocation.py:61-66; v2.out check4 and train fits |
| S6 | Free δ wrongly called an upper bound | **applied** (§6 priors, oracles, K-graft, M2g) | gemini review:291-296 correction |
| S7 | No controlled native-vs-graft comparison | **applied** (§1, §6, M3, §3.1) | graft-beside-native marked unverified, tested in M0 first |
| S8 | Alias / surface generality not measured | **applied** (§5.1 dedup, §6 cohorts and metrics) | engram-graft-assessment.md:96-97; gemini review:300 |
| S9 | Native short subjects have no bar/kill | **applied** (§5.2, K-native-short) | gemini review §8 (0.066 / 0.000) |
| S10 | Sequential test misses same-subject conflicts; train-set retention; kill/pass gap | **applied** (§5.1, §6, K-scale, M6) | tier-1 keys are relation-blind by construction |
| S11 | Novelty claims overlap prior art | **applied** (§9) | BadEngram and EngramEdit/LongCat abstracts re-read (arXiv); User-as-Engram and Cross-Model Memory Transfer from the relaunch proposal, abstracts not re-read. The old "in-place native writing is untested" sentence is removed. |
| S12 | Baselines incomplete | **applied** (§6 baselines) | MEMIT/AlphaEdit feasibility on MoE/GDN marked unverified |
| S13 | Gemma alias KL ≈ 0 tautological; alias ≈ SWEA | **applied** (§4.2, §6, M4, M5) | proof f fires only on aliased ids by construction |
| S14 / F4 | M1 bitwise criteria vs batched serve | **applied** (§6 I1/I3, M1) | PERF_PUSH_CONTINUANCE.md:128-136 |
| S15 | SSOR KL bar uncalibrated; ripple mixed in | **applied** (§5.4 per-token KL, §6) | in-context leak KL 0.512 |
| F1 | Iteration N and FLOP formula understated | **applied, partly** (§5.1, §7) | plan_calc.py:34 uses 4·active·N. Full prompts are 674–1548 tokens (`P/review-apply/budget.out`), which matches the review. With the prefix caching of INTEGRATION.md §4.3, which the review did not account for, N is 251–497. Both are budgeted, and FLOP now uses 6·active·N. |
| F2 | Campaign cost inconsistent; no iteration counts | **applied** (§7 cost table) | new total ≈ $1.1k–14k (est.) |
| F3 | No scoring path on the serve; eval throughput unbudgeted | **applied** (§3.4, §7, M1) | api_server.py:415-420, 1637-1638; serve.sh:692-694, 1145-1148 |
| F5 | Thesis-deciding milestone gated on avoidable engineering | **applied** (M2-lite, §7 venues) | `hub_quant.out`: non-table files 255.7 GB; NVFP4 repo public at 135.2 GB. "Rows shipped from home" is kept as a fallback if the owner's served table repo is not public. |
| F7 | Corpus pass has no budget | **applied** (§5.3, §7, M0) | `count_bench.out` 0.21 M tok/s |
| F8 | Evidence lives in ephemeral /tmp | **applied** (proofs/ copies, M0 fixtures and fetch script) | .pt blobs and tokenizers not copied (size); scripts still contain scratchpad paths |
| F9 | pytest missing; slow proofs in the default suite | **applied** (§3.1, M0) | `import pytest` → ModuleNotFoundError; times.txt |
| F10 | 178 GFLOP/s uncited | **applied** (§7) | not present in act.out or any output; replaced with `cpu_flops.out` figures, end-to-end marked unmeasured |

No finding was rejected outright. F1 was applied only in part: its N assumed no prefix caching.
