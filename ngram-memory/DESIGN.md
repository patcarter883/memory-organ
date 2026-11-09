# ngram-memory — design

Status: founding design, 2026-11-09. Nothing here is implemented yet. Every number below cites the script that produced it or a code location. Anything not measured is marked **(unverified)** or **(est.)**.

Conventions used throughout:

- **`SP/`** = `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/`. All proof scripts and their outputs live there. M0 ports them into `tests/`.
- **Code citations** are relative to the installed transformers 5.22.2 (`…/site-packages/transformers/`):
  - `q4:` = `models/qwen4_exp/modeling_qwen4_exp.py`, `q4c:` = `models/qwen4_exp/configuration_qwen4_exp.py`;
  - `g4:` = `models/gemma4/modeling_gemma4.py`;
  - `q35:` / `q35moe:` = `models/qwen3_5/modeling_qwen3_5.py` / `models/qwen3_5_moe/modeling_qwen3_5_moe.py`;
  - `cu:` = `cache_utils.py`, `cm:` = `conversion_mapping.py`, `cml:` = `core_model_loading.py`.
- **`R/`** = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/` at commit `b98eb99`. This is the minisgl fork that is our *best guess* for the owner's Flash-Next serve. The owner has not confirmed it.
- **Metric names** follow memory-organ's conventions: validity gate, efficacy, generality, locality. Nothing else from memory-organ (its `cam/` package or its framing) is reused.

---

## 1. Thesis and scope

### Thesis

Language models have started to ship **native n-gram lookup tables**: large embedding tables, keyed on the recent tokens, whose output is added to the residual stream near the bottom of the network.

- **Qwen3.8-Flash-Next** has an Engram-style "PLE" module. It is 51.2 G parameters, 102.4 GB in bf16, and is keyed by hashed 2- and 3-grams (`SP/compute/count.out`).
- **Gemma 4 E4B** has Per-Layer Embeddings. It is 2.82 G parameters and is keyed by token id.

In such a model a fact can be added by **writing rows**. An edit is a small, auditable, reversible diff to a named tensor. It is not a weight-space edit spread through MLPs.

The durable asset is therefore a **portable table editor**: one key/row addressing scheme, one solver, one edit-record and patch format, and one evaluation harness. Each target model plugs in through a thin backend.

Models without a table get a **graft**: a module in the same format, zero-initialised so it is an exact no-op, after which the same editor writes into it. Grafting is the only model-specific part. Everything else carries over unchanged to any model that ships a compatible table.

### Targets

| Tier | Model | Table | Key | Source |
|---|---|---|---|---|
| Native | Qwen/Qwen3.8-Flash-Next (`qwen4_exp`; 48 layers; d=2560; 512 experts top-10; GDN hybrid; 360 GB) | `layers.1.ple`: 16 heads × ~20M prime-sized rows × 160 | hashed 2-grams (8 heads) and 3-grams (8 heads) | §2 |
| Native | google/gemma-4-E4B(-it) | `embed_tokens_per_layer` [262144, 42·256] | token id, per layer | §4.2 |
| Graft | Qwen/Qwen3.8-27B (`qwen3_5`, 64 layers, d=5120) | Flash-Next format, sparse | **same keys as Flash-Next** | §4.3 |
| Graft | Qwen/Qwen3.6-35B-A3B (`qwen3_5_moe`, 40 layers, d=2048) | Flash-Next format, sparse | **same keys as Flash-Next** | §4.3 |
| Graft | google/gemma-4-26B-A4B-it, gemma-4-31B-it (`gemma4`, PLE disabled) | Flash-Next format with a Gemma key instance (primary), or Gemma-PLE format with alias ids (alternative) | own tokenizer | §4.4 |

### Out of scope

Training any weights other than table rows (a graft's fixed init is part of the format); retrieval or in-context memory; multimodal keys (text `input_ids` only); weight-space editors except as baselines; memory-organ's `cam/` code; building engines (we define a patch interface and write small overlay adapters).

---

## 2. The format spec — the contract shared by native tables, grafts and the editor

Derived from `q4` and checked byte-for-byte against the real checkpoint's hash buffers (`SP/t1/fetch_buffers.py`: `MATCH`, `sizes match: True`, `offsets match: True`). minisgl's independent `R/ple/hashing.py` is bit-identical to HF on 8,112 real-parameter ids including chunked prefill and EOS edge cases (`SP/serve/ple_path/hash_parity.py`).

### 2.1 KeySpec

Parameters: `(V, eos, seed, ple_layer_index, ngram_size, heads_per_ngram, base, divisor)`. Flash-Next's values are `(248320, 248044, 1234, 0, 3, 8, 20_000_000, 128)`.

1. **Multipliers.** For `i = 0..ngram_size−1`:
   - `m_i = 2·(splitmix64(seed + 10007·ple_layer_index + GAMMA·(i+1)) mod ⌊⌊(2^63−1)/V⌋/2⌋) + 1` (q4:1033-1050).
   - Flash-Next's values are `[23703573157769, 20109073645365, 8052911324071]`.
   - The bound keeps `x·m < 2^63`, so int64 products never overflow.
2. **Heads.** There are `(ngram_size−1)·heads_per_ngram = 16` heads: heads 0–7 hash the 2-gram and heads 8–15 hash the 3-gram. **There is no unigram head**; the loop starts at n=2 (q4:1154).
3. **Head sizes.** Head h has `p_h` = the k-th prime after `base−1`, with `k = 16·ple_layer_index + h + 1` (q4:1091-1096). This gives 20000003 … 20000171. Offsets `off_h` are prefix sums. The total is 320,001,446 rows, padded to a multiple of `divisor` (320,001,536).
4. **History.** `x_k` is the token k positions back within the current EOS segment. Wherever that crosses an EOS or the start of the sequence, `x_k := eos` (`_shift_right_ignore_eos`, q4:1107-1123).
   - Only `eos_token_id = 248044` (`<|endoftext|>`) resets.
   - `<|im_end|>` (248046) does not reset, so n-grams span chat turns.
   - Pad ids are replaced by eos before hashing (q4:1467-1470).
5. **Mixing.**
   - `mix2 = (x0·m0) XOR (x1·m1)`, `mix3 = mix2 XOR (x2·m2)`.
   - `row_h = off_h + (mix_n mod p_h)` (q4:1154-1166).
6. **Layout.**
   - The logical table is `[padded_rows, row_dim]` with `row_dim = ple_embed_dim / 16`; for Flash-Next that is 160 (2560 values per position).
   - On disk it is stored as `divisor` equal row shards named `…ngram_embedding.shard_{s}.weight`, each [2,500,012, 160] bf16, in files model-00005…00037.
   - HF concatenates the shards in numeric order at load (cm:2131-2140; cml:1319-1337, 1701) and re-chunks them on save (cml:112-143).
   - Address of a row: `shard, local = divmod(row, 2500012)`; byte offset `8 + header_len + data_offsets[0] + local·320` (`SP/t1/a_rows_for.out`).
7. **Second PLE layer.** A second PLE layer would use `ple_layer_index = 1`, which gives new multipliers and the next 16 primes.

The fingerprint of a KeySpec is `sha256(V, eos, seed, ple_layer_index, multipliers, head sizes)` plus the tokenizer.json sha256. Every edit record and every patch carries it.

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

### 2.3 Decode-state spec

The layer-1 cache entry holds `number_of_conv_states = 3` slots (q4c:171):

- slot 2: the last 2 token ids, eos-seeded (q4:1127-1148);
- slot 1: the last 9 positions of `N_conv(u)`, each 4·d values (q4:1210-1226; cu:1192-1230).

An engine that caches only attention KV and the GDN state is wrong: wiping these two slots changed logits by up to 0.32 (`SP/t1/bcd_proofs.out`). **Any cached prefix whose tokens read an edited row is stale after an edit**, because slot 1 and every downstream KV page embed row values.

### 2.4 Graft profile ("editor init")

A graft is a ModuleSpec instance with:

- `hc_count = 1`. This is the one documented simplification: one stream and one gate.
- `ple_embed_dim = host d`.
- table = 0, `W_k = 0` (gate ≡ 0.5), `conv1d = 0`, norm weights = 0 (scale 1), `W_v` = fixed seeded random orthogonal matrix.

Then `out_t = 0.5·W_v e_t`, which is **linear in the rows and independent of the query**. The minimum-norm rows for a target residual change δ are `e = 2·W_vᵀδ`.

- **No-op.** An empty table gives exactly zero output for *any* projections, because `key_proj(0)`, `norm(0)`, `conv(0)` and `silu(0)` are all 0. This was bit-identical in fp32 and bf16 on all four hosts (`SP/graft/results_all.json`, `SP/graft/bf16_noop.out`).
- **Format extension (graft-only, flagged in the spec).** `orders = {1,2,3}` adds 8 unigram heads (`mix1 = x0·m0`) at primes k = 17…24 — the range a second PLE layer would use, which grafts never have. In a synthetic allocation, this raised held-out fit for 1-token subjects from 0.05–0.17 to 0.66–0.81 (`SP/solver/v2.out`). Native tables cannot have it.
- **Format option.** `ple_embed_dim = 2·d` lets 2-gram-only writes reach the whole residual. A 2-gram-only write reached 0.46 of a random target at width d (`SP/graft/results_0.json`). Reasoned, not tested end to end.

### 2.5 Gemma-PLE format (second table kind)

Key = token id t and layer l. Row = `W[t, 256l : 256(l+1)]`. The module math is in §4.2. The editor treats it as `KeySpec(kind="unigram", V=262144)` with `n_layers` value slots per key.

---

## 3. Architecture

### 3.1 Package layout

```
ngram-memory/
  DESIGN.md
  ngram_memory/
    spec.py          KeySpec / TableSpec / ModuleSpec dataclasses, fingerprints, JSON (de)serialisation
    hashing.py       transformers-free rows_for(ids)->[T,H] rows, rows_for_ngram(), checkpoint_loc();
                     port of SP/t1/src/ngram_format.py; cross-checked against q4 and R/ple/hashing.py
    tables.py        RowStore implementations:
                       HFTensorStore      (a loaded nn.Embedding weight)
                       ShardMmapStore     (safetensors shard files, pread by byte offset; never loads the table)
                       SparseRowStore     (sorted row ids + fp32 values; graft storage; SP/graft/graft_proto.py:52)
                       QuantRowCodec      (fp8-global-scale / int6-g32 decode, matching the serve)
    ledger.py        EditRecord, RowLedger, materialise(), rollback(), conflict sets
    patch.py         ngram-edit/v1 manifest + safetensors payload; apply/verify against any RowStore
    backends/
      base.py        Backend protocol (3.2)
      qwen4exp_native.py
      gemma4_ple.py   (+ alias.py: reserved-id alias manager)
      graft/
        module.py     editor-init PLE built from q4.Qwen4ExpTextPLELayer on meta + SparseRowStore
        qwen35.py     hooks + cache slot extension for qwen3_5 / qwen3_5_moe
        gemma4.py     hooks + side LinearAttentionLayer for gemma4
    solver/
      prompts.py     X_rw / X_ess / X_col / X_ret builders
      anchors.py     candidate-row policy (tiers), corpus row-load counters
      penalties.py   per-row output-space ridge + reuse/collateral weights
      rowsolve.py    projected Adam in row space through the frozen model; GN warm start; joint ridge for linear grafts
      quantise.py    bf16 / serve-codec straight-through materialisation
    serve/
      hf_adapter.py      overlay hook on the embedding output (also the graft runtime)
      minisgl_adapter.py client for the /ple/* control plane + overlay patch for R/ple/source.py
      offline_patch.py   in-file shard patching (bf16 tables only; refused for quantised tables)
    eval/
      cohorts.py  metrics.py  parity.py  capability.py  report.py
  tests/   tiny-config proofs (M0), one file per SP/ proof
  scripts/ solve.py, apply.py, evaluate.py, fetch_rows.py (HTTP-range row fetch)
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

Design rule: **gradients reach only a sparse leaf added after the gather**. No backend ever makes the table trainable; on Flash-Next a full-table fp32 gradient would be about 205 GB (`SP/compute/count.out`). This pattern was checked on a tiny Qwen4Exp with GDN layers, and per-layer checkpointing gave exactly the same gradient:

- 561 trainable rows out of 16,896;
- finite-difference relative error 4.7e-6;
- gradient-checkpointing difference 0.0 (`SP/serve/v2-solve-venue/autograd_coverage.txt`).

### 3.3 Edit record and ledger

An `EditRecord` holds: `edit_id`, time, author; `spec_fingerprint`, model repo and commit; the fact `(s, r, o_true, o_new)` and prompt sets; `keys` (token-id n-grams with tier); `rows` (tensor, row, col0, ncols) with `base_bits_sha256` each; fp32 `delta`; solver config and solve-time metrics; `parent_epoch` and `conflict_set` (earlier edits sharing rows).

The `RowLedger` maps row → [(edit_id, fp32 delta)] and saves each row's original bits on first write. Materialisation is `codec(orig + Σ active deltas)` in canonical edit-id order. Naive in-place subtraction is not exact: after sequential bf16 edits it left residue on 50/64 rows (max 4.9e-4), against 0.0 for ledger re-materialisation (`SP/solver/v5.out`). Removal is exact as state, but later edits were solved conditioned on it, so its conflict set is re-validated.

### 3.4 Engine-agnostic patch interface `ngram-edit/v1`

Manifest (JSON): model repo and commit; `table_spec` (tensor-name template, rows per shard, row width, base codec); `key_spec`; `mode` (`delta` for native, `absolute` for grafts); a fingerprint of the dequantised base rows; per-edit provenance. Payload (safetensors): `row_ids` int64 [R], `values` fp32 [R, W].

Served row = `codec_decode(base[r]) + delta[r]`, then the engine's own cast. **Edits are never re-quantised into a quantised base table.** Measured on 8,192 real bf16 rows (`SP/serve/v2-deltaverify/requant_survival.txt`):

| Delta size (× row norm) | bf16 fp32-overlay keep | int6 in-file keep |
|---|---|---|
| 0.003 | 0.98 | 0.00 |
| 0.03 | 1.00 | 0.88 |
| 0.1 | 1.00 | 0.98 |

The serve's fp8 global scale (about 2e-4) also clips values above about 0.09 (`SP/serve/INTEGRATION.md` §3.5).

Operations: `apply(patch) → epoch`, `revert(edit_id)`, `list()` (per-rank compiled hash), `read_rows(ids)`, `invalidate(row_ids | all)`, plus an **epoch guard**. The guard stamps requests at admission, and a request admitted before an edit cannot insert prefix-cache state that read edited rows.

Adapters:

1. **HF:** an overlay on the embedding output. This is also the graft runtime.
2. **minisgl:** an fp32 overlay applied right after `decode_raw` in `PLEEmbeddingSource.finish_rows` (`R/ple/source.py:242-244`), before the H2D copy.
   - Graph-captured buffers keep their pointers (`R/ple/graph_capture.py:132-147`).
   - Measured overhead is 7–24 µs per decode call and 1.5–2.6 ms per 2048-token prefill chunk at realistic hit rates (`SP/serve/integration/bench_overlay_hits.txt`).
3. **Any engine with an unquantised table on the device:** row `index_copy_` over RPC, sliced by columns under tensor parallelism (Flash-Next's TP plan is `embedding_colwise`, q4c:99).
4. **Offline:** patch bf16 shard bytes at the §2.1 byte offset, then restart. Allowed only for bf16 tables.

Cache invalidation is part of the contract. A radix node is stale if any position on its path reads an edited row id. Nulling the recurrent snapshot alone re-adopts stale KV (`R/kvcache/radix_cache.py:299-312`; `SP/serve/INTEGRATION.md` §3.4). Version 1 therefore **drains traffic and flushes the whole tree**. Targeted detach plus the epoch guard is the steady-state design.

---

## 4. Backends

### 4.1 `qwen4exp_native` (T1)

- **Tensor.** `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight`.
- **Loading.** Never `from_pretrained` with the table. HF concatenates it on CPU (cm:2133-2139) and skips it from `device_map` (`_no_placement_params`, q4:1321-1324). Build the model on meta, load the body, and replace `ngram_embedding` with `ShardMmapStore` (or the serve's codec files) plus a sparse delta leaf. The lookup already moves results back to the model's device (q4:1169-1174).
- **Rows per anchor.** 16 (5 KB in bf16). Gradient reaches exactly the rows the prompt reads: 380 of 380 in the tiny model, with finite differences at 1.27628e-3 against autograd at 1.27670e-3 (`SP/t1/bcd_proofs.out`).
- **Locality.** Exact at the PLE output: the hit positions plus +3, +6, +9 through the dilated conv. Logits are bit-identical elsewhere (100/100 sequences, `SP/t1/bcd_proofs.out`).
  - Real-size per-head collision rate is about 1/2e7 per position.
  - The real collateral comes from *the same bigram in any other context*, which hits all 8 2-gram heads.
- **Decode.** Cached decode equals a full forward with an edited table to 2.7e-8 (`SP/t1/bcd_proofs.out`). The HF cache handles state natively; the serve's slot state (`R/ple/state.py`) handles it there.
- **Real-weight facts that shape the solver** (`SP/t1/real_stats.out`, `SP/t1/e_express.out`):
  - RMS(v) ≈ 0.0058 vs 0.0042 for a token embedding: PLE is a first-order input, not an adapter. `W_v` is near full rank (2541/2560); 2-gram head slices are 2× the 3-gram ones (Frobenius 12.3 vs 6.3).
  - The conv's current-position tap on stream 1 (mean |w| ≈ 0.071) depends only on the direction of v and is about 10× embedding scale at the anchor (2.14 vs 0.21).
  - Row edits reach R² 0.29–0.36 of 0.2–1× targets with the conv on, 0.80–0.91 with it off, 0.89 only at 25× (proxy queries). Per-stream targets cap at about 1/4: the reachable change is `(σ1·v′, …, σ4·v′)`. So the solver optimises final NLL in row space, not a free residual target.

### 4.2 `gemma4_ple` (T2)

- **Tensor.** `model.language_model.embed_tokens_per_layer.weight` [262144, 10752] bf16, 5.64 GB.
- **Math** (g4 lines; E4B: H=2560, D=256, L=42):
  - token term `τ_l = 16·W[t, lD:(l+1)D]` (g4:1580-1585, 1443-1444, 1740-1744);
  - context term `c_l = RMSNorm((W_M·√H·E[t])/√H)_l` (g4:1587-1593, 1768-1774);
  - combined `π_l = (c_l+τ_l)/√2` (g4:1586, 1779);
  - per layer `h ← (h + N_{w_l}(P_l(gelu_tanh(G_l h) ⊙ π_l)))·layer_scalar_l` (g4:1358-1363, 1420-1429).
  - `π_l` depends on t alone: a captured per-layer input matches `effective_ple(t)` within 8.3e-7.
- **Write limits.** The post-norm fixes the branch norm (11.33–11.37 while the row delta sweeps 0→1000, `SP/t2-gemma4-ple/tiny_results.json`): edits rotate, never grow. Layers 24–41 share KV (`num_kv_shared_layers=18`; g4:1171-1175, 1225-1229, 1241-1244; `e4b_meta.py`), so a non-final-position write reaches later tokens only through layers 0–22; the tiny analogue's shared layers had exactly zero downstream effect. Row 0 (pad and multimodal, g4:2316-2320) is never edited.
- **Decode.** `per_layer_inputs` are recomputed from each step's ids (g4:1637-1640, 2683-2685). No extra state is needed. Cached and full decode agree to 4.1e-5 (`SP/t2-gemma4-ple/tiny_run.log`).
- **Unigram problem → alias ids.** A row fires on every occurrence of its token. Gemma 4 has 6,227 reserved `<unusedN>` ids with untrained E4B PLE rows (std exactly 1/16). A serving-loop matcher rewrites the subject's last token in context to an alias whose main and PLE rows start as copies; installing it is an exact no-op and edits then change only aliased occurrences (proof f, `SP/t2-gemma4-ple/tiny_results.json`). The tied output head masks alias columns.
- **Capacity (tiny, random weights; `SP/t2-gemma4-ple/capacity_results.json`).** Row edits ranked between a free residual delta and a norm-clamped one. Optimised rows grew to norm 58–108 vs about 2.4 natively, destroying the token's own content, so row trust regions are mandatory. Real efficacy is **unverified**.

### 4.3 `graft/qwen35` (Qwen3.8-27B, Qwen3.6-35B-A3B)

- **Module.** `q4.Qwen4ExpTextPLELayer`, imported and built on meta, with the editor init from §2.4 and `SparseRowStore`. The sparse store matched a dense zero table with the same rows written to max |Δlogit| 0.0 (`SP/graft/test_graft.py`).
- **KeySpec.** Exactly Flash-Next's: V=248320, eos=248044, `ple_layer_index` 0. The graft's buffers equal the real checkpoint's (`SP/graft/fetch_fn_buffers.out`), so **row ids are identical to Flash-Next's** and do not depend on hidden size.
- **Tokenizers** (`SP/graft/tok_check.out`):
  - Qwen3.8-27B's tokenizer files are byte-identical to Flash-Next's.
  - Qwen3.6-35B-A3B differs only by 7 extra Flash-Next special tokens (ids 248070–248076). 310 of 310 texts gave identical ids.
  - So Flash-Next keys are valid on both hosts.
- **Injection.** A forward-pre-hook on `layers[1]` (a GDN layer, as in Flash-Next) adds the output before the sublayer norm. A top-level pre-hook captures `input_ids` and the 2-D mask, because `ForConditionalGeneration` calls the language model with `input_ids=None` (q35:1644, q35moe:1754).
  - The padding mask reproduces `create_recurrent_attention_mask`; it was equal to the host's GDN mask.
  - **Unverified:** the multimodal class path itself. Only `ForCausalLM` was tested.
- **Cache.** Extend layer 1's `LinearAttentionLayer` from 1 to 3 state slots in place (`SP/graft/graft_proto.py:96`). The PLE uses slots 1 and 2 as natively (q4:1128, 1146, 1216), and beam reorder, crop and reset work through the normal cache code. Cached and full decode agree to 1.8e-7, against a host baseline of 2.1e-7.
- **Value transfer from Flash-Next.** Keys and edit specs transfer; values do not (trained `W_v`, query gates and 4 streams vs d=5120/2048 with a fixed orthogonal `W_v`). A fitted map M (host layer-1 activations regressed on Flash-Next's, aligned position-for-position by the shared tokenizer) is an optional warm start only, since matching shallow states failed before (§6). Default: re-solve per host. **Unverified:** no map was fitted.
- **Gotchas.** `W_k=0` is a dead point: its gradient is exactly 0 through `sign`/`clamp_min` (q4:1241; `SP/graft/calib_grad.out`), so v1 does not calibrate gates. MTP draft heads never see the graft, which only lowers acceptance (reasoned).

### 4.4 `graft/gemma4` (gemma-4-26B-A4B-it, gemma-4-31B-it)

- **Primary format.** The Flash-Next format with a **Gemma key instance**: V=262144, eos=1 (`<eos>`), and therefore different multipliers.
  - **To confirm:** the reset token could instead be 106 `<end_of_turn>`.
  - All checks passed on tiny 26B-like (MoE, `k_eq_v`) and 31B-like hosts: no-op bit-identical; cached/full decode 4.1e-6 vs a host baseline of 3.4e-6; left-padded batches match; gradient on 192 of 192 rows read (`SP/graft/results_all.json`).
- **Gemma-specific handling.**
  - `Gemma4Model` passes a dict mask (g4:2422), so the 2-D mask is captured at the top level.
  - `layer_scalar` multiplies the whole residual (g4:1429) and `embed_scale=√d` makes the residual large (g4:1565); per-fact solving absorbs both. In the 26B the router reads the injected residual (g4:1404-1406), so edits also move routing.
  - Layer 1 is a sliding-window KV layer, so a side `LinearAttentionLayer` hangs off the Cache with `reorder_cache`/`reset`/`batch_*` forwarded (`SP/graft/graft_proto.py:126`). `crop()` is not forwarded: speculative decoding must raise until it is.
- **Alternative format (A/B in M5): Gemma's own PLE** via `hidden_size_per_layer_input>0`. Stock `Gemma4ForCausalLM` builds it, reports only the PLE tensors missing, and is an exact no-op with table, `per_layer_model_projection` and post-norm weight at 0 (proof g, `SP/t2-gemma4-ple/tiny_results.json`). Pros: stock-loadable, maybe engine-supported (**unverified**), exact alias keys. Cons: needs the serving-loop alias matcher; dense table (4.0 GB 26B, 8.05 GB 31B at D=256); on/off write strength to calibrate via `w_l`.

### 4.5 Graft physical size

Each host keeps the full logical row space, so primes and row ids are unchanged, but stores only the rows that have been written. Each edited anchor writes 16 rows. Computed by formula, assuming uniform n-grams (`SP/compute/collide.out`):

| Edited anchors | Overlay (fp32, width 160) | Positions hitting an edited row by collision |
|---|---|---|
| 1e3 | 10 MB | 0.08% |
| 1e4 | 102 MB | 0.8% |
| 1e5 | 1.0 GB | 7.7% |

A dense graft at width d would be 204.8 GB in bf16 on the 27B. This is why grafts are sparse.

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

- **X_rw:** the rewrite prompt plus K=4 paraphrases, each also under C=5 prefix contexts. Context diversity raised held-out fit from 0.65 to 0.82 (`SP/solver/v2.out`).
- **X_ess:** `"{s} is a"`, `"{s} is"`, `"{s},"`, `"{s} was"`, plus at least 4 same-subject other-relation (SSOR) templates.
- **X_col:** corpus windows that read a candidate row through a different n-gram.
- **X_ret:** rewrite prompts of earlier edits that share rows with this one.
- Prompts are right-padded or batched one per sequence, **never left-padded**. Left padding changes the PLE's padding semantics relative to the serve (`SP/serve/INTEGRATION.md` §2.1).

### 5.2 Anchor policy (n-gram selection)

Subject-token statistics from the real Qwen tokenizer (`SP/t1/coverage.out`): of 30 subjects, 10 are 1 token, 8 are 2, 5 are 3, 5 are 4 and 2 are 5.

- **Tier 1 — subject-internal n-grams.** The window lies wholly inside the subject span, mainly at the last subject token q.
  - A 2-token subject has an internal 2-gram only.
  - A subject of 3 or more tokens has both an internal 2-gram and an internal 3-gram.
- **Tier 2 — subjects with no internal key at q.**
  - (a) Right-crossing (s_last, r1) and (s_last, r1, r2). These are relation-specific, which helps SSOR, but they fail when r1 is " is".
  - (b) Left-crossing (c, s) for the top left contexts covering ≥90% of corpus P(c | s), plus the segment-start (eos, s) key. Example: Paris or Toyota in 9 contexts gave 7 distinct preceding tokens.
- **Grafts only — tier 0.** Unigram heads (§2.4).
- **Gemma.** Prefer the alias of the last subject token. Prefer layers 0–22 with large rms(w_post); real values range 0.05–8.6 (`SP/t2-gemma4-ple/real_stats.json`).
- **Tokenisation varies with position.** "Nokia" is 1 token mid-sentence but 2 at sentence start, so keys are enumerated for every surface form the prompt set produces.
- **Prior evidence** (§6): keys reaching into pre-subject context killed generalisation for short subjects.

### 5.3 Penalties and trust regions

- **Per-row ridge in output space.** The regulariser is `Σ_j λ_j ‖V_j u_j‖² / ‖h̄‖²`, with `λ_j = λ_ridge + λ_reuse·w_j`. Output space is used rather than row space because `W_v` head slices differ by 2×.
- **Reuse weight.** `w_j = min(w_len + w_freq − 1 + γ·occ_j, w_max)`.
  - `w_freq` is the percentile of row j's **collateral load**: corpus counts of *other* n-grams that hash to row j. These come from streaming a 1–10B-token corpus through the real hash into 320M uint32 counters, about 1.3 GB.
  - `occ_j` counts earlier edits on the row.
  - On Gemma, unigram counts are used instead.
- **Starting values.** EngramEdit's defaults: λ_ridge 0.01, λ_reuse 0.05, γ 9, w_freq_max 10, w_max 64.
- **Length weight.** The n-gram length weight is flattened for crossing keys. EngramEdit's η = 8/2/1 pushed 74–81% of the edit into crossing 3-grams and cut 2-token held-out fit from 0.75 to 0.59 (`SP/solver/v2.out`).
- **Per-row versus per-n-gram penalties.** A per-row penalty cut collateral by 25–60% compared with a per-n-gram one.
- **Trust regions.**
  - Residual change: `‖Δm_t‖ ≤ ρ‖h_t‖`, ρ ∈ [0.5, 1], by projection after each step.
  - Native rows also: `‖u_j‖ ≤ κ‖e_j‖`, κ ≈ 10.
  - EngramEdit's clamp relative to the memory output is meaningless for a zero graft, because the memory output is 0 there.

### 5.4 Objective and optimisation

```
min_U  mean_{X_rw} NLL(o_new | x; U)                        all object tokens, teacher-forced, FINAL layer
     + λ_KL  · mean_{X_ess} Σ_{t≥q} KL(p_0 ‖ p_U)            every position from subject end + 3 greedy tokens
     + λ_col · KL on X_col  + λ_ret · NLL of overlapping earlier edits
     + Σ_j λ_j ‖V_j u_j‖²/‖h̄‖²
```

- **W1 native and W2.** Projected Adam in row space through the real frozen model, with gradients only on the sparse leaf.
  - Optional warm start: Gauss–Newton matching of a stage-1 δ* restricted to the reachable family.
  - **Never** use the EngramEdit Eq. 28 one-shot linearisation as the answer. On the real `Qwen4ExpTextPLELayer` with hc=4 it left 0.12–0.23 residual on reachable targets, where GN and Adam reached about 0. On Gemma's gated, normed PLE it left 0.18–0.79 (`SP/solver/v1.out`, `SP/solver/v4.out`).
  - The frozen-gate least-squares solve gives negative R² with the conv on (`SP/t1/e_express.out`).
- **W1 graft.** The module is linear, so the batch closed form `U* = (AᵀA+Λ)⁻¹AᵀB` is exact; it matched L-BFGS to 1.5e-7 (`SP/solver/v2.out`). After it, 5–10 direct refinement steps on the full objective.
  - On a tiny model the direct solve beat the two-stage one: Δlog P of +0.29 against +0.16 (`SP/solver/v3.out`). Directional only.
- **Quantisation-aware finish.** Materialise in the serve's codec (bf16 cast after the fp32 overlay), re-evaluate, and take straight-through steps if efficacy drops. Diffs below about 3% of the row norm partly vanish (`SP/solver/v5.out`).

### 5.5 Multi-token objects

Subject-anchored rows carry all object tokens through the NLL. The fallback is object-internal keys, for example (o1, o2) → o3, or o1 unigrams in grafts.

On native Flash-Next the key for o2 is necessarily relation-crossing, (r_last, o1). These rows usually carry a large collateral load, so the penalty normally forbids them. When they are used, X_col must contain corpus windows of that n-gram.

### 5.6 Sequential edits and rollback

- Batches of about 100 edits, targets recomputed on the current model. The conflict set (earlier edits sharing candidate rows; mostly same subject, other relation) feeds X_ret. Conflict clusters are periodically re-solved from the original table; stored diffs make this order-free.
- **Fallback XK mode (exact-key overlay):** private rows behind a 64-bit n-gram fingerprint check (EngramEdit B.4.3) when collisions dominate. Needs an engine hook rather than a tensor diff; only κ changes in the solver.
- Rollback = ledger re-materialisation (§3.3).

---

## 6. Evaluation protocol

### Priors from earlier work

From `docs/research/gemini-engram-code-review.md`, on frozen Qwen2.5-0.5B:

- **Free delta.** A free layer-2 delta at the last subject token, optimised for **final-layer** NLL, reached:
  - efficacy 1.000;
  - paraphrase 0.875–0.917 (base 0.208);
  - same-subject other-relation leakage of about 21–25%, mostly on `"{s} is a"`.
  - This is an **upper bound** for a layer-1 table, which can express less than a free delta (§4.1).
- **Shallow targets fail.** Matching shallow teacher states fails: the fact arrives at blocks 20–23 of 24, and the block-6 oracle gave P(new) 0.012.
- **Short subjects.** Keys reaching into pre-subject context gave held-out generalisation of 0.066 for 2-token and 0.000 for 1-token subjects (§8 there).
- **Cached decode** must carry the n-gram and conv state.

EngramEdit's own strong results come from an exact-key overlay, not from in-place hashed writes (B.4.3, per `SP/solver/` notes). In-place native writing is untested territory.

### Cohorts

CounterFact (2,000 sequential; 500 for ablations), zsRE (2,000), CounterFact+ prefixed neighbours, SSOR (CounterFact subjects × other-relation templates, `"{s} is a"` reported separately), and a **short-subject cohort**: at least 150 subjects each of 1, 2 and ≥3 tokens per tokenizer, crossed with anchor tier.

### Metrics

- **Efficacy:** P(new) > P(old) on the rewrite prompt; also top-1 and P(new). **Generality:** held-out paraphrases never in X_rw, novel wordings reported separately.
- **Locality:** neighbourhood specificity and NS+; SSOR KL and top-1 = new; generic-text KL per token split by whether the window reads an edited row; collision-window KL. Fluency/consistency as in ROME; zsRE token accuracy.
- **Capability retention:** MMLU/GSM8K/HellaSwag subsets and WikiText perplexity at 0/1k/2k/5k edits. **Sequential curves:** early-edit retention, row occupancy, conflicts.
- **Baselines:** no edit, in-context ceiling, free-δ oracle, ROME/MEMIT/AlphaEdit on graft hosts. Bootstrap 95% CIs throughout.

### Hard invariants (a failure is a bug and stops the run)

- **I1.** Windows that read no edited row are **bitwise** unchanged. This held on all tiny proofs.
- **I2.** Cached decode equals a full forward at the host's own noise floor, over 50 prompts × 32 greedy tokens with identical tokens. This also holds on the live serve after a patch, including for a prefix cached before the edit.
- **I3.** Rollback restores pre-edit outputs exactly (bitwise on HF; within batch-composition noise on the serve).
- **I4.** `rows_for` matches the model's lookups on at least 10k positions, including EOS edge cases.

### Serve-transfer checks (P0–P5, from `SP/serve/INTEGRATION.md` §5)

- Row ids, base row bytes and edit transport are bitwise equal between the solve and the serve.
- Pre-edit top-1 agreement between the solve model and the serve is at least 95%. Otherwise add fp8 activation emulation: the serve's per-token fp8 expert activations have 4.09e-2 relative error per layer (`R/weights/cpu_tier.py:38-40`).
- Realised ΔNLL / predicted ΔNLL ≥ 0.7 on targets.
- Cache hygiene: warm, then edit, then query must equal a cold run.

### Proposed accept bars (unverified; to be revised after M2)

| Metric | Bar |
|---|---|
| Efficacy | ≥ 0.95 |
| Generality, overall | ≥ 0.80 |
| Generality, 1-token subjects | ≥ 0.60 |
| Neighbourhood drop | ≤ 0.05 |
| `"{s} is a"` top-1 = new | ≤ 0.05 |
| SSOR KL | ≤ 0.3 |
| Capability retention | ≥ 96% at 1k edits (EngramEdit's bar) |

### Ablations

λ_KL ∈ {0, 0.0625, 0.25, 1}; anchor tier; per-row vs per-n-gram penalty; in-place vs exact-key overlay (XK mode); ρ and κ; grafts: orders {2,3} vs {1,2,3} and width d vs 2d; Gemma: Flash-Next-format graft vs Gemma-PLE plus aliases.

### Kill criteria

- **K-native.** On Flash-Next, with ρ ≤ 1 and κ ≤ 10, median efficacy is below 0.5 on 50 validity-gated facts across tiers 1 and 2. Conclusion: a layer-1 hashed table cannot hold facts through in-place writes. Stop native in-place editing and move to the XK exact-key overlay, and record the negative result.
- **K-leak.** `"{s} is a"` top-1 = new stays above 0.25 at λ_KL = 1 on every tier. Conclusion: the keying is relation-blind. Restrict to right-crossing keys; if those fail efficacy, stop.
- **K-transfer.** Realised/predicted ΔNLL stays below 0.5 even with fp8 emulation. Conclusion: offline solves are not valid for the serve. Use forward-only acceptance or zero-order steps through the serve, or stop.
- **K-scale.** Retention of early edits below 0.8, or capability below 96%, at 1k sequential edits with conflict re-solving.
- **K-graft.** A graft fails I1 or I2 on real weights, or its efficacy is below 0.5 where the free-δ oracle on the same host exceeds 0.9.

---

## 7. Compute plan

### Sizes (exact, from safetensors headers; `SP/compute/count.out`)

| Target | Weights (bf16) | Table | Active parameters per token | Solve iteration at N=320 tokens (≈4·active·N) | Activations at N=320 with per-layer checkpointing (upper bound, `SP/compute/plan_calc.out`) |
|---|---|---|---|---|---|
| Flash-Next | 360 GB (body 251 GB bf16, 73 GB at ~4.5-bit) | 102.4 GB | 6.67 G | 8.5 TFLOP | 0.59 GB |
| Gemma E4B | 16 GB | 5.64 GB | 4.70 G | 6.0 TFLOP | 0.13 GB |
| Qwen3.8-27B | 55.6 GB | sparse graft | 25.6 G | 32.8 TFLOP | 0.41 GB |
| Qwen3.6-35B-A3B | 71.9 GB | sparse graft | 2.95 G | 3.8 TFLOP | 0.18 GB |
| gemma-4-26B-A4B | 51.6 GB | sparse graft | 3.82 G | 4.9 TFLOP | 0.14 GB |
| gemma-4-31B | 62.6 GB | sparse graft | 30.7 G | 39.3 TFLOP | 0.34 GB |

Weights, not activations, decide the hardware. The table never needs to be resident: a batch reads about 5 KB of rows per token.

### Per-venue plan

- **This container** (4 CPUs, 15 GB RAM, no GPU, about 30 GB disk): M0 — all tiny proofs, the full pipeline, and patch round-trips through the HF adapter and the overlay prototype (`SP/serve/ple_path/ple_overlay.py`). Measured bf16 CPU throughput is about 178 GFLOP/s (`SP/compute/act.out`); use `OMP_NUM_THREADS=1` when the CPUs are shared. Optional: Gemma E2B real weights (9.3 GB text-only fetch, **needs the owner's go-ahead**), est. 14–17 s per iteration — the only real-weights native proof possible here.
- **Owner, 2×16 GB RDNA4 (ROCm).** Gemma E4B native fits one card (about 9.4 GB body + up to 1.9 GB activations, table on CPU; est. 0.2–0.5 s per iteration). Grafted hosts at about 4.5 bits per weight (15–21 GB) need both cards and a differentiable dequant matmul on ROCm (**unverified**). **Conflict:** the Flash-Next serve very likely occupies both cards at TP=2 (`SP/serve/INTEGRATION.md` §4.1, inferred), so local GPU work may need serve downtime.
- **Owner's Flash-Next serve: forward-only.** It runs under `torch.inference_mode` with graph capture (`R/scheduler/scheduler.py:1260`), so it serves the validity gate, evaluation, P0–P5 and forward-only acceptance, and applies patches through the overlay. Measured headroom while serving is 11.1 GiB available with 33.9 GiB swap in use (`tools/serve.sh:964-966`), so no concurrent local HF solves.
- **Flash-Next solve venues** (`SP/serve/INTEGRATION.md` §4.2, `SP/serve/v2-solve-venue/cost_model.txt`; all est.):
  - **(d2)** 1× cloud H200, quantised body dequantised per layer (about 84 GB download; rows shipped from home): 1–6 s per iteration; the serve stays up.
  - **(f)** Local downtime, both cards, experts in RAM: 3.6–30 s per iteration plus about 10 min reboot (`R/weights/config.py:46`).
  - **(b)** Local CPU streaming beside the serve: 32–50 s per iteration at N=256 with a native kernel; overnight steady state only.
  - Expert codec: INTEGRATION.md says NVFP4, `SP/serve/v2-solve-local/bench_dequant2.T4.txt` benchmarks MXFP4 bit-exact against the fork; confirm. A dequantised-expert module matched dense experts exactly (`SP/serve/v2-solve-venue/autograd_coverage.txt`).
- **Cloud, RunPod** ($/GPU-hour community/secure, read 2026-11-09 from a page summary; reconfirm). Graft hosts on 1× A100 80 GB ($1.39/$1.79) or RTX PRO 6000 96 GB ($1.69/$2.49), about $3–12 per proof session. Flash-Next on 1× H200 ($3.59/$5.29) for (d2), or 4× H100 ($10.76/$15.96) for a bf16 body, about $35–65 per session. Whole campaign about $60–120 (est.) plus checkpoint storage.

### Questions for the owner about the Flash-Next serve

1. **Engine.** Is the serve `patcarter883/minisglang-rdna4` at or after `b98eb99`? Has anything changed in `R/ple/source.py` (`finish_rows`), `graph_capture.py`, `radix_cache.py` or `_process_one_msg`?
2. **Hardware.** Does it run on the 2×16 GB RDNA4 cards, with TP=2? What are RAM, swap and NVMe, and what do `free -g`, `swapon --show` and `arc_summary` show under load?
3. **Table codec.** fp8 E4M3 with a global scale, or int6-g32? What is the exact scale and any `scale_override`? Is `MINISGL_PLE_META_FILES` set? Where does the table live (mmap from NVMe, with the ZFS ARC)?
4. **Body.** Is the expert codec NVFP4 or MXFP4, and which checkpoint and HF commit are served? Which tokenizer and chat template?
5. **Caches.** Is `--gdn-radix` on, and what is `max_rec_snapshots`? Is MTP speculative decoding on? Is traffic single-user, so drain-and-flush on every edit is acceptable?
6. **Control plane.** Is it acceptable to add a `/ple/*` control plane behind `MINISGL_PLE_EDIT=1` and an overlay directory `MINISGL_PLE_OVERLAY`? Should edits persist across restarts?
7. **Scoring.** Is there a prompt-logprob or scoring endpoint? Can the serve export layer-1 hidden states (for an optional value map)?
8. **Solve venue.** Is a cloud solve with an 84 GB body download acceptable, or scheduled downtime of about 10 min plus the session?

---

## 8. Milestones

| M | What | Where | Pass | Kill / stop |
|---|---|---|---|---|
| **M0** | Package skeleton (§3.1). Port the SP/ proofs into `tests/`: hashing parity (I4) against real buffers; native locality, decode and gradient on tiny qwen4_exp; graft no-op, decode, padding and gradient on tiny qwen3_5, qwen3_5_moe and gemma4 (26B- and 31B-like); Gemma PLE locality and alias; ledger rollback; `ngram-edit/v1` round-trip HF adapter ↔ overlay prototype with bitwise equal logits. Plus a `scripts/fetch_rows.py` HTTP-range reader for real Flash-Next rows. | this container | All invariants bitwise as in SP/ outputs; `pytest` under 10 min on 4 CPUs | none (engineering) |
| **M1** | **First write into the REAL Flash-Next table, through the owner's serve, forward-only.** Overlay patch to `R/ple/source.py:242-244` plus drain-and-flush. Run P0–P2 against the deployment's files. Apply a *synthetic* diff (random direction, norms 0.1×/1×/10× row norm) on the rows of a chosen 2-gram, then measure: I1 on 200 prompts that do not read it; ΔNLL / KL on prompts that do (the real conv-saturation curve); I3 after revert; P5 cache hygiene. | owner's serve (forward-only); patch built here | I1 bitwise on non-hit prompts; I3 exact; P5 equal to a cold run; dose-response curve recorded | If I1 or P5 fails: engine integration bug; fix before any solve |
| **M2** | **First native fact solve on Flash-Next.** 10, then 50 validity-gated CounterFact facts, tiers 1 and 2, λ_KL ∈ {0, 0.0625}. Run P3 and P4; ship deltas home; apply through M1's path; evaluate on the serve. | (d2) 1× H200 cloud, or (f) local downtime | Efficacy ≥ 0.9; generality ≥ 0.6; realised/predicted ≥ 0.7; I1 holds | K-native, K-transfer |
| **M2g** (parallel) | Real-weights solver validation on a cheap graft host: Qwen3.6-35B-A3B, then Qwen3.8-27B, with Flash-Next keys. Includes the multimodal-class path check, and re-uses M2's edit specs (keys transfer). Optional: fit the value map M from paired layer-1 activations and measure the warm-start gain. | RunPod RTX PRO 6000 / A100, about $5–12 each | I1 and I2 on real weights; efficacy ≥ 0.9; within 0.1 of the free-δ oracle | K-graft |
| **M3** | Leakage and short subjects: SSOR plus the short-subject cohort, the λ_KL sweep, tier ablation, and unigram-order grafts. | same venues as M2 / M2g | `"{s} is a"` top-1 = new ≤ 0.05; 1-token generality ≥ 0.6 (graft) | K-leak |
| **M4** | Gemma E4B native (W2) with aliases: 200 facts; layer selection 0–22. | 1× RDNA4 card (during serve downtime) or cloud A100; E2B on CPU here as a fallback (needs the owner's go-ahead) | Efficacy ≥ 0.9; plain-token contexts KL ≈ 0 (alias); row-norm trust region respected | Efficacy < 0.5 at the trust region: Gemma PLE is not a fact store |
| **M5** | Gemma 26B/31B grafts: Flash-Next-format (Gemma keys) vs Gemma-PLE plus aliases, A/B. | RunPod A100 | Pick the format with the better generality at equal locality | K-graft for both formats |
| **M6** | Scale: sequential 100 → 2k (→ 10k) edits with conflict re-solving and capability retention; collision leakage on real text frequencies. | cloud (solve) + serve (eval) | Retention ≥ 0.8 of early edits at 2k; capability ≥ 96% at 1k | K-scale: fall back to the exact-key overlay (XK mode) |
| **M7** | Steady-state serve integration: targeted radix detach plus epoch guard, persistence, `/ple/list` per-rank hash agreement (P6), batched edits with forward-only acceptance. | owner's serve | P5 holds with targeted detach equal to drain-and-flush; no serve regression | — |

M1 deliberately comes before any solve. It edits the real native table with no gradient machinery, tests the engine contract, and measures the one quantity the tiny models cannot give: how the real model's output responds to row-delta magnitude through the conv-dominated path.

---

## 9. Risks, open questions, prior art

### Risks

1. **Conv dominance on native Flash-Next.** Small edits cannot steer the direction-only conv term. Row edits reached only R² 0.29–0.36 with the conv on (`SP/t1/e_express.out`), but with proxy queries, not real layer-1 states. The real-weight answer comes from M1 (dose-response) and M2. *Mitigation:* larger κ, accepting the conv spill to t+3, t+6, t+9 (measured at 1–8% of the target size).
2. **Shared bigram rows leak.** A 2-gram edit fires in every context containing the bigram. The prior estimate of same-subject leakage is 21–25%. *Mitigation:* KL term, right-crossing keys, per-row collateral penalty.
3. **Short subjects have no subject-only native key.** One third of the subjects are 1 token. *Mitigation:* the unigram extension (grafts only) and left-context coverage on native tables (**unverified** on real text).
4. **Collisions as edits accumulate.** About 0.8% of positions are touched at 1e4 anchors and 7.7% at 1e5, under a uniform assumption. *Mitigation:* the exact-key overlay (XK mode).
5. **Solve/serve numeric mismatch.** fp8 table, fp4 experts and fp8 activations. Gated by P3 and P4.
6. **Stale prefix caches.** Contract plus invalidation (§3.4).
7. **Hardware contention.** The owner's GPUs are probably busy serving.
8. **Engine support for grafted serving** (vLLM, SGLang, llama.cpp) does not exist. Grafts are HF-only until a model-class patch is written.

### Open questions

- Real layer-1 query states and gate sparsity on Flash-Next.
- Whether the fitted Flash-Next → host value map M beats re-solving per host.
- Which reset token Gemma grafts should use (`<eos>` vs `<end_of_turn>`).
- Whether to calibrate the graft gate and conv (dead point at `W_k = 0`).
- Whether alias handling belongs in the tokenizer or the serving loop.

### Prior art (from `docs/research/engram-relaunch-proposal.md`)

- **EngramEdit (2610.10533).** Two-stage: δ* with the model frozen, including an essence term and a norm clamp, then regularised least squares into n-gram rows with a reuse and frequency penalty; App. A.3.2 covers the gated case. Its sequential results use an exact-key overlay (B.4.3).
- **NeuralDB (2507.18028).**
- **SWEA (2401.17809):** subject-word embedding edits.
- **Engram Adapter (2608.29327):** a frozen-base Engram adapter whose residual is about 0.08% of the hidden norm.

### What would be novel

1. In-place fact writes into a **shipped native hashed n-gram table** of a frontier-scale MoE, applied live to a running serve through an engine-agnostic, reversible row-diff contract with cache invalidation.
2. One format and editor spanning native hashed tables, native per-layer token tables and zero-init grafts, with **keys that transfer across models sharing a tokenizer** (Flash-Next → Qwen3.8-27B / Qwen3.6-35B-A3B).
3. Row-space solving through the real gated and conv module, rather than the one-shot linearisation.
4. Collateral-load-priced row penalties computed from the real hash over a corpus.

### Sources

Every script cited inline has a matching `.out`/`.txt`/`.json` beside it. The folders are:

- `SP/t1/`: native Flash-Next.
- `SP/t2-gemma4-ple/`: Gemma PLE.
- `SP/graft/`: grafts.
- `SP/solver/`: V1–V5.
- `SP/compute/`: sizes and costs.
- `SP/serve/`: serve integration, overlay, requantisation and venue benchmarks.

Prior evidence: `/home/user/memory-organ/docs/research/gemini-engram-code-review.md` (§5, §8) and `/home/user/memory-organ/docs/research/engram-relaunch-proposal.md`.
