# ngram-memory — design

Status: founding design, 2026-11-09, revised that day after three red-team reviews (correctness, science, feasibility). **Restructured 2026-11-10** around the owner's new direction (Stage 1 = offline ingestion of a codebase; everything local; grafts first; Flash-Next as the native reference for dissection). **Stage-1 revision 2026-11-10** after a science and a feasibility red-team of that restructure. See the **Review log** at the end for all three revisions. Nothing here is implemented yet. Every number cites the script that produced it or a code location. Anything not measured is marked **(unverified)** or **(est.)**.

Conventions used throughout:

- **`P/`** = `/home/user/memory-organ/ngram-memory/proofs/`. All proof scripts and their outputs are copied there from the session scratchpad, in the same sub-folders (`.py` plus their `.out`/`.txt`/`.log`/`.json` outputs, and `serve/INTEGRATION.md`).
  - The copies are kept for provenance. Most scripts still hard-code the scratchpad paths they ran from, and none of the binary fixtures (`t1/real/*.pt`, `t2-gemma4-ple/hf/real_small.pt`, tokenizer files, the trained 0.5B rows `rows_armA.pt`, `.npy`/`.npz` NLL arrays) were copied. M0 makes them re-runnable (§8).
  - `P/redteam-correctness/` and `P/redteam-feasibility/` hold the reviewers' re-runs and new checks. `P/review-apply/` holds the budget script written for the first revision.
  - **`S1/`** = `P/stage1/`: the 2026-11-10 Stage-1 research (prior art, objective and row counts, local compute, evaluation protocol, the Qwen2.5-0.5B ingestion experiment and its skeptic re-run, this revision's own checks in `S1/designrev/`, and the Stage-1 red-team re-runs in `S1/rt-science/` and `S1/rt-feas/`).
  - **`GF/`** = `P/graftfirst/`: the three graft-first reports with their skeptic verdicts (`GF/REPORTS.md`; where a report and its skeptic disagree, the skeptic is used here) and their scripts (`train_local/`, `skeptic_local/`, `serve-port/`, `serve-skeptic/`, `dissect/`, `skeptic_dissect/`).
- **Code citations** are relative to the installed transformers 5.22.2 (`…/site-packages/transformers/`):
  - `q4:` = `models/qwen4_exp/modeling_qwen4_exp.py`, `q4c:` = `models/qwen4_exp/configuration_qwen4_exp.py`;
  - `g4:` = `models/gemma4/modeling_gemma4.py`;
  - `q35:` / `q35moe:` = `models/qwen3_5/modeling_qwen3_5.py` / `models/qwen3_5_moe/modeling_qwen3_5_moe.py`;
  - `cu:` = `cache_utils.py`, `cm:` = `conversion_mapping.py`, `cml:` = `core_model_loading.py`, `tp:` = `distributed/tensor_parallel.py`.
- **`R/`** = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/`, and `R/../../tools/` for its tools.
  - Citations are against local **HEAD `df29117`** (2026-11-06). That commit is 293 commits after `b98eb99`, which is an ancestor.
  - `R/ple/`, `R/kvcache/radix_cache.py` and `R/weights/config.py` are unchanged between the two commits (`git diff --stat` is empty). `scheduler.py`, `weights/cpu_tier.py` and `tools/serve.sh` have changed, and are cited at HEAD.
  - The owner confirmed the serve runs on this fork and pushed the latest code before `df29117` was checked out (2026-11-09); local uncommitted changes, if any, are unknown.
- **Metric names** follow memory-organ's conventions: validity gate, efficacy, generality, locality. Nothing else from memory-organ (its `cam/` package or its framing) is reused.
- **Section labels.** The numbered sections §2–§9 keep the numbers that other documents (`serve/INTEGRATION.md`, the research reports) cite. The two sections added on 2026-11-10 carry letter labels: **§I Ingestion (Stage 1)**, placed after §4, and **§D Dissection program**, placed after §6. Milestones were re-sequenced; the old→new milestone map is in §8.
- **Naming clash.** "Project cartridge" here means a namespaced sparse row overlay. It is unrelated to HazyResearch's *Cartridges* (a trained KV prefix, arXiv 2506.06266), which §9 cites as prior art.

---

## 1. Thesis and scope

### Thesis

Language models have started to ship **native n-gram lookup tables**: large embedding tables, keyed on the recent tokens, whose output is added to the residual stream near the bottom of the network.

- **Qwen3.8-Flash-Next** has an Engram-style "PLE" module. It is 51.2 G parameters, 102.4 GB in bf16, and is keyed by hashed 2- and 3-grams (`P/compute/count.out`).
- **Gemma 4 E4B** has Per-Layer Embeddings. It is 2.82 G parameters and is keyed by token id.

In such a model, knowledge can be added by **writing rows**. A write is a small, auditable, reversible diff to a named tensor, not a weight-space edit spread through MLPs. Models without a table get a **graft**: a module that matches the native format in keys and module math, zero-initialised so that it is an exact no-op, after which the same writer fills it.

The durable asset is a **portable row writer**: one key/row addressing scheme, one sparse-row trainer and solver, one patch format, one evaluation harness. Each target model plugs in through a thin backend.

What the rows are used for is now staged (owner direction, 2026-11-09):

| Stage | What is written | Unit | Where it runs | Status |
|---|---|---|---|---|
| **1. Offline ingestion** | a **project cartridge**: rows trained on a codebase (first target: the owner's minisgl fork, `minisglang-rdna4/python`, 1.44 M tokens) | a namespaced, versioned, switchable, reversible sparse overlay (`ngram-edit/v1` patch + manifest, §3.5) | the owner's machine, serve stopped for the build (§7) | **first** (§I) |
| **2. Incremental updates** | row diffs from git commits: dirty rows reset and refit, dead rows deleted | one child patch per commit batch (`parent_version`) | nightly local GPU window, or CPU beside the serve for small graft hosts (§7) | after a served cartridge exists (§I.6) |
| **3. Online per-fact edits** | CounterFact-style facts via RowSolve, solved against the active cartridge | an EditRecord per fact (§3.3, §5) | local GPU or CPU beside the serve, applied live through the overlay | later (§5, §6.B) |

Stage 3 comes last because a cartridge defines the namespace and the row occupancy that per-fact edits are solved against (§I.7).

**What 2/3-gram rows can and cannot hold** decides what Stage 1 can promise (§I.3): identifier spellings, import paths, config and env-var names, local API completions, comment and docstring style — yes, keyed on the last 2–3 tokens. Long-range semantics (what a function does, cross-file types, which call is right here) — not by keying; only through attention reading rows written at earlier positions, which is unproven. The claim to test is "additive with retrieval (E6) at a general-code retention cost of at most +1.0% NLL (R2)", not "replaces RAG" (§9 prior art). It has a kill (E6, §6.A.7). "Near-zero" retention cost, in the sense of Engram Adapter's 99.4–100.1%, would mean R2 ≤ +0.2%; nothing measured so far supports it (the proxy costs +0.5–0.8% on stdlib, §I.8).

"The same writer gives the same results on native and grafted tables" is a hypothesis. A controlled comparison on Flash-Next tests it (§6.A.6, M6). It is not an assumption of the design.

### Priority (owner, 2026-11-10; supersedes the earlier ordering)

**Grafts first.** Flash-Next serves too slowly to be the daily driver (12.9 tok/s decode at TP=2, about 20 tok/s with EP; `R/../../tools/serve.sh:692-694, 1145-1148`), against 102.6 tok/s for the default Qwen3.6-35B-A3B AWQ serve under EP (`serve.sh:148-149`). Flash-Next is kept as the **native reference**: a model trained *with* an n-gram table, dissected forward-only (§D) to inform grafts onto models trained *without* one. The order is (§8):

1. M0 tiny proofs here; M1 the Qwen2.5-0.5B graft proxy (the first data point, already run once, §I.8).
2. M2-0 one-layer feasibility on the owner's cards (AWQ loader, GDN backward, per-card VRAM), then M2 Stage-1 ingestion of the owner's repo into a graft on **Qwen3.6-35B-A3B** (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, the fork's default serve) on the owner's GPUs, serve stopped.
3. M3 the **Qwen3.8-27B** graft (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`; tokenizer byte-identical to Flash-Next, so keys and rows are directly comparable).
4. M4 serving a grafted model in minisgl.
5. The dissection track (§D) runs alongside from the start.
6. Later: Stage 2 on the served graft (M5), Flash-Next native ingestion (M6), Stage-3 online per-fact editing (M7), Gemma (M8), scale (M9).

### Targets

| Tier | Model | Role | Table | Key | Source |
|---|---|---|---|---|---|
| Graft (first) | Qwen/Qwen3.6-35B-A3B (`qwen3_5_moe`, 40 layers, d=2048, 256 experts top-8, GDN 3:1) via `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (24.97 GB) | daily driver; architectural twin of Flash-Next | Flash-Next format, sparse, row width 128 | **same keys as Flash-Next** | §4.3, §4.6 |
| Graft (second) | Qwen/Qwen3.8-27B (`qwen3_5`, 64 layers, d=5120, dense) via `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (19.80 GB) | same generation and tokenizer as Flash-Next | Flash-Next format, sparse, row width 320 | **same keys as Flash-Next** | §4.3, §4.6 |
| Native (reference) | Qwen/Qwen3.8-Flash-Next (`qwen4_exp`; 48 layers; d=2560; 512 experts top-10; GDN hybrid; 360 GB) | dissection; later native ingestion and per-fact edits | `layers.1.ple`: 16 heads × ~20M prime-sized rows × 160 | hashed 2-grams (8 heads) and 3-grams (8 heads) | §2, §4.1, §D |
| Native | google/gemma-4-E4B(-it) | token-keyed comparison; one-card pilot | `embed_tokens_per_layer` [262144, 42·256] | token id, per layer | §4.2 |
| Graft | google/gemma-4-26B-A4B-it, gemma-4-31B-it (`gemma4`, PLE disabled) | later | Flash-Next format with a Gemma key instance (row width 176 / 336), or Gemma-PLE format with alias ids | own tokenizer | §4.4 |

Checkpoint sizes: `GF/train_local/hub_detail.out`, re-checked in `GF/skeptic_local/hub_recheck.json`.

### Out of scope

- Training any weights other than table rows, with one exception: on **grafts only**, a rows + `value_proj` arm is an ablation (§I.1). A graft's fixed init is part of the format.
- Retrieval or in-context memory as *our* mechanism (BM25 RAG is a baseline and a composition partner, §6.A.4).
- Multimodal keys (text `input_ids` only); weight-space editors except as baselines; memory-organ's `cam/` code.
- Building engines. We define a patch interface and write small adapters. **The engine (minisgl) stays torch-free on the row path.** Training and solving run in ngram-memory's own process, where torch (including torch-ROCm on the owner's GPUs) is fine. **Owner decision (2026-11-10):** that training process may import the fork's differentiable GDN path, `minisgl.gdn.hf_patch.patch_qwen3_5_gdn` (native `gdn_hip`, recompute-backward; validated on Qwen3.5-4B; prefill from a zero state only, so prefix-cached solves use HF's torch fallback). This makes the fork a training-time dependency of ngram-memory, never the reverse.
- Cloud venues, except as an optional speed-up (§7 footnote).

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
- **Revisit after dissection (2026-11-10): the linear profile is not Flash-Next's regime.** In Flash-Next, `N_conv` normalises the gated value per token and per stream *before* the conv, so the conv branch depends only on the **direction** of v; the gate cancels in it except through eps (1e-6 against a variance of about 8e-6). On real weights with the gate at 0.5, the conv branch is larger than the direct term on every stream: **1.4×, 13.9×, 2.7×, 1.7×** (stream 1 conv RMS 0.041 vs direct 0.0029), and the earlier taps (t−3, t−6, t−9) carry 47–75% of the conv on streams 0, 2, 3; only stream 1 is effectively pointwise (`GF/skeptic_dissect/branch_share.out`; consistent with `P/t1/e_express.out`, |silu(conv)| 2.14 vs |u| 0.256). So the editor-init graft (conv = 0, gate fixed 0.5, output linear in row norm) runs in a **different regime** from the model trained with the table: it has no direction-only, norm-saturated path and no temporal taps. Grafts therefore carry **two arms** in M1–M3:
  - **G-lin** (default, above): conv = 0. Exact closed form, output bounded by row norm, served by the minisgl "linear" mode (§4.7).
  - **G-dir** (ablation): a **fixed, untrained** direction branch `c_t = silu(Σ_j w_j ⊙ N_conv(u)_{t−3(3−j)})` with `N_conv` weight 0 (scale 1) and fixed taps — either current-position only (`w_3 = α·1`, others 0) or Flash-Next-like dilated taps — with α set from §D's site probe (S) so that ‖c‖/‖h‖ matches the measured per-stream ratio. Still an exact no-op on empty rows (`N_conv(0) = 0`, `silu(0) = 0`), but the output no longer scales with row norm, so the trust region moves from row norm to a cap on how many rows fire (§I.1). Served by the minisgl "general" mode (§4.7). Not yet tested; M0 adds its no-op, decode-state and gradient tests.
  - Output scale for both arms is set from the measured ‖PLE‖/‖X‖ (§D, experiment S), **not** from embedding-RMS ratios: frequency-weighted token-embedding RMS on real code is 0.0038 (Flash-Next), 0.0138 (27B), 0.0094 (35B-A3B) (`GF/skeptic_dissect/embed_freq.out`), while Flash-Next's PLE output on stream 1 is about 0.04, about 10× its embedding scale.

### 2.5 Gemma-PLE format (second table kind)

Key = token id t and layer l. Row = `W[t, 256l : 256(l+1)]`. The module math is in §4.2. The editor treats it as `KeySpec(kind="unigram", V=262144)` with `n_layers` value slots per key.

---

## 3. Architecture

### 3.1 Package layout

```
ngram-memory/
  DESIGN.md
  requirements-dev.txt   pytest, pytest-xdist (pytest is not in the venv today)
  proofs/                provenance copies of the founding proofs (P/), incl. stage1/ and graftfirst/
  ngram_memory/
    spec.py          KeySpec / TableSpec / ModuleSpec dataclasses (own schema, not HF configs), fingerprints, JSON
    hashing.py       transformers-free rows_for(ids)->[T,H] rows, rows_for_ngram(), checkpoint_loc(),
                     decode_row_ids_batch() (vectorised decode-step hash, GF/serve-port/vec_decode_hash.py);
                     port of P/t1/src/ngram_format.py; cross-checked against q4 and a vendored copy of
                     R/ple/hashing.py at a recorded commit (test skips cleanly if absent)
    tables.py        RowStore implementations:
                       HFTensorStore      (a loaded nn.Embedding weight)
                       ShardMmapStore     (safetensors shard files, pread by byte offset; never loads the table)
                       SparseRowStore     (sorted row ids + fp32 values; graft and cartridge storage; P/graft/graft_proto.py:52)
                       QuantRowCodec      (fp8-global-scale / int6-g32 decode, matching the serve)
    ledger.py        EditRecord, RowLedger, materialise(), rollback(), conflict sets
    patch.py         ngram-edit/v1 manifest + safetensors payload; cartridge manifest fields (§3.5); apply/verify
    cartridge.py     CartridgeManifest, namespace/version chain, per-file reader index (sidecar), compile to served bf16
    backends/
      base.py        Backend protocol (3.2)
      qwen4exp_native.py
      gemma4_ple.py   (+ alias.py: reserved-id alias manager and matcher)
      graft/
        module.py     editor-init PLE: q4.Qwen4ExpTextPLELayer built from a ModuleSpec namespace (not
                      Qwen4ExpTextConfig, which rejects hc_count=1); G-lin and G-dir arms (§2.4);
                      UnigramNGramEmbedding subclass for orders {1,2,3}; SparseRowStore
        qwen35.py     hooks + cache slot extension for qwen3_5 / qwen3_5_moe
        host_quant.py frozen 4-bit host layers: AWQ int4-g32 / MXFP4 dequant-to-bf16 autograd.Function that
                      returns only dX (no weight grad); chunked MoE experts; optional fp8 activation fake-quant
        gdn.py        wrapper over the fork's minisgl.gdn.hf_patch (owner decision, §1), widened to
                      Qwen3_5MoeGatedDeltaNet (§4.6); falls back to HF's torch delta rule for cached calls
        gemma4.py     hooks + side LinearAttentionLayer for gemma4
        beside_native.py  graft installed next to Flash-Next's native PLE (native-vs-graft arm, §6.A.6)
    ingest/          Stage 1 and Stage 2 (§I)
      corpus.py      git-blob reader at a pinned commit, filters, secret scan, dedup (sha + MinHash), splits A/B,
                     framings (bare file, chat fence, cat -n), 4,096-token windows with true n-gram history
      rowpolicy.py   row census, count>=k / enrichment policies, reuse weights omega_j, collision stats
      kltargets.py   base-model top-64 log-prob targets for anchor batches (cartridge off)
      train.py       sparse-row trainer: host-RAM SparseRowStore + row-wise Adagrad, proximal group lasso,
                     trust-region projection, GPU step overlap, checkpoint/resume
      increments.py  git-diff reader index, dirty/new/dead rows, reset-and-refit, child patches
    solver/          Stage 3 (§5)
      prompts.py     X_rw / X_ess / X_col / X_ret builders, template train/held-out split, dedup
      anchors.py     candidate-row policy (tiers), corpus row-load counters
      penalties.py   per-row output-space ridge + reuse/collateral weights
      rowsolve.py    projected Adam in row space through the frozen model (all backends); GN warm start;
                     closed-form joint ridge as a warm start for linear grafts
      quantise.py    serve-codec straight-through materialisation (overlay semantics, §3.3)
    serve/
      hf_adapter.py      overlay hook on the embedding output (also the graft runtime)
      minisgl_adapter.py client for the /ple/* and /ngram/* control planes (incl. /ple/score) + overlay patch
      offline_patch.py   in-file shard patching (bf16 tables only; refused for quantised tables)
    dissect/         §D: row_stats.py (G), slice_probe.py (S), ablate_client.py (A), maps.py (M/E/F)
    eval/
      cohorts.py  metrics.py  parity.py  capability.py  matcher_eval.py  report.py
      stage1/    splits.py (A by file, B by commit time), scoring.py (teacher-forced windows), tokclass.py,
                 qa.py (AST-generated repo QA), baselines/ (bm25_rag.py, lora.py, knm.py, ngm.py, controls.py)
  tests/   tiny-config proofs (M0), one file per P/ proof; solver proofs marked slow
    fixtures/  small fixtures (headers, hash buffers, .out references); .pt blobs via scripts/fetch_fixtures.py
  scripts/ ingest.py, update.py, solve.py, apply.py, evaluate.py, fetch_rows.py (HTTP-range row fetch),
           fetch_fixtures.py, count_rows.py (corpus row-load counters)
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

Operations: `apply(patch) → epoch`, `revert(edit_id)`, `list()` (per-rank compiled hash), `read_rows(ids)`, `invalidate(row_ids | all)`, `score(ids, positions) → log-softmax` (forward-only, §8 M6a), plus an **epoch guard**. The guard stamps requests at admission, and a request admitted before an edit cannot insert prefix-cache state that read edited rows.

Adapters:

1. **HF:** an overlay on the embedding output. This is also the graft runtime.
2. **minisgl:** an fp32 overlay applied right after `decode_raw` in `PLEEmbeddingSource.finish_rows` (`R/ple/source.py:242-244`), before the H2D copy.
   - Graph-captured buffers keep their pointers (`R/ple/graph_capture.py:132-147`).
   - Measured overhead is 7–24 µs per decode call and 1.5–2.6 ms per 2048-token prefill chunk at realistic hit rates (`P/serve/integration/bench_overlay_hits.txt`).
3. **Any engine with an unquantised table on the device:** row `index_copy_` over RPC. The adapter reads the engine's **actual** shard layout for the table and slices accordingly: column-sharded, row (vocab-parallel) sharded, or replicated.
   - HF's own plan is `embedding_colwise` (q4c:99, `EmbeddingParallel(shard_dim=1)` at tp:650), but that is HF's choice. Other engines may differ; this is owner question 9.
4. **Offline:** patch bf16 shard bytes at the §2.1b byte offset, then restart. Allowed only for bf16 tables.

Cache invalidation is part of the contract. A radix node is stale if any position on its path reads an edited row id. Nulling the recurrent snapshot alone re-adopts stale KV (`R/kvcache/radix_cache.py:299-312`; `P/serve/INTEGRATION.md` §3.4). Version 1 therefore **drains traffic and flushes the whole tree**. Targeted detach plus the epoch guard is the steady-state design.

### 3.5 Project cartridges (Stage 1 and 2 artifacts)

A cartridge is an `ngram-edit/v1` patch whose manifest gains (S1/ingest-algo design §6):

- `namespace`, `version`, `parent_version`; `kind = "cartridge" | "increment" | "fact"`;
- `source`: repo, commit (or commit range for increments), file-list sha, tokenizer.json sha, framings used;
- `row_policy` (e.g. `count>=2 on split-A train`), `train_config`, `prune_floor`, and the graft arm (G-lin / G-dir);
- `mode = delta` on native tables, `absolute` on grafts;
- a sidecar **reader index**: per-file sparse row counts, used by Stage 2 to find dirty rows (§I.6).

**Graft file vs cartridge file.** `W_v` (and any fixed G-dir weights) belong to the host, not the cartridge, so they ship as two artifacts (`GF/REPORTS.md` graft-serve §3): `graft-<host>.safetensors` (value_proj bf16 [d, d], manifest with KeySpec fingerprint, TableSpec, ModuleSpec, host config sha, W_v sha) and `cartridge-<name>-v<k>.safetensors` (`row_ids` int64 sorted [U], `rows` bf16 [U, w], plus the manifest of the graft it was trained against). Loaders refuse on any fingerprint mismatch.

**Layering on a request** (fixed order, fp32 sum, one cast; §3.3): `served(r, ns) = cast(decode(base_r) + δ^cart_ns[r] + Σ_increments δ[r] + Σ_{facts in ns} δ^fact[r])`, with `decode(base_r) = 0` on grafts. A request names at most one cartridge namespace; requests without one are exactly base (invariant I1).

**Switching and reversal.** Each (cartridge, version) is a namespace on the serve; the prefix cache is rooted per namespace (§4.7). Rollback drops patches from the chain. Deleting a row returns it exactly to base.

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
- **Alternative format (A/B in M8): Gemma's own PLE** via `hidden_size_per_layer_input>0`. Stock `Gemma4ForCausalLM` builds it and reports only the PLE tensors missing.
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

### 4.6 Graft hosts on the owner's machine: checkpoints and training setup

Source: `GF/REPORTS.md` "graft-hosts-local" and its skeptic verdict (the skeptic wins where they differ). All throughput and memory figures are **est.** until M2's first-layer measurement.

**Checkpoints: train against what is served**, so solve/serve parity is a numerics question (as P3/P4 are for Flash-Next).

| Host | Checkpoint | Format | GB | Serve arm | Notes |
|---|---|---|---|---|---|
| 35B-A3B | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | AWQ int4 g32 on routed experts only; GDN, attention, shared expert and lm_head F16 | 24.97 | `qwen35b-awq`, the default `MODEL` (`tools/serve.sh:62, 128-171`) | serve runs experts W4A8 (`MINISGL_MOE_W4A16` off, `R/quant/kernels.py:61`) → optional fp8 fake-quant on expert inputs; serve computes bf16 (`R/server/args.py:534-545`), so F16 dense weights are cast to bf16 in training too |
| 27B | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | Quark MXFP4 g32, served W4A16 | 19.80 | `qwen38-27b-mxfp4`, the fastest 27B arm, 46.5–47.9 tok/s, 60.8 with MTP (`serve.sh:396-419`) | bf16 forward over dequantised weights is the same numerics class |

FP8 (31–37 GB) and bf16 checkpoints do not fit 2×16 GB with activations; bf16 is a CPU/offload reference only. Body sizes for the AWQ 35B: 2.86 GB F16 dense + 18.62 GB experts + 1.02 GB lm_head = 22.5 GB without embeddings (skeptic), about 15.0 GB on card 0 at a 57:43 split, so **split layers across the cards by memory, not by card speed**.

**Training process** (ngram-memory's own torch-ROCm process, serve stopped, every job under `gpu-lease -n 2`; fork `CLAUDE.md:3-40`):

- torch nightly for ROCm 7.2 with gfx1201 (`Dockerfile:31-52`; measurements on `torch=2.15.0.dev20260827+rocm7.2`). Set `TORCH_BLAS_PREFER_HIPBLASLT=0` (intermittent INTERNAL_ERROR, `docs/IMAGE.md:68`).
- **Frozen 4-bit weights stay resident on the cards**; each layer is dequantised to bf16 just in time inside an `autograd.Function` that returns only dX = dY·W, never a weight gradient. Reuse the fork's `R/quant/mxfp4.py:262` and `R/quant/nvfp4.py:374` reference dequantisers. **AWQ needs a custom loader, not just an unpack:** HF's AWQ path requires gptqmodel (`quantizers/quantizer_awq.py:46`) and converts only `nn.Linear` (`integrations/awq.py:93-96`), while Qwen3.5-MoE experts are fused parameters built by the `qwen2_moe` merge (`cm:2116-2117`), so `from_pretrained` cannot place per-expert qweight/qzeros/scales (inferred from code, not run). The loader reads safetensors directly and keeps experts packed; this is M2-0 (§8). Dequant cost: **about 180 s per 1M tokens on the 35B and 135 s on the 27B** (once per microbatch per pass; `GF/skeptic_local/recheck.out`), not the 10–13 s first estimated.
- **MoE.** HF `Qwen3_5MoeExperts` takes stacked [E, 2I, d] weights and has a `grouped_mm` fallback with a backward (`integrations/moe.py:189-262`); skip its weight gradient. Push **large per-layer chunks** (8k–16k tokens, i.e. 256–512 tokens per expert): at 2,048-token microbatches each expert sees 64 tokens and is bandwidth-capped at about 41 TF/s (skeptic). Whether `grouped_mm` runs on gfx12 is **unverified**.
- **GDN: the fork's `gdn_hip`, not flash-linear-attention** (fla hangs on gfx1201: `R/gdn/hf_patch.py:1-3`). `patch_qwen3_5_gdn` runs native prefill with recompute-backward (`R/gdn/layer.py:350-370`). **Only the forward is native:** the backward re-runs a pure-torch reference, a chunked scan of width `max(4, 64//B)` with a Python forward-substitution loop per chunk (`R/../../tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py:108-191, 305-347, 502-535`); native backward ops exist only for conv1d and the gated rmsnorm (`:369`, `:403`). At 35B shapes (HV=32, K=V=128) that is **24.1 aten ops per token per GDN layer at B=4** (7.3 of them kernels; 126 at B=1), against **0.41** for HF's `torch_chunk_gated_delta_rule` (`S1/rt-feas/gdn_ref_ops*.out`, `hf_gdn_ops.out`). At 5–10 µs per kernel and 1–2 µs per view from one Python thread (**est.**), the launch overhead alone is 0.44–0.89 h per 1M tokens on the 35B (30 GDN layers) and 0.71–1.42 h on the 27B (48). **Fix before M2's build:** pair the native forward with HF's chunked torch recompute in the backward (one small `autograd.Function`), or capture the reference backward (hipGraph / `torch.compile`), or write a native backward; §7.5 #1 times it. Correctness so far: forward within about 1e-3 of fla, input-gradient cosine > 0.9999 (`tools/gdn_backward_validate.py`, `tools/cam_native_gdn_validate.py`), validated on Qwen3.5-4B only. Ingestion windows all start from a zero state, which suits its prefill-only contract. **Gap found in this revision:** the patch matches layers by the exact class name `"Qwen3_5GatedDeltaNet"` (`R/gdn/hf_patch.py:104`), so on the 35B-A3B (`Qwen3_5MoeGatedDeltaNet`) it would replace **0 layers**. The two HF classes have identical source modulo the `Moe` prefix (`S1/designrev/gdn_class_check.out`), so widening the name test is the expected fix; M2 must validate it (forward and dX against HF's torch fallback). The 27B (`qwen3_5`) matches the class name, **but not the weights**: in the Quark checkpoint the non-embedding body is U8 12.936 GB plus only 0.005 GB BF16 (`GF/skeptic_local/hub_recheck.json`), so the GDN projections are MXFP4-packed, while `_build_native_from_hf` copies `in_proj_qkv`/`in_proj_z`/`out_proj` as dense weights (`R/gdn/hf_patch.py:28-56`). A bf16 copy is 115.8 M params × 48 layers ≈ 11.1 GB, about +8.2 GB over 4-bit, which does not fit the two cards. **M3 prerequisite:** keep the MXFP4 weights packed and use the just-in-time dequant `autograd.Function` inside the native layer's projections. Flash-Next's `Qwen4ExpTextGatedDeltaNet` differs and is not covered.
- **Activation checkpointing** per layer; logits and cross-entropy in 512-position chunks, with `lm_head` and the cross-entropy off the cards if VRAM requires it; `embed_tokens` gathered on the CPU. **Mixed chunking:** GDN and attention run in 4k-token pieces while the MoE layers get the 8–16k-token chunks (next bullet).
- **Rows in host RAM** as a `SparseRowStore` (sorted int64 ids + fp32 master values + one fp32 row-wise Adagrad accumulator; §I.2). Each step: hash on the CPU (0.21 M tok/s, `P/redteam-feasibility/count_bench.out`), unique rows, H2D, embedding over remapped ids, dense row gradient back, sparse update on the CPU overlapped with the next GPU step.
- **VRAM** (skeptic, per card, at two 2,048-token microbatch inputs per layer, `GF/skeptic_local/recheck.py:29`): **35B 15.0 GB (card 0) / 13.8 GB (card 1)**; 27B 12.1 / 13.2 GB (`recheck.out`; an earlier version of this line gave the 27B pair for the 35B). At the 8–16k-token chunks recommended above, the 35B adds about +1.0 GB of layer checkpoints per card plus about 3.6 GB transient for the GDN reference backward (≈ 2.15 GB of saved per-chunk fp32 state + ≈ 1.5 GB of fp32 q/k/v and gradient copies; `autograd.py:174-190, 516-517`), i.e. about **16.6 GB at 8k and 18.7 GB at 16k per card against a 16.82 GB cap** (**est.**), before non-flash SDPA scores (≈ 2.1 GB per 16k chunk at head_dim 256). Hence the mixed chunking above. The report's two budgets (27.2/24.7 GB total in `vram_fit.out`, no script; 30.3/34.2 GB in `cost_model.out`) disagree; measure per-card peak at the real chunk size in M2-0.

**CPU beside a running serve** (4 cores, `MemoryMax=15G`, 4-bit weights read per layer from NVMe with O_DIRECT at about 3 GB/s, INTEGRATION §4.1–§4.2): possible for the 35B but slow (§7); not practical for the 27B. The GDN ratio and CPU rates were measured on this sandbox's Sapphire Rapids (AMX), not the owner's Zen 4, so CPU figures are optimistic.

### 4.7 Serving a grafted model in minisgl

Source: `GF/REPORTS.md` "graft-serve" and its skeptic. Scripts in `GF/serve-port/`, recomputation in `GF/serve-skeptic/recompute.out`.

**Device math exists.** The fork's `Qwen4ExpPLE.compute` (`R/models/qwen4exp.py:290-467`) matches HF `Qwen4ExpTextPLELayer` at `hc_count=1` on CPU: random init max |Δ| 1.7e-6 fp32, 0.039 bf16 (about 2.5 ulp); with the editor init the output is **bit-exact `0.5·(e@W_vᵀ)`** in bf16, and an empty cartridge gives exactly +0.0 (`GF/serve-port/parity_hc1.out`). On GPU, `minv_linear` sends M ≤ 16 to `dense_bf16_gemv` and larger M to `dense_gemm`, which are not bit-identical to each other (`R/layers/minv.py:256-264`), so **GPU graft-vs-HF parity needs a tolerance**, set to the ungrafted minisgl-vs-HF noise floor.

**Port plan** (12 steps, `GF/REPORTS.md` graft-serve §7):

- New `R/models/ngram_graft.py` (`NGramGraftBlock`; a new file because `qwen3_5.py` cannot import `qwen4exp.py`, `qwen4exp.py:110`), with `mode="linear"` (G-lin: `p = value_proj(e)` with W_v pre-scaled by 0.5; chosen only if W_k ≡ 0 and conv ≡ 0 in the graft file) and `mode="general"` (G-dir or any future gate/conv: the unmodified `Qwen4ExpPLE` behind a config shim `hc_count=1, ple_embed_dim=d`).
- Inject in `Qwen3_5Model.forward` after layer 0 by **folding into `residual`** (the CAM-tap pattern, byte-exact at zero update; `R/models/qwen3_5.py:545-582`, fused norm `R/layers/norm.py:144-159`), placed **after** the `MINISGL_DUMP_LAYERS` dump and DFlash aux capture so those still equal HF's `hidden_states[1]`. Held as `self._ngram_graft` to stay out of `state_dict` (`R/layers/base.py:113`). One edit covers `qwen3_5_moe` and `clover` through the `mlp_factory` seam (`R/models/qwen3_5_moe.py:222-251`).
- New numpy-only `R/ple/cartridge.py`: sorted composite keys `(ns << 32) | row_id` (row ids < 2^29) plus bf16-bit rows; base + ordered overlays (increments, facts), latest wins; one namespace id per (cartridge, version), pinned per request with a refcount, so no drain is needed; `/dev/shm` memmap + mlock shared across ranks (the container already has `ipc: host`, 16 GB shm, unlimited memlock: `docker-compose.yml:64-82`).
- bf16-only staging in `R/ple/source.py` straight into `embeddings.view(torch.int16)` (device staging `max_tokens·d·2` instead of `·6`: 32/80 MiB vs 96/240 MiB at 8,192 tokens), vectorised decode-step hash (64,000 rows, 0 mismatches; 9.9 µs at bs1, and 11.5 µs vs 392 µs for the per-sequence loop at bs16; `GF/serve-port/vec_decode_hash.out`), `build_graft_runtime()` beside `build_ple_runtime` (`R/ple/runtime.py:363-438`).
- Per-request selection: `SamplingParams.ngram_cartridge` → `Req.ngram_ns`, resolved and pinned at admission; a serve default and a model alias for stock OpenAI clients.
- Prefix cache: **one radix root per namespace** (`RadixPrefixCache.roots`, `R/kvcache/radix_cache.py:199, 299, 390`). Because every Stage-2 increment is a new namespace, **prefix carry-over of clean subtrees (INTEGRATION §3.4(a)) or per-conversation version pinning is required, not optional** (skeptic).
- Engine plumbing (scheduler staging, ledger, spec rollback, radix snapshots, graph-capture pointer gate) is already keyed on `ctx.ple`, `mc.ple_layer_ids` and `model.ple_block()` (`R/engine/engine.py:737`, `R/scheduler/scheduler.py:426, 474-486, 1500`, `R/engine/config.py:106-109`, `R/engine/graph.py:290-299`).

**Corrections adopted from the skeptic.**

- **The 35B serves EP by default** (`serve.sh:152`). In this fork EP means separate replicas running different requests (`R/scheduler/ep.py:17-41`), not TP lockstep: each replica has its own radix tree, control ops must reach both, and the per-step cross-rank digest test does not apply. **PLE has never run under EP**: the idle-replica dummy prefill (`ep.py:94-97`) goes through `_stage_ple`, which raises for a request without a GDN slot (`scheduler.py:3210-3225`) — whether the dummy gets a slot is untraced. M4 either tests it or starts with `EP=0`.
- Drop the per-token `active` mask in linear mode: it protects only the sign of zero, which `torch.equal` ignores. Instead **skip the GEMV on steps where every request is on namespace 0** (eager, or a second captured graph), so non-cartridge traffic pays nothing.
- The bitmap prefilter (5–7× faster on misses, `GF/serve-port/bench_bitmap.out`) does not help at the dense hit rates of project text; the host benchmarks used 0.16–1.6 M rows, not the realistic 2–7.6 M, and dense staging at 100% hits is a single-threaded gather (29–78 ms per 8,192 tokens) worth threading.

**Overhead (est., skeptic-recomputed).**

| Host | Decode: graft cost per step | Current decode step | Decode overhead | Prefill overhead |
|---|---|---|---|---|
| 35B-A3B | 47–68 µs | 9.75 ms EP (`serve.sh:148-149`) – 10.5 ms TP (`ROCM10_UPGRADE.md:59`) | **0.5–0.7%** | 0.9–3.0% per 2,048-token chunk (`serve.sh:34`), 1.1–4.1% per 8,192 |
| 27B | 160–200 µs | about 21.5 ms (MXFP4 W4A16) | **0.75–0.95%** (about half with W_v sharded over 2 ranks) | 0.5–2.1% (backbone prefill rate unmeasured) |

RAM per rank for a 1.42 M-token cartridge, bf16 + int64 id (2w+8 B/row): count ≥ 2 on the train split (2.04 M rows) 0.50 GiB (35B) / 1.23 GiB (27B); a production cartridge trained on all 250 files at count ≥ 2 (2,385,780 rows, `S1/ingest-algo/count_rows.out` line 6, reproduced in `S1/rt-feas/count_rows.out`) is **0.59 / 1.44 GiB**; all rows (7.56 M, all files) 1.86 / 4.56 GiB (`S1/designrev/budget_local.out`). Not measured on GPU: all device times, PLE under EP, the dummy-prefill path.

---

## I. Ingestion (Stage 1): building a project cartridge from a codebase

Sources: `S1/ingest-algo/` (objective, row counts, optimiser bench, increments; the counts were re-run and reproduce the earlier `S1/design/` numbers exactly), `S1/priorart/`, `S1/evalproto*/`, `S1/experiment/` and `S1/verify-experiment/` (the 0.5B proxy and its skeptic re-run). Counts use the Flash-Next tokenizer (sha256 `0997f410…`, equal to `AutoTokenizer` on 5 files) and the fork's `R/ple/hashing.py` with the real multipliers; transformers' `Qwen4ExpTextNGramEmbedding.forward` with an index stub agreed on **11,405 of 11,405 positions** over 21 files, including a mid-sequence EOS (`S1/ingest-algo/extras.out` line 1). Wall-clock figures are **est.**

### I.0 Summary

- **Artifact.** One namespaced, versioned `ngram-edit/v1` patch (§3.5): `row_ids` int64 [R], values [R, w]. Trained by next-token NLL on the project's files plus KL anchors, backbone frozen, **rows only** (grafts: rows + `value_proj` is an ablation arm).
- **Which rows.** Rows read by training positions, filtered to n-grams with **count ≥ 2** in the training split, with a soft per-row reuse weight for rows general text also reads. For minisgl `python/` (1.42 M tokens): **7,555,967 rows if every row read is kept** (all 250 files), **2.04 M with count ≥ 2 on the train split**, 2.39 M with count ≥ 2 on all files (the production cartridge) (§I.5).
- **Regularisers.** KL to the base model on non-project text; row-wise group lasso + L2 with a trust region relative to the row scale; injected-residual budget ρ.
- **Optimiser.** fp32 master deltas with row-wise Adagrad in host RAM, 4w+12 B per row (§I.2).
- **First build** on the 35B-A3B graft: about **0.7–2.6 h** for 3 epochs including anchors and a serve stop/boot **if the GDN backward is fixed first** (§4.6); about 3.1–7.5 h with the fork's reference backward as it stands (§7.2). Before that, M2 needs several days of engineering (M2-0, §8).
- **What to expect** (§I.3, §I.8): at the fixed scoring context (≥ 1,024 in-file tokens) the 0.5B proxy's held-out-file gain is about **0.5%** (−0.52%, file-cluster CI [−0.67, −0.33]) at 49k training tokens, against a **+0.5–0.6%** NLL cost on stdlib; with cold 512-token windows it was −1.14% vs +0.51–0.83%. The gain is mostly comment/docstring prose and identifier continuations. The retention-safe part of the rows (the long tail, ≈ 97% of rows) is worth only about 0.3% even at 512 tokens, so the planned row policies are expected to land near **E1 ≈ 0.15–0.3% at retention-safe settings (est.)** unless retraining with KL anchors and ω_j beats the pruning frontier (§I.8). Retention on general *code* is the binding constraint, not general text.

### I.1 Objective

**Trainable set.** The only leaf is a sparse fp32 delta `δ ∈ R^{|J|×w}` added after the gather (the §3.2 rule, `P/serve/v2-solve-venue/autograd_coverage.txt`). The row read is `e_j = decode(base_j) + δ_j` on native tables and `e_j = δ_j` on grafts.

- **Native tables are rows-only, always.** Training `value_proj` would change every position of every request, because native base rows are nonzero. That is not a row diff and would need a per-request device weight, which breaks the torch-free overlay (INTEGRATION §3.1).
- **Grafts: rows-only by default; rows + `value_proj` is an ablation.** A graft's base is zero and `value_proj(0) = 0`, so a cartridge-private ΔW_v changes only positions that read nonzero rows. In the 0.5B proxy it gained 0.002 nats and slightly raised collateral (§I.8), so it is not the default. If used, ΔW_v ships low-rank (r ≈ 16) in a `module_delta` manifest extension and is valid only when nothing else nonzero is active in that graft.

**Loss.**

```
min_δ   E_{x∈D_train}  mean_t  NLL(x_{t+1} | x_{≤t}; δ)                  next-token, all positions, per-n-gram weight capped
      + λ_near · E_{x∈D_near} mean_t KL(p_0 ‖ p_δ)                        other Python (the project's own dependencies)
      + λ_chat · E_{x∈D_chat} mean_t KL(p_0 ‖ p_δ)                        prose / chat / tool text with no project-specific
                                                                          n-grams (never the base's answers to project prompts)
      + Σ_j ω_j · [ λ_2 ‖δ_j‖²/s_j²  +  λ_1 ‖δ_j‖/s_j ]                  L2 to base + row group lasso
 s.t.   ‖δ_j‖ ≤ κ·s_j                      trust region relative to row scale; projected after each step
        ‖Δm_t‖ ≤ ρ‖h_t‖ on anchor batches  injected residual vs hidden norm; checked, then penalised
```

- **Row scale `s_j`.** Native: `‖decode(base_j)‖` (median 0.096 on the served int6 table, INTEGRATION §3.5). Graft G-lin: the output-space scale `‖½ W_{v,h(j)} δ_j‖/‖h̄‖` (§5.3), because a zero base has no norm. Graft G-dir: the output does not scale with row norm (§2.4), so the budget is on **how many rows fire per position** (row policy) plus ρ.
- **Trust region.** Native κ = 1 to start (not the per-fact κ ≈ 10): a cartridge row fires at every occurrence of its n-gram in project traffic, not at one anchor. Grafts **ρ ∈ {0.003, 0.01, 0.03}, starting at 0.01**: Engram Adapter's injected residual is 0.078% of hidden norm (mean over OOD tokens; 0.28–0.29% on activated tokens, 1.1–1.2% at P95) with 99.4–100.1% OOD retention, while the 0.5B proxy injected a median 3.8%, mean 9.1%, p99 44% on held-out project text and median 2.7%, mean 7.3% on stdlib, and lost 0.8% on stdlib (§I.8; `S1/experiment/run_full.stdout` line 25). At ρ = 0.05 the bound would not bind at the median position of either set, so 0.05 is not a starting point; 0.01 only matches Engram Adapter's P95.
- **Pruning floor.** After training, drop native rows with ‖δ_j‖ < 0.03·s_j; smaller deltas mostly vanish in the serve's bf16 cast (realised error 0.69 at 0.3% of row norm, 0.22 at 1%, 0.07 at 3%; INTEGRATION §3.5). The group lasso is a proximal step, `δ_j ← max(0, 1 − lr·λ₁ω_j/(s_j‖δ_j‖√a_j))·δ_j`, so rows reach exactly zero. **λ₁ is set relative to the measured gradient scale** (fix 2026-11-10): λ₁ = λ̃₁·G₀, with G₀ the median over the warm-up steps of the per-step median ‖g_j‖·s_j (the loss gradient of a read row, in row-scale units). For a row starting at zero the Adagrad metric cancels, and the prox keeps it at exactly zero iff ‖g_j‖·s_j < λ₁ω_j. So λ̃₁ = 1 holds a median-gradient row (ω = 1) at zero, and λ̃₁ = 0.1 blocks only rows 10× below the median. An absolute λ₁ is not portable: per-row gradients of a per-token-mean loss scale with reads per window (≈ 1e-5 on the 0.5B proxy at 4k windows), so the M0 smoke's λ₁ = 1e-5 had no stated meaning; measured against G₀ ≈ 9.1e-4 on the proxy it was λ̃₁ ≈ 0.011, a weak penalty whose 'zeroed' rows were mostly the low-gradient tail (rows read in anchor windows or at near-zero-gradient positions). Real 0.5B check at λ̃₁ = 0.3 (`S/nm/l1fix/run/metrics.jsonl`, 4 steps of 4k windows): G₀ = 9.0–9.1e-4, stable from step 0; 28–33% of the rows read in a step are held at zero (13,559–15,889 of 43,857–48,445), so the per-row gradient distribution has a heavy lower tail. The prox uses the same √(a_j + ε) as the step, and ε = 1e-30 (accumulators start at a₀ ≈ 1e-10 on the proxy, so the old ε = 1e-10 was not negligible). M1b grid: λ̃₁ ∈ {0, 0.1, 0.3}, selected on dev (`TrainConfig.lambda_l1_rel`, `--lambda-l1-rel`).
- **Reuse weight** `ω_j = 1 + γ·r_j/(p_j + r_j)`, with `p_j` the row's read rate in the project and `r_j` its rate in the reference mix (near Python + chat) scaled by π_other, the share of non-project text inside project requests (unmeasured; 0.3–0.6 for agentic coding traffic, **est.**). It targets shared, heavily read rows: the 14.9% of the project's distinct 2-grams also in stdlib carry **48.2%** of its 2-gram occurrences, and **70.4%** of stdlib positions and **74.3%** of `transformers` positions read at least one project row (`S1/ingest-algo/count_rows.out` lines 10–11). Start γ = 4, λ_near, λ_chat ∈ {0.1, 0.3, 1} (sweep).
- **KL anchors exclude project-relevant behaviour.** KL to the base on project prompts (or on the base's answers to them) penalises exactly what the cartridge should change (§6.A.5 QA, E4), so D_chat is filtered to text with no project-specific n-gram (absent from the background, §I.3).
- **KL targets.** Base logits for the anchor batches, computed once with the cartridge off, stored as top-64 log-probs plus a log-sum-exp remainder (≈ 512 B per token; ≈ 0.3 GB at 0.5× the training tokens).
- **Stopping and selection.** Early-stop, and choose lr, ρ, row policy and arm, on the **dev** files of Split A only (§6.A.1); the test files are scored once, for the chosen configuration, against a grid pre-registered in the run manifest. Split B is reported alongside as a co-primary endpoint (§6.A.2).
- **Objective upgrade path (Stage-1b, from prior art).** Cartridges/Knowledge-Modules results show distillation from the same model **with the corpus in context** beats next-token training (MTOB 33.5 vs 24.9 chrF; "memorizes well but generalizes poorly" for next-token), and EntiGraph shows synthetic augmentation beats raw continued pretraining on a small corpus (§9). The arm "teacher = host + BM25-retrieved repo files in the window, student = host + rows, loss = KL on the target file" is the first upgrade to test once a next-token baseline exists (M1b on the proxy, M2 on the 35B), and the mandated next step when E1 lands in the 0.5–1% zone (§6.A.7).

### I.2 Sparse optimiser

- **State.** fp32 master δ in a `SparseRowStore` (sorted row ids + [R, w] values) plus one fp32 Adagrad accumulator per row: `a_j += mean(g_j²)`, `δ_j −= lr·g_j/√(a_j + ε)`. The initial accumulator is the median g² of the first 50 steps, so once-seen rows do not take full-size first steps. All 16 heads' rows are separate entries.
- **Why not (Sparse)Adam.** Row visits are very skewed: 7,069 rows serve 50% of 2-gram-head reads and 151,181 serve 90%; 56.7% of 2-gram rows and 73.7% of 3-gram rows are read once (`S1/ingest-algo/extras.out`, row-load skew). Lazy Adam keeps stale moments on rare rows, normalises their single noisy gradients to full-size steps, and costs about 3× the memory. Prior art agrees on simple optimisers for row-sparse updates (Sparse Memory Finetuning uses SGD; §9).
- **Throughput.** An 8,192-token microbatch of real project windows touches 96,169 distinct rows (25,323 at 2k tokens; 341,230 at 32k). Coalescing + Adagrad on 4 CPU cores: 0.35 + 0.25 s per 8k microbatch, 0.46 + 0.31 s at 32k (`S1/ingest-algo/optim_bench.out`). That hides under a 3.5–26 s GPU step (**est.**), so the store stays in host RAM; H2D/D2H about 62 MB per microbatch each way.
- **Memory** (fp32 δ + accumulator + id = 4w+12 B/row; `S1/designrev/budget_local.out`):

| Host (w) | B/row | all rows, all 250 files (7.56 M) | count ≥ 2, all files (2.39 M; production) | count ≥ 2, train split (2.04 M) | count ≥ 5, train split (0.55 M) |
|---|---|---|---|---|---|
| 35B-A3B (128) | 524 | 3.96 GB | 1.25 GB | 1.07 GB | 0.29 GB |
| Flash-Next (160) | 652 | 4.93 GB | 1.56 GB | 1.33 GB | 0.36 GB |
| 27B (320) | 1,292 | 9.76 GB | 3.08 GB | 2.64 GB | 0.71 GB |
| SparseAdam fp32 m, v (w = 160, ref.) | 1,928 | 14.6 GB | 4.60 GB | 3.93 GB | — |

Row counts: 7,555,967 and 2,385,780 from all files, 2,040,221 and 550,891 from the Split-A train files (`S1/ingest-algo/count_rows.out` line 6, `row_policy.out`; both re-run identically in `S1/rt-feas/`). The train-split "all rows" figure is 6,721,239. The all-files count ≥ 2 column is arithmetic on those counts.

### I.3 What 2/3-gram rows can and cannot hold

A 3-gram row at position t is keyed on `(x_{t−2}, x_{t−1}, x_t)`. It shifts the prediction of `x_{t+1}` directly, later tokens through attention, and (native, G-dir) the conv taps at +3, +6, +9.

**Can, by keying:** identifier spellings and sub-token completion (`min|is → gl`); import paths (`from minisgl.` → module); env-var and config-key names; local API completions after `.`, `(` or `self.`; argument names and order; decorator and boilerplate idioms; the project's comment and docstring style. The heaviest project-specific n-grams are exactly these: `' min|is|gl'` 736, `'MIN|IS|GL'` 515, `' from| min|is'` 337, `'.get|("|MIN'` 256, `' NV|FP|4'` 181, `':| List|[int'` 172 (`S1/ingest-algo/count_rows.out`). Project-specific 2-grams (absent from 2 M tokens of stdlib, 2 M of `transformers` and 2.5 M of WikiText) fire at 35.9% (2-gram) and 66.4% (3-gram) of project positions.

**Lookup coverage** (a count-based n-gram LM on held-out files, Split A): the 2-gram context was seen in training at 81.5% of positions; "confident" contexts (count ≥ 2, top share ≥ 0.5) cover 24.1% of positions at 58.2% top-1 (3-gram 22.6%, 60.3%); on the time split, confident 3-gram top-1 is 65.4% (`count_rows.out` lines 19, 23). This is **not** an upper bound on what rows can do (rows also act on later positions through attention), and the base model already predicts many of these tokens. M1 adds the **headroom** number (positions where the confident n-gram is right and the base's top-1 is wrong), and the same counts become a zero-training baseline: the host interpolated with a project Kneser–Ney / cache n-gram LM (§6.A.4), which a cartridge must beat.

**Cannot, by keying:** what a function does, cross-file types, invariants, which call is correct in a given context — and **reversed lookups**. Code always orders `from <module> import X`, so ingestion never trains X → module. Measured reach on 1,632 AST-generated repo-QA items (`S1/evalproto2/qa_reach*.out`): "which module defines X" (n=981) and "which file reads env var" (n=239) have a project-specific first-answer key **0.0** of the time; whole-answer recall by pure 3-gram chaining is about 0 for every kind. Partly reachable: signatures after `def name(` (first key project-specific 0.81, 3-gram argmax correct 0.225), import completion (0.46, 0.158), constant values `NAME =` (0.59, **0.646**, but 0.138 if the prompt ends in a space). Any QA gain must come through attention reading rows written at earlier cue positions — the same free-early-delta pathway that leaked in the earlier fact work. Docs prose (`docs/**/*.md`) is the only source that states name → module in cue-first order; ingesting it is the QA ablation.

**Hypothesis, untested:** rows could inject "project identifier with feature F" vectors that later layers use. Matching shallow teacher states failed before (§6.B priors), so there is no evidence for it yet.

### I.4 Data pipeline

1. **Source.** Git blobs at a pinned commit (the working tree plays no part). Filter by extension; exclude vendored, generated, minified and lock files, binaries, files > 256 KB. **Secret scan** and exclude `.env`, keys and tokens: rows memorise code, and the cartridge ships as a file. Scope barely matters for this repo: all 261 text files under `python/` give 1,440,935 tokens and 7,673,035 rows vs 1,418,418 and 7,555,967 for the 250 `.py` files (`extras.out`).
2. **Dedup.** Exact by blob sha; near-duplicate by MinHash over token 5-gram shingles, each cluster wholly in train or held-out. Cap each n-gram's loss weight at the 99.9th percentile of counts (boilerplate such as `'────────|────────|────────'`, 186 occurrences).
3. **Splits** (§6.A.1). By file: 15% held-out by `hash(path)`, stable across versions (measured 38 files / 200,897 tokens in the design split). By time: the added lines of the last N commits form a forward held-out set (80 commits `b80007f..df29117`: 60,590 added positions).
4. **No training on git history.** Every distinct `.py` blob on first-parent history (205 commits, 566 blobs) is 7.41 M tokens, 5.2× HEAD, yet only **163,312 rows (2.1%)** are not live at HEAD. History adds stale APIs, not coverage; use it only for the time split and for Stage 2.
5. **Framing: train on the forms in which files reach the serve.** Inside a chat turn in a ```` ```python ```` fence (no EOS reset): 9,870 new rows, 99.5% of positions fully covered by the bare-file rows — framing inside chat does not matter. `cat -n`-style numbered tool output: **+36% tokens, 647,624 new rows (+8.6%)**, only 77.2% of positions fully covered (`extras.out`). Mix bare and numbered framings in proportion to the owner's real agent traffic.
6. **Chunking.** 4,096-token windows; 90 files exceed 4,096 tokens and the largest is 96k (`S1/compute/tokcount.out`). Long files: stride 3,584, the 512-token overlap is context only. Each window keeps its **true two-token n-gram history** at the start, never an EOS seed, or its first row ids differ from what the serve reads. Right-pad only (§5.1). Every window starts from a zero GDN state, which matches the `gdn_hip` patch's contract (§4.6).
7. **Tokenisation.** The host's own tokenizer, `add_special_tokens=False`, checked against `AutoTokenizer`. Qwen grafts reuse the Flash-Next keys (§4.3; the 35B-A3B tokenizer lacks only 7 special tokens). Gemma needs its own key instance and a recount.

### I.5 Measured row counts for the owner's codebase

Flash-Next keys, minisgl `python/**/*.py` at `df29117` (`S1/ingest-algo/count_rows.out`, `row_policy.out`):

| Quantity | Value |
|---|---|
| Files / bytes / tokens | 250 / 5,414,867 / **1,418,418** (3.82 B/token) |
| Distinct 2-grams / 3-grams | **295,195 / 662,289** (singletons 57.2% / 74.9%) |
| count ≥ 2 / ≥ 5 / ≥ 20 (2-grams; 3-grams) | 126,438 / 43,168 / 8,403; 166,172 / 37,095 / 4,992 |
| Distinct rows per 2-gram head / 3-gram head | 292,967–293,114 / 651,345–651,617 (gap to distinct n-grams = within-head collisions) |
| **Total distinct (head, row)**, all 250 files | **7,555,967** (2.36% of 320 M logical rows) |
| Overlay at w = 160: fp32 / bf16 / int6-g32 (+ ids) | 4,669 / 2,364 / 994 MiB |

Trainable-row policies on the training split (reach = share of positions reading a selected row, exact-2-gram / any-row incl. collisions):

| Policy | Rows | bf16 + ids (w=160) | Collision rate per head (2g / 3g) | Held-out reach | Near (transformers) reach | General (WikiText) reach |
|---|---|---|---|---|---|---|
| all rows read | 6,721,239 | 2,102 MiB | 1.33% / 2.87% | 0.815 / 0.868 | 0.612 / 0.723 | 0.229 / 0.452 |
| **count ≥ 2 (default)** | **2,040,221** | **638 MiB** | 0.56% / 0.72% | 0.751 / 0.776 | 0.566 / 0.614 | 0.198 / 0.277 |
| count ≥ 5 | 550,891 | 172 MiB | 0.19% / 0.16% | 0.649 / 0.658 | 0.505 / 0.519 | 0.167 / 0.188 |
| enrich ≥ 4 vs stdlib + wiki (ablation) | 1,760,641 | 551 MiB | 0.46% / 0.64% | 0.472 / 0.587 | 0.272 / 0.403 | 0.005 / 0.093 |

Enrichment removes nearly all exact general-text exposure but costs 28 points of held-out reach, so it is an ablation arm; count ≥ 2 plus ω_j is the default. Prior art suggests TF-IDF against a background (Sparse Memory Finetuning, §9) as a third arm. A general-code background of 25.6 M tokens already contains 75% of the fork's bigram **occurrences** and 41% of its trigram occurrences (`S1/priorart/repo_ngram_stats.out`, Qwen2.5 tokenizer); against pretraining-scale data the overlap would be higher (**est.**). **Choosing which rows to write is the main retention problem.**

Graft hosts touch the same row ids; bytes scale with width (§I.2, §4.7). Gemma E4B's table is token-keyed: 18,208 distinct token ids appear in the repo (about 196 M trainable parameters; `S1/local-compute/local_cost.out`), so a cartridge there is a project-wide token shift (SWEA-like), with its own row count.

### I.6 Incremental updates from git diffs (Stage 2)

1. **Reader index.** The cartridge's sidecar holds per-file sparse row counts. On each commit: `git diff --name-status`, re-tokenise only the changed files in every framing, compute each row's count before and after.
2. **Measured per commit** (40 of the 80 commits `b80007f..df29117` touch `.py` files; `count_rows.out`, `S1/ingest-algo/commit_tokens.json`):

| Per commit | Median | p90 | Max |
|---|---|---|---|
| dirty rows (count changed) | 4,796 | 26,442 | 167,183 |
| new rows | 1,312 | 7,277 | 73,998 |
| dead rows (count → 0) | 128 | 1,381 | 10,352 |
| tokens of the changed files | 18.8 k | 103 k | 452 k |

   Over the window: 546,621 dirty rows (7.2% of the cartridge), 235,047 new, 32,511 dead. Replaying all commits reproduced the HEAD census exactly (7,555,967 rows).
3. **Forgetting.** Dead rows and rows that fall below the count threshold are **deleted** (they return exactly to base). Other dirty rows are **reset to zero and refit** (warm start only when speed matters) on the changed files' windows, a replay sample of unchanged windows that read them, and one anchor KL pass; all other rows stay frozen. Rows not marked dirty but co-trained with deleted text keep second-order traces; a **full rebuild** bounds this, triggered when cumulative dirty rows exceed about 20% of the cartridge, or weekly.
4. **Versioning.** Each increment is a child patch with `parent_version` and its commit range; rollback drops patches; the served value is recompiled exactly as in INTEGRATION §3.3. On the serve each version is a new namespace, so prefix carry-over or per-conversation pinning is required (§4.7).
5. **Cost** (**est.**; `S1/ingest-algo/walltime.out`): a GPU window is dominated by the serve stop and boot (2 × 300 s): about 0.17–0.31 h per median-to-p90 commit on the 35B-A3B graft, 0.2–1.1 h on the 27B. CPU beside the serve: 0.8–18 h (35B), 6–121 h (27B). So **commits are batched nightly**. "Online" in the strict sense needs either the CPU path on the small graft host for small diffs, or an in-serve backward, which is excluded.

### I.7 Composition with Stage-3 per-fact edits

- **Layering:** cartridge, then increments, then facts in edit-id order, fp32 sum, one cast (§3.5). Facts are either global or scoped to a cartridge namespace.
- **Project facts** (e.g. "this API was renamed") are solved **with the cartridge active**; their EditRecord gets `parent = cartridge@version`. RowSolve's reuse weight treats cartridge rows as occupied (`occ_j` counts them; `w_freq` uses the project load), and X_col windows come from the project corpus.
- **Global facts** are solved on the base; their conflict set gains `rows(fact) ∩ rows(cartridge)`, and they are re-validated forward-only under every cartridge they will run with.
- **Increments and rebuilds** re-validate every fact in the namespace on its held-out paraphrases; facts below the K-scale bar (retention 0.8) are re-solved. The ledger stores diffs, so re-solving is order-free.

### I.8 Evidence: ingestion through a graft on frozen Qwen2.5-0.5B (CPU; first data point)

Setup (`S1/experiment/ingest.py`, `run_full.stdout`, `results.json`): frozen Qwen2.5-0.5B fp32; graft = `Qwen4ExpTextPLELayer` from `P/graft/graft_proto.py` at the input of block 1 with the §2.4 editor init (G-lin), Flash-Next-style KeySpec on the Qwen2.5 vocabulary with ~200k-row heads (3,201,408 logical rows, width 56); 483,519 reachable rows held compactly (identical to a dense zero table in smoke tests). Corpus: minisgl 250 `.py` files, split by file with seed 0 into 212 train / 38 held-out. Training: **49,152 tokens** (96 × 512) for 2 epochs, SparseAdam lr 1e-2, penalty 1e-3·mean‖e‖². One seed, no sweep. **None of the §I.1 objective was tested:** SparseAdam rather than row-wise Adagrad, a plain L2 penalty, and no KL anchors, ω_j, count ≥ 2 filter or ρ budget. Re-run independently with the same data and seeds by a skeptic (`S1/verify-experiment/verify.py`, `verify.stdout`, 36 min on 4 idle threads): **headline numbers reproduce** (held-out −0.03656, stdlib +0.01267, WikiText −0.00079, train −0.5801).

| Set (nats/token) | Base | Rows only (A) | Δ (95% chunk bootstrap) | Rows + value_proj (B) | Δ |
|---|---|---|---|---|---|
| **Held-out files** (cold 512-token windows) | 3.2492 | 3.2127 | **−0.0366 (−1.1%)** [−0.0446, −0.0289]; file-cluster [−0.0467, −0.0294]; 31/32 chunks improve | 3.2108 | −0.0384 |
| Train chunks | 3.0830 | 2.5029 | −0.580 (−18.8%) | 2.4944 | −0.589 |
| stdlib Python (6,132 tokens) | 1.5304 | 1.5431 | **+0.0127 (+0.83%)** [+0.0079, +0.0178], 11/12 chunks worse | 1.5452 | +0.0147 |
| WikiText-2 | 2.7363 | 2.7355 | −0.0008 [−0.0044, +0.0022] | 2.7354 | −0.0009 |

**Eval coverage.** "Held-out files" are 32 chunks (16,352 tokens) drawn from only 15 of the 38 held-out files; one file carries 22% of the tokens. Chunks are cut from the concatenated stream, so they start cold mid-file and cross EOS (`S1/verify-experiment/verify.py:79-82`; `S1/rt-science/cluster_ci.out`). R2's +0.83% rests on 6,132 stdlib tokens from 12 chunks.

Checks that held: only the row tensor had `requires_grad` (27,077,064 = 483,519 × 56), host `state_dict` sha256 identical before and after; removing the graft restores base NLL exactly (Δ 0.0); cached decode vs full forward 4.2e-5 (held-out) and 4.5e-5 (train), while dropping the n-gram side state per step gives 2.2 and 8.7, so the check has power; no meaningful train/held-out leakage (0.0% of evaluated target 16- and 32-grams occur in the training chunks used; 8-grams 0.52%, worth 3% of the gain).

**Skeptic corrections (adopted):**

1. **Collisions contribute nothing.** Exact-key rows only: held-out −0.0366, stdlib +0.0119; colliding rows only: −0.0002 and +0.0005. The 32% of the gain on tokens whose 2-gram was never seen comes from rows of seen n-grams at **earlier** positions, carried forward by attention. The stdlib drift is also exact-key, not collision.
2. **The gain is mostly frequent prose n-grams, not project API knowledge.** By Python token class: string/docstring 41% of held-out tokens, 51% of the gain; comment 22%, 29%; identifiers 18%, 14%; other code 19%, 6% (`S1/verify-experiment/tokclass.json`). Identifier continuation sub-tokens −2.27% (n = 1,182; **all** identifiers), first sub-tokens −0.48% (n = 1,702) (`S1/evalproto2/tokdelta.out` lines 19–20). Project-specific only: continuations −2.15% (226 + 576 tokens), first sub-tokens −0.58% (n = 100), and project repeat first sub-tokens get **worse** (+0.54%, n = 258); general identifier first sub-tokens improve more (−2.52%, n = 179) than project ones (lines 9, 14–17). This is consistent with generic Python/docstring re-specialisation of a weak base rather than project knowledge. Row norm tracks training frequency (corr(log count, norm) = 0.83); the top rows are `", so"` (7 of the top 9), `"(self,"`, `'    """'`, `" which is"`, `" at the"`, `". The"`, `" minis"`. Zeroing the top-100 2-grams + top-100 3-grams (1,598 rows, 0.3%) removes 37% of the held-out gain and halves the stdlib drift; top-1,000 + 1,000 (3.3% of rows) removes 73% of the gain and nearly all the drift (+0.0006).
3. **A trivial global bias explains about 5%**: adding the mean injected vector (norm 0.048) everywhere gives held-out −0.0019.
4. **Injection is far larger than Engram Adapter's.** Engram Adapter's 0.078% is a mean over **out-of-distribution** tokens; the like-for-like proxy figure is stdlib: median 2.7%, mean 7.3% (about 34× and 93×). On in-domain held-out text the proxy is median 3.8%, mean 9.1% (`run_full.stdout` line 25).
5. Uncontended step time about 3.2 s per 512 tokens on 4 threads; the "seen" sets are from the 49k tokens actually trained, not the 1.17 M-token pool.

**Stage-1 red-team re-scoring (same arm-A rows, `S1/rt-science/`):**

6. **The gain halves at the fixed scoring context; the cost does not.** On 24 blocks from 10 held-out files of ≥ 2,048 tokens, the same targets score −1.14% (file-cluster CI [−1.31, −0.92]) with cold 512-token windows and **−0.52% [−0.67, −0.33] with ≥ 1,024 in-file tokens of context**; stdlib (12 blocks, 9 files) costs +0.51% and +0.58% (`ctx_len.out`). The gain-to-cost ratio falls from about 2.2 to about 0.9. Under §6.A.1's context rule the one data point sits on the E1 kill line and fails M1's joint kill; **the number to beat is −0.5%, not −1.1%.**
7. **The pruning frontier** (512-token context, `S1/verify-experiment/verify.stdout` lines 29–30):

| Rows kept | Held-out gain | stdlib cost |
|---|---|---|
| all | 1.13% | +0.83% |
| top-100 2-grams + 3-grams zeroed (1,598 rows) | 0.71% | +0.41% |
| top-1,000 + 1,000 zeroed (15,949 rows) | 0.31% | +0.04% |

   The retention-safe long tail (≈ 97% of rows) is worth about 0.3% at 512 tokens, below E1's kill before the context shrinkage in 6. Row policies that only prune move along this frontier; M1b must move **above** it (KL anchors, ω_j, ρ during training).
8. **Row-hit conditioning is not an invariant.** Only 4.0% of held-out positions read no trained row; there mean |Δ| is 0.127 nats and only 0.15% are bitwise equal, because rows at earlier positions reach them through attention. Only 0.006% of positions have a row-free prefix, and those are exactly equal (`nohit.out`).
9. **Noise floor.** Between-file SD of the per-file Δ is 0.018–0.023 nats, so with 31 test files the file-cluster MDE (80% power) is about 0.009–0.011 nats, **0.35–0.43%** (`cluster_ci.out`; est., from 13–15 files).

**Not run:** the LoRA arm (budget gate), any sweep, a second seed, the n-gram-interpolation and foreign-repo controls (§6.A.4). **Memorisation:** the train gain is 16× the held-out gain; at 2 epochs this mostly reflects training length, and since served traffic is mostly the ingested files and edits to them, MEM is a diagnostic, not a kill (§6.A.7).

### I.9 Defaults carried into M1–M3, and what the evidence changes

- **Row policy:** count ≥ 2 plus ω_j by default; the frequency ablation above says the high-count generic rows (`", so"`, `"(self,"`) carry both most of the gain and most of the drift, so **enrich ≥ 4 / TF-IDF and a per-row cap on the most generic rows** are first-class arms, not afterthoughts. Pruning alone only moves along the measured frontier (correction 7: ≈ 0.3% gain at +0.04% cost); expect **E1 ≈ 0.15–0.3% at retention-safe settings (est.)** unless training-time regularisation beats it.
- **Budget:** ρ ∈ {0.003, 0.01, 0.03}, start 0.01 (§I.1); the proxy's 3.8% (held-out) and 2.7% (stdlib) medians are far above the retention-safe regime.
- **Scoring:** always at the fixed context of §6.A.1, on the dev files for selection and the test files once.
- **Metrics:** report identifier, comment and docstring NLL separately; a docstring-style gain is not codebase knowledge.
- **Controls:** a same-recipe **foreign-repo cartridge** and a **train-time key-shuffle** arm (the content controls), random-row and eval-time permuted-key controls (Cross-Model Memory Transfer found random memory matched its perplexity gain), an NGM-style untrained 2/3-gram average, the constant-bias control, and a zero-training **n-gram/cache-LM interpolation** baseline (§6.A.4).

---

## 5. The solver ("RowSolve"): Stage 3 per-fact edits

This is the per-fact solver of the 2026-11-09 design, unchanged. It is used from M7 (§8), solved against the active cartridge when an edit is project-scoped (§I.7). The Stage-1 trainer (§I.1–§I.2) shares its sparse-leaf rule (§3.2), backends and quantisation-aware finish, but not its prompt sets, anchors or objective.

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
  - The per-row versus per-n-gram comparison ("collateral cut by 25–60%") is **not at matched fit**: train fit fell from 0.841 to 0.748 and from 0.935 to 0.879. M7 re-runs it at equal train fit, with full-load collateral.
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
  - XK rows go through the same module (same W_v, gate and conv) as in-place rows. So XK removes hash co-tenancy and collisions, but **not** a layer-1 expressivity limit. M7's three-arm design (§6.B) separates the two.
- Rollback = ledger re-materialisation (§3.3).

---

## 6. Evaluation protocol

Two protocols share the invariants and the serve-transfer checks (§6.B "Hard invariants", "Serve-transfer checks"): **§6.A** for Stage-1/2 cartridges (primary now) and **§6.B** for Stage-3 per-fact edits (kept from the 2026-11-09 design, used from M7).

### 6.A Stage 1 and 2: ingesting a codebase

Source: `S1/evalproto/`, `S1/evalproto2/` (scripts and outputs cited as [ep], [power], [tokdelta], [reach]). Proxy numbers are from the 0.5B experiment (§I.8).

#### 6.A.1 Corpus, document format, splits

- **Corpus v0:** `python/minisgl/**/*.py` at `df29117`: 250 files, 5.41 MB, 1,418,418 Flash-Next tokens. v1 adds `tools/`, `tests/` (about 6.9 MB more `.py`) and optionally `docs/**/*.md` (157 files, 2.87 MB; the QA ablation, §I.3). The local clone is shallow (first commit 2026-09-25).
- **Document format (training and eval must match):** `# file: <path relative to python/>\n` + source, files separated by EOS (`<|endoftext|>` 248044). EOS is the only history reset (§2.1); `<|im_end|>` does not reset, so chat wrappers are not used in scoring.
- **Split A, by file (co-primary):** **test** if `h = sha256(path)[:8]/2^32 < 0.15` and size > 400 B: 31 test files, about 200k Flash-Next positions [ep]; **dev** if `0.15 ≤ h < 0.25` (about 10% of files, est. ≈ 25); the rest train (219 − dev ≈ 194 files, est.). Stable as files are added. All early stopping, lr, ρ, row-policy and arm selection use dev; test is scored once per pre-registered configuration. 31 test files give a file-cluster MDE of about 0.35–0.43% (§6.A.7), so corpus v1 (`tools/`, `tests/`) is added to dev and test before any E1 gate is applied to a real host. **Dedupe gate:** held-out files with > 50% verbatim lines (≥ 20 chars) in train move to a near-duplicate diagnostic (currently `minisgl/mamba2/__init__.py`, `minisgl/models/llama.py`; median overlap 9%). Report the share of held-out targets whose ending 16-gram appears in train. (The §I.8 proxy used a different 85/15 shuffle split; M1 re-runs on Split A.)
- **Split B, by commit time (co-primary: it is the deployment case, edits to files the cartridge has seen; uncontaminated; the bridge to Stage 2):** train on the tree at `T_cut = a4a5543` (2026-10-28); test on lines added in `(T_cut, df29117]`: 110 commits, 4,142 added `.py` lines in 52 files, 6 brand-new files [ep]. At the near-identical cut `b80007f`, about 60.6k added positions, of which 81.7% read a trained 2-gram row and 57.0% a trained 3-gram row. Score each added line in its HEAD-file context; only targets on added lines count. Sub-report the 6 new files (held out by both file and time).
- **Scoring context (fixed):** sliding window, left context C = 2,048 in-file tokens, stride 1,024, identical windows for every arm. Also report C = 512 and 8,192: the gain should shrink with context as in-file repeats take over, and that curve is a result. (Proxy base NLL is 2.661 nats at 1,024-token context vs 3.249 with 512-token windows from cold, which is why C is fixed.)

#### 6.A.2 Metrics (one teacher-forced forward per condition)

- **Held-out NLL**: co-primary endpoints **E1** (Split A test files) and **E2** (Split B added lines), both at the fixed context: Δ nats/token, relative %, perplexity; paired, **file-clustered** bootstrap (10k resamples), per-window CIs secondary. Breakdowns:
  - lexical class (Python `tokenize`): string/docstring, comment, keyword, operator/whitespace, identifier;
  - identifiers: project-specific (absent from stdlib + torch + numpy + transformers; 2,358 of 2,921 defined names) vs general vs builtin; first vs continuation sub-token; first occurrence in file vs repeat; cross-file;
  - key category: unseen 2-gram / seen 2-gram only / seen 3-gram with new next / seen 3-gram with seen next — memorised continuation vs generalisation;
  - row-hit conditioning is **not** a code-retention diagnostic: about 96% of held-out positions read a trained row, and the 4% that do not still move (mean |Δ| 0.127 nats) through attention to earlier rows (§I.8 correction 8). The only position-level identity is: **positions with no trained row anywhere in their prefix** (since the last EOS) have ON = OFF bitwise in the graft (0.006% of proxy positions; exact). I1 (§6.B) stays window-level.
- **Completion (teacher-forced EM = greedy EM up to ties):** next identifier (headline: project-specific first-in-file and cross-file first use; the base proxy already gets repeats at EM 0.708 vs first-in-file 0.108); next line (non-comment lines ≥ 8 chars); RepoBench-style cross-file-first lines. Edit similarity on a 200-line subset once a decode path with the cartridge exists (report-only).
- **Cost and operations (reported, not gated):** rows touched, artifact bytes, training and eval wall-clock, serve overlay overhead; for Stage 2, rows dirtied and update wall-clock per commit.

#### 6.A.3 Retention

- **R0, OFF is bit-exact (hard invariant).** HF: sha256 of all logits on 64 fixed prompts (code and prose) equals base, cartridge unloaded and loaded-but-disabled; native: the original row bytes' sha256 too. Serve: the I1/I3 rule of §6.B. Plus I2 (cached decode = full forward, with the broken-state power check; proxy 4.2e-5 vs 2.2) and the host-weight hash.
- **R1, general text ON:** WikiText-2 test, 64k tokens: ΔNLL, per-token KL(base‖ON), top-1 agreement.
- **R2, general code ON (binding).** 64k tokens of stdlib modules not used by any regulariser, plus 64k of an unrelated ML codebase (e.g. `transformers/models/*`); replay/KL sets disjoint by package; report with and without replay. **Unlike per-fact edits, retention ON is not near-guaranteed by I1 here**: at Flash-Next scale the project's rows are read by 70.4% of stdlib, 74.3% of ML-library and 47.5% of WikiText positions (`S1/ingest-algo/count_rows.out` lines 10–12). The proxy already shows +0.83% on stdlib (§I.8).
- **R3, capability (forward-only, about 3k items, about 300k tokens):** 1k each of MMLU, ARC-Challenge, HellaSwag by log-likelihood, paired McNemar; HumanEval + MBPP-sanitized pass@1 once the cartridge decodes, with batched generation (on the 0.5B proxy via HF; on real hosts from M4 through the serve, because the GDN shim refuses a cache, `R/gdn/hf_patch.py:68-75`, and eager HF decoding of a 4-bit MoE is too slow). Tencent's repo fine-tune lost 24% HumanEval (§9), the failure to avoid.
- **REG, regurgitation and secrets (diagnostic):** greedy continuations from 200 train-file prefixes: longest verbatim match with train, share of ≥ 64-token verbatim runs, vs base; plus canary strings planted in a train file and a scan of continuations for secret-shaped strings (the §I.4 secret scan is the primary control).

#### 6.A.4 Baselines and controls (same tokens, split, scoring)

| Arm | Definition | Note |
|---|---|---|
| **B0 base** | frozen host, OFF | reference |
| **RAG-BM25** | index train files (Split B: tree at T_cut), 40-line chunks stride 20; query = last 20 prefix lines; prepend `# retrieved from <path>` chunks up to 1k / 4k tokens | zero training, zero retention cost; beat a full fine-tune in the Tencent study (EM 53.8 vs 44.2); report extra prefill and latency |
| **RAG-oracle** | prepend definitions of the names the file imports | retrieval upper bound |
| **RAG + cartridge** | both | the claim to test is additivity (precedent: FT + RAG 57.4 > RAG 53.8); E6 has a kill |
| **N-gram / cache-LM interpolation** | host interpolated with a count-based project LM: modified Kneser–Ney 2/3-gram on the train files plus an in-file cache component (Tu et al. 2014; Hellendoorn & Devanbu 2017), λ tuned on dev; zero training, zero retention cost (it is off outside project text only if gated like the cartridge, so report both gated and always-on) | the "can counting do what trained rows do" arm; E1 must beat it |
| **LoRA r=16** | attention + GDN projections, all layers; same tokens/steps; lr sweep | in-domain comparator, **not an upper bar** (MLP/expert LoRA matters for knowledge); offline only (minisgl has no LoRA path) |
| **LoRA all-linear r=16** | attention + GDN + MLP (dense host); on the MoE host also the shared expert and the routed experts' projections (frozen router) | the stronger reference for E5; learn/forget trade-off on code per Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (arXiv 2405.09673; from memory, not re-read) |
| **LoRA / cartridge param-matched** | proxy: arm A touched 482,756 rows × 56 = 27.0 M params ≈ all-linear LoRA r=48 (26.4 M). On real hosts the cartridge is larger (count ≥ 2 on Flash-Next keys: 2.39 M × 160 = 382 M params), so match the other way: a cartridge **budgeted to LoRA's parameter count** (top-K rows) | |
| **Engram-Adapter-style** | replicate its layout: injection at several depths (its layers {1, 3, 7, 15} of 36, scaled to the host's depth), 12 fresh hashed tables of 90,007 rows per injected layer, trainable W_v, W_k/gate, norms, conv and a scalar gate, occupancy masks | graft only; the same-family reference (Engram Adapter reached about 55–65% of LoRA's in-domain gain at 99–101% OOD retention). A single-site table is reported as a separate ablation, not as "Engram Adapter" |
| **kNM-LM datastore** | forward-only datastore of in-project tokens the base mispredicts | frozen-backbone corpus memory; +8 EM on Android API lines in the source paper, at 2.4× throughput cost |
| **Foreign-repo cartridge** (main content control) | same recipe, rows trained on an unrelated Python repo with matched tokens and docstring density (e.g. a same-size slice of an ML library not used by R2 or the regularisers) | separates "project knowledge" from generic Python/docstring re-specialisation (the proxy's gain looks like the latter, §I.8); C1 is judged on project-specific tokens against it |
| **Train-time key shuffle** | same recipe, with each training position's row ids replaced by those of a random other position (fixed permutation of n-gram → row) | content-free rows with real training; tests whether the keys carry the gain |
| **Controls** | random rows at matched norms (can only hurt, so a sanity check, not a C1 control); **eval-time** permuted keys (trained rows read through a fixed random permutation of row ids); NGM (untrained causal 2/3-token embedding averages, about +1–2 points generic gain with no corpus); constant mean-injection bias (5% of the proxy gain) | a Stage-1 claim must beat all of them on project text |

#### 6.A.5 Repository QA (report-only in Stage 1)

1,632 AST-generated seed items ([qa] `S1/evalproto/qa_seed.jsonl`): 1,065 "which module defines X", 317 "signature of Y", 250 "which file reads env var `MINISGL_*`"; generated import-completion and constant-value items added. JSONL schema: `kind, question, cue/answer (code form), answer, aliases, evidence file:line, commit, split, reach_class`; the owner appends lines. Scoring: teacher-forced EM, 4-way MC by answer log-likelihood with same-package distractors, answer NLL. Base proxy: EM 0.0 on all kinds; MC 0.323–0.407. Reach (§I.3) says only signature, import and constant items are plausibly helped; track them separately.

#### 6.A.6 Native vs graft (controlled on Flash-Next) and cross-host

- **N:** native-row overlay on `layers.1.ple` (base rows + Δ through the trained W_v, gate and conv). **G:** a zero-init graft **beside** the native PLE at decoder index 1, same KeySpec, orders {2,3}, output into all 4 streams (needs its M0 tiny test), in both arms G-lin and G-dir.
- Held fixed: row ids (the same set S), tokens, data order, optimiser, steps; a 3-point lr sweep **per arm** (natural row scales differ: RMS(v) 0.0058, §4.1).
- Report the paired ΔNLL_N − ΔNLL_G (file-cluster bootstrap), the same for R1/R2, and ‖Δh‖/‖h‖.
- Cross-host grafts (35B-A3B, 27B, Gemma) are **not** controlled; report each as a fraction of that host's own LoRA all-linear r=16 gain.

#### 6.A.7 Gates and numeric criteria (per host; provisional, revised after M1)

**Gates (any failure is a bug):** R0 bit-exact OFF; I2 at the host's repeat-run noise floor with a broken-state control ≥ 100× larger; host hash unchanged; I4 row-id parity on ≥ 10k positions; dedupe audit (held-out 16-gram-in-train ≤ 2%, near-duplicate files excluded).

All E-criteria are scored at the fixed context of §6.A.1 (C = 2,048 in-file tokens) on the Split-A **test** files (or Split B), once, for the configuration chosen on dev; CIs are file-cluster bootstrap 95%. A "bound" below means the CI bound, not the point estimate.

| ID | Metric | PASS | KILL |
|---|---|---|---|
| E1 | Held-out by-file NLL (test files), relative reduction | ≥ 2.0%, lower bound > 0.5% | upper bound < 0.5% at the operating point (below) after the lr pilot + 1 epoch in **every** arm (G-lin, G-dir, and N where run) → stop ingesting into layer-1 rows on that host |
| E2 | Split-B added lines (co-primary; the deployment case) | ≥ 1.5%, lower bound > 0 | report; with E1 killed, E2 alone cannot rescue the host |
| MEM | Memorisation: held-out vs train gain (diagnostic) | report; train ≤ 10× held-out | none (training length dominates train gain at 2–3 epochs; REG covers the risk) |
| REG | Regurgitation / secrets (§6.A.3) | ≥ 64-token verbatim runs ≤ 2× base; no canary or secret-shaped string reproduced | any canary reproduced → fix the secret scan before shipping |
| E3 | Project-identifier continuation sub-tokens | ≥ 3% (proxy 2.15%, 512-token windows) | report; first sub-token report-only (proxy −0.58%) |
| E4 | Teacher-forced EM: project identifier first-in-file / next line | ≥ +2 pts / ≥ +1 pt, paired CI excludes 0 | report |
| E5 | vs LoRA all-linear r=16 at equal tokens | cartridge E1 gain ≥ 50% of LoRA's | < 25% of LoRA's **and** R2 no better than LoRA's → LoRA dominates on quality; the cartridge case rests only on servability, OFF-exactness and per-commit cost |
| E6 | vs RAG-BM25 (4k) at the operating point | "additive" if RAG + cartridge minus RAG ≥ 0.5% E1, lower bound > 0 | RAG + cartridge minus RAG < 0.25% E1, or its CI includes 0 → drop the additivity claim (§1) |
| NGI | vs n-gram/cache-LM interpolation | E1 gain ≥ 1.5× the interpolation's | cartridge ≤ interpolation → counting does what the rows do; ship the count LM instead |
| C1 | vs foreign-repo cartridge (on project-specific identifier tokens) and train-time key shuffle, eval-time permuted keys, NGM (on all test tokens) | gain ≥ 3× the best control's | best control ≥ 50% of the cartridge gain → the gain is content-free / generic re-specialisation |
| R1 | WikiText-2 ON: ΔNLL / mean KL | ≤ +0.2% and ≤ 0.005 nats | > +0.5% or > 0.02 nats at the operating point |
| R2 | General code ON (stdlib + unrelated ML repo) | ≤ +1.0% (proxy +0.5–0.8%) | judged jointly with E1 at the operating point (below) |
| R3 | Capability, forward-only MC, paired | ≥ 99% of base (cf. Engram Adapter 99.4–100.1%) | < 97%, or a significant paired drop > 2 pts |
| NG | Native vs graft on Flash-Next (paired) | G ≥ 80% of N's E1 gain at ≤ N's R2 cost → "format portable for ingestion" | G < 50% of N → grafts need trained projections; promote the Engram-Adapter arm |
| QA | Reachable kinds, train-file items, 4-way MC | +5 pts (soft) | none in Stage 1 |

**One operating point, one decision.** The **operating point** is the dev-selected configuration with the largest dev E1 among those with dev R2 ≤ +1.0% (and R1 within PASS). The host decision is joint, at that one point:

| Zone (test, at the operating point) | Action |
|---|---|
| no configuration has R2 ≤ +1.0%, or E1 upper bound < 0.5% | **stop** rows-only layer-1 ingestion on this host (record the negative result) |
| otherwise below the next row (upper bound ≥ 0.5% but point < 1.0%, or lower bound ≤ 0.5%) | **do not scale**: the only next step is the objective upgrade (Stage-1b distillation with context, §I.1) on the same host, once; if it stays in this zone, stop |
| E1 1.0–2.0%, lower bound > 0.5%, C1 and NGI pass | proceed to the next milestone, with E1 recorded as "partial" |
| E1 ≥ 2.0% (PASS) with R2 ≤ +1.0%, C1, NGI pass | PASS |

E5's and E6's gaps (25–50% of LoRA; 0.25–0.5% over RAG) are reported as "dominated on quality" / "not shown additive" and do not stop the host by themselves.

**Power.** The paired per-token Δ has window-level SD 0.023 nats (design effect 2.2); window-level MDEs are 0.0056 nats at 64k tokens and 0.0028 at 262k [power]. **The unit that matters is the file:** between-file SD of the proxy's per-file Δ is 0.018–0.023 nats, so with 31 test files the file-cluster MDE (80% power) is about **0.009–0.011 nats, 0.35–0.43%** (`S1/rt-science/cluster_ci.out`; est., from 13–15 files). The 0.5% E1 kill is therefore only about 1.2–1.4× the noise floor, which is why kills are CI-based and why corpus v1 adds test files before gating real hosts. For fine identifier classes (n ≈ 100–1,200 on the proxy), pool classes or raise held-out tokens before gating. Split B has 52 files; its file-cluster MDE is reported with it.

**Stage 2 criteria** (M5): replaying the post-`T_cut` commits incrementally reaches ≥ 80% of a full retrain's Split-B gain; forgetting on unchanged files ≤ 0.2% NLL; rollback bit-exact.

#### 6.A.8 Public benchmarks and the private equivalent

Offline-usable after one download (counts and licences to verify on download): RepoEval (≈ 14 repos, function level with tests; best per-repo fit), EvoCodeBench (≈ 25 repos with execution), RepoBench v1.1 Python (`cross_file_first` / `cross_file_random` / `in_file`), CrossCodeEval Python (StarCoder-15.5B code EM: in-file 8.82 → BM25 15.72 → with reference 21.01; `S1/priorart/cceval.txt:331-336`), Long Code Arena project-level completion (closest public analogue of Split B), CodeUpdateArena (54 updated APIs, 670 tasks; the natural test for Stage 2/3). **Caveat:** 2022–2024 public repos are very likely in the hosts' pretraining, so gains there measure re-specialisation, not new knowledge. The decisive test is the private **MSB** built from `minisglang-rdna4`: frozen split manifest (file list, `T_cut`, per-file sha256, tokenizer sha256, KeySpec fingerprint), AST-generated RepoBench-style and RepoEval-API-style items, the QA JSONL, a contamination probe (anomalously low base NLL per file), and the Stage-2 replay of the 110 post-cut commits.

### 6.B Stage 3: per-fact edits (kept from the 2026-11-09 design)

The rest of this section is the per-fact protocol, unchanged except for milestone references (old → new IDs in §8). It applies from M7. Its hard invariants and serve-transfer checks also apply to cartridges.

#### Priors from earlier work

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

#### Oracles (defined)

- **O-site (primary):** one free δ ∈ R^d per fact, shared across positions, added at the PLE-output site (decoder layer 1 input on Flash-Next and Qwen grafts; the graft site on Gemma) at **the same hit positions as the anchor keys**, with the same ρ trust region and the same objective.
- **O-last (comparability):** one δ at the last subject token only, same site and ρ, for comparison with the earlier result.
- On Flash-Next, O-site is also the expressivity arm (c) of M7.

#### Cohorts

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

#### Metrics

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

#### Baselines

- No edit; in-context ceiling (same run, same prompts); O-site and O-last oracles.
- **EngramEdit**, verbatim configuration (two-stage δ* then ridge, its η and λ), on every W1 host.
- **SWEA** (token-id-keyed input-embedding edit) on every host. It needs no graft, so it is the null hypothesis for "is a graft needed".
- **NeuralDB** control, and **GRACE / WISE / MEMOIR** for the sequential runs.
- ROME/MEMIT/AlphaEdit on graft hosts, and MEMIT/AlphaEdit at small n on Flash-Next (MoE/GDN feasibility **unverified**).

#### Native vs graft (controlled)

On Flash-Next itself, install a zero-init editor-init graft **beside** the native PLE at decoder layer 1, with the same KeySpec, orders {2,3}, and output added into all 4 streams (**unverified**: needs an M0 tiny test). Solve the same facts both ways with the same solver code (Adam path). Report the paired difference in efficacy, generality and leakage, normalised by the same-host O-site oracle. This is the only test of "same editor, same results" with the host held fixed.

#### Hard invariants (a failure is a bug and stops the run)

- **I1.** Windows that read no edited row are unchanged.
  - On HF: **bitwise**. This held on all tiny proofs.
  - On the serve: bs=1, sequential requests, fixed chunk size, compared against a no-edit control under identical conditions. Bitwise only where that control is itself bitwise across repeat runs. Otherwise the tolerance is the control's measured repeat-run noise floor, and the report says which was used. Concurrent batched runs are not bitwise reproducible even without edits (`R/../../docs/journal/PERF_PUSH_CONTINUANCE.md:128-136`).
- **I2.** Cached decode equals a full forward at the host's own noise floor, over 50 prompts × 32 greedy tokens with identical tokens. This also holds on the live serve after a patch, including for a prefix cached before the edit.
- **I3.** Rollback restores pre-edit outputs: bitwise on HF; on the serve, under the same pinned conditions and tolerance rule as I1.
- **I4.** `rows_for` matches the model's lookups on at least 10k positions, including EOS edge cases.

#### Serve-transfer checks (P0–P5, from `P/serve/INTEGRATION.md` §5)

- Row ids, base row bytes and edit transport are bitwise equal between the solve and the serve.
- Pre-edit top-1 agreement between the solve model and the serve is at least 95%. Otherwise add fp8 activation emulation: the serve's per-token fp8 expert activations have 4.09e-2 relative error per layer (`R/weights/cpu_tier.py:38-40`).
- Realised ΔNLL / predicted ΔNLL ≥ 0.7 on targets, read through `/ple/score` (§8 M6a).
- Cache hygiene: warm, then edit, then query must equal a cold run, under the I1 pinned conditions.

#### Proposed accept bars (unverified; to be revised after M7)

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

#### Ablations

λ_KL ∈ {0, 0.0625, 0.25, 1}; anchor tier; per-row vs per-n-gram penalty at equal train fit; in-place vs exact-key overlay (XK mode); ρ and κ; grafts: orders {2,3} vs {1,2,3} (each paired with its lexical-neighbour and SSOR leakage), width d vs 2d, closed-form vs Adam; native vs graft on Flash-Next; Gemma: Flash-Next-format graft vs Gemma-PLE plus aliases vs SWEA.

#### Kill criteria

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

## D. Dissection program: Flash-Next (trained with a table) vs grafted hosts (trained without)

Source: `GF/REPORTS.md` "dissection" and its skeptic (the skeptic wins). Scripts: `GF/dissect/` (range reads, row pilot, embedding scale), `GF/skeptic_dissect/` (branch shares, frequency-weighted embedding RMS). Purpose: settle the graft design questions that rows-only ingestion cannot settle by itself — gate, conv/direction branch, output scale and trust region, which n-grams to store, injection site, whether transplanting Flash-Next rows is worth anything — using **forward-only** access to Flash-Next (row range reads, a 2-layer slice, the owner's serve). No backward pass on Flash-Next is needed until M6.

### D.1 Mechanics the experiments rely on (checked in code)

- **Zero rows are an exact, full ablation**: `key_proj`/`value_proj` have no bias (q4:1196-1197) and RMSNorm(0) = 0 (eps inside the rsqrt), so e = 0 → output 0 (bit-exact in `P/graft/bf16_noop.out`). Zero-all, zero-2-gram-heads, zero-3-gram-heads, mean-row and shuffled-id ablations are all **numpy-only** on the serve: fill `_host_rows` after the decode at `R/ple/source.py:243`, before the H2D at `:244` (the overlay hook), or remap ids in `begin_rows` (`source.py:227`).
- **Gate and conv ablations are torch-side** in `Qwen4ExpPLE.compute` (`R/models/qwen4exp.py:421-443`): a boot-time env flag, graph-capture-safe, one restart (about 300 s, `R/weights/ckpt_read.py:5-10`) per variant; owner approval needed.
- **The gate is one scalar per (token, stream)** — 4 per token — computed from the concatenation of all 16 heads (q4:1240-1241). Per-order effects need leave-one-order-out recomputation.
- **The gate depends only on what enters layer 1** (`embed(x)` ×4 → layer 0 GDN + MoE → PLE; q4:1278), so gate, value, conv share and output scale can be measured on a **2-layer slice**: Flash-Next layer 0 is 5.19 GB bf16, embedding 1.27 GB, the PLE block without the table 0.066 GB (`P/compute/hdr/*.json`, re-checked by the skeptic).
- **Scoring.** No teacher-forced path exists today (`R/server/api_server.py:415-420, 1637-1638`); A needs `/ple/score` (M6a). Fallback without engine changes: one `max_tokens=1, top_logprobs=20` request per prefix (NLL censored outside the top 20; pilot only).
- **Range reads of the public bf16 shards** (`GF/dissect/range_probe.out`, `row_pilot.out`): one 320 B row ≈ 0.20 s serial; 200 rows/s at 48 threads; a contiguous 2,048-row block is one 0.32 s request; multi-range requests are refused (HTTP 416). About 12 n-grams/s, so 5k n-grams ≈ 7 min; the whole-repo 7.7 M rows would take 11–16 h (**est.**; HF rate limits at 10^4–10^5 requests untested), so whole-codebase row reads belong on the owner's Optane through `R/weights/row_table.py` (numpy, int6). The public checkpoint is bf16; the serve uses int6-g32 (median relative error 1.4e-3), fine for statistics.

### D.2 Findings so far

1. **Row coherence vs frequency (pilot G0, `GF/dissect/row_pilot.out`).** Centre each head's row on its head mean and map through the real W_v slice; coherence = mean pairwise cosine over the n-gram's 8 heads (co-tenant noise is independent across heads, the n-gram's own signal is shared). Frequent code/prose 2-grams: 0.150 / 0.194; 3-grams 0.161; count 5–20 code 2-grams 0.017; once-seen 0.003 (2-gram), 0.013 (3-gram); random buckets 0.000. Derived SNR of the summed 8-head value ≈ 8ρ/(1−ρ): 1.4–1.9 for top n-grams, about 0.02 for the long tail. **Flash-Next's rows for locally rare (project-specific) n-grams are mostly co-tenant noise**, so a cartridge should spend rows on n-grams frequent in the project but rare in general, and a Flash-Next → graft transplant is predicted to be worth about 0 (hypothesis; F tests it). Caveats: frequency from a 400k-token local corpus, n = 32–48 per bucket, value path only.
2. **The conv/direction branch dominates the PLE output** (`GF/skeptic_dissect/branch_share.out`): conv/direct = 1.4, 13.9, 2.7, 1.7 per stream at gate 0.5; it depends only on the direction of v. Consequences adopted in §2.4 (two graft arms) and below (what A can and cannot show).
3. **Output scale** (`GF/skeptic_dissect/embed_freq.out`): frequency-weighted embedding RMS 0.0038 (Flash-Next) / 0.0138 (27B) / 0.0094 (35B-A3B); Flash-Next's stream-1 PLE output is about 0.04. Uniform-vocabulary sampling doubles Flash-Next's embedding RMS, so the report's "0.6× host embedding RMS ≈ 0.008" calibration is withdrawn. Set the graft scale from S's measured ‖PLE‖/‖X‖.
4. **Hints only:** the value aligns with lm_head of the next token (z +1.03) almost exactly as much as with the current token (+1.02), so no next-token-specific "direct logit" claim; never-read padding rows have median norm 8e-18 but mean 0.015, so "trained with weight decay" is not shown.
5. **Placement.** Flash-Next's `ple_layer_ids=[2]` (1-based) is the Engram paper's best single-module site, "Layer 2" (arXiv 2601.07372 §6.2), the same site the grafts use (decoder index 1).

### D.3 Experiments

| ID | What | Where / cost (**est.**) | Measures | Decides |
|---|---|---|---|---|
| **G** (first) | Scale G0 to ≈ 5k n-grams per order; frequency from the 1B-token row-load counters (§7) or a public proxy | CPU here (≈ 7 min of range reads) or numpy on the owner's Optane; ≈ 1 day of code | coherence-vs-frequency curve and the crossing frequency f* (coherence ≈ 0.05); 2- vs 3-gram value share (W_v slices 12.3 vs 6.3 Frobenius); row norm vs load | cartridge row policy (spend rows below f*); graft row norm prior; whether F is worth building |
| **S** (first) | 2-layer Flash-Next slice (embed + layer 0 + PLE), sparse rows for 20–50k tokens of the owner's code + prose | here from a 6.5 GB public bf16 range fetch (≈ 8 GB peak RAM if kept bf16); on the owner's box only with a dequantising loader for the MXFP4 local checkpoint; 1–2 days of code | per token and stream: gate sigmoid(g_s), direct vs conv share, ‖PLE out‖/‖X‖, leave-one-order-out Δgate and Δv; cut by token class, coherence, frequency; within- vs between-n-gram gate variance | graft output scale and ρ; the G-dir arm's α and taps; whether the gate is context selection (needs a calibrated, non-zero W_k init: W_k = 0 is a dead point, `P/graft/calib_grad.out`) or between-n-gram |
| **A** (after M6a) | Serve ablation sweep through `/ple/score`: none, zero-all, zero-2g, zero-3g, mean-row, shuffle, gate-constant (boot flag), conv-off (boot flag) | owner's serve, forward-only; ≈ 0.5 M tokens per variant at 10–14 min each (single 16k-prompt prefill rates of 1,250–1,430 tok/s, `serve.sh:1145-1148`, are optimistic for 5.6k-token files with full-vocab scoring), + 300 s per boot-flag restart; radix and GDN/conv state flushed between variants; measures the quantised (MXFP4, W4A8) model | ΔNLL per token by class, seen-before, coherence bin | how much a co-trained model depends on its table and where (Engram §6.3: suppressing the table kept 29–44% of factual-QA vs 81–93% of reading comprehension). **Not** a test of the graft's fixed gate: gate-constant only touches the direct term (≈ 7% of stream 1), and conv-off removes most of the PLE output plus a train/test mismatch |
| D | Downstream reach: per-layer ‖Δh_L‖ zero-rows vs none; logit-lens KL; CKA of Flash-Next's post-PLE layer-1 state vs every layer of the 27B and 35B-A3B | serve debug dump (prefill only) or an HF forward-only session during downtime | | injection site |
| M | Ridge map from Flash-Next layer-1 inputs (2560) to 27B layer-1 inputs (5120), position-aligned by the shared tokenizer | CPU, hours (27B slice 3.3 GB) | held-out R² | prerequisite for E, F |
| E | Ingested 27B graft rows vs native rows on the same n-grams: RSA/CKA (map-free) and cos(M·v_FN, v_27B) per frequency bin | CPU minutes after M3 | | whether the graft learns what the native table holds above f*, and something new below |
| F | Transplant: graft rows `e = 2·W_vᵀ·M·v_FN` (min-norm for G-lin); arms zero / transplant / shuffled / above-f* only / ×0.5, ×1, ×2 | 27B harness, GPU, serve down; 2–12 min per arm | held-out code ΔNLL | a free generic init if positive (predicted ≈ 0) |
| N | Native cartridge on Flash-Next (= M6b) | downtime, §7 | | the with/without contrast with the same keys |

**Order:** G and S run now, alongside M1–M3 (cheap, feed the graft design directly). A waits for the Flash-Next control plane (M6a) and a graft baseline. D/M/E/F follow M3; N is M6b.

**Decision rules (numeric).**

- **G-dir is mandatory in M2/M3** if S confirms conv ≥ 2× direct on the median token in ≥ 2 of 4 streams (the skeptic's real-weight check already shows it on all 4 at gate 0.5).
- **Calibrated gate arm** if S finds ≥ 50% of gate-logit variance *within* n-grams (context selection); otherwise the fixed 0.5 gate stays.
- **Row policy:** if G finds f* and project rows below f* carry ≥ 70% of the M2 cartridge's E1 gain, the default row policy becomes "count ≥ 2 in project, below f* in general".
- **Transplant F is dropped** if it reaches < 10% of the trained cartridge's E1 gain at its best scale.

---

## 7. Compute plan (local-first, wall-clock)

Everything runs on the owner's machine. Cloud is at most an optional speed-up (footnote at the end of this section). All throughput and wall-clock figures are **est.** unless a measurement is cited; the efficiency brackets are replaced by M2's first-layer measurement.

### 7.1 The machine and its constraints

- **Owner's machine.** 2 × 16 GB AMD RDNA4 (gfx1201): card 0 RX 9070 XT, card 1 RX 9070 (fork `CLAUDE.md`). Measured hipBLASLt dense bf16 115.3 TF/s at M=2048 on card 0 (`docs/journal/BLOCKWISE_FP8_BRIEF_2026-09-22.md:167`); PCIe H2D 28.93 / 14.48 GB/s (`R/weights/cpu_tier.py:6`); DDR ceiling 57 GB/s (`cpu_tier.py:420`). 96 GB RAM: about **20 GB MemAvailable while the Flash-Next serve runs** (it pins a 53.75–55.9 GiB host arena), all of it when the serve is down; host headroom beside a 35B-A3B or 27B serve is likely larger (unmeasured). The Flash-Next n-gram table and swap live on an Optane drive; checkpoints on other NVMe.
- **The GPUs are not in constant use; the serve can be stopped for jobs.** A Flash-Next stop + boot costs about 300 s (`R/weights/ckpt_read.py:5-10`, 300.1 s; boots of up to 618 s were recorded earlier, INTEGRATION §4.2). Every GPU job goes through `gpu-lease -n N -- <cmd>` (fork `CLAUDE.md:3-40`).
- **CPU beside a running serve:** 4 cores pinned away from the serve's, `systemd-run --scope -p MemoryMax=15G -p MemorySwapMax=0`, NVMe O_DIRECT reads, no Optane, no GPU (INTEGRATION §4.2(A)). CPU rates here were measured on this sandbox's Sapphire Rapids (AMX); the owner's Zen 4 is probably slower.
- **This container** (4 CPUs, 15 GB RAM, no GPU, about 25 GB free disk): M0 and M1 (the 0.5B proxy), §D's G and S experiments, row censuses, packaging. Measured: the 0.5B graft trains at about 3.2 s per 512 tokens on 4 idle threads (`S1/verify-experiment/verify.stdout`); bf16 GEMM 2,235 GFLOP/s (`P/redteam-feasibility/cpu_flops.out`). The container is **shared**: during the Stage-1 review a sibling job used 227% CPU and fp32 GEMM fell to 7.8 GFLOP/s, ruining a timing run (`S1/rt-feas/cpu05b.out`), so CPU wall-clock here is a lower bound. A 0.5B fp32 forward + backward at 4,096-token windows was reported OOM-killed in 15 GB (the run ended without output; not reproduced), so 4k windows need activation checkpointing.

### 7.2 Stage 1 and 2 on each host

Inputs: graft hosts from the skeptic verdict in `GF/REPORTS.md` (graft-hosts-local; includes the ~180 s / 135 s per 1M tokens of dequant the first estimate missed; 27B efficiency capped at about 35% of peak). **The Qwen GDN-host GPU figures assume a GDN backward at kernel speed** (the old "GDN multiplier 1.05–1.5"); with the fork's pure-torch reference backward as it stands, add 0.44–0.89 h (35B) / 0.71–1.42 h (27B) per 1M tokens (§4.6; **est.**), shown in brackets; Flash-Next and Gemma from `S1/local-compute/local_cost.out` (not skeptic-checked); cartridge arithmetic in `S1/designrev/budget_local.out` (3 epochs over the 1.22 M-token Split-A train set, +0.5× anchor KL tokens, + 2 × 300 s serve stop/boot); increments from `S1/ingest-algo/walltime.out`.

| Host | GPU, serve stopped: per 1M training tokens | **One cartridge, 3 epochs (minisgl `python/`)** | CPU beside the serve: per 1M tokens | Per commit (median–p90), GPU incl. stop/boot | Fits? |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B graft** (AWQ, both cards) | **0.1–0.45 h** [reference GDN bwd: 0.54–1.34 h] | **0.7–2.6 h** [3.1–7.5 h] (skeptic: 0.6–2 h without anchors/boot) | ≥ 11–35 h (a cartridge takes **2.5–8 days**: 60.4–192.2 h with anchors, `S1/designrev/budget_local.out` line 17) | 0.17–0.31 h | 15.0 / 13.8 GB per card at 2k-token microbatches, ≈ 16.6–18.7 GB at 8–16k (cap 16.82; §4.6, est.) → mixed chunking required; rows + Adagrad 1.25 GB (count ≥ 2, all files) to 4.0 GB (all) in host RAM |
| **Qwen3.8-27B graft** (Quark MXFP4, both cards) | **0.5–1.0 h** [1.2–2.4 h] | **2.9–5.7 h** [6.8–13.5 h] (skeptic: 2–4.5 h without anchors/boot) | 62–205 h (not practical) | 0.2–1.1 h | 12.1 / 13.2 GB per card **only if the GDN projections stay MXFP4-packed** (§4.6; a bf16 copy adds ≈ 8.2 GB); rows 3.1–9.8 GB |
| Flash-Next native (both cards, MXFP4 experts streamed from pinned host RAM, 8,192-token batches) | 0.15–1.0 h (9–59 min) | 1.0–5.6 h; **0.8–4.8 h of downtime** in `local_cost.out`'s recipe | 27 h – 4.7 d | 0.18–0.53 h | experts 59.77 GiB + trainer 1.3–4.9 GB in 96 GB; data-transfer bound |
| Gemma 4 E4B native (one card) | 0.12–0.32 h (7–19 min) | about 0.6–2.6 h | 14 h – 2.3 d | 0.17–0.34 h | fits one card; the other card stays free |
| gemma-4-26B-A4B / 31B grafts | 0.07–0.35 h / 0.4–1.8 h | 0.6–2.6 h / 3.4–17 h (`walltime.out`, unreviewed) | 14 h – 2.3 d / 4–15 d | — | later (M8) |

So: **once the GDN backward is fixed (§4.6), a codebase cartridge on the 35B-A3B is one evening of serve downtime**, the 27B one night, and a 6-configuration sweep on the 35B fits in one or two evenings. With the reference backward a 35B cartridge is 3.1–7.5 h (over M2's 4 h bar) and a 27B one 6.8–13.5 h (reaching M3's 12 h demotion trigger), so the fix is an M2 prerequisite. These are training hours; M2 also needs several days of engineering first (M2-0, §8). The CPU-beside-the-serve path is for small increments on the 35B graft only.

**Evaluation points.** One Stage-1 evaluation point is about 0.65–0.7 M tokens (Split A + Split B in context + R1 + R2; R3 adds about 0.3 M). In ngram-memory's process with the serve stopped: about 1–3 min on the 35B, 4–19 min on the 27B, 1–9 min on Flash-Next (`local_cost.out`, unreviewed). Through a served graft (after M4) the 35B prefill is about 7,000 tok/s (`serve.sh:149`), about 2 min per point. Through the Flash-Next serve, scoring needs `/ple/score` (M6a) and runs at roughly 1,250–1,430 tok/s for single long prompts, so 8–20 min per point including the overheads in §D.3.

### 7.3 Stage 3 per-fact solves (later)

Sizes are exact, from safetensors headers (`P/compute/count.out`). FLOP is 6·active·N (forward, recompute under per-layer checkpointing, backward), with N the per-iteration token count of §5.1: 251–497 with prefix caching (used), 674–1548 for full prompts (`P/review-apply/budget.out`).

| Target | Weights (bf16) | Table | Active params / token | TFLOP / iter, prefix-cached N | TFLOP / iter, full N | Ckpt activations (GB) |
|---|---|---|---|---|---|---|
| Flash-Next | 360 GB (body 251 GB bf16, 73 GB at ~4.5-bit) | 102.4 GB | 6.67 G | 10–20 | 27–62 | 0.46–2.86 |
| Gemma E4B | 16 GB | 5.64 GB | 4.70 G | 7–14 | 19–44 | 0.10–0.63 |
| Qwen3.8-27B | 55.6 GB | sparse graft | 25.6 G | 39–76 | 104–238 | 0.32–1.98 |
| Qwen3.6-35B-A3B | 71.9 GB | sparse graft | 2.95 G | 4.4–8.8 | 12–27 | 0.14–0.88 |
| gemma-4-26B-A4B | 51.6 GB | sparse graft | 3.82 G | 5.8–11 | 16–36 | 0.11–0.70 |
| gemma-4-31B | 62.6 GB | sparse graft | 30.7 G | 46–92 | 124–285 | 0.27–1.65 |

Weights, not activations, decide the hardware. The table never needs to be resident: a batch reads about 5 KB of rows per token.

**Iterations per solve: 20–30**, not the 100–300 the 2026-11-09 cost table assumed. The audited EngramEdit stage 1 is 20 Adam steps plus a closed-form ridge and about 2 forward-only checks, about 21–25 iteration-equivalents (INTEGRATION §4.2). M7 logs NLL per step on the first solves and replaces this.

| Host | Venue | Per edit (21–25 iterations) | 2,000-fact cohort |
|---|---|---|---|
| 35B-A3B graft | GPU, serve stopped | 10–30 s (0.5–1.5 s/iter; skeptic: no script, re-derivation agrees) | 5.6–17 h (`budget_local.out`) |
| 35B-A3B graft | CPU beside the serve | 12–38 min (expert dequant dominates) | not practical; use nightly batches of a few dozen |
| 27B graft | GPU, serve stopped | 10–40 s | 5.6–22 h |
| Flash-Next native | (A) CPU beside the serve, prefix-cached, batched | 19–40 min single; 3.2–6.9 min/edit at 16 × 4 in RAM; 2.2–4.8 min at 32 × 4 with NVMe-spilled prefix state; 70–200 edits per 8-h night (INTEGRATION §4.2–4.3) | 7–20 nights |
| Flash-Next native | (B) both GPUs, serve stopped | 3–10 s/iter; 1–2 min single, seconds per edit at K = 32 (INTEGRATION §4.2) | one or two downtime windows |

### 7.4 Venue decision (replaces the 2026-11-09 "open conflict" note)

The conflict was between INTEGRATION.md (first native solve on the local CPU beside the serve) and this document (cloud M2-lite first). **Resolved in favour of local** by the owner's direction: all stages run on the owner's machine. Concretely: Stage 1/2 builds and cohort-scale solves use the GPUs in serve-downtime windows; small Stage-2 increments (35B graft) and single native per-fact solves use INTEGRATION's option (A) beside the serve; forward-only checks use the serve itself. The cloud M2-lite arm is removed from the milestone path (footnote). The per-iteration disagreement (32–91 s here vs 54–115 s in INTEGRATION) is settled the same way as everything else: by the first measured layer on the owner's machine (M2 for grafts; M6/M7 for Flash-Next), and INTEGRATION's numbers are used meanwhile.

### 7.5 First measurements on the owner's machine (they replace the assumed brackets)

1. **(M2-0)** One real-size 35B-A3B decoder layer of each kind (GDN, attention, MoE), loaded from safetensors with the custom AWQ unpack, forward + input-gradient backward on gfx1201, with `gdn_hip` (widened to `Qwen3_5MoeGatedDeltaNet`, §4.6). Includes **a GDN timing at B=4, T=4096 on one layer, forward and backward**, with the reference backward and with the fix, and **per-card peak VRAM at the real chunk size** (4k GDN/attention pieces, 8–16k MoE chunks) at the planned layer split.
2. Whether `torch.nn.functional.grouped_mm` runs on gfx12; otherwise a chunked bmm fallback.
3. The fp8-activation gap between the W4A8 expert serve and a bf16-activation training forward (top-1 agreement ≥ 95% or add fake-quant; §6.B P-checks).
4. Both cards streaming from host at once vs the 57 GB/s DDR ceiling (Flash-Next only).
5. Decode and prefill overhead with the graft (M4), and PLE under EP.

### 7.6 Corpus row-load pass (§5.3, §I.1)

Measured at 0.21 M tok/s per process for the vectorised 16-head hash plus `np.add.at`: 1.3 h per 1B tokens per core, 13 h for 10B, before tokenisation (`P/redteam-feasibility/count_bench.out`). Plan: 1B tokens on 4 processes with per-process counters (about 5 GB RAM) and a sort-and-bincount kernel, est. under an hour on the owner's CPU; 10B only if the 1B percentiles are unstable. The Gemma key instance needs a second counter set. These counters feed ω_j (§I.1), RowSolve's `w_freq` (§5.3) and §D's frequency axis.

### 7.7 Questions for the owner

1. **Engine.** The serve is `patcarter883/minisglang-rdna4` (confirmed). Is the running build still `df29117`, and has anything changed since in `R/ple/source.py` (`finish_rows`), `graph_capture.py`, `radix_cache.py`, `qwen3_5.py` or `_process_one_msg`?
2. **Hardware.** `free -g`, `swapon --show` and `arc_summary` under load with (a) the Flash-Next serve and (b) the 35B-A3B serve. Which arm runs by default for each (EP or TP)?
3. **Downtime.** *Answered 2026-11-10: yes, the serve may be down for an evening (35B cartridge build) or a night (27B).* Still open: whether the 35B serve may run on one card while a job uses the other. Note that the AWQ 35B checkpoint is 24.97 GB (22.5 GB of body without embeddings) against one 16 GB card, so a one-card serve would run host-offloaded at PCIe-bound speed (est. far below the 102.6 tok/s EP serve); one-card 35B training with experts streamed from host is an option in `S1/local-compute/local_cost.out`.
4. **Corpus.** Which repos to ingest after minisgl, and which file types and framings match the owner's agent traffic (bare files vs `cat -n` output; §I.4 step 5)? Any paths that must never be ingested (secrets)?
5. **Flash-Next table codec and body** (for M6): fp8 E4M3 global scale or int6-g32, any `scale_override`, `MINISGL_PLE_META_FILES`; expert codec (MXFP4 vs NVFP4), HF commit, tokenizer and chat template.
6. **Caches.** `--gdn-radix`, `max_rec_snapshots`, MTP on/off for each serve; is traffic single-user, so a per-namespace flush (and, for Flash-Next v1, drain-and-flush) is acceptable?
7. **Control plane.** *Answered 2026-11-10: yes,* `/ngram/*` (cartridge load/unload/overlay/list/drop_ns) may be added to the qwen3_5 serve behind an env flag, with authentication. Still open: `/ple/*` on the Flash-Next serve (incl. the forward-only `/ple/score`), and whether cartridges should persist across restarts.
8. **Hidden states.** Can the serve export layer-1 hidden states (for §D's D and M experiments)?
9. **Table placement** on the Flash-Next serve: replicated per rank, column- or row-sharded?

### Footnote: cloud (optional speed-up only)

Nothing in the plan needs the cloud. It could shorten hyperparameter sweeps on the 27B/31B grafts or a Flash-Next per-fact cohort. RunPod prices read 2026-11-09 from a page summary (reconfirm): 1× A100 80 GB $1.39/$1.79 per GPU-hour (community/secure); RTX PRO 6000 96 GB $1.69/$2.49; 1× H200 $3.59/$5.29; 4× H100 $10.76/$15.96. The 2026-11-09 per-fact campaign table ($1.1k–14k, assuming 100–300 iterations per solve) rescales at **20–30 iterations per solve** to about **$0.2k–1.4k** (low end ×0.2, high end ×0.1, setup hours not rescaled; `S1/designrev/budget_local.out`), of which the 5k sequential run is $165–1,058. Cloud Flash-Next options remain as INTEGRATION §4.2 (C): 1× H200 with the MXFP4 body resident, 1–5 s per iteration; prompts leave the machine.

---

## 8. Milestones (re-sequenced 2026-11-10: grafts first)

Order: M0 → M1 → M2-0 → M2 → M3 → M4 → M5, with the dissection track (§D) alongside from M0; M6 (Flash-Next native) and M7 (Stage-3 online per-fact edits) later; M8 Gemma, M9 scale. Criteria IDs (E1, MEM, R1–R3, C1, NG…) are defined in §6.A.7; invariants I1–I4 and P-checks in §6.B. Wall-clock is **est.** (§7).

| M | What | Where | Pass | Kill / stop |
|---|---|---|---|---|
| **M0** | Package skeleton (§3.1) with `requirements-dev.txt`. Port the P/ proofs into `tests/`: hashing parity (I4) against real buffers; native locality/decode/gradient on tiny qwen4_exp; graft no-op, decode, padding and gradient on tiny qwen3_5, **qwen3_5_moe** and gemma4; graft built without `Qwen4ExpTextConfig`; **G-dir arm** no-op, decode-state and gradient tests (§2.4); unigram-head subclass; graft-beside-native on tiny qwen4_exp; graft `crop()`; Gemma PLE editor-init; ledger rollback; `ngram-edit/v1` + cartridge manifest round-trip HF adapter ↔ overlay prototype with bitwise equal logits. `ingest/` pipeline on the minisgl corpus: blob reader, secret scan, dedup, Splits A/B, framings, windows with true history; row census reproduces `S1/ingest-algo/count_rows.out` (7,555,967 rows) and the 11,405/11,405 HF stub parity. Fixtures + `fetch_fixtures.py`; vendored `R/ple/hashing.py`. Start the 1B-token row-load count. | this container | All invariants bitwise as in P/; census exact; default suite < 10 min on 4 CPUs; slow solver proofs marked | none (engineering) |
| **M1** | **Proxy ingestion, Qwen2.5-0.5B graft (first data point).** Done once (§I.8): at the fixed context E1 −0.52% [−0.67, −0.33] with stdlib +0.58% (cold 512-token windows: −1.14% / +0.51–0.83%); the pruning frontier tops out at ≈ 0.3% gain for retention-safe rows. **This is the number M1b must beat, not 1.1%.** **M1b** (pre-registered grid; selection on Split-A dev, one test read; Split A on the Flash-Next-style KeySpec over the Qwen2.5 vocabulary; activation checkpointing for 4k windows): about **8 trained arms, one seed, G-lin unless stated**, one factor at a time — count ≥ 2 + ω_j + KL anchors at ρ = 0.01 (base arm), the same at ρ = 0.003 and 0.03, enrich ≥ 4 at the best ρ, G-dir at the best setting, a **foreign-repo cartridge** and a **train-time key-shuffle** arm (C1), LoRA all-linear r=16; eval-only: base, n-gram/cache-LM interpolation, RAG-BM25 1k/4k, kNM-lite, NGM, eval-time permuted keys, constant bias; the headroom number (§I.3); Split-B scoring of the chosen arm. Deferred: second seed, LoRA r=48, HumanEval/MBPP (to M4, or batched generation here if time allows). | CPU here: ≈ 45–65 min per trained arm uncontended, so ≈ 6–9 h of training plus ≈ 8–10 h of eval-only conditions (RAG 4k ≈ 3× an evaluation point) — **about one day**, more when the container is shared; or one RDNA4 card in a few hours | Gates; operating-point decision of §6.A.7 with these bars: E1 ≥ 1.5% (lower bound > 0.5%) **with** R2 ≤ +0.5% and R1 ≤ +0.2%, and the operating point lies **above the pruning frontier** (more E1 than the frontier at equal R2); C1 and NGI pass; REG; report E2, E5, MEM | no configuration with R2 ≤ +1% reaches E1 ≥ 1% (lower bound > 0.5%) at the fixed context → rows-only layer-1 ingestion does not transfer from the proxy; stop before M2 and report. In the 0.5–1% zone: one Stage-1b (distillation-with-context) arm, then decide |
| **M2-0** | **One-layer feasibility on the owner's machine** (written in M0, run before any M2 build): custom AWQ loader from safetensors (experts packed; gptqmodel-free), one real 35B layer of each kind (GDN, attention, MoE) forward + dX backward with timings and per-card peak VRAM at the real chunk sizes; GDN at B=4, T=4096 with the reference backward and with the fixed backward (§4.6); `grouped_mm` on gfx12. | owner's 2 cards, serve stopped; ≈ 1–2 h to run (est.) | (provisional) the fixed GDN backward within 1.5× of the native forward's cost per token and dX cosine ≥ 0.9999 vs HF's torch delta rule; per-card peak ≤ 15.5 GB at the planned split | GDN backward cannot be brought below 0.2 h per 1M tokens on the 35B → re-cost M2 before starting it |
| **M2** | **Stage-1 cartridge of the owner's repo on a Qwen3.6-35B-A3B graft** (`cyankiwi/…-AWQ-4bit`). Prereqs: M2-0 passed; `gdn_hip` widened to `Qwen3_5MoeGatedDeltaNet` and validated (forward rel. err ≤ 1e-3 and dX cosine ≥ 0.9999 vs HF's torch delta rule on real layer shapes); AWQ dequant bit-exact vs a reference unpack; W4A8 vs bf16 expert top-1 agreement ≥ 95% (else fp8 fake-quant); mixed chunking (4k GDN/attention, 8–16k MoE). Then: 3-point lr pilot × 100k tokens per arm on dev, 1–3 epochs of Split A train; arms G-lin and G-dir (and rows + W_v); best M1 row policy and ρ; LoRA all-linear r=16 baseline; RAG and RAG + cartridge; n-gram interpolation and foreign-repo controls; all §6.A metrics except HumanEval/MBPP (M4). | owner's 2 cards, serve stopped, `gpu-lease -n 2`; **several days of engineering** (loader, GDN backward, chunking, eval harness) plus ≈ 0.7–2.6 h per 3-epoch arm with the fixed GDN backward (3.1–7.5 h without), ≈ one or two evenings for the sweep | Gates (R0 bit-exact OFF, I2, host hash, I4); §6.A.7 operating-point decision: E1 ≥ 2.0% (lower bound > 0.5%) with R2 ≤ +1.0%, R1, R3 ≥ 99%; E2 ≥ 1.5%; E3 ≥ 3%; C1; NGI; REG; E5 ≥ 50% of LoRA's E1 gain; 3-epoch build ≤ 4 h wall-clock | **joint, at the operating point:** no configuration with R2 ≤ +1% reaches E1 ≥ 1% (lower bound > 0.5%) after pilot + 1 epoch in both G-lin and G-dir → stop rows-only graft ingestion on this host; the 0.5–1% zone → one Stage-1b arm, then decide; E5 dominated (< 25% of LoRA **and** R2 no better) → LoRA wins on quality, re-scope the cartridge case to servability/reversibility/per-commit cost |
| **M3** | **Qwen3.8-27B graft** (`amd/…-Quark-AWQ-MXFP4`), same recipe and arms. Prereq: the GDN patch keeps the MXFP4 projections packed with just-in-time dequant (§4.6; a bf16 copy does not fit); keys and rows directly comparable with Flash-Next (same tokenizer): I4 against Flash-Next's buffers on ≥ 10k positions; feeds §D experiments E and F. | owner's 2 cards, serve stopped; ≈ 2.9–5.7 h per 3-epoch arm with the fixed GDN backward (6.8–13.5 h without) | Same bars as M2; cross-host consistency: E1 as a fraction of the host's own LoRA gain within ×0.5–2 of the 35B's | as M2; additionally, a 3-epoch build > 12 h measured → demote the 27B to dissection-only (E/F) |
| **M4** | **Serve a grafted model in minisgl** (35B-A3B first, 27B second): the §4.7 port — `ngram_graft.py` (linear + general modes), numpy `cartridge.py`, bf16 staging, vectorised decode hash, per-request `ngram_cartridge`, per-namespace radix roots with carry-over or per-conversation pinning, `/ngram/*` control plane, all-ns-0 GEMV skip; EP: test the dummy-prefill path or start `EP=0`. | owner's serve | Bit-exact no-op: empty cartridge and ns-0 requests vs the ungrafted serve (`torch.equal` logits + 512-token greedy over prefill, chunked prefill, decode, eager and graph replay, radix hits); mixed batch: ns-0 request identical to running alone; GPU parity with the HF graft within the ungrafted minisgl-vs-HF noise floor; realised/predicted ΔNLL on held-out project text ≥ 0.7; decode overhead ≤ 1.5% (est. 0.5–0.7% on the 35B), prefill ≤ 5%; both EP replicas on the same cartridge version; spec-verify with rejections equals non-spec greedy | realised/predicted < 0.5 even with fp8 activation emulation (K-transfer) → train with W4A8 emulation and bf16 STE on rows, then re-test; decode overhead > 5% → move to sharded W_v / skip paths before Stage 2 |
| **M5** | **Stage 2: incremental updates on the served 35B cartridge.** Replay the 110 post-`T_cut` commits in nightly batches (§I.6): reader index, delete dead rows, reset-and-refit dirty rows, child patches, namespace bump with carry-over. | GPU window (≈ 0.17–0.31 h per median–p90 commit incl. stop/boot) or CPU beside the serve for small diffs | Split-B gain ≥ 80% of a full retrain; forgetting on unchanged files ≤ 0.2% NLL; rollback bit-exact; nightly batch ≤ 30 min wall-clock including stop/boot | incremental < 50% of full retrain → nightly full rebuilds only (and record the negative result) |
| **D-track** (from M0) | §D: **G** (row statistics at ≈ 5k n-grams/order) and **S** (2-layer Flash-Next slice) now; **A** after M6a; D/M/E/F after M3. | this container (G, S); owner's serve (A) | G: coherence-vs-frequency curve with f* and a bootstrap CI; S: per-stream gate, direct/conv share and ‖PLE‖/‖X‖ on ≥ 20k tokens. Each experiment's decision rule (§D.3) is applied to M2/M3 arms | F dropped if < 10% of the trained cartridge's gain; A deferred if M6a slips |
| **M6** | **Flash-Next native.** **M6a** (was M1): overlay patch at `R/ple/source.py:242-244`, drain-and-flush, forward-only `/ple/score`; P0–P2; a synthetic diff (0.1×/1×/10× row norm) on one 2-gram's rows: I1 on 200 non-reading prompts, ΔNLL/KL on reading prompts (dose-response through the conv-dominated path), I3, P5 — bs=1, sequential, fixed chunk, vs a no-edit control. **M6b**: native cartridge N and beside-graft G (G-lin, G-dir) on the same rows (§6.A.6), during downtime. | owner's serve (M6a, forward-only); both cards, serve stopped (M6b, ≈ 1–5.6 h per 3-epoch arm) | M6a: I1, I3, P5 under the §6.B serve rule; `/ple/score` matches HF on a shared prompt set. M6b: §6.A.7 gates and E1/R1–R3; NG ≥ 80% → "format portable for ingestion" | M6a: I1/P5 fail → engine bug, fix before any native write. M6b: N and G both < 0.5% E1 → stop native ingestion at layer 1; N < 0.5% with G ≥ 1% → native co-tenants/gate are the problem, use the beside-graft |
| **M7** | **Stage 3: online per-fact edits** (was M2-lite, M2, M2g, M3). First on the 35B graft with its cartridge active (project facts; EditRecord `parent = cartridge@version`), then global CounterFact facts; then Flash-Next native (arms a in-place, b XK, c O-site). Logs iterations per solve (20–30 expected). | 35B: GPU windows (10–30 s/edit) or CPU beside the serve (12–38 min/edit); Flash-Next: INTEGRATION (A) nightly, (B) downtime | §6.B bars: efficacy ≥ 0.9; subject-preserved generality ≥ 0.6 (≥ 0.8 by the end of M7); realised/predicted ≥ 0.7; I1. Composition: the cartridge's E1 retained ≥ 95% after 100 facts in its namespace; facts retain ≥ 0.8 after a cartridge increment (re-solve otherwise) | K-native (with attribution), K-transfer, K-leak, K-native-short, K-graft (§6.B) |
| **M8** | Gemma (was M4, M5): E4B native with aliases (200 facts; SWEA baseline) and an E4B token-keyed cartridge; 26B/31B grafts, Flash-Next-format vs Gemma-PLE + aliases vs SWEA. | one card (E4B); both cards (26B/31B) | §6.B M4/M5 bars; for the E4B cartridge, §6.A.7 E1/R2 | Efficacy < 0.5 at the trust region → Gemma PLE is not a fact store; K-graft for both formats |
| **M9** | Scale and steady state (was M6, M7): several cartridges (≥ 3 repos) switched per request; sequential 100 → 2k (→ 5k) facts with conflict re-solving; targeted radix detach + epoch guard, persistence, per-rank hash agreement (P6). | owner's machine | §6.B K-scale bars; per-request switching overhead within M4's bounds; P5 with targeted detach equal to drain-and-flush | K-scale → exact-key overlay (XK) if M7 attributed failures to hashing |

**Old → new milestone IDs** (old IDs still appear in §5 and §6.B text and in `serve/INTEGRATION.md`): M0 → M0; M1 → **M6a**; M2-lite, M2, M2g, M3 → **M7**; M4, M5 → **M8**; M6, M7 → **M9**. The cloud M2-lite arm is dropped (§7.4).

Why this order: M1 can kill the approach cheaply on CPU; M2 is the cheapest real-weight host that is also the daily driver and the architectural twin of Flash-Next; M3 shares Flash-Next's tokenizer, so its rows feed the dissection; M4 turns a cartridge into something the owner uses daily; Stage 2 needs a served cartridge; native and per-fact work depend on the Flash-Next control plane and on the cartridge namespace they are solved against.

---

## 9. Risks, open questions, prior art

### Risks

1. **Retention on general code (Stage 1, binding).** Project rows are read by 70–74% of positions of other Python (§I.1); the proxy already shows +0.5–0.8% stdlib NLL, all from exact-key rows, against a fixed-context gain of only 0.5% (§I.8), and pruning alone trades gain for cost along a frontier that is ≈ 0.3% at near-zero cost. *Mitigation:* row policy (count ≥ 2, enrichment/TF-IDF, cap on the most generic rows), reuse weight ω_j, KL anchors, ρ ∈ {0.003, 0.01, 0.03}, per-request switching (the cost is paid only inside project traffic). R2 and E1, jointly at one operating point, decide.
2. **Memorisation instead of ingestion.** Train gain 16× held-out gain on the proxy (§I.8). For a cartridge that serves the files it ingested this is partly the point; the risks are verbatim regurgitation and secrets. *Mitigation:* frequency filter, group lasso, early stopping on Split-A dev; REG (§6.A.3) and the secret scan; MEM is reported as a diagnostic.
3. **The gain is style, not knowledge.** 80% of the proxy gain is docstring/comment prose; project identifier first sub-tokens move −0.58% while general ones move −2.52%, and project repeats get worse. *Mitigation:* report by token class; E3/E4 are gated separately; the foreign-repo cartridge (C1) and n-gram interpolation (NGI) controls; the distillation-with-context objective (§I.1) is the first upgrade.
4. **Graft regime mismatch.** Flash-Next's PLE output is dominated by a direction-only conv branch; the G-lin graft has none (§2.4, §D.2). *Mitigation:* the G-dir arm and §D's S experiment.
5. **Local toolchain on gfx1201.** `gdn_hip` patch misses the MoE class name and densifies the 27B's MXFP4 GDN weights; its backward is a pure-torch Python-loop reference (≈ 60× HF's op count; §4.6); the 35B does not fit at 8–16k-token chunks without mixed chunking; AWQ needs a custom loader; `grouped_mm`, SDPA kernels for head_dim 256/512 and real per-layer efficiency are unverified. *Mitigation:* M2-0 first; fallbacks exist (HF torch delta rule in the backward, chunked bmm, chunked attention).
6. **Serve integration of grafts.** PLE never ran under EP (the 35B default); GPU parity needs a tolerance; every Stage-2 version cold-misses the prefix cache without carry-over (§4.7). *Mitigation:* M4 gates; `EP=0` first if needed.
7. **Conv dominance on native Flash-Next** (Stage 3, M6–M7). Small edits cannot steer the direction-only conv term; row edits reached only R² 0.29–0.36 with the conv on (`P/t1/e_express.out`, proxy queries). XK does not remove this limit (§5.6). M6a's dose-response and M7's arm (c) decide.
8. **Shared bigram rows leak** (Stage 3). A 2-gram edit fires in every context containing the bigram; the prior same-subject leakage is 19–25%. *Mitigation:* KL term, right-crossing keys, per-row full-load penalty.
9. **Short subjects have no subject-only native key** (Stage 3). One third of subjects are 1 token. K-native-short decides.
10. **Collisions as writes accumulate.** Within one cartridge the per-head collision rate is 0.56–0.72% at count ≥ 2 and 1.3–2.9% for all rows (§I.5); collisions contributed nothing measurable in the proxy (§I.8), but several cartridges plus facts in one namespace raise occupancy. *Mitigation:* per-request namespaces, the XK overlay for facts.
11. **Solve/serve numeric mismatch.** 4-bit experts, W4A8 activations on the 35B, fp8/int6 tables on Flash-Next. Gated by the P-checks and M4's realised/predicted ratio.
12. **Downtime contention.** Builds need serve downtime and both cards; the cards are shared across repos through `gpu-lease`.
13. **Secrets in cartridges.** Rows memorise code and a cartridge ships as a file: secret scan before ingestion (§I.4), cartridges treated like source.
14. **Ephemeral evidence.** Binary fixtures (tokenizers, trained rows, NLL arrays) still live only in the session scratchpad; M0's fetch script removes the dependency.
15. **Security.** A writable table is a backdoor surface (BadEngram). Patches carry provenance and fingerprints; the serve's control planes must be authenticated (**owner decision**).

### Open questions

- Does any objective beyond next-token NLL (distillation with the repo in context, synthetic queries) move identifier first-use and cross-file metrics, which keying alone cannot reach?
- What share of a project request is non-project text (π_other in ω_j)?
- Does the G-dir arm (fixed direction branch) ingest better than G-lin at equal R2, as Flash-Next's structure suggests?
- Is Flash-Next's gate context selection or a collision denoiser (S)?
- Real torch-ROCm throughput per layer on gfx1201, and PLE under EP.
- Whether the fitted Flash-Next → host map M (§D) beats training each host from scratch (predicted: no, for project n-grams).
- Which reset token Gemma grafts should use (`<eos>` vs `<end_of_turn>`); whether alias handling belongs in the tokenizer or the serving loop.
- How many iterations a per-fact solve needs (M7 measures; 20–30 assumed).

### Prior art

From `docs/research/engram-relaunch-proposal.md` and the reviews; arXiv abstracts re-read for BadEngram, EngramEdit and LongCat-Flash-Lite only.

- **EngramEdit (2610.10533).** Edits the natively pretrained n-gram memory of LongCat-Flash-Lite (2601.21204). Two-stage: δ* with the model frozen, including an essence term and a norm clamp, then regularised least squares into n-gram rows with a reuse and frequency penalty. App. A.3.2 covers the gated case; B.4.3 describes exact-key side vectors. The XK mode here is B.4.3, and the per-row penalty here is its A.3.3 frequency penalty moved from per-n-gram to per-row.
- **BadEngram (2609.13478).** Modifies only the memory parameters of Flash-Next's native PLE, with the backbone unchanged, to plant a trigger (47.8% HarmBench ASR). Same table, a trigger-keyed write, optimised through the frozen model.
- **User-as-Engram (2606.19172).** Post-hoc per-user row overlays on an Engram-native base (as described in engram-relaunch-proposal.md:467; abstract not re-read).
- **Cross-Model Memory Transfer (2608.17050).** Moves a frozen Engram memory to a different model with a trained reader.
- **Engram Adapter (2608.29327):** a frozen-base Engram adapter whose residual is about 0.08% of the hidden norm.
- **SWEA (2401.17809):** token-id-keyed subject embeddings at the input. **NeuralDB (2507.18028).**
- **GRACE (2211.11031), WISE (2405.14768), MEMOIR (2506.07899):** discrete keyed edit codebooks on frozen models.

**Prior art for Stage 1 (ingesting a corpus into an n-gram or embedding memory of a frozen model; repository-specialised code models).** Source: the 2026-11-10 prior-art report (papers read that day). **Provenance:** `S1/priorart/` holds only `pdf/knm.txt`, `cceval.txt` and the n-gram statistics, so only the kNM-LM and CrossCodeEval figures are backed by saved excerpts; the Engram Adapter, Cartridges, EntiGraph, Tencent, Memory Decoder, Sparse Memory FT and other figures below are **as reported, not re-checked** until excerpts are saved (M0 task).

- **No published work we found trains n-gram rows on a codebase with the backbone frozen, and none writes a corpus into a model's *native* pretrained n-gram table.** Count-based n-gram and cache language models of code are, however, old prior art and the obvious zero-training competitor (next item).
- **Count-based n-gram / cache LMs of code** (recalled from memory, not re-read here): Hindle et al. 2012, "On the naturalness of software" (ICSE; n-gram LMs capture code's repetitiveness); Tu, Su & Devanbu 2014, "On the localness of software" (FSE; a cache component for local repetition); Hellendoorn & Devanbu 2017, "Are deep neural networks the best choice for modeling source code?" (FSE; a nested-cache n-gram model competitive with neural LMs on in-project code); infini-gram (arXiv 2401.17377; ∞-gram interpolation improves neural LMs' perplexity). Source of the NGI baseline (§6.A.4, §6.A.7).
- **Engram Adapter (Hou & Wang, 2608.29327; closest prior art).** Fully frozen Qwen3-4B/8B; trains 12 fresh hashed tables (90,007 rows × 128/256), a zero-init projection and a scalar gate per injected layer (layers {1, 3, 7, 15} of 36), with non-trainable occupancy masks (a lookup counts only if all k hashes are occupied). About 154 M parameters, 1.2× LoRA r=64; 1k–10k examples, one epoch, AdamW 1e-3. In-domain: 4B AG-News 85.14 → 87.93 (LoRA r=32 90.15), MedMCQA 53.65 → 55.79 (57.10), 8B AG-News 79.49 → 86.47 (LoRA r=64 90.29): about **55–65% of the best LoRA-family gain**. OOD retention 99.38% / 100.07% (4B), 101.0% (8B; LegalBench 72.53 vs LoRA's 44.67); MBPP 97.7%. Injected residual 0.078% of hidden norm over all OOD tokens, 0.28–0.29% on activated tokens (P95 1.1–1.2%), 26–28% of OOD tokens activate. Limits: classification/MCQ domains, one family, ≤ 8B, no composition, Python hashing costs +9–11% decode and up to +40% prefill. **Positioning:** its tables are fresh and small with a trained projection; ours are a native pretrained table or a graft with a fixed projection, at 2–7.6 M rows per repo — so its numbers are a prior to calibrate against (the E5 bar uses its 55–65%), not a prediction.
- **DeepSeek Engram (2601.07372).** Co-trained (262B tokens; table Adam at 5× lr, no weight decay; conv zero-init); zeroing it at inference keeps 29–44% of factual-knowledge scores but 81–93% of reading comprehension (§6.3). No table-only or frozen-backbone training.
- **LongCat-Flash-Lite (2601.21204).** 31.4 B n-gram parameters co-trained from scratch; mixed coding gains (HumanEval+ 28.7 → 31.1, MultiPL-E 30.2 → 30.0). No evidence that the n-gram axis is unusually good for code.
- **STAPLE (2610.12599).** Token-id tables in the FFN; only the up-projections frozen, ≈ 100 B tokens of warm-up; no fully frozen run. No support for "backbone frozen, tables only" at our budget.
- **User as Engram (2606.19172).** Per-user row writes on self-pretrained 178 M–1.22 B Mini-Engram models: +0.00005 bpb contamination vs +1.784 for per-user LoRA, at about 16 rows per fact and < 1% occupancy. Row writes keep locality **when occupancy is sparse**; a repo write hits 0.4–2.2 M n-gram types, a different regime.
- **Cross-Model Memory Transfer (2608.17050).** Hashed memory trained on 10–30 M tokens, then frozen with a small trained reader: Mistral-7B QA 32.1 → 38.5–38.8 (RAG 33.4, LoRA 32.3). **Random memory scores 34.6 and matches the perplexity gain**; permuted keys 32.5. Hence the random-row and permuted-key controls; since there the random memory had a trained reader, while untrained random rows here can only hurt, C1's content controls are the foreign-repo cartridge and the train-time key shuffle (§6.A.4).
- **NGM (2605.16893).** Training-free causal 2/3-token embedding averages at layers 2 and 15: +0.98 to +1.72 average on Qwen3 0.6–8B. A Stage-1 cartridge must beat this generic floor on project text.
- **Sparse Memory Finetuning (Lin et al., 2510.15103).** Updates only top-t slots of a *pretrained* memory layer by TF-IDF against background access counts, with SGD: NaturalQuestions F1 −11% vs −71% for LoRA and −89% for full FT at equal new-fact learning. Closest analogue to rows-only updates on a native table; source of the TF-IDF row-policy arm and the simple-optimiser default.
- **Memory Grafting (2605.20948).** Forward-only memory values from a frozen donor (recipient pretrained with the memory): a precedent for forward-only row fills (§D experiment F).
- **kNN-LM / kNM-LM (1911.00172; ASE 2023, 2308.09313).** Forward-only datastore of in-project tokens the LM mispredicts: Android API line EM 5.99 → 13.74 (CodeGPT), intra-project token accuracy 54.6 → 69.7, at 2.4× lower throughput; FT2Ra (2404.01554) beats it (26.32 vs 13.93 EM). The kNM baseline in §6.A.4.
- **Memory Decoder (2508.09874)** (a plug-in 0.5B decoder trained to imitate kNN; Qwen2.5 domain perplexity 17.0 → 3.7 at 0.5B) and **Cartridges (HazyResearch, 2506.06266)** (a trained KV prefix via self-study and context distillation; distillation beats next-token, MTOB 33.5 vs 24.9; MMLU 54.7 → 54.3 vs LoRA → 45.3; ≈ 30 min on 8×H100 per corpus). Strong frozen-backbone baselines that need a second model or a KV prefix in the serving path. Knowledge Modules / Deep Context Distillation (2503.08727) is the same lesson for LoRA.
- **EntiGraph (2409.07431).** 1.3 M source tokens → 455 M synthetic; Llama-3-8B QuALITY 39.49 → 56.22 (RAG 60.35; both 62.60); raw continued pretraining on the small corpus scored below base. Augmented data matters more than epochs (inference for code, untested).
- **Repository fine-tuning vs RAG.** Tencent (2505.15179, FSE'25 Industry): 401 M tokens of internal C++, six models, EM base 24.79 / FT 44.20 / **BM25 RAG 53.76** / FT + BM25 57.43; FT cost HumanEval −24.1%, MBPP −11.7%. Semantic Scopes (2602.05780): FT beats RAG on Levenshtein distance in 24/24 settings (forgetting not measured). Benchmarks: CrossCodeEval, RepoBench, RepoEval/RepoCoder, EvoCodeBench, ExecRepoBench, REPOCOD, Long Code Arena, CodeUpdateArena (§6.A.8).
- **Expected outcome (est., from these and §I.8):** a small perplexity drop on held-out repo code at realistic context (the proxy: 0.5%), mostly idiom and docstring style, with identifier gains on continuations; unlikely to beat BM25 repo-RAG on cross-file completion; whether it is **additive with RAG** at R2 ≤ +1.0% is the test (E6), and whether it beats a count-based n-gram interpolation is the first thing to establish (NGI).

### What would be novel (narrowed)

Stage 1–2:

1. **Ingesting a codebase into a production model's native, pretrained n-gram table** (Flash-Next PLE) with the whole backbone frozen, and the same into grafts on models trained without one (35B-A3B, 27B), **with one row format, one trainer and identical keys**, including the controlled native-vs-graft comparison on one host (§6.A.6) and the dissection that explains it (§D).
2. **Code repositories as the target of an Engram-family memory**, with background-relative row selection on a dense hashed table, a residual budget, content-free controls and a commit-time split on private repos.
3. **Namespaced, switchable, reversible cartridges with per-commit row diffs**, served through a torch-free overlay on consumer AMD GPUs.

Stage 3 (unchanged from 2026-11-09):

4. **Hashed co-tenant in-place writes** into a shipped native table, measured against EngramEdit-style exact-key (XK) writes on the same table and the same facts, with the three-arm attribution of §6.B.
5. A **live-serve diff and cache-invalidation contract**: engine-agnostic, reversible row diffs with an epoch guard and radix invalidation, applied to a running Flash-Next serve.

Not claimed as novel: key transfer across models that share a tokenizer (true of any token-id hash); optimising memory parameters through a frozen model (BadEngram, EngramEdit, Engram Adapter); frequency-priced penalties (EngramEdit A.3.3); TF-IDF slot selection (Sparse Memory Finetuning).

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
- `P/stage1/` (= `S1/`): Stage-1 research of 2026-11-10 — `priorart/` (n-gram overlap with a general-code background, CrossCodeEval text), `ingest-algo/` (row census, policies, optimiser bench, per-commit increments, wall-clock), `design/` (the earlier attempt whose counts `ingest-algo/` reproduced), `compute/` and `local-compute/` (local throughput models, token counts, CPU calibration), `evalproto/` and `evalproto2/` (splits, QA seed, power, token-class deltas, QA reach), `experiment/` (0.5B graft ingestion) and `verify-experiment/` (its skeptic re-run, leak and token-class checks), `designrev/` (this revision's GDN class check and budget arithmetic).
- `P/graftfirst/` (= `GF/`): `REPORTS.md` (three reports + skeptic verdicts) and their scripts and outputs.

Prior evidence: `/home/user/memory-organ/docs/research/gemini-engram-code-review.md` (§5, §8), `/home/user/memory-organ/docs/research/engram-relaunch-proposal.md`, `/home/user/memory-organ/docs/research/engram-graft-assessment.md`.

---

## Review log

### Revision 2026-11-09: three red-team reviews

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

### Revision 2026-11-10: owner's new direction (Stage 1 ingestion, local-first, grafts first)

| # | Change | Where | Basis |
|---|---|---|---|
| N1 | Project restructured into Stage 1 (offline codebase ingestion into project cartridges), Stage 2 (incremental git-diff updates), Stage 3 (online per-fact RowSolve) | §1, §3.5, §I, §5 title | owner direction 2026-11-09 |
| N2 | New section §I Ingestion: objective, regularisers, row-wise Adagrad, data pipeline, measured row counts (7,555,967 rows; 2.04 M at count ≥ 2), increments, composition, what 2/3-gram rows can and cannot hold | §I | `S1/ingest-algo/*.out`, `S1/evalproto2/qa_reach*.out` |
| N3 | 0.5B graft ingestion added as evidence with the skeptic's corrections (collisions contribute nothing; gain mostly frequent prose n-grams; injection 47–114× Engram Adapter's) | §I.8 | `S1/experiment/run_full.stdout`, `S1/verify-experiment/verify.stdout`, `tokclass.json` |
| N4 | Evaluation split into §6.A (Stage-1 protocol: Splits A/B, token-class and key-category breakdowns, R0–R3 with general code as the binding retention constraint, RAG/LoRA/Engram-Adapter/kNM/NGM baselines and random-row/permuted-key controls, numeric criteria with power) and §6.B (per-fact protocol, unchanged) | §6 | `S1/evalproto*/` |
| N5 | Compute plan rewritten local-first in wall-clock; dollar table moved to a footnote and rescaled to 20–30 iterations per solve ($1.1k–14k → about $0.2k–1.4k); the venue conflict resolved in favour of local | §7 | `GF/REPORTS.md` skeptic figures, `S1/local-compute/local_cost.out`, `S1/designrev/budget_local.out`, INTEGRATION §4.2–4.3 |
| N6 | Milestones re-sequenced, grafts first: M1 0.5B proxy, M2 35B-A3B AWQ graft, M3 27B Quark MXFP4 graft, M4 graft serving in minisgl, M5 Stage 2, D-track alongside, M6 Flash-Next native, M7 per-fact, M8 Gemma, M9 scale; numeric pass/kill per milestone; old → new ID map | §8 | owner priority 2026-11-10 |
| N7 | Graft profile: two arms (G-lin, G-dir) because Flash-Next's PLE output is dominated by a direction-only conv branch (1.4–13.9× the direct term); output scale from the site probe, not embedding RMS | §2.4, §D | `GF/skeptic_dissect/branch_share.out`, `embed_freq.out` |
| N8 | Graft hosts locally: checkpoints, dequant-per-layer training with dX-only autograd, ~180 s / 135 s per 1M tokens of dequant, large per-layer MoE chunks, memory-based layer split; owner decision to import the fork's `gdn_hip` patch | §1, §4.6 | `GF/REPORTS.md` graft-hosts-local + skeptic |
| N9 | **New check:** `patch_qwen3_5_gdn` matches the exact class name `Qwen3_5GatedDeltaNet`, so it patches 0 layers on the 35B-A3B (`Qwen3_5MoeGatedDeltaNet`, identical source modulo the prefix); M2 prerequisite | §4.6, §8 M2 | `S1/designrev/gdn_class_check.out`; `R/gdn/hf_patch.py:104` |
| N10 | Graft serving in minisgl: port plan, numpy cartridge store, per-namespace radix roots with required carry-over, EP as the 35B default (PLE never run under EP), GPU parity needs a tolerance, all-ns-0 GEMV skip instead of the active mask, ~0.5–0.75% decode overhead | §4.7, M4 | `GF/REPORTS.md` graft-serve + skeptic |
| N11 | New section §D Dissection program (G, S, A, D, M, E, F, N), with the skeptic's corrections on conv, gate-constant interpretation and output scale, and numeric decision rules | §D | `GF/REPORTS.md` dissection + skeptic |
| N12 | Prior-art positioning for Stage 1 (Engram Adapter as closest; Sparse Memory FT, Cross-Model Memory Transfer, NGM, kNM-LM, Cartridges, Memory Decoder, EntiGraph, Tencent FT-vs-RAG) and narrowed novelty | §9 | `S1/priorart/` |
| N13 | Experiment scripts and outputs copied into `P/stage1/` and `P/graftfirst/` (binaries excluded) | header, §9 Sources | — |

Kept unchanged: the format spec (§2.1–§2.3, §2.5), architecture and patch contract (§3.1–§3.4, extended), native and Gemma backends (§4.1–§4.5), RowSolve (§5), the per-fact protocol and invariants (§6.B), the 2026-11-09 review log above.

### Stage-1 revision 2026-11-10: science and feasibility red-team of the restructure

Each finding was checked against the cited scripts, outputs or code before it was applied. The reviewers' scripts and outputs are copied to `S1/rt-science/` (`ctx_len`, `nohit`, `cluster_ci`; they reuse the skeptic's `rows_armA.pt`, not copied) and `S1/rt-feas/` (`count_rows`, `row_policy` re-runs, `gdn_ref_ops*`, `hf_gdn_ops`, `cpu05b`). "S" = science, "F" = feasibility. The pre-revision file was saved (ephemeral) as `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/DESIGN.pre-stage1-revision.md`.

| # | Finding | Verdict | Check / reason |
|---|---|---|---|
| S1 | Proxy gain halves at the design's fixed context (−1.14% → −0.52%), stdlib cost does not (+0.51/+0.58%) | **applied** (§I.0, §I.8 corr. 6, §8 M1, §9 risk 1) | `ctx_len.out`: short −1.135% [−1.31, −0.92], long −0.517% [−0.67, −0.33]; `ctx_len.py:66-85` uses the same targets with ≥ 1,024 in-file tokens vs cold 512. Caveat kept: 24 blocks / 10 files ≥ 2,048 tokens |
| S2 | Gain/cost frontier never stated; M1b/M2 bars lie beyond it | **applied** (§I.0, §I.8 corr. 7, §I.9, M1 pass) | `verify.stdout` lines 29–30: −0.0231 / +0.0062 and −0.0099 / +0.00058 nats = 0.71% / +0.41% and 0.31% / +0.04%. Bars kept; M1b must land above the frontier. "0.15–0.3% at retention-safe settings" is marked est. |
| S3 | Early stopping and sweeps selected on the test split | **applied** (§I.1, §6.A.1, §6.A.7, M1, M2) | grep finds no dev/validation split. Dev = `0.15 ≤ h < 0.25`, test unchanged, grid pre-registered, one test read |
| S4 | Noise floor in the wrong unit | **applied** (§6.A.7 Power, §I.8 corr. 9, §6.A.1) | `cluster_ci.out`: between-file SD 0.0228, MDE 0.0114 nats @31 files = 0.43% / 0.35% of base. Kills made CI-based; corpus v1 adds test files before real-host gating |
| S5 | Row-hit "invariant" false at position level | **applied** (§6.A.2, §6.A.3 R1, §I.8 corr. 8) | `nohit.out`: 4.0% no-hit, mean \|Δ\| 0.127, 0.15% bit-equal; prefix-clean 0.006%, \|Δ\| 0.0 |
| S6 | Undecided gaps; M2 kills not joint | **applied** (§6.A.7 operating-point table, M1, M2) | the old E1 KILL and R2 KILL were judged at different settings; one operating point (best dev E1 with dev R2 ≤ +1%) with an action per zone |
| S7 | Central claim has no kill; "near-zero" undefined | **applied** (§1, E6 kill, §9 expected outcome) | E6/QA KILL were "none"; E6 now kills at < 0.25% or CI including 0; "near-zero" defined as R2 ≤ +0.2% and not claimed |
| S8 | No foreign-repo content control; "permuted keys" undefined | **applied** (§6.A.4, C1, §I.9, M1b) | `tokdelta.out` lines 9, 15, 17: gen first −2.52% (n 179), proj first −0.58% (n 100), proj repeat first +0.54% (n 258). Eval-time permutation defined; train-time key shuffle added; random rows demoted to a sanity check |
| S9 | Count-based n-gram/cache LMs missing as baseline and prior art; "upper bound" wrong | **applied** (§I.3, §6.A.4, NGI, §9) | absent from §6.A.4 and §9. Citations recalled from memory, marked not re-read. Headroom number added to M1 (not computed in this revision) |
| S10 | MEM kill penalises the deployment case; Split B should be co-primary | **applied** (§6.A.1, §6.A.2, §6.A.7, §9 risk 2) | served traffic is the ingested files plus edits = Split B. MEM → diagnostic; REG (regurgitation, canaries, secrets) added. Split B's smaller file count (52) noted for power |
| S11 | ρ default contradicts the design's reasoning | **applied** (§I.1, §I.9, M1b) | `run_full.stdout` line 25: stdlib inj median 2.69%, mean 7.27%; held-out 3.77% / 9.14%. Grid {0.003, 0.01, 0.03}, start 0.01 |
| S12 | Proxy numbers overstated or mislabelled | **applied** (a–e, §I.8, E3) | (a) `tokdelta.out` line 20 is `ID\|cont (all)`; project-only (226 × −0.0752 + 576 × −0.0508) / (226 × 4.006 + 576 × 2.159) = −2.15%. (b) `cluster_ci.out`: 15 files in eval, max token share 0.219; `verify.py:79-82` chunks the concatenated stream. (c) 6,132 stdlib tokens. (d) stdlib 0.0727 / 0.00078 = 93×, 0.0269 / 0.00078 = 34×. (e) §I.8 setup line |
| S13 | KL anchor on the base's own project answers works against Stage 1 | **applied** (§I.1) | dropped; D_chat restricted to text with no project-specific n-grams |
| S14 | Baselines under-specified | **applied** (§6.A.4, E5, §6.A.6, §9) | (a) EA layout from §9 (layers {1,3,7,15}, 12 × 90,007) now replicated; single-site kept as an ablation. (b) all-linear / expert LoRA added and used for E5; Biderman et al. cited from memory. (c) `ls S1/priorart/` = `cceval.txt`, `pdf/knm.txt`, n-gram stats/census only; other figures marked "as reported, not re-checked" |
| F1 | GDN backward is a pure-torch Python-loop recompute | **applied** (§4.6, §7.2 brackets, §7.5 #1, M2-0, M2, M3, §I.0, §9 risk 5) | `autograd.py:305-347` backward calls `ref_gdn_prefill_core` under `enable_grad`; `:502-535` chunk `max(4, 64//B)`; native bwd only at `:369`, `:403`. `gdn_ref_ops.out` 24.1 ops/token at B=4 vs `hf_gdn_ops.out` 0.41. Time conversion is est. |
| F2 | 35B VRAM figure is the 27B's; 35B does not fit at 8–16k chunks | **applied** (§4.6, §7.2, M2) | `recheck.out` lines 13–14 (35B 15.0 / 13.8) vs 22–23 (27B 12.1 / 13.2); `recheck.py:29` 2 × 2,048-token inputs. 16.6–18.7 GB at 8–16k is the reviewer's est. |
| F3 | 27B GDN weights are MXFP4; the patch densifies them | **applied** (§4.6, §7.2, M3 prereq) | `hub_recheck.json`: 27B "other" U8 12.936 GB + BF16 0.005 GB; `hf_patch.py:28-56` copies dense weights |
| F4 | AWQ loader unscoped; M2 not runnable in hours | **applied** (§4.6, M2-0, M2, §6.A.3 R3) | `quantizer_awq.py:46` (gptqmodel), `integrations/awq.py:93-96` (`nn.Linear` only), `cm:2116-2117` (`qwen2_moe` merge); `hf_patch.py:68-75` refuses a cache. M2-0 pass/kill thresholds are provisional |
| F5 | M1 "5–6 h total" contradicts M1b scope | **applied** (M1, §7.1) | M1b pruned to ≈ 8 trained arms, one seed, one factor at a time; ≈ 1 day CPU. The 4,096-token OOM is recorded as reported (the run's output stops, `cpu05b.out`; not reproduced) |
| F6 | CPU cartridge "2–6 days" uses a different recipe | **applied** (§7.2) | `budget_local.out` line 17: 60.39–192.15 h = 2.5–8.0 days |
| F7 | Row counts from different corpora paired | **applied** (§I.0, §I.2, §I.5, §4.7) | `count_rows.out` line 6: 2,385,780 rows at count ≥ 2 on all files, bf16 + ids 746.3 MiB; 2,385,780 × 524 / 1,292 B = 1.25 / 3.08 GB; × 264 / 648 B = 0.59 / 1.44 GiB |
| F8 | One-card 35B serve cannot hold the model | **applied** (§7.7 Q3) | 24.97 GB checkpoint, 22.5 GB body vs 16 GB |

No finding was rejected. Four were applied with qualifications: S2's 0.15–0.3% expectation and F1/F2's times and VRAM are kept as estimates; S9's headroom number is an M1 task, not computed here; F5's OOM is recorded as reported.
