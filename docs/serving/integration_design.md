# CAM knowledge-editing → `minisgl-rdna4` serve engine — integration design spec

Status: design/analysis only. No code was modified and no GPU workload was run to produce this.
Scope: fold the CAM "memory tap + product-key store" editing mechanism (today: offline HF-transformers
in `cam/recall_mag.py`) into the minisgl serve engine (port 1919) as a first-class, capture-capable
serving path.

Target base for MVP: **Qwen3.5-4B GDN-hybrid** — the exact base CAM's tap is trained against.
Confirmed shape (`~/.cache/huggingface/.../Qwen--Qwen3.5-4B/config.json`): `hidden_size=2560`,
`num_hidden_layers=32`, so the trained **tap layer 24** is a valid decoder-layer index. minisgl serves
this model via `python/minisgl/models/qwen3_5.py` (`Qwen3_5ForConditionalGeneration`).

Production serve config we must remain compatible with (project memory):
`--attn hip` (native HIP, Triton-free, the only capture-capable attention path) + `--graph N` +
`MINISGL_MOE_SCATTER=0`; graph ≈20% faster TPOT than eager. The 4B is dense (no MoE), so the MoE knob
is moot for MVP but the graph and attn constraints hold.

---

## 0. What CAM is, in serving terms

Two independent objects, both trained per-base at ONE tap layer (layer 24) against Qwen3.5-4B, with the
conf-gate and learned key-pool on:

1. **The tap** — `cam/gated_tap.py::GatedMemoryTap`. A zero-init gated cross-attention module:
   `h' = h + tanh(gamma) ⊙ to_o( softmax(to_q(h)·to_k(bank)ᵀ)·to_v(bank) )`, with a learnable null
   sink slot (zero value) and an optional store-confidence gate `c = sigmoid(conf_scale·(conf/EMA −
   conf_bias))` that scales the whole injection by retrieval strength. Params live in **fp32**; the
   additive update is cast back to the base dtype (`h.dtype`, bf16) so `gamma=0` is a byte-exact no-op.
   Today it is attached to a frozen HF decoder layer by a `register_forward_hook` (`MAGInjector`).

2. **The store** — `cam/pk_store.py::ProductKeyStore` wrapped by `cam/pk_store_adapter.py::PKStoreAdapter`.
   A product-key top-k addressed value bank operating in `mem_dim` (default 512) space. The learned
   projections are `in_proj` + `norm` (base-embed→mem_dim), `subj_pool_q` (learned attention key pool),
   `store.to_wkey`/`to_wval` (write projections), `store.codebook1/2` (sub-key codebooks),
   `store.read_q[h]`/`read_o[h]`/`read_norm[h]`/`read_out_norm`/`head_bias` (multi-head read),
   `readout_q` (K-slot attention pool), `out_proj` (mem_dim→base_hidden). The **value bank `V`
   `[B, N, mem_dim]`** is episodic state, NOT a parameter (`store.init_state`).

The **production-relevant path** is Track 4 persistent (`recall_mag.py`): a standing store with **B
disjoint value banks** (`CAM_DISJOINT_BANKS`, default 32 recommended) routed by a stable hash of the
subject token-ids (`_subject_bank` → `hashlib.md5(subject_tids) % B`). Edits are written once each into
their subject's bank (`_persistent_write_one` → `adapter.persistent_write`); at query time the subject
is pooled into a read query, the bank is read (`adapter.persistent_bank` → pooled `[1,K,mem]`), the tap
injects at layer 24, and the base's next-token logit is read.

The whole read pipeline that produces the tap bank at inference:

```
subj_tids  --_e (embed→in_proj→norm)-->  [1,S,mem]
           --_pool_subject (learned attn or mean)-->  q [1,1,mem] (or [1,H,mem])
b = _subject_bank(subj_tids, B)                       # pick disjoint bank
persistent_bank(V[b], q):
    read, _, _last_conf = store.read(V[b], q, return_conf=True)   # product-key top-k, multi-head
    read = _maxsim_reduce(read)                                   # multi-vector: best head
    bank = softmax(readout_q @ readᵀ) @ read                     # [1,K,mem]
injector.set_bank(bank, conf=_last_conf, relidx=0)               # feeds the tap
logits = base(prompt)                                             # tap injects at L24
```

`out_proj` from `inject()` is NOT used on the tap path — the tap's own `to_k`/`to_v` consume the
`mem_dim` bank directly. `out_proj` is only for the Stage-1 `direct_logits` diagnostic. **So the
serving store surface is exactly: `_e`, `_pool_subject`, `persistent_write`, `store.read(...,
return_conf=True)`, `_maxsim_reduce`, `readout_q` pool — plus `store.write`.** The K-slot pooled
`[1,K,mem]` bank and the scalar `conf` are the only two tensors that cross into the model forward.

---

## 1. Where the tap lives (injection point)

**Turn the HF forward-hook into a first-class minisgl module inside `Qwen3_5Model.forward`.**

The HF hook rewrites `layers[L].output[0]` — i.e. it injects into the **residual stream after decoder
layer L**. In minisgl (`python/minisgl/models/qwen3_5.py`), `Qwen3_5DecoderLayer.forward` returns
`(x, residual)` where the *residual stream after the layer* is reconstructed by the NEXT layer's
`input_layernorm.forward(x, residual)` (fused add-then-norm, `RMSNormFused`). The exact HF-equivalent
of "layer L output hidden" is the value `residual.clone()` captured right after `layer.forward` — this
is already done verbatim for spec-decode aux capture in `Qwen3_5Model.forward`:

```python
for lid, layer in enumerate(self.layers.op_list):
    x, residual = layer.forward(x, residual)
    if cap_set is not None and lid in cap_set:
        grabbed[lid] = residual.clone()          # == HF layer-L output hidden
```

**Injection point (concrete):** in `Qwen3_5Model.forward`, after `x, residual = layer.forward(...)`
for `lid == tap_layer`, apply the tap to the residual stream. Because the residual stream is carried as
the un-normed pair `(x, residual)`, and the HF tap adds to the *post-layer hidden* `h = x + residual`,
inject by adding the tap update onto `residual` (or `x`; the subsequent `input_layernorm` sums them, so
adding to either is equivalent up to which tensor carries it — add to `residual` to mirror the HF
"output[0]" semantics and keep `x` as the just-computed block delta):

```python
if self._tap is not None and lid == self._tap_layer:
    # tap sees the FULL post-layer hidden and returns an additive update
    residual = self._tap.apply(x, residual)   # h = x+residual; residual += tap_update(h)
```

The tap module is a `BaseOP`/`nn.Module` sub-op living on `Qwen3_5Model` (constructed only when memory
is enabled), NOT a `register_forward_hook` (hooks are opaque to graph capture and to minisgl's
state-dict walk). Port `GatedMemoryTap.forward` almost verbatim; the only shape change is that minisgl's
forward is **flat `[num_tokens, hidden]`**, not `[B, T, H]`. Two adaptations:

- **Reshape:** operate on `h[num_tokens, H]` → treat as `[num_tokens, 1, H]` (T=1 per row) or keep the
  token axis and broadcast the per-row bank. The tap's attention is per-query-position over K bank
  slots, so it is naturally per-token: `q = to_q(h)  [N_tok, H]`, and the bank must be indexed
  **per token → per request**. See §3 for how a row maps to its request's bank.
- **Per-request bank fan-out:** in training the bank is `[B, K, mem]` with one row per sequence. In
  serving a batch is a flat concat of tokens from multiple requests (prefill: variable extend_len per
  req; decode: 1 token per req). The tap needs, for each token row, the bank of the request that owns
  it. Build a **per-token gather index** `row_bank_idx[num_tokens]` from `batch.reqs`/extend_lens
  (prefill) or identity (decode, 1 token/req) and gather `bank[row_bank_idx]` → `[num_tokens, K, mem]`.
  This mirrors exactly how GDN threads `state_indices` per sequence (`gdn_slots.py`).

**Prefill vs decode.** The tap is dtype/shape-identical in both; the only difference is which token
positions get injected and how the bank index is built:
- **Prefill:** inject at every prompt position of a memory-enabled request (or only the generated /
  last position — see §3 for the semantic choice). Bank index expands the per-req bank across that
  req's `extend_len` rows.
- **Decode:** one row per request; `row_bank_idx == arange(bs)`. This is the hot path and the one that
  must be graph-capturable (§5).

**Non-memory requests:** `row_bank_idx = -1` (or a reserved NULL bank row of zeros) → the tap contributes
0 for those rows (null bank ⇒ `to_k/to_v` of zeros ⇒ softmax over null sink ⇒ zero value ⇒ no-op even
before the gate). This lets memory and non-memory requests **share one batch** (essential — the engine
batches across requests indiscriminately).

**Programming the tap layer:** add `Qwen3_5ForConditionalGeneration.set_tap(tap, layer)` mirroring the
existing `set_capture_layers(ids)` (which already programs `self.model._capture_layer_ids`). The tap and
its layer index become model state, so graph capture (which calls `model.forward()`) exercises the same
injection.

---

## 2. Store as engine state

The store's **trained projections are global, read-only weights**; the **value banks are mutable engine
state**. Two distinct lifetimes:

### 2a. Trained weights (global, immutable, loaded once)

`PKStoreAdapter` (`in_proj`, `norm`, `subj_pool_q`, `store.to_wkey/to_wval/codebook1/codebook2/read_q/
read_o/read_norm/read_out_norm/head_bias`, `readout_q`) and the `GatedMemoryTap` (`to_q/to_k/to_v/to_o/
gamma/null_key/conf_scale/conf_bias/conf_ema`). These are ~tens of MB in `mem_dim=512`.

**Where they live:** a new engine-owned object, `CAMState`, constructed in `Engine.__init__` right after
`self.model.post_load()` and the KV/GDN caches (`engine.py` ~L131–216, alongside the `is_gdn_hybrid`
GDN-state block). It holds:
- `self.tap: GatedMemoryTap` (fp32, on `self.device`), handed to the model via `model.set_tap(...)`.
- `self.store: ProductKeyStore` (the read/write projections + codebooks, fp32, on device).
- `self.mem_proj: {in_proj, norm, subj_pool_q, readout_q}` (the `PKStoreAdapter` read-front-end pieces).
  Simplest: keep the whole `PKStoreAdapter` instance but drive it through `_e/_pool_subject/
  persistent_bank/persistent_write` only (its `embed` table is the FROZEN base embed — reuse
  `model.model.embed_tokens.weight`, do NOT reload the 3GB tied table; see §4).

Wire it onto `Context` as `ctx.cam_state` (mirroring `ctx.gdn_state`/`ctx.cca_state` in `core.py`), so
the tap sub-op inside the model reaches per-forward bank/conf via `get_global_ctx().cam_state`, exactly
how `GDNLinearAttn.forward` reaches `ctx.gdn_state`.

### 2b. Value banks (mutable, per-server-global standing store)

The **B disjoint banks** `V: List[Tensor[1, N, mem]]` (production-relevant path) are a **single global
standing store shared by every request**, NOT per-request. They ARE the "edited knowledge" — an edit is
a write into a bank; a query is a read. This is the correct serving semantics: edits persist across
requests until overwritten.

`CAMState.banks = [store.init_state(1, device, dtype=fp32) for _ in range(B)]` with `B =
CAM_DISJOINT_BANKS` (default 32). At `N = n_sub²` slots × `mem_dim` × fp32, a bank is small
(e.g. `n_sub=32 ⇒ N=1024`, `1024·512·4 B ≈ 2 MB`; ×32 banks ≈ 64 MB) — negligible next to the KV pool.

**Lifecycle:**
- **Init:** empty banks at engine start (or loaded from a persisted edit-set — see §4/§6).
- **Write (edit):** a control-plane op (§6) calls `store.write(V[b], key, val)` where `b =
  _subject_bank(subject_tids, B)`, `key = _pool_subject(_e(subject_tids))`, `val = _e([[new_tid]])`.
  This is the exact `_persistent_write_val` body from `recall_mag.py`. Writes happen on the scheduler
  process, OUTSIDE any forward, guarded so they don't race an in-flight capture replay (writes mutate
  `V` in place; a decode graph that reads `V` via `store.read` must not run concurrently — trivially
  true in the single-threaded scheduler loop, but see §5 for the captured path).
- **Read:** happens per request at prefill (§3), producing the K-slot bank fed to the tap.

**Per-request vs global — decision:** banks are **global**; the *read result* (the `[1,K,mem]` tap bank
+ conf scalar) is **per-request**, computed once at prefill and reused for all of that request's decode
steps. This matches training (`_persistent_preds` reads once per edit, then one base forward). A
per-request bank override (a request supplying its own ephemeral edits) is a Phase-3 extension (§6),
layered as an extra bank the read merges in.

---

## 3. Read path per forward (query formation + when)

Training has a clean "subject span" (the builder hands exact token offsets). Serving has no builder.
**Clarifications, grounded in `_persistent_preds`:**

### 3a. What the query is

In the persistent eval the read query is the **subject token-ids** (`r.subject_tids`), pooled — NOT the
whole prompt. `_persistent_preds` does `q = _e(subject_tids); q = _pool_subject(q, keepdim=True)`. So at
inference we need a **designated subject**, not the free prompt. Two serving options:

- **(MVP) Explicit subject:** the request supplies the subject span (a string or token range). The API
  (§6) carries `memory: {subject: "Danielle Darrieux"}` (or `subject_token_span: [i, j]` into the
  prompt). The server tokenizes the subject, forms `subject_tids`, computes the bank at prefill. This is
  faithful to training (subject-keyed) and unambiguous. **Recommended for MVP.**
- **(Phase 3) Implicit subject extraction:** derive the subject from the prompt (NER / last
  noun-phrase / a learned span picker). Out of scope for the buildable MVP; flagged as a risk (§7)
  because the whole store is addressed by subject identity — a wrong subject reads the wrong bank.

The subject-bank routing hash (`_subject_bank`) is on **discrete token-ids**, so write-time and
read-time must tokenize the subject identically (same tokenizer, `add_special_tokens=False`) — the
server already owns the tokenizer process; reuse it.

### 3b. When the read happens

**Prefill only, once per request.** The read (`persistent_bank`) is `mem_dim` product-key ops on B=1 —
cheap but variable-shape (subject length varies) and involves top-k + gather, which we do NOT want on
every decode step or inside a graph. `_persistent_preds` computes the bank once and reuses it across the
(batched) base forward. So:

- At **prefill** for a memory-enabled request, after tokenization, compute `(bank[1,K,mem], conf[1])`
  on the scheduler/engine and **stash it on the `Req`** (new fields `req.mem_bank`, `req.mem_conf`,
  default `None`). The tap consumes it at the prefill forward AND at every subsequent decode step.
- At **decode**, no store read — the tap reads the already-computed per-request bank. The engine gathers
  `bank = stack([r.mem_bank for r in batch.reqs])` → `[bs,K,mem]` and `conf = stack([r.mem_conf ...])`
  → `[bs]`, sets them on `ctx.cam_state` before the forward. This keeps all dynamic product-key work OUT
  of the decode hot path and out of the captured region.

### 3c. How the bank feeds the tap for generated positions

- Prefill: inject at **all** prompt positions of the memory request (cheap, and matches the HF path
  which taps every position of the teacher-forced forward). Simpler and strictly more faithful than
  last-position-only; the null-sink + conf-gate already suppress injection where the store has nothing.
- Decode: inject at the single new token per step. Same per-request bank.

Concretely the tap sub-op, per forward, does:
```python
cam = get_global_ctx().cam_state
if cam.active_bank is None: return residual              # memory globally off this batch
bank = cam.active_bank            # [num_tokens, K, mem]  (already gathered per row)
conf = cam.active_conf            # [num_tokens] or None
# GatedMemoryTap.forward body, flat [num_tokens, H]:
h = x + residual
upd = gamma_gated_cross_attention(h, bank, conf)         # zero for null/zero-bank rows
return residual + upd.to(residual.dtype)
```
The per-row gather (`bank[row_bank_idx]`, `conf[row_bank_idx]`) is done by the engine when it sets
`cam_state`, so the captured forward sees a static-shape `[padded_tokens, K, mem]` tensor whose
CONTENTS are refreshed each step (the GDN-capture pattern, §5).

---

## 4. Checkpoint loading

### What `save_ckpt` currently saves (`cam/recall_mag.py::save_ckpt`, L385)

A single `torch.save` dict:
- `"adapter"`: `PKStoreAdapter.state_dict()` **minus** `embed.*` and `unembed` (the frozen ~3GB tied
  table is dropped and rebuilt from the base's embedding on load).
- `"taps"`: `injector.taps.state_dict()` — the `nn.ModuleDict{str(L): GatedMemoryTap}` (so keys are
  `"24.to_q.weight"`, etc.).
- Scalars/knobs: `tap_layer`, `tap_heads`, `conf_gate`, `n_rel`, `mem_dim`, `heads`, `chunk`,
  `expansion`, `k`, `n_sub`, `topk`, `sub_topk`, `addr_sup_weight`, `pk_read_heads`, `store="pk"`,
  `mt_value`, `readout`, `perpos_key`, `base1`, `embed_shape`, plus counterfactual eval metadata.

`load_ckpt` (L429) rebuilds `PKStoreAdapter(embed_weight, H, ...)` from those knobs, `load_state_dict(...,
strict=False)` (embed/unembed allowed-missing), then rebuilds `MAGInjector([L], mem_dim, ...)` and
`injector.taps.load_state_dict(ck["taps"])`. It asserts the donor embed table shape matches.

### What's needed on the serve side (and what's missing)

- **Reuse, don't re-embed:** the serve engine already holds the base's tied embedding at
  `model.model.embed_tokens.weight` (`VocabParallelEmbedding`). Pass `embed_weight =
  model.model.embed_tokens.weight` into `PKStoreAdapter` so the dropped `embed.*`/`unembed` are rebuilt
  from the live serve weights — **no separate 3GB load, and it guarantees the same donor table**
  (`embed_shape` assert enforces it). Note: under TP the embed is vocab-parallel (sharded across ranks);
  the store needs the FULL table. For MVP the 4B runs TP=1, so `embed_tokens.weight` is the full table.
  For TP>1 (future) the store weights + banks should live on rank 0 and the produced `[1,K,mem]` bank
  broadcast to all ranks (small), keeping the store replicated-not-sharded (like GLM's replicated MLA
  latent). Flag as a TP follow-up.

- **A loader on the engine side.** Add `CAMState.load(ckpt_path, embed_weight, device)` that does exactly
  what `load_ckpt` does but WITHOUT constructing an HF `base` — it only needs the adapter + tap, since
  the base IS the serve model. It returns `(store_adapter, tap, tap_layer, conf_gate, n_rel)`. The
  builder-dependency in `load_ckpt` (`assert builder is not None`) is **only** for the episodic
  `memory_bank` path; the persistent path (`_e`/`_pool_subject`/`persistent_write`/`persistent_bank`)
  does NOT use the builder, so the serve loader passes `builder=None` and never calls episodic methods.

- **Missing today, add to the ckpt for reproducible serving:**
  1. `CAM_DISJOINT_BANKS` / `n_sub` are enough to reconstruct empty banks, but the **edit set** (the
     written associations) is NOT in the tap/adapter ckpt — the banks are runtime state. For a server to
     come up with knowledge already loaded, add an **optional companion artifact**: a list of
     `(subject_string, new_object_string)` edits (tokenizer-agnostic, like `save_ckpt` already does for
     `cf_facts`) that the server replays through `store.write` at boot to rebuild `V`. Alternatively,
     serialize the raw `V` banks — but the string-edit list is tokenizer-portable and much smaller.
     **Recommend: persist the edit list; rebuild banks at boot** (deterministic, and lets you re-key if
     `B` changes).
  2. The **env-gated CAM knobs** that change addressing (`CAM_LEARNED_KEY_POOL`, `CAM_POOLED_SUBJ_KEY`,
     `CAM_KEY_HEADS`, `CAM_KEY_MAXSIM`, `CAM_DISJOINT_BANKS`) are read from `os.environ` inside the
     adapter at call time. These MUST be pinned to the training values at serve time (write vs read use
     them symmetrically). Bake them into the ckpt dict and have `CAMState.load` set them (or pass them as
     explicit adapter kwargs instead of env — a cleaner serve-side refactor, since a shared server
     shouldn't depend on process env for correctness). **This is the highest-value ckpt-format change.**

---

## 5. Eager-first, then graph capture

### 5a. Eager MVP

The whole path works eager with zero capture concerns: the tap is a plain module add; the store read is
a normal PyTorch call done at prefill on the scheduler side. Turn OFF cuda-graph (`--graph 0`) for the
MVP, exactly as GDN-hybrid and spec-decode already force eager in places (`engine.py::_adjust_config`).
Deliver correctness first: reproduce `eval_persistent` cf-delivery numbers through the server for a
fixed edit set + prompt cohort.

### 5b. What blocks graph capture of the DECODE tap

Graph capture (`engine/graph.py::_capture_graphs`) records `model.forward()` on a dummy decode batch and
replays with static input buffers whose CONTENTS are refreshed (`GraphCaptureBuffer.copy_from`). Three
capture hazards, mapped to the CAM decode path:

1. **The store read (`persistent_bank`)** — product-key `topk`, `torch.gather`, variable subject length,
   softmax over dynamic candidate sets. **Mitigation: it is NOT in the decode path.** Per §3b the read
   runs at prefill only, on the scheduler, eager. The captured decode forward only sees the pre-computed
   `[bs,K,mem]` bank + `[bs]` conf. **No product-key op is ever captured.** This is the key design move.

2. **The disjoint-bank gather + per-row bank index** — `bank[row_bank_idx]`. **Mitigation:** for decode
   the gather is done by the engine BEFORE the forward into a **static `[max_bs, K, mem]` bank buffer**
   (and a `[max_bs]` conf buffer), refreshed in place each replay — precisely the
   `GDNGraphCapture._state_indices`/`gdn_capture.prepare_for_replay` pattern
   (`gdn/graph_capture.py`). Add a `CAMGraphCapture` holding `self._bank_buf[max_bs,K,mem]`,
   `self._conf_buf[max_bs]`, wired into `GraphRunner.__init__` next to `gdn_capture`/`cca_capture`, with
   `prepare_for_capture` (fill padding rows with a NULL/zero bank) and `prepare_for_replay` (copy the
   batch's real per-req banks in, zero the padded tail). The tap sub-op reads these static buffers via
   `ctx.cam_state`. Padding rows point at a zero bank ⇒ tap no-op ⇒ garbage-safe (same as GDN NULL slot).

3. **The tap's cross-attention itself** — `to_q/to_k/to_v/to_o`, softmax, `tanh(gamma)`, conf sigmoid.
   All **static-shape** given a fixed `[max_bs, K, mem]` bank and `K` fixed. `K` = `readout_q` slot count
   (a constant from the ckpt), `mem` fixed, `H` fixed. **Fully capturable** — it is just matmuls +
   softmax on fixed shapes, like any attention block already captured. The only subtlety is the
   `conf_ema` buffer update is **training-only** (`if self.training:`), so in eval it is a pure read —
   capture-safe.

**Net:** with the read hoisted to prefill and a `CAMGraphCapture` static bank buffer, the **decode tap
is graph-capturable** and gets the ~20% TPOT win. No product-key top-k, no gather-by-hash, no dynamic
shape is ever inside the captured region.

### 5c. Prefill

Prefill is already eager in minisgl (only decode is captured). So the prefill tap injection AND the
prefill store read both run eager with no capture work. The static-shape constraint only bites decode.

### 5d. Options summary (as the task asked)

- **Run store read outside the captured region** — CHOSEN (read at prefill, per §3b).
- **Precompute bank at prefill** — CHOSEN (`req.mem_bank`, reused across decode).
- **Static-shape the top-k** — NOT NEEDED, because top-k never enters decode. (If a future design wanted
  per-decode-step re-reads, static-shaping would require fixed subject length + fixed `topk`/`sub_topk`
  — already constants — plus a fixed candidate count; feasible but unnecessary for the edit-serving use
  case where the subject is fixed for the whole generation.)

---

## 6. API surface delta (`server/api_server.py`)

Follow the **exact precedent of the in-engine RSA field** (`OpenAICompletionRequest.rsa: bool|dict|None`
+ `rsa_defaults` on `ServerArgs` + `merge_params`). Two additions:

### 6a. Per-request opt-in (read side) — completions

Add to `OpenAICompletionRequest` (api_server.py L56–96):
```python
memory: bool | dict | None = None
# null/false -> plain completion (tap off for this request).
# true        -> memory ON, subject = auto/whole-prompt (Phase 3) — for MVP require the object form.
# object      -> {"subject": "<string>"}  (MVP: explicit subject; the server tokenizes it,
#                routes to its disjoint bank, reads, and injects at L24 for this request).
```
In `v1_completions`, when `memory` is set and enabled, mark the outgoing `TokenizeMsg` (or a new field
on it) with the subject so the scheduler computes `req.mem_bank`/`req.mem_conf` at prefill (§3). A
`SamplingParams`-adjacent carrier is cleanest: add `memory_subject: str | None` to the tokenize message
and thread it to `Req` (new field), mirroring how `grammar` rides `SamplingParams` from api_server →
scheduler → `req.sampling_params.grammar`. When absent, `req.mem_bank=None` ⇒ tap no-op for that row ⇒
memory and non-memory requests coexist in one batch (§1).

Server-level default `memory_defaults` on `ServerArgs` + `--memory-*` flags (`--memory-enable`,
`--memory-ckpt <path>`, `--memory-edits <path>`, `--memory-disjoint-banks N`) parallel to `--rsa-*`
(`args.py::add_rsa_args`). `--memory-ckpt` loads the tap+store; `--memory-edits` seeds the banks at boot.

### 6b. Edit ingress (write side) — a control-plane route

Edits are writes into the global standing store, not generations. Add a dedicated endpoint:
```
POST /v1/memory/edit   {"edits": [{"subject": "...", "object": "..."}], "mode": "add"|"overwrite"}
POST /v1/memory/reset  {}                      # re-init empty banks
GET  /v1/memory/stats  {}                      # #edits, per-bank occupancy, gate stats
```
`/v1/memory/edit` sends a new backend control message (a `MemoryEditMsg`, sibling of `AbortMsg`) to the
scheduler, which — between forward steps, single-threaded, no race with capture replay — runs
`store.write(V[b], key, val)` per edit exactly as `_persistent_write_val`. `mode:"overwrite"` uses
`_persistent_write_val(..., val_tid=new)` on an existing key (the store's delta write is error-correcting,
per `eval_persistent_overwrite`). This is the online-edit path; when a formal online-API spec is written,
cross-reference it here (none exists in `docs/` yet — this section defines the initial contract).

**TP note:** the write executes on rank 0's store; the banks are replicated-or-rank0-only (§4). Keep the
store single-rank to avoid cross-rank write divergence (same reasoning as the structured-output
cross-rank commit fix in `forward_batch`).

---

## 7. Phasing + top 3 risks

### Phasing

- **Phase 0 — plumbing (CPU-only, no GPU):** port `GatedMemoryTap` → a minisgl `BaseOP` tap sub-op;
  add `ctx.cam_state`; add `model.set_tap`; add `CAMState` + `CAMState.load` reusing `load_ckpt` logic
  against the serve embed table; add `req.mem_bank/mem_conf`. No behavior change when memory is off
  (tap is `None` ⇒ zero-cost, like `_capture_layer_ids`).
- **Phase 1 — MVP eager, single request, `--graph 0`:** explicit-subject `memory` field; prefill store
  read → per-req bank; tap injects eager at L24; `/v1/memory/edit` + `--memory-edits` boot seeding.
  **Acceptance:** serve a fixed edit set, reproduce `eval_persistent` cf-delivery / prior-recall for a
  cohort through HTTP within tolerance of the offline number.
- **Phase 2 — graph capture of decode:** add `CAMGraphCapture` static bank/conf buffers (GDN pattern);
  enable `--graph N`; verify byte-identical logits eager-vs-graph for memory requests, and the ~20%
  TPOT win. Verify padded/non-memory rows stay no-op.
- **Phase 3 — concurrency + ergonomics:** many concurrent memory+non-memory requests in one batch
  (per-row bank gather already supports it); online edits under load; then optionally implicit-subject
  extraction and per-request ephemeral edits (an extra bank merged at read). TP>1 store replication.

### Top 3 risks + mitigations

1. **Subject formation at inference (no builder span).** The store is addressed by subject identity; a
   wrong/misaligned subject reads the wrong disjoint bank (`_subject_bank` hashes token-ids) and delivers
   nothing or the wrong edit. *Mitigation:* MVP requires an EXPLICIT subject string in the request and
   tokenizes it identically to write time (same tokenizer, `add_special_tokens=False`); defer
   implicit-subject extraction to Phase 3 behind a flag. Add `/v1/memory/stats` to surface per-bank
   occupancy and the tap's `last_conf`/`last_cgate` so misroutes are observable.

2. **Env-var-driven addressing correctness in a shared server.** The adapter reads `CAM_LEARNED_KEY_POOL`,
   `CAM_POOLED_SUBJ_KEY`, `CAM_KEY_HEADS`, `CAM_KEY_MAXSIM`, `CAM_DISJOINT_BANKS` from `os.environ` at
   call time (`pk_store_adapter.py::_pool_subject`, `_maxsim_reduce`; `recall_mag.py::_n_disjoint_banks`).
   Write and read MUST use identical settings, and a server shouldn't depend on process env for numeric
   correctness. *Mitigation:* bake these into the ckpt and set them once in `CAMState.load` (or refactor
   the adapter to take them as explicit kwargs — preferred). Fail loudly if a boot env disagrees with the
   ckpt-recorded value.

3. **Capture/write concurrency + static-buffer aliasing.** A decode graph replay reads the global banks
   `V` transitively (only via the pre-computed per-req bank, so `V` itself is not in the graph) AND reads
   the static `CAMGraphCapture` bank buffer. An online `/v1/memory/edit` mutates `V`; a prefill computes
   a new per-req bank. *Mitigation:* the scheduler is single-threaded — sequence edits and bank
   computation BETWEEN forward steps, never concurrent with a replay (same discipline GDN uses for state
   slots). Padding rows in the capture buffer point at a zero/NULL bank (tap no-op), mirroring
   `GDNGraphCapture`'s NULL slot 0, so rounded-up batch sizes never inject garbage. Because the read is
   hoisted out of decode, an edit only affects requests whose prefill runs AFTER the write — the intended
   online semantics, and race-free.

---

## Appendix — concrete file/symbol touch-list

Serve engine (`minisgl-rdna4`):
- `python/minisgl/models/qwen3_5.py` — `Qwen3_5Model.forward` (inject after tap layer), `set_tap`,
  new `MemoryTap(BaseOP)` (port of `GatedMemoryTap`, flat `[num_tokens,H]`).
- `python/minisgl/core.py` — `Context.cam_state`; `Req.mem_bank`/`Req.mem_conf`;
  `SamplingParams`/tokenize-msg `memory_subject`.
- `python/minisgl/engine/engine.py` — build `CAMState` in `__init__`; gather per-req bank/conf into
  `ctx.cam_state` before `forward_batch`; `CAMState.load`.
- `python/minisgl/engine/graph.py` — `CAMGraphCapture` (static bank/conf buffers), wired into
  `GraphRunner` next to `gdn_capture`/`cca_capture` (`prepare_for_capture`/`_for_replay`).
- `python/minisgl/scheduler/scheduler.py` — at prefill, compute `req.mem_bank/mem_conf` via the store;
  handle `MemoryEditMsg` (write into banks between steps).
- `python/minisgl/server/api_server.py` — `memory` field on `OpenAICompletionRequest`;
  `/v1/memory/edit|reset|stats` routes.
- `python/minisgl/server/args.py` — `memory_defaults` + `--memory-*` flags (mirror `add_rsa_args`).

CAM research (`memory-organ`) — reused as-is / lightly refactored:
- `cam/gated_tap.py::GatedMemoryTap.forward` — ported verbatim (fp32 compute, cast-back).
- `cam/pk_store.py::ProductKeyStore.{read,write,init_state,_address}` — used unchanged.
- `cam/pk_store_adapter.py::{_e,_pool_subject,persistent_write,persistent_bank,_maxsim_reduce}` — the
  serving surface.
- `cam/recall_mag.py::{save_ckpt,load_ckpt,_subject_bank,_persistent_write_val,_init_banks}` — the ckpt
  format + persistent write/route logic (adapt `load_ckpt` to skip the HF base; add env-knob baking + an
  edit-list companion artifact).
