# Online Store API — knowledge-editing memory in production

Concrete API design for the editable CAM memory served alongside a frozen LLM. The store holds
facts as `(subject-key → new-object-value)` associations injected into the residual stream by a
trained tap. This document specifies the **edit-plane** API (create / update / delete / list /
persist). It does **not** re-specify how the injected bank reaches the model's forward pass —
that is the **data-plane / serve-integration** concern of the companion doc
[`integration_design.md`](./integration_design.md) (not yet written at time of authoring); this
spec cross-references it rather than duplicating it.

Everything here is grounded in code that already exists and is validated offline in
`cam/recall_mag.py` (the persistent-store harness) and `cam/pk_store.py` /
`cam/pk_store_adapter.py` (the store itself). The serve engine
(`minisgl-rdna4/python/minisgl/server/api_server.py`) currently has **no** memory integration; this
API is new. Request/response style mirrors that server's FastAPI + pydantic `BaseModel` conventions.

---

## 0. What actually exists (the ground truth this API wraps)

The production data layout and every store call named below are real:

- **B disjoint value banks.** `_init_banks(adapter, B)` (`cam/recall_mag.py:1019`) builds a Python
  **list of B banks**, each `adapter.store.init_state(1, DEV, dtype=torch.float32)`. Each bank is a
  tensor `[1, N, d_hub]` where `N = n_sub²` (default `n_sub=32` → **N=1024 slots**) and
  `d_hub = mem_dim` (default **512**). Default production `B` is env-driven
  (`CAM_DISJOINT_BANKS`, `_n_disjoint_banks()` at `:1000`); the task's target design point is
  **B=32**. Disjoint banks broke the shared-store capacity ceiling (**N=137**) by turning one
  crowded store into ~B parallel low-crowding stores (+106%).
- **Subject-hash routing.** `_subject_bank(subject_tids, B)` (`:1009`) routes each edit to a bank by
  `int(md5(",".join(subject_tids)), 16) % B` — a **stable hash of the discrete subject token-ids**,
  so write and read route identically regardless of encoder state.
- **Error-correcting delta write.** `_persistent_write_val(adapter, V, r, val_tid, pooled)`
  (`:1025`) embeds the subject (`adapter._e`), pools its key (`adapter._pool_subject`, or last
  token), embeds the value token, routes to bank `b = _subject_bank(...)`, and calls
  `V[b] = adapter.persistent_write(V[b], key, val)`. `persistent_write`
  (`pk_store_adapter.py:472`) forwards to `ProductKeyStore.write` (`pk_store.py:150`), a
  scatter-add delta `v_s += beta·w·(val − v_s)` into the product-key-addressed slots. **Repeated
  writes accumulate; a second value for the same subject cleanly UPDATES in place** (validated:
  `eval_persistent_overwrite`, B beats stale A ~1.7×).
- **Subject-keyed read.** `adapter.persistent_bank(V[b], q)` (`pk_store_adapter.py:478`) reads the
  routed bank with the subject query and returns the pooled `[1, K, mem_dim]` bank the tap consumes,
  plus a store-confidence scalar `self._last_conf` (factual-head pre-norm retrieval magnitude — the
  honest "is this subject actually written" signal, `pk_store.py:207`).
- **Validated online properties.** `eval_persistent` (`:1102`) writes N edits into one standing
  store and queries each by subject: **no catastrophic forgetting**, cf-delivery holds. `curve`
  tracks retention/interference. `eval_persistent_overwrite` (`:1158`) validated **clean
  update-in-place**.

The store is **additive and error-correcting**. This single fact drives the honest limits on
DELETE (§5) and on capacity (§6).

---

## 1. Data model & identity

### 1.1 An edit

```
Edit {
  subject:      string | int[]      # subject text OR pre-tokenized subject token-ids
  relation:     string | null       # relation/prompt template context; needed to elicit & to scope
  new_value:    string | int[]      # the object to deliver (single-token today; see §1.4)
  edit_id:      string              # server-assigned, stable handle (see §1.3)
}
```

- **Subject → tokens → bank.** The subject string is tokenized (server-side, same tokenizer as the
  base) to `subject_tids`. Identity and routing are on the **token-ids**, never the string, matching
  `_subject_bank`. Two strings that tokenize identically are the same subject.
- **Key vector.** Written/read key = pooled subject span (`_pool_subject`, mean; or the learned
  attention pool under `CAM_LEARNED_KEY_POOL=1`) or the subject's **last token** embed. The pooling
  choice is a **store-wide invariant fixed at load** (it must match how the tap was trained) — it is
  **not** a per-edit knob. Expose it read-only in store metadata (§4).
- **Value.** Today the store slot is a single `mem_dim` vector and delivery is validated for
  **single-token values** (`new_tid`). Multi-token values are §1.4.

### 1.2 Multiple facts about one subject

The store keys on `(subject)`, not `(subject, relation)`. Two different relations for the same
subject **address the same product-key slots** and will interfere (the second delta partially
overwrites the first). Honest handling:

- **MVP:** one live value per subject. A write to an existing subject is an **update** (§1.3),
  not an additional fact. Document this loudly.
- **Full:** scope by folding the relation into the key. `_pos_key(name_key, t)`
  (`pk_store_adapter.py:243`) already folds a position tag into the key; the same mechanism can fold
  a **relation tag** so `(subject, relation)` addresses a distinct slot-neighborhood. This requires
  the tap/store to have been trained relation-aware (see `counterfactual_multi`,
  `setup_counterfact_multi` at `:658`, which edits multiple relations together). Until that training
  exists in the served checkpoint, **multi-relation-per-subject is not offered.**

### 1.3 Referencing an edit later (update / delete)

The store has **no native per-edit index** — it is content-addressed by subject. So identity is
**derived, not stored in the tensor**:

- `edit_id = hash(subject_tids [, relation_id])`, deterministic and stable. Given the same subject
  you can always recompute it; the client may also keep the returned `edit_id`.
- The server maintains a small **side index** (a plain dict/SQLite, NOT in the bank):
  `edit_id → { subject_tids, relation, new_value, bank_index, created_at, version }`. This is the
  authoritative record of *what was intended*; the bank is the *delta-compressed realization*. The
  side index is what makes LIST, INSPECT, and (rebuild-based) DELETE possible — see §5.
- `bank_index = _subject_bank(subject_tids, B)` is recorded for observability and rebuild routing.

### 1.4 Multi-token subjects & values

- **Multi-token subjects** are already supported: routing hashes the whole `subject_tids` list; the
  key pools the whole span (or takes the last token). `setup_counterfact_multi` keys on the
  subject's **last token** by default and validated multi-token subjects.
- **Multi-token values** are **not** in the validated serve path. The single value slot carries one
  `mem_dim` vector; multi-token answers are a research feature (`mt_value`, per-position `perpos`
  stores, `direct_logits` returning `[B,K,vocab]`). **MVP accepts single-token `new_value` only** and
  rejects multi-token with a 422 (`unsupported: multi_token_value`). Mark it a full-surface item.

---

## 2. Operations & REST surface

All routes live under `/v1/memory`. They mutate the edit-plane; visibility rules (§3) govern when a
mutation is observable by concurrent `/generate` traffic. Style matches `api_server.py` pydantic
models.

### 2.1 Upsert (create-or-update) — `POST /v1/memory/edits`

The primary op. Idempotent on `(subject, relation)`.

Request:
```
UpsertRequest {
  subject:    string | int[]
  relation:   string | null = null
  new_value:  string | int[]         # single-token today
  mode:       "upsert" | "create" | "update" = "upsert"
}
```
- `create` → 409 if the subject already has a live value.
- `update` → 404 if it does not.
- `upsert` (default) → write regardless.

Response:
```
EditResponse {
  edit_id:     string
  subject:     string
  bank_index:  int
  version:     int                    # incremented on each write to this subject
  op:          "created" | "updated"
}
```

**Store mapping.** Tokenize subject+value → `subject_tids`, `val_tid`. Build the edit record `r`.
`_persistent_write_val(adapter, V, r, val_tid, pooled)` — routes to `V[_subject_bank(...)]` and
delta-writes. On `update`, this is a **second write to the same key**: validated to cleanly replace
(UPDATED high, STALE low, `eval_persistent_overwrite`). Update side index, bump `version`.

### 2.2 Update — `PATCH /v1/memory/edits/{edit_id}`

Sugar over upsert with `mode:"update"`; body carries only `new_value`. Same store call (a fresh
delta write of the new value into the same subject-routed slots). 404 if `edit_id` unknown.

### 2.3 Forget / delete — `DELETE /v1/memory/edits/{edit_id}`

See §5 for the honesty analysis. Request may carry a strategy:
```
DeleteRequest { strategy: "counter_write" | "tombstone" | "rebuild" = "tombstone" }
```
Response:
```
DeleteResponse { edit_id, strategy, guarantee: "best_effort" | "exact", cost: "O(1)"|"O(N_bank)" }
```

### 2.4 Bulk import — `POST /v1/memory/edits:bulkImport`

```
BulkImportRequest {
  edits:       UpsertRequest[]
  on_error:    "abort" | "skip" | "collect" = "collect"
  atomic:      bool = false          # if true, stage into a shadow bank set, swap on success (§3)
}
```
Response returns per-edit `EditResponse` (or error) plus counts and **post-import per-bank
occupancy** (§6). Loops `_persistent_write_val` per edit exactly as `eval_persistent`'s write phase
(`:1143`). Bulk is where bank imbalance shows up first, so always return occupancy.

### 2.5 List / inspect — `GET /v1/memory/edits`, `GET /v1/memory/edits/{edit_id}`

List is served entirely from the **side index** (the bank cannot be enumerated — it is a
delta-compressed slot tensor, not a key→value map). Supports `?bank=`, `?relation=`,
`?limit=&cursor=`. Inspect returns the edit record plus a **live probe** option
(`?probe=true`): run `_persistent_preds` for this one subject and report whether the store currently
delivers `new_value` (`predicted_tid == new_tid`) and the store-confidence scalar `_last_conf`. Probe
is the only *ground-truth* readback; the side index is intent, the probe is reality.

### 2.6 Clear — `POST /v1/memory:clear`

Re-init all banks: `V = _init_banks(adapter, B)` (`:1019`) and empty the side index. `O(B·N·d_hub)`
allocation, instant logically. Guard behind a confirm token in the request to avoid foot-guns.

### 2.7 Snapshot / restore — `POST /v1/memory:snapshot`, `POST /v1/memory:restore`

Persistence lifecycle, §4.

---

## 3. Consistency & concurrency

The B banks are **shared mutable fp32 tensors read on every inference forward** (each decode step
that fires the tap calls `persistent_bank` on the subject's bank). Writes mutate slots in place
(`ProductKeyStore.write` scatter-adds; `Vnew = V` unless `V.requires_grad`, which it is **not** at
serve time — `init_state` returns a non-grad tensor). So a naive concurrent write during a forward
read is a genuine data race on a tensor.

**Design: per-bank copy-on-write swap + a snapshot-at-request-start of the bank list.**

- **Read path.** At the start of a `/generate` request (or each forward that will inject), the
  engine takes a reference to the **current bank list** `V` (a Python list of B tensor handles) —
  cheap, no copy. It reads `V[b]` for whatever subject it needs. Because it holds the handle, a
  concurrent writer that *replaces* `V[b]` with a new tensor does not perturb the in-flight read.
- **Write path (copy-on-write per bank).** An upsert does **not** scatter into the live tensor.
  It clones the target bank, delta-writes into the clone, then atomically **rebinds the list slot**:
  `Vnew_b = store.write(V[b].clone(), key, val); V[b] = Vnew_b`. The list-slot rebind is a single
  Python reference assignment (atomic under the GIL). Cost is one `[1, N, d_hub]` clone per write
  (**N·d_hub·4 B = 1024·512·4 ≈ 2 MB** at defaults) — negligible vs a base forward.
- **Visibility.** A write becomes visible to any request whose bank-list snapshot is taken **after**
  the rebind. In-flight requests finish against the pre-write bank (read-your-writes is not
  guaranteed within a single already-started generation; it is guaranteed for the next request).
  This is the right default: an edit landing mid-decode should not tear a partially-generated answer.
- **Write serialization.** Two upserts to the **same bank** must serialize (both read-modify-write
  that bank). Use a **per-bank lock** (B locks, default 32) so writes to different banks proceed in
  parallel — this matches the disjoint-bank design and keeps the hot path lock-free across subjects.
  The per-bank lock guards the `clone → write → rebind` critical section only.
- **Bulk atomic import** (§2.4, `atomic:true`) stages into a **shadow list** of cloned banks, writes
  all edits, then swaps the whole list reference in one assignment — all-or-nothing visibility.

This keeps the inference forward **lock-free** (it only reads handles) and bounds writer cost to one
small clone. The full wiring of how the engine obtains `V` and injects the read into the residual
stream is the data-plane concern of `integration_design.md`.

---

## 4. Persistence & lifecycle

### 4.1 What must persist

1. **The B value banks.** `B` tensors of `[1, N, d_hub]` fp32. At defaults **B=32, N=1024,
   d_hub=512**: `32·1024·512·4 B = 64 MiB` total. bf16 storage (supported: `init_state(dtype=…)`,
   `pk_store.py:106`) halves it to 32 MiB. This is the *editable* state.
2. **The trained adapter projections.** `PKStoreAdapter` state (`in_proj`, `norm`, `_pool_subject`
   params, `readout_q`, `out_proj`) **and** the `ProductKeyStore` params (`codebook1/2`,
   `to_wkey/to_wval`, `read_q/read_o/read_norm`, `head_bias`) **and** the MAG tap. These are
   **frozen** and shipped with the model checkpoint — see `save_ckpt`/`load_ckpt`
   (`recall_mag.py:385`/`:429`), which already persist adapter + taps + all store knobs
   (`n_sub, topk, sub_topk, mem_dim, heads, ...`) minus the ~3 GB tied embed (rebuilt from the base
   table on load). **The banks are NOT in that checkpoint** — they are runtime state and get their
   own snapshot.

### 4.2 Snapshot format

A snapshot is `{ banks: safetensors, side_index: json, meta: json }`:
- `banks.safetensors` — the B tensors, keyed `bank_{i}`, dtype as stored. Small (≤64 MiB).
- `side_index.json` — the §1.3 records (subject_tids, relation, new_value, bank_index, version,
  created_at). Human-inspectable, the authoritative intent log.
- `meta.json` — `{ B, n_sub, N, d_hub, pooled_key: bool, adapter_ckpt_id, base_model_id, tokenizer,
  schema_version }`. The `adapter_ckpt_id` + `base_model_id` **must match** the loaded model on
  restore (same donor guard `load_ckpt` already enforces via `embed_shape`/`base1`, `:439`); a
  mismatch is a hard error, because a bank is only meaningful against the projections that wrote it.

### 4.3 Warm start & versioning

- **Warm start** = restore the bank snapshot into a freshly loaded (frozen) adapter. Because banks
  are decoupled from adapter weights, the same edit set warm-starts any server that loads the
  matching adapter checkpoint.
- **Versioning:** `schema_version` in meta gates format evolution; per-edit `version` (§1.3) tracks
  update history. Snapshots are content-addressable by a hash of `banks + side_index`.
- **Restore is the reference implementation of `rebuild`** (§5): re-init banks, replay the side
  index through `_persistent_write_val`. This makes exact DELETE achievable by replay-minus-one.

---

## 5. Delete honesty (the load-bearing caveat)

**The store is an error-correcting *additive* memory. There is no true erase.** A write folds a
delta into a set of product-key slots (`v_s += beta·w·(val − v_s)`, `pk_store.py:169`); slots are
**shared** across subjects that address overlapping product-keys. You cannot subtract one subject's
contribution without knowing every other subject that touched those slots. So "forget" has three
mechanisms, each with an honest guarantee:

| Strategy | Mechanism | Guarantee | Cost | Collateral |
|---|---|---|---|---|
| **`tombstone`** (MVP default) | Mark `edit_id` deleted in the **side index**; at read time the engine suppresses injection for a subject on the tombstone list (skip the tap / zero the bank for that subject). The bank tensor is **untouched**. | The subject **stops being delivered** from the serve path. The *residue stays in the tensor*. | O(1) | None to other subjects. Reversible. |
| **`counter_write`** | Write the subject → **its base-prior value** (`true_tid`), or → a null/neutral value, so the delta pushes the slot back toward "no edit". | **Best-effort.** Pushes delivery back toward the prior but does **not** guarantee removal: shared slots and the error-correcting `(val − v_s)` term mean the subject may still read weakly, and the counter-write perturbs *other* subjects sharing those slots. | O(1) | **Yes** — perturbs slot-neighbors. |
| **`rebuild`** | Re-init the banks and **replay the side index minus this edit** (`_init_banks` + loop `_persistent_write_val`, exactly §4.3 restore). | **Exact** — the resulting banks are byte-for-byte what they would be had the edit never been written. | O(N_bank writes) for the affected bank (only that bank needs replay, since banks are disjoint) | None. |

**Recommendation.** MVP DELETE = **tombstone** (O(1), no collateral, reversible, and honest that the
tensor residue remains — which is fine because the serve path never reads a tombstoned subject).
Offer **rebuild** as the "hard delete / compliance erase" path; it is exact and, thanks to disjoint
banks, only replays the *one* affected bank (~N/B ≈ 4 edits at B=32), so it is cheap in practice.
**`counter_write` is documented but not recommended** — it is the only O(1) option that touches the
tensor, and it is neither exact nor collateral-free. Never claim `counter_write` erases.

The API surfaces this via `DeleteResponse.guarantee` (`best_effort` vs `exact`) so callers cannot be
misled.

---

## 6. Failure modes & limits

### 6.1 Bank load imbalance (the primary failure mode)

Routing is a hash (`_subject_bank`), so occupancy is **balls-into-bins**, not uniform. With B=32 and
N_total edits, expected load is N_total/B but the **max** bank runs meaningfully hotter; hash
collisions crowd some banks while others stay empty.

- **Delivery drops when a bank crowds.** The B-sweep knee is at **~4 subjects/bank** — the code's
  own rationale (`_n_disjoint_banks` docstring, `recall_mag.py:1000`) is that disjoint banks work
  precisely because each parallel store sits in the **low-crowding regime (~0.5 delivery @ N≈9 per
  bank)**; the **shared-store ceiling was N=137**. Empirically each bank degrades as it exceeds
  **~4–9 subjects**. So the operational budget is roughly **B·4 ≈ 128 comfortable edits** at B=32,
  degrading toward the crowding wall beyond that — and imbalance means the *hottest* bank hits the
  wall before the average does.
- **Capacity per bank:** the tensor has **N=1024 slots** (n_sub=32), but useful capacity is set by
  **addressing crowding**, not slot count — product-key top-k means many subjects share slots, so the
  binding wall (~4–9) is far below N. Do not advertise "1024 facts/bank".
- **At overflow:** no crash. Writes still succeed (scatter-add always lands), but **delivery for
  crowded subjects silently degrades** — the dangerous failure mode. This is why observability is
  mandatory.

### 6.2 Observability (required, not optional)

Expose per-bank occupancy and a health signal:

```
GET /v1/memory:stats  ->
{
  B: 32,
  banks: [ { index, n_edits, n_slots_touched, est_crowding: n_edits/4 }, ... ],
  total_edits, max_bank_load, imbalance: max_bank_load / mean_bank_load,
  crowded_banks: [indices where n_edits > 9],
}
```
- `n_edits` per bank comes from the side index (exact). `n_slots_touched` (optional, from a probe)
  approximates real slot pressure.
- **Alert** when any `n_edits > ~9` (the crowding knee) or `imbalance > ~2×`. The mitigation is to
  **raise B** (re-shard: re-init with larger B and replay the side index — the same rebuild path),
  which is why `B` is a store-level, snapshot-recorded parameter, not fixed forever.
- Per-subject health is the `?probe=true` inspect (§2.5): does the store still deliver this edit,
  and what is `_last_conf`.

### 6.3 Other limits

- **Single-token values only** in the validated path (§1.4) — reject multi-token at the edge.
- **One value per subject** in MVP (§1.2) — multi-relation-per-subject needs a relation-aware trained
  checkpoint.
- **Locality/collateral:** even a correct edit can perturb neighbors (the store is not surgical by
  construction; the `--locality-weight` / `--conf-gate` training levers exist to narrow this but are
  a *training-time* property of the shipped checkpoint, not an API knob).
- **Tokenizer coupling:** subject/value tokenization must use the **exact** base tokenizer; a
  tokenizer mismatch silently misroutes and misbinds. Enforced via `meta.tokenizer` on restore.

---

## 7. MVP vs full surface

**MVP (ship first) — the smallest honest edit plane:**
- `POST /v1/memory/edits` (upsert; single-token value; one value/subject)
- `PATCH /v1/memory/edits/{edit_id}` (update — validated clean overwrite)
- `DELETE /v1/memory/edits/{edit_id}` with **tombstone** default (+ **rebuild** for exact erase)
- `GET /v1/memory/edits`, `GET /v1/memory/edits/{edit_id}?probe=true` (list/inspect from side index +
  live probe)
- `POST /v1/memory:clear`
- `POST /v1/memory:snapshot` / `:restore` (banks safetensors + side_index json)
- `GET /v1/memory:stats` (per-bank occupancy + crowding alerts) — **required in MVP**, it is the only
  guard against silent overflow
- Concurrency: per-bank COW swap + bank-list snapshot-at-request-start (§3)

**Full surface (later):**
- `POST /v1/memory/edits:bulkImport` with `atomic` shadow-swap
- `counter_write` delete strategy (documented, de-emphasized)
- Multi-relation-per-subject (relation-tag folded into the key via `_pos_key`; needs a
  relation-aware checkpoint à la `counterfactual_multi`)
- Multi-token values (`mt_value`/perpos stores)
- Online re-sharding (raise B) as a first-class op with progress reporting
- Retention/interference telemetry surfaced from the `eval_persistent` sweep machinery

---

## 8. Cross-references

- **Data-plane / serve integration** (how `V` reaches the forward pass, tap wiring, decode-loop
  injection, graph-capture interaction): [`integration_design.md`](./integration_design.md). This
  online-API spec deliberately does **not** cover the injection mechanics.
- **Store internals:** `cam/pk_store.py` (`ProductKeyStore.init_state/write/read`),
  `cam/pk_store_adapter.py` (`_e`, `_pool_subject`, `persistent_write`, `persistent_bank`).
- **Validated online semantics:** `cam/recall_mag.py` — `_init_banks` (`:1019`),
  `_subject_bank` (`:1009`), `_persistent_write_val` (`:1025`), `eval_persistent` (`:1102`),
  `eval_persistent_overwrite` (`:1158`), `_n_disjoint_banks` (`:1000`).
- **Serve API style:** `minisgl-rdna4/python/minisgl/server/api_server.py` (FastAPI + pydantic
  `BaseModel`; `/v1/...` route conventions).
