"""Native backend for qwen4_exp (Qwen3.8-Flash-Next) tables (DESIGN.md §4.1, protocol §3.2).

* The table is never trainable. Applied edits are an fp32 overlay (`serve.hf_adapter.RowOverlay`) on the table
  module's output, materialised from a `RowLedger` (served = base + sum of active deltas, canonical order); training
  and solving use a SEPARATE trainable delta leaf (`delta_leaf`) added after the gather (§3.2 design rule).
* `ShardTableEmbedding` replaces `ngram_embedding` by a byte-offset reader over the safetensors shards
  (`tables.ShardMmapStore`), so the 51 GB table is never loaded (§4.1 "Loading"). Rows come back in the model's
  dtype; the lookup moves them to the ids' device as q4 does.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..hashing import hasher_for
from ..ledger import CallableBase, EditRecord, RowLedger
from ..patch import Patch, make_patch, verify_patch
from ..serve.hf_adapter import RowOverlay
from ..spec import FormatSpec, flash_next_format
from ..tables import ShardMmapStore, SparseRowStore
from .base import Key, ParityReport, RowAddr, RowDiff, StateSpec, decode_state_spec


# ------------------------------------------------------------------------------------------ model helpers
def text_model(model: nn.Module) -> nn.Module:
    m = model
    for name in ("model", "language_model"):
        if hasattr(m, name) and not hasattr(m, "layers"):
            m = getattr(m, name)
    if not hasattr(m, "layers"):
        raise TypeError(f"cannot find decoder layers on {type(model).__name__}")
    return m


def text_config_dict(model: nn.Module) -> dict:
    cfg = model.config
    cfg = getattr(cfg, "text_config", None) or cfg
    return cfg.to_dict()


def ple_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    ple = getattr(text_model(model).layers[layer_idx], "ple", None)
    if ple is None:
        raise ValueError(f"decoder layer {layer_idx} has no PLE module")
    return ple


class ShardTableEmbedding(nn.Module):
    """Read-only `nn.Embedding` stand-in over safetensors shards (pread by byte offset, never loads the table)."""

    def __init__(self, store: ShardMmapStore, dtype: torch.dtype = torch.bfloat16, num_rows: int | None = None):
        super().__init__()
        self.store = store
        self.num_embeddings = int(num_rows if num_rows is not None else store.num_rows)
        self.embedding_dim = store.row_width
        self.dtype = dtype
        self.register_buffer("_dev", torch.zeros(0, dtype=dtype), persistent=False)

    @property
    def weight(self) -> torch.Tensor:  # q4 reads `.weight.device`
        return self._dev

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.store.gather(ids.detach().cpu().numpy())
        return rows.to(device=ids.device, dtype=self.dtype)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, shards={len(self.store._locs)}, dtype={self.dtype}"


def replace_table_with_shards(model: nn.Module, shards: Sequence[tuple[str, str]], layer_idx: int = 1,
                              dtype: torch.dtype | None = None) -> ShardTableEmbedding:
    ple = ple_layer(model, layer_idx)
    old = ple.ple_embedding.ngram_embedding
    store = ShardMmapStore(shards, old.embedding_dim)
    if store.num_rows < old.num_embeddings:
        raise ValueError(f"shards hold {store.num_rows} rows, the table has {old.num_embeddings}")
    dt = dtype or (old.weight.dtype if old.weight.device.type != "meta" else torch.bfloat16)
    new = ShardTableEmbedding(store, dt, old.num_embeddings)
    ple.ple_embedding.ngram_embedding = new
    return new


def _codec_for(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp32"


def _table_base(table: nn.Module, w: int):
    """CallableBase reading original row bits from the live table (fp64 tables are read as fp32: lossy, reported
    rows only; the forward overlay never round-trips the base)."""
    if isinstance(table, ShardTableEmbedding):
        c = table.store.codec.codec
        if c not in ("bf16", "fp32"):
            raise NotImplementedError(f"ledger base codec {c}")
        return CallableBase(table.store.gather_raw, c, w)
    W = table.weight
    codec = _codec_for(W.dtype)

    def fn(ids: np.ndarray) -> np.ndarray:
        r = W.detach()[torch.from_numpy(ids)].cpu()
        r = r if r.dtype == torch.bfloat16 else r.to(torch.float32)
        if r.dtype == torch.bfloat16:
            return r.view(torch.int16).numpy().view(np.uint8).reshape(ids.size, -1)
        return r.numpy().view(np.uint8).reshape(ids.size, -1)

    return CallableBase(fn, codec, w)


# --------------------------------------------------------------------------------------------- backend
class Qwen4ExpNativeBackend:
    """Backend protocol (§3.2) over an HF qwen4_exp model whose layer `spec.module.layer_idx` has the native PLE."""

    def __init__(self, model: nn.Module, spec: FormatSpec | None = None, engine_cast: torch.dtype | None = None,
                 tensor_name: str | None = None):
        self.model = model
        cfg = text_config_dict(model)
        self.spec = spec or flash_next_format(cfg)
        if self.spec.module.kind != "native":
            raise ValueError("native backend needs a native ModuleSpec")
        self.ple = ple_layer(model, self.spec.module.layer_idx)
        emb = self.ple.ple_embedding
        self._check_key(emb)
        self.table = emb.ngram_embedding
        if (self.table.num_embeddings, self.table.embedding_dim) != (self.spec.padded_rows, self.spec.row_width):
            raise ValueError(f"table {(self.table.num_embeddings, self.table.embedding_dim)} != spec "
                             f"{(self.spec.padded_rows, self.spec.row_width)}")
        self.tensor = tensor_name or self.spec.table.tensor_name
        self.hasher = hasher_for(self.spec.key)
        self.ledger = RowLedger(self.spec.padded_rows, self.spec.row_width, _table_base(self.table, self.spec.row_width),
                                spec_fingerprint=self.spec.fingerprint(), tensor=self.tensor)
        self.overlay = RowOverlay(SparseRowStore.empty(self.spec.padded_rows, self.spec.row_width),
                                  engine_cast=engine_cast).install(self.table)
        self._ids = itertools.count()

    def _check_key(self, emb) -> None:
        k = self.spec.key
        ok = (emb.layer_multipliers.tolist() == list(k.multipliers)
              and list(emb.head_vocab_sizes) == list(k.head_sizes)
              and emb.ngram_heads_offsets.tolist() == list(k.head_offsets) and emb.eos_token_id == k.eos_token_id)
        if not ok:
            raise ValueError("KeySpec does not reproduce the model's own hash buffers")

    # ------------------------------------------------------------------------------------------ keys
    def keys(self, ids, positions: Sequence[int]) -> list[Key]:
        """n-gram keys (oldest token first, eos-substituted across EOS/sequence start) at each position, one per
        order in KeySpec head order. `ids` is one sequence [T] (a fresh segment)."""
        x = np.asarray(ids.detach().cpu() if isinstance(ids, torch.Tensor) else ids, dtype=np.int64).reshape(1, -1)
        sh = [self.hasher.shifted(x, s)[0] for s in range(self.spec.key.ngram_size)]
        out: list[Key] = []
        for p in positions:
            for n in self.spec.key.orders:
                out.append(tuple(int(sh[s][p]) for s in reversed(range(n))))
        return out

    def rows_for(self, keys: Sequence[Key]) -> RowAddr:
        rows = np.concatenate([self.hasher.rows_for_ngram(k) for k in keys]) if keys else np.empty(0, np.int64)
        return RowAddr(self.tensor, torch.from_numpy(rows), torch.zeros(rows.size, dtype=torch.long),
                       self.spec.row_width)

    def row_ids(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, H] rows an uncached batch reads (fresh segments)."""
        r = self.hasher.rows_for(ids.detach().cpu().numpy(), mask=None if mask is None else mask.cpu().numpy())
        return torch.from_numpy(np.ascontiguousarray(r))

    def read_rows(self, addr: RowAddr | torch.Tensor, cast: str | None = None) -> torch.Tensor:
        """Effective rows as served now (base + active deltas), fp32; cast='bf16' applies the engine cast."""
        rows = addr.rows if isinstance(addr, RowAddr) else addr
        _, v = self.ledger.materialise(rows.cpu().numpy(), cast=cast)
        return torch.from_numpy(v)

    # --------------------------------------------------------------------------------------- forward
    def delta_leaf(self, row_ids, sparse_grad: bool = True) -> RowOverlay:
        """A trainable zero delta over `row_ids` (the only thing gradients reach; read `.delta.values.grad`)."""
        st = SparseRowStore.reserved(self.spec.padded_rows, self.spec.row_width, torch.as_tensor(row_ids).reshape(-1))
        return RowOverlay(st, trainable=True, sparse_grad=sparse_grad)

    def forward(self, batch: dict, delta: RowOverlay | SparseRowStore | None = None):
        """Frozen model forward with `delta` added after the gather (on top of the applied edits)."""
        if delta is None:
            return self.model(**batch)
        ov = delta if isinstance(delta, RowOverlay) else RowOverlay(delta)
        ov.install(self.table)
        try:
            return self.model(**batch)
        finally:
            ov.remove()

    # ---------------------------------------------------------------------------------------- edits
    def _publish(self) -> None:
        ids, d = self.ledger.delta_sum()
        self.overlay.delta.load_store(SparseRowStore(self.spec.padded_rows, self.spec.row_width,
                                                     torch.from_numpy(ids), torch.from_numpy(d)))

    def apply(self, diff: RowDiff | EditRecord) -> int:
        if isinstance(diff, RowDiff):
            if diff.mode != "delta":
                raise ValueError("native tables take 'delta' diffs")
            rec = EditRecord(diff.edit_id or f"edit{next(self._ids):06d}", diff.row_ids.cpu().numpy(),
                             diff.values.detach().cpu().numpy(), spec_fingerprint=self.spec.fingerprint(),
                             tensor=self.tensor)
        else:
            rec = diff
        ep = self.ledger.add(rec)
        self._publish()
        return ep

    def revert(self, edit_id: str) -> int:
        ep = self.ledger.remove(edit_id)
        self._publish()
        return ep

    def apply_patch(self, patch: Patch, edit_id: str | None = None) -> int:
        verify_patch(patch, self.spec, base_rows_fn=lambda ids: self.ledger.orig_rows(ids))
        return self.apply(EditRecord.from_patch(patch, edit_id))

    def export_patch(self, **kw) -> Patch:
        return self.ledger.to_patch(self.spec, **kw)

    # ----------------------------------------------------------------------------------- invariants
    def stale(self, token_seqs: Sequence[torch.Tensor], row_ids) -> torch.Tensor:
        """Which (fresh-segment) cached prefixes read any of `row_ids`."""
        rid = np.asarray(torch.as_tensor(row_ids).reshape(-1).cpu())
        return torch.tensor([bool(np.isin(self.hasher.rows_for(np.asarray(s.cpu()).reshape(-1)), rid).any())
                             for s in token_seqs])

    def decode_state(self) -> StateSpec:
        return decode_state_spec(self.spec)

    @torch.no_grad()
    def parity(self, ids: torch.Tensor | None = None, prefill: int | None = None) -> ParityReport:
        """rows_for (numpy hasher) vs the model's own lookup; cached decode vs full forward (current edits)."""
        if ids is None:
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(0, self.spec.key.vocab_size, (2, 20), generator=g)
        cap: dict = {}

        def grab(mod, inp, out) -> None:  # must return None (a value would replace the output)
            cap.setdefault("ids", inp[0].detach().clone())

        h = self.table.register_forward_hook(grab)
        try:
            full = self.model(input_ids=ids, use_cache=False).logits.float()
        finally:
            h.remove()
        match = bool(torch.equal(cap["ids"].cpu(), self.row_ids(ids)))
        P = prefill or ids.shape[1] // 2
        out = self.model(input_ids=ids[:, :P], use_cache=True)
        d = [(out.logits.float() - full[:, :P]).abs().max().item()]
        cache = out.past_key_values
        for t in range(P, ids.shape[1]):
            o = self.model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            d.append((o.logits.float()[:, -1] - full[:, t]).abs().max().item())
        return ParityReport(rows_for_matches_model=match, cached_vs_full_max_abs=max(d),
                            notes=[f"prefill {P} + {ids.shape[1] - P} decode steps, {len(self.ledger.active)} edits"])

    def remove(self) -> None:
        self.overlay.remove()


__all__ = ["Qwen4ExpNativeBackend", "ShardTableEmbedding", "replace_table_with_shards", "ple_layer", "text_model",
           "text_config_dict"]
