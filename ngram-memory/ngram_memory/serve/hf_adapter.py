"""HF adapter (§3.4 adapter 1): an fp32 row overlay added right after an embedding-like gather.

    served(r) = engine_cast(base(r) + delta(r))          (§3.3 overlay rule; base bits are never rewritten)

`RowOverlay` wraps a SparseRowEmbedding holding the delta. Installed as a forward hook on the table module
(`...ple_embedding.ngram_embedding`, native or graft), it adds `delta[ids]` to the module output, so:

* on a graft (base 0) it is the graft runtime: a zero table + overlay == the same rows stored in the graft;
* on a native table it is the "sparse leaf added after the gather" of §3.2: with `trainable=True` gradients reach
  only the delta rows, never the base table.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..tables import SparseRowEmbedding, SparseRowStore


class RowOverlay(nn.Module):
    def __init__(self, delta: SparseRowStore, trainable: bool = False, sparse_grad: bool = True,
                 engine_cast: torch.dtype | None = None):
        """engine_cast: if set (e.g. torch.bfloat16), the hooked output becomes
        cast(fp32(base) + delta) -> output dtype, i.e. the serve's one-cast rule `served = bf16(base + delta)`
        (§3.3). Default None: output + delta in the output dtype (unchanged behaviour)."""
        super().__init__()
        self.delta = SparseRowEmbedding(delta.copy(), sparse_grad=sparse_grad, requires_grad=trainable)
        self._handle = None
        self.enabled = True
        self.engine_cast = engine_cast

    @property
    def num_rows(self) -> int:
        return self.delta.num_embeddings

    @property
    def row_width(self) -> int:
        return self.delta.embedding_dim

    def store(self) -> SparseRowStore:
        return self.delta.store

    def _hook(self, module: nn.Module, inputs, output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        ids = inputs[0]
        d = self.delta(ids.to(self.delta.row_ids.device))
        if self.engine_cast is not None:
            s = output.to(torch.float32) + d.to(device=output.device, dtype=torch.float32)
            return s.to(self.engine_cast).to(output.dtype)
        return output + d.to(device=output.device, dtype=output.dtype)

    def install(self, table_module: nn.Module) -> "RowOverlay":
        """Hook onto an embedding-like module whose forward takes ids and returns [..., row_width]."""
        if self._handle is not None:
            raise RuntimeError("overlay already installed")
        n = getattr(table_module, "num_embeddings", None)
        w = getattr(table_module, "embedding_dim", None)
        if (n, w) != (self.num_rows, self.row_width):
            raise ValueError(f"overlay shape {(self.num_rows, self.row_width)} != table {(n, w)}")
        self._handle = table_module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def ple_table_module(ple_layer: nn.Module) -> nn.Module:
    """The table module of a q4 PLE layer (native `layers.1.ple` or a graft's `.ple`)."""
    return ple_layer.ple_embedding.ngram_embedding


def served_rows(base, delta: SparseRowStore, ids, cast: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
    """Rows as an engine would serve them: one cast of the fp32 sum (base may be None for grafts)."""
    v = delta.gather(ids)
    if base is not None:
        v = base.gather(ids) + v
    return v if cast is None else v.to(cast)


__all__ = ["RowOverlay", "ple_table_module", "served_rows"]
