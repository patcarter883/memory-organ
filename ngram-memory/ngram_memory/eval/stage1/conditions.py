"""Arms that act inside the host: a cartridge's rows through a graft (`RowsCondition`) and residual injections at a
decoder input (`LayerInjection`, the base of NGM and the constant-bias control). See scoring.Condition."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ...backends.graft.module import NGramGraft
from ...backends.graft.qwen35 import decoder_layers, install_graft
from ...spec import FormatSpec
from ...tables import SparseRowStore
from .scoring import Condition


class RowsCondition(Condition):
    """A graft (G-lin / G-dir per `fmt`) with the given rows, installed while active. Rows never get gradients."""

    def __init__(self, fmt: FormatSpec, rows: SparseRowStore | None, name: str = "rows", graft: NGramGraft | None = None):
        self.name = name
        self.fmt = fmt
        self.graft = graft if graft is not None else NGramGraft(fmt, rows=rows)
        for p in self.graft.parameters():
            p.requires_grad_(False)
        self.n_rows = len(self.graft.rows_store())

    @classmethod
    def from_file(cls, fmt: FormatSpec, path: str | Path, name: str | None = None) -> "RowsCondition":
        sd = torch.load(path, map_location="cpu", weights_only=True)
        return cls(fmt, SparseRowStore.from_state_dict(sd), name=name or Path(path).stem)

    def activate(self, model: nn.Module):
        return install_graft(model, self.graft, record=False)

    def describe(self) -> dict:
        return dict(name=self.name, rows=self.n_rows, arm=self.fmt.module.arm, layer_idx=self.fmt.module.layer_idx,
                    format_fingerprint=self.fmt.fingerprint())


class LayerInjection(Condition):
    """Adds inject(input_ids [B, T], hidden [B, T, d]) -> [B, T, d] to the input of decoder `layer_idx`."""

    def __init__(self, layer_idx: int, inject: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], name: str):
        self.layer_idx, self.inject, self.name = layer_idx, inject, name
        self.last_ratio: list[float] = []

    @contextlib.contextmanager
    def activate(self, model: nn.Module):
        box: dict = {}

        def cap(_m, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise NotImplementedError("LayerInjection needs input_ids")
            box["ids"] = ids

        def inj(_m, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            add = self.inject(box["ids"], hidden).to(hidden.dtype)
            self.last_ratio.append(float((add.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1e-12)).median()))
            hidden = hidden + add
            if args:
                return (hidden,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = hidden
            return args, kwargs

        hs = [model.register_forward_pre_hook(cap, with_kwargs=True),
              decoder_layers(model)[self.layer_idx].register_forward_pre_hook(inj, with_kwargs=True)]
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    def describe(self) -> dict:
        r = sorted(self.last_ratio)
        return dict(name=self.name, layer_idx=self.layer_idx,
                    median_injection_ratio=r[len(r) // 2] if r else None)


__all__ = ["RowsCondition", "LayerInjection"]
