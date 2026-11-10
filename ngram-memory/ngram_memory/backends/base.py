"""Backend protocol (DESIGN.md §3.2) and the small value types it exchanges.

Design rule: gradients reach only a sparse leaf added after the gather; no backend makes the table trainable.
Only the protocol and value types live here; implementations are per backend (M0-core ships none that satisfy
the full protocol: the graft module + hooks + overlay are the pieces, the ledger-backed apply/revert is not done).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import torch

from ..spec import FormatSpec
from ..tables import SparseRowStore

#: An n-gram key: the token ids (x_{t-n+1}, ..., x_t); for Gemma-PLE (token, layer).
Key = tuple[int, ...]


@dataclass(frozen=True)
class RowAddr:
    """Where a key's rows live: one entry per head (or per layer slot)."""

    tensor: str
    rows: torch.Tensor  # int64 [N]
    col0: torch.Tensor  # int64 [N]
    ncols: int


@dataclass
class RowDiff:
    """A set of row writes. mode 'absolute' (grafts) or 'delta' (native)."""

    row_ids: torch.Tensor  # int64 [R]
    values: torch.Tensor  # fp32 [R, w]
    mode: str = "absolute"
    edit_id: str | None = None


SparseDelta = SparseRowStore


@dataclass(frozen=True)
class StateSpec:
    """What a cache must carry for exact decode (§2.3)."""

    layer_idx: int
    slots: dict = field(default_factory=dict)  # slot index -> (description, length)


@dataclass
class ParityReport:
    rows_for_matches_model: bool | None = None
    cached_vs_full_max_abs: float | None = None
    host_cached_vs_full_max_abs: float | None = None
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class Backend(Protocol):
    spec: FormatSpec

    def keys(self, ids: torch.Tensor, positions: Sequence[int]) -> list[Key]: ...

    def rows_for(self, keys: Sequence[Key]) -> RowAddr: ...

    def read_rows(self, addr: RowAddr) -> torch.Tensor: ...

    def forward(self, batch: dict, delta: SparseDelta | None): ...

    def apply(self, diff: RowDiff) -> int: ...

    def revert(self, edit_id: str) -> int: ...

    def stale(self, token_seqs: Sequence[torch.Tensor], row_ids: torch.Tensor) -> torch.Tensor: ...

    def decode_state(self) -> StateSpec: ...

    def parity(self) -> ParityReport: ...


def decode_state_spec(fmt: FormatSpec) -> StateSpec:
    """§2.3 for any Flash-Next-format module (native or graft)."""
    m, k = fmt.module, fmt.key
    return StateSpec(layer_idx=m.layer_idx, slots={
        1: (f"last {m.conv_state_len} positions of N_conv(u), {m.hc_count * m.hidden_size} values each",
            m.conv_state_len),
        2: (f"last {k.context_len} token ids, eos-seeded ({k.eos_token_id})", k.context_len)})


__all__ = ["Backend", "Key", "RowAddr", "RowDiff", "SparseDelta", "StateSpec", "ParityReport", "decode_state_spec"]
