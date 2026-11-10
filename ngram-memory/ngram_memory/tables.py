"""Row stores (§3.1): where table rows live.

* SparseRowStore -- sorted unique int64 row ids + fp32 values [U, w]; every other row of the logical
  [num_rows, w] table is exactly zero. Graft and cartridge storage (P/graft/graft_proto.py:52).
* HFTensorStore  -- a loaded dense nn.Embedding weight (tiny tests, Gemma tables).
* ShardMmapStore -- safetensors shard files read by byte offset (never loads the table) (§2.1b address).
* QuantRowCodec  -- bf16 / fp32 / fp8-e4m3 global-scale / int6-g32 row decode (conventions of the serve's
  R/weights/row_table.py at df29117).

All stores share: `num_rows`, `row_width`, `gather(ids) -> fp32 tensor [..., w]`.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def _as_long(ids) -> torch.Tensor:
    if isinstance(ids, np.ndarray):
        ids = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64))
    return torch.as_tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------------------------- SparseRowStore
class SparseRowStore:
    """Logical [num_rows, row_width] table that is zero except at `row_ids` (sorted, unique, int64).

    `values` is fp32 [U, w]. Rows may be *reserved* (present with value 0): a reserved zero row behaves exactly
    like an absent one in `gather`, which is what lets training hold only the reachable rows (the 0.5B experiment's
    compact trick) and still equal a dense zero table.
    """

    def __init__(self, num_rows: int, row_width: int, row_ids: torch.Tensor | None = None,
                 values: torch.Tensor | None = None):
        self.num_rows, self.row_width = int(num_rows), int(row_width)
        if row_ids is None:
            row_ids = torch.empty(0, dtype=torch.long)
        row_ids = _as_long(row_ids).reshape(-1)
        if values is None:
            values = torch.zeros(row_ids.numel(), self.row_width, dtype=torch.float32)
        values = torch.as_tensor(values).to(torch.float32).reshape(-1, self.row_width)
        if values.shape[0] != row_ids.numel():
            raise ValueError(f"{row_ids.numel()} ids vs {values.shape[0]} value rows")
        if row_ids.numel():
            order = torch.argsort(row_ids)
            row_ids, values = row_ids[order], values[order]
            if bool((row_ids[1:] == row_ids[:-1]).any()):
                raise ValueError("row_ids must be unique")
            if int(row_ids[0]) < 0 or int(row_ids[-1]) >= self.num_rows:
                raise IndexError("row id out of range")
        self.row_ids = row_ids.contiguous()
        self.values = values.contiguous()

    # ---- constructors
    @classmethod
    def wrap(cls, num_rows: int, row_width: int, row_ids: torch.Tensor, values: torch.Tensor) -> "SparseRowStore":
        """Share (not copy) already sorted-unique ids and fp32 values [U, w] (no validation beyond shapes)."""
        self = cls.__new__(cls)
        self.num_rows, self.row_width = int(num_rows), int(row_width)
        if values.shape != (row_ids.numel(), self.row_width):
            raise ValueError(f"values {tuple(values.shape)} vs ids {row_ids.numel()} x {self.row_width}")
        self.row_ids, self.values = row_ids, values
        return self

    @classmethod
    def empty(cls, num_rows: int, row_width: int) -> "SparseRowStore":
        return cls(num_rows, row_width)

    @classmethod
    def reserved(cls, num_rows: int, row_width: int, ids) -> "SparseRowStore":
        """Zero rows at the unique ids given (e.g. every row a training corpus can read)."""
        return cls(num_rows, row_width, torch.unique(_as_long(ids).reshape(-1)))

    @classmethod
    def from_dense(cls, dense: torch.Tensor, keep_zero_rows: bool = False) -> "SparseRowStore":
        d = dense.detach().to(torch.float32)
        ids = torch.arange(d.shape[0]) if keep_zero_rows else (d != 0).any(-1).nonzero().flatten()
        return cls(d.shape[0], d.shape[1], ids, d[ids].clone())

    # ---- queries
    def __len__(self) -> int:
        return self.row_ids.numel()

    def lookup(self, ids) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos, hit): pos indexes `values` (clamped, valid only where hit)."""
        ids = _as_long(ids)
        n = self.row_ids.numel()
        if n == 0:
            return torch.zeros_like(ids), torch.zeros(ids.shape, dtype=torch.bool)
        pos = torch.searchsorted(self.row_ids, ids.contiguous()).clamp_max(n - 1)
        return pos, self.row_ids[pos] == ids

    def gather(self, ids) -> torch.Tensor:
        ids = _as_long(ids)
        pos, hit = self.lookup(ids)
        if self.row_ids.numel() == 0:
            return torch.zeros(*ids.shape, self.row_width, dtype=torch.float32)
        return torch.where(hit.unsqueeze(-1), self.values[pos], torch.zeros((), dtype=torch.float32))

    def contains(self, ids) -> torch.Tensor:
        return self.lookup(ids)[1]

    def nonzero_row_ids(self) -> torch.Tensor:
        return self.row_ids[(self.values != 0).any(-1)]

    # ---- mutation (rebuilds the arrays; call sites holding `values` must re-bind)
    @torch.no_grad()
    def ensure(self, ids) -> None:
        """Reserve zero rows for ids not yet present."""
        ids = torch.unique(_as_long(ids).reshape(-1))
        if ids.numel() and (int(ids[0]) < 0 or int(ids[-1]) >= self.num_rows):
            raise IndexError("row id out of range")
        new = ids[~self.contains(ids)]
        if new.numel() == 0:
            return
        all_ids = torch.cat([self.row_ids, new]).sort().values
        vals = torch.zeros(all_ids.numel(), self.row_width, dtype=torch.float32)
        if self.row_ids.numel():
            vals[torch.searchsorted(all_ids, self.row_ids)] = self.values
        self.row_ids, self.values = all_ids, vals

    @torch.no_grad()
    def scatter(self, ids, values: torch.Tensor, accumulate: bool = False) -> None:
        """Write (or add) rows; duplicate ids accumulate when accumulate=True, last write wins otherwise."""
        ids = _as_long(ids).reshape(-1)
        values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, self.row_width)
        self.ensure(ids)
        pos = torch.searchsorted(self.row_ids, ids)
        if accumulate:
            self.values.index_add_(0, pos, values)
        else:
            self.values[pos] = values

    @torch.no_grad()
    def delete(self, ids) -> None:
        """Remove rows (they return exactly to zero)."""
        keep = ~torch.isin(self.row_ids, _as_long(ids).reshape(-1))
        self.row_ids, self.values = self.row_ids[keep].contiguous(), self.values[keep].contiguous()

    @torch.no_grad()
    def prune_zero(self) -> None:
        keep = (self.values != 0).any(-1)
        self.row_ids, self.values = self.row_ids[keep].contiguous(), self.values[keep].contiguous()

    # ---- export
    def to_dense(self, max_bytes: int = 1 << 30) -> torch.Tensor:
        if self.num_rows * self.row_width * 4 > max_bytes:
            raise MemoryError(f"dense table would be {self.num_rows * self.row_width * 4} bytes")
        w = torch.zeros(self.num_rows, self.row_width, dtype=torch.float32)
        w[self.row_ids] = self.values
        return w

    def to_bf16(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(row_ids int64 [U], rows bf16 [U, w]) -- the cartridge payload (§3.5)."""
        return self.row_ids.clone(), self.values.to(torch.bfloat16)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"row_ids": self.row_ids, "values": self.values,
                "shape": torch.tensor([self.num_rows, self.row_width], dtype=torch.long)}

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor]) -> "SparseRowStore":
        n, w = (int(x) for x in sd["shape"])
        return cls(n, w, sd["row_ids"], sd["values"])

    def copy(self) -> "SparseRowStore":
        return SparseRowStore(self.num_rows, self.row_width, self.row_ids.clone(), self.values.clone())

    def __repr__(self) -> str:
        return f"SparseRowStore(num_rows={self.num_rows}, row_width={self.row_width}, stored={len(self)})"


# --------------------------------------------------------------------------------------- SparseRowEmbedding
class SparseRowEmbedding(torch.nn.Module):
    """Drop-in for `nn.Embedding(num_rows, w)` backed by a SparseRowStore (sorted ids buffer + compact fp32
    Parameter `values` [U, w]). Ids not stored read exactly 0 and get no gradient; with `sparse_grad=True` the
    gradient of `values` is a sparse COO tensor over the compact rows (use torch.optim.SparseAdam / a sparse
    row optimiser). Mathematically identical to a dense zero-initialised table with the same rows written.
    """

    def __init__(self, store: "SparseRowStore", sparse_grad: bool = True, requires_grad: bool = False):
        super().__init__()
        self.num_embeddings, self.embedding_dim = store.num_rows, store.row_width
        self.sparse_grad = sparse_grad
        self.register_buffer("row_ids", store.row_ids)
        self.values = torch.nn.Parameter(store.values, requires_grad=requires_grad)

    @property
    def weight(self) -> torch.Tensor:  # Qwen4ExpTextNGramEmbedding.forward reads `.weight.device`
        return self.values

    @property
    def store(self) -> "SparseRowStore":
        """A SparseRowStore VIEW sharing this module's tensors (in-place optimiser steps are visible)."""
        return SparseRowStore.wrap(self.num_embeddings, self.embedding_dim, self.row_ids, self.values.data)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = self.row_ids.numel()
        if n == 0:
            return torch.zeros(*ids.shape, self.embedding_dim, dtype=self.values.dtype, device=ids.device)
        pos = torch.searchsorted(self.row_ids, ids.contiguous()).clamp_max(n - 1)
        hit = self.row_ids[pos] == ids
        rows = torch.nn.functional.embedding(pos, self.values, sparse=self.sparse_grad)
        return rows * hit.unsqueeze(-1).to(rows.dtype)

    def _rebind(self, st: "SparseRowStore") -> None:
        self.row_ids = st.row_ids
        self.values = torch.nn.Parameter(st.values, requires_grad=self.values.requires_grad)

    @torch.no_grad()
    def reserve(self, ids) -> None:
        """Hold zero rows for `ids` (e.g. every row a corpus can read) so they can be trained. Re-creates the
        Parameter: build optimisers after reserving."""
        st = self.store.copy()
        st.ensure(ids)
        self._rebind(st)

    @torch.no_grad()
    def write(self, ids, values: torch.Tensor, accumulate: bool = False) -> None:
        st = self.store.copy()
        st.scatter(ids, values, accumulate)
        self._rebind(st)

    @torch.no_grad()
    def load_store(self, store: "SparseRowStore") -> None:
        if (store.num_rows, store.row_width) != (self.num_embeddings, self.embedding_dim):
            raise ValueError("store shape mismatch")
        self._rebind(store.copy())

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, stored={self.row_ids.numel()}, sparse_grad={self.sparse_grad}"


# ---------------------------------------------------------------------------------------------- HFTensorStore
class HFTensorStore:
    """A loaded dense table (e.g. `nn.Embedding.weight`); gather returns fp32 copies."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.num_rows, self.row_width = int(weight.shape[0]), int(weight.shape[1])

    def gather(self, ids) -> torch.Tensor:
        return self.weight.detach()[_as_long(ids)].to(torch.float32)

    @torch.no_grad()
    def scatter(self, ids, values: torch.Tensor, accumulate: bool = False) -> None:
        ids = _as_long(ids).reshape(-1)
        v = torch.as_tensor(values).reshape(-1, self.row_width).to(self.weight.dtype)
        if accumulate:
            self.weight.index_add_(0, ids, v)
        else:
            self.weight[ids] = v


# --------------------------------------------------------------------------------------------- QuantRowCodec
def _e4m3_lut() -> np.ndarray:
    b = np.arange(256, dtype=np.uint32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    mant = (b & 0x7).astype(np.float64)
    out = sign * np.where(exp == 0, mant * 2.0 ** -9, (1.0 + mant / 8.0) * np.power(2.0, exp - 7.0))
    out[(exp == 0xF) & (mant == 0x7)] = np.nan
    return out.astype(np.float32)


_E4M3 = _e4m3_lut()


@dataclass(frozen=True)
class QuantRowCodec:
    """Decode raw row bytes to fp32.

    codec: "bf16" | "fp32" | "fp8_e4m3_global" (one global scale) | "int6_g32" (four 6-bit codes per 24-bit
    little-endian word, LSB first, bias +32; one fp16 scale per 32 values -- R/weights/row_table.py:109-151).
    """

    codec: str
    row_width: int
    scale: float = 1.0

    @property
    def row_bytes(self) -> int:
        if self.codec == "bf16":
            return 2 * self.row_width
        if self.codec == "fp32":
            return 4 * self.row_width
        if self.codec == "fp8_e4m3_global":
            return self.row_width
        if self.codec == "int6_g32":
            return self.row_width * 6 // 8
        raise ValueError(self.codec)

    def decode(self, raw: np.ndarray, group_scales: np.ndarray | None = None) -> np.ndarray:
        raw = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1, self.row_bytes)
        n = raw.shape[0]
        if self.codec == "bf16":
            u = raw.view("<u2").astype(np.uint32) << 16
            return u.view("<f4").reshape(n, self.row_width) * np.float32(self.scale)
        if self.codec == "fp32":
            return raw.view("<f4").reshape(n, self.row_width) * np.float32(self.scale)
        if self.codec == "fp8_e4m3_global":
            return _E4M3[raw] * np.float32(self.scale)
        if self.codec == "int6_g32":
            if group_scales is None:
                raise ValueError("int6_g32 needs per-row group scales [n, w/32]")
            if self.row_width % 32 or raw.shape[1] % 3:
                raise ValueError("int6_g32 needs row_width % 32 == 0")
            words = raw.reshape(n, -1, 3).astype(np.uint32)
            w = words[:, :, 0] | (words[:, :, 1] << 8) | (words[:, :, 2] << 16)
            codes = np.stack([(w >> (6 * i)) & 0x3F for i in range(4)], axis=-1).reshape(n, -1).astype(np.int16) - 32
            s = np.asarray(group_scales).astype(np.float32).reshape(n, self.row_width // 32, 1)
            return (codes.reshape(n, -1, 32).astype(np.float32) * s).reshape(n, self.row_width)
        raise ValueError(self.codec)


# -------------------------------------------------------------------------------------------- ShardMmapStore
_ST_DTYPE = {"BF16": ("bf16", 2), "F32": ("fp32", 4), "F8_E4M3": ("fp8_e4m3_global", 1)}


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """(header dict, data start byte) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


@dataclass(frozen=True)
class _ShardLoc:
    path: str
    data_start: int  # absolute byte offset of the tensor's first row
    rows: int


class ShardMmapStore:
    """Row reads from safetensors shards by byte offset (§2.1b): never loads the table.

    `shards[s]` = (file path, tensor name) of shard s, all [rows_per_shard, w] (the last may be shorter).
    Byte offset of (shard, local) = 8 + header_len + data_offsets[0] + local * row_bytes.
    """

    def __init__(self, shards: Sequence[tuple[str, str]], row_width: int, scale: float = 1.0):
        self.row_width = int(row_width)
        self._locs: list[_ShardLoc] = []
        codec = None
        for path, name in shards:
            hdr, start = read_safetensors_header(path)
            t = hdr[name]
            c, nbytes = _ST_DTYPE[t["dtype"]]
            if codec not in (None, c):
                raise ValueError("mixed dtypes across shards")
            codec = c
            rows, w = t["shape"]
            if w != self.row_width:
                raise ValueError(f"{name}: width {w} != {self.row_width}")
            self._locs.append(_ShardLoc(path, start + t["data_offsets"][0], rows))
        self.codec = QuantRowCodec(codec or "bf16", self.row_width, scale)
        self.rows_per_shard = self._locs[0].rows
        self.num_rows = sum(l.rows for l in self._locs)
        self._fds: dict[str, int] = {}

    def address(self, row: int) -> tuple[str, int]:
        """(file, absolute byte offset) of a global row."""
        s, local = divmod(int(row), self.rows_per_shard)
        loc = self._locs[s]
        if local >= loc.rows:
            raise IndexError(row)
        return loc.path, loc.data_start + local * self.codec.row_bytes

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY)
        return fd

    def gather_raw(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        rb = self.codec.row_bytes
        out = np.empty((ids.size, rb), dtype=np.uint8)
        for i, r in enumerate(ids.tolist()):
            path, off = self.address(r)
            buf = os.pread(self._fd(path), rb, off)
            if len(buf) != rb:
                raise IOError(f"short read at row {r}")
            out[i] = np.frombuffer(buf, dtype=np.uint8)
        return out

    def gather(self, ids) -> torch.Tensor:
        shape = np.asarray(ids).shape
        vals = self.codec.decode(self.gather_raw(ids))
        return torch.from_numpy(np.ascontiguousarray(vals)).reshape(*shape, self.row_width)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __del__(self) -> None:  # best effort
        try:
            self.close()
        except Exception:
            pass


__all__ = ["SparseRowStore", "SparseRowEmbedding", "HFTensorStore", "ShardMmapStore", "QuantRowCodec", "read_safetensors_header"]
