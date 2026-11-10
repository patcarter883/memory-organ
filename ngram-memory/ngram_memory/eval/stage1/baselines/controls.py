"""Content controls for a trained cartridge (§6.A.4 "Controls"); all evaluated like the cartridge itself.

* eval-time permuted keys: the trained rows are read through a fixed random permutation of row ids. The
  permutation is a per-head affine bijection r -> off_h + (a_h (r - off_h) + b_h) mod p_h (head sizes are primes),
  so rows stay in their head (same n-gram order) and the full 320 M-row space needs no permutation table.
* random rows at matched norms: same row ids, Gaussian directions, each row's norm kept.
* constant bias: the cartridge's mean injected vector (mean graft output over calibration positions) added at
  every position (`ConstantBiasCondition`).
The train-time key shuffle is a training arm (ingest side), not an eval transform.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ....backends.graft.module import NGramGraft
from ....backends.graft.qwen35 import install_graft
from ....spec import KeySpec
from ....tables import SparseRowStore
from ..conditions import LayerInjection
from ..scoring import Doc, Window


def head_permutation(key: KeySpec, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(a_h, b_h) per head, a_h in [1, p_h)."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(key.head_sizes, np.int64)
    a = np.array([rng.integers(1, p) for p in sizes], np.int64)
    b = np.array([rng.integers(0, p) for p in sizes], np.int64)
    return a, b


def permute_rows(ids, key: KeySpec, seed: int = 0, inverse: bool = False) -> np.ndarray:
    """pi(r) (or pi^-1) for global row ids."""
    r = np.asarray(ids, np.int64)
    off = np.asarray(key.head_offsets, np.int64)
    sizes = np.asarray(key.head_sizes, np.int64)
    h = np.searchsorted(off, r, side="right") - 1
    if (h < 0).any() or (r - off[h] >= sizes[h]).any():
        raise ValueError("row id outside the key's logical rows")
    a, b = head_permutation(key, seed)
    loc = r - off[h]
    if inverse:
        ainv = np.array([pow(int(x), -1, int(p)) for x, p in zip(a, sizes)], np.int64)
        out = (ainv[h] * ((loc - b[h]) % sizes[h])) % sizes[h]
    else:
        out = (a[h] * loc + b[h]) % sizes[h]
    return off[h] + out


def permuted_keys_store(store: SparseRowStore, key: KeySpec, seed: int = 0) -> SparseRowStore:
    """A store S' with S'.gather(r) = S.gather(pi(r)): reading row r returns the trained row stored at pi(r)."""
    new_ids = permute_rows(store.row_ids.numpy(), key, seed, inverse=True)
    return SparseRowStore(store.num_rows, store.row_width, torch.from_numpy(new_ids), store.values.clone())


def random_rows_store(store: SparseRowStore, seed: int = 0) -> SparseRowStore:
    g = torch.Generator().manual_seed(seed)
    v = store.values
    r = torch.randn(v.shape, generator=g)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12) * v.norm(dim=-1, keepdim=True)
    return SparseRowStore(store.num_rows, store.row_width, store.row_ids.clone(), r)


@torch.no_grad()
def mean_graft_output(model, graft: NGramGraft, docs: Sequence[Doc], windows: Sequence[Window],
                      max_tokens: int = 50_000) -> torch.Tensor:
    """Mean residual addition [d] of the installed graft over the windows' forward positions."""
    tot, n = None, 0
    with install_graft(model, graft, record=True) as h:
        for w in windows:
            if n >= max_tokens:
                break
            ids = torch.from_numpy(docs[w.doc].ids[w.a:w.e])[None]
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            o = h.last["out"][0].float()
            tot = o.sum(0) if tot is None else tot + o.sum(0)
            n += o.shape[0]
    if tot is None:
        raise ValueError("no positions")
    return tot / n


class ConstantBiasCondition(LayerInjection):
    def __init__(self, bias: torch.Tensor, layer_idx: int = 1, name: str = "const_bias"):
        b = bias.detach().float().reshape(1, 1, -1)
        super().__init__(layer_idx, lambda ids, hidden: b.expand_as(hidden), name)
        self.bias_norm = float(b.norm())

    def describe(self) -> dict:
        d = super().describe()
        d["bias_norm"] = self.bias_norm
        return d


__all__ = ["head_permutation", "permute_rows", "permuted_keys_store", "random_rows_store", "mean_graft_output",
           "ConstantBiasCondition"]
