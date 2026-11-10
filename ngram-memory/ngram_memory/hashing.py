"""Transformers-free (and torch-free) n-gram hashing: token ids -> global table row ids.

Port of P/t1/src/ngram_format.py, vectorised in numpy; the decode-step path follows
P/graftfirst/serve-port/vec_decode_hash.py. Semantics are q4:1107-1166 (`_shift_right_ignore_eos`, mixing,
per-head prime modulus, head offsets) generalised to an explicit head order (`KeySpec.orders`):

    x_k[p]   = ids[p-k] if p-k is inside the current EOS segment (an EOS belongs to the segment it ends), else eos
    mix_n[p] = XOR_{k<n} x_k[p] * m_k                     (int64; m_k < (2^63-1)/V so no overflow, mix >= 0)
    row[p,h] = off_h + mix_{order(h)}[p] mod p_h

Padding (q4:1465-1470): positions where `mask == 0` are replaced by eos BEFORE hashing.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .spec import KeySpec, TableSpec


class NGramHasher:
    """Vectorised hasher for one KeySpec. All outputs are int64 numpy arrays."""

    def __init__(self, key: KeySpec):
        self.key = key
        self.eos = np.int64(key.eos_token_id)
        self.mult = np.asarray(key.multipliers, dtype=np.int64)
        self.sizes = np.asarray(key.head_sizes, dtype=np.int64)
        self.offsets = np.asarray(key.head_offsets, dtype=np.int64)
        self.head_orders = np.asarray(key.head_orders, dtype=np.int64)
        self.hpn = key.heads_per_ngram
        self.n_heads = key.n_heads
        self.context_len = key.context_len

    # ------------------------------------------------------------------------------------------------ history
    def shifted(self, hist: np.ndarray, shift: int) -> np.ndarray:
        """`_shift_right_ignore_eos` on [B, L] (oldest first)."""
        if shift == 0:
            return hist
        B, L = hist.shape
        pos = np.arange(L, dtype=np.int64)
        eos_pos = np.where(hist == self.eos, pos[None, :], np.int64(-1))
        prev_incl = np.maximum.accumulate(eos_pos, axis=1)
        prev = np.concatenate([np.full((B, 1), -1, dtype=np.int64), prev_incl[:, :-1]], axis=1)
        in_seg = pos[None, :] - (prev + 1)
        src = pos - shift
        out = hist[:, np.maximum(src, 0)]
        valid = (in_seg >= shift) & (src[None, :] >= 0)
        return np.where(valid, out, self.eos)

    def _mix_table(self, xs: list[np.ndarray]) -> dict[int, np.ndarray]:
        mixes: dict[int, np.ndarray] = {}
        acc = xs[0] * self.mult[0]
        mixes[1] = acc
        for k in range(1, self.key.ngram_size):
            acc = np.bitwise_xor(acc, xs[k] * self.mult[k])
            mixes[k + 1] = acc
        return mixes

    def _rows_from_mix(self, mixes: dict[int, np.ndarray]) -> np.ndarray:
        cols = [mixes[o] for o in self.key.orders]  # each [...]
        m = np.repeat(np.stack(cols, axis=-1), self.hpn, axis=-1)  # [..., H]
        return self.offsets + np.remainder(m, self.sizes)

    # ------------------------------------------------------------------------------------------------- public
    def rows_for(self, ids, *, context=None, mask=None) -> np.ndarray:
        """Row ids for every position. ids [T] -> [T, H]; ids [B, T] -> [B, T, H].

        context: optional [n-1] / [B, n-1] tokens that preceded `ids` (a decode-state slot, eos-seeded); None means
        the sequence starts here. mask: optional [T] / [B, T] padding mask (0 -> hashed as eos)."""
        a = np.asarray(ids, dtype=np.int64)
        one = a.ndim == 1
        if one:
            a = a[None]
        if mask is not None:
            m = np.asarray(mask)
            m = m[None] if m.ndim == 1 else m
            a = np.where(m.astype(bool), a, self.eos)
        B, T = a.shape
        if context is None:
            ctx = np.full((B, self.context_len), self.eos, dtype=np.int64)
        else:
            ctx = np.asarray(context, dtype=np.int64)
            ctx = ctx[None] if ctx.ndim == 1 else ctx
            if ctx.shape != (B, self.context_len):
                raise ValueError(f"context shape {ctx.shape} != {(B, self.context_len)}")
        hist = np.concatenate([ctx, a], axis=1)
        xs = [self.shifted(hist, k)[:, -T:] for k in range(self.key.ngram_size)]
        rows = self._rows_from_mix(self._mix_table(xs))
        return rows[0] if one else rows

    def rows_for_ngram(self, ngram: Sequence[int]) -> np.ndarray:
        """Rows of an explicit n-gram (x_{t-n+1}, ..., x_t): only that order's heads_per_ngram rows, in head order.
        The n-gram is assumed to lie inside one EOS segment (no eos among its context tokens)."""
        toks = [int(t) for t in ngram]
        n = len(toks)
        if n not in self.key.orders:
            raise ValueError(f"order {n} not in key orders {self.key.orders}")
        x = list(reversed(toks))
        mixed = x[0] * int(self.mult[0])
        for k in range(1, n):
            mixed ^= x[k] * int(self.mult[k])
        hs = np.asarray(self.key.heads_of_order(n), dtype=np.int64)
        return self.offsets[hs] + np.remainder(np.int64(mixed), self.sizes[hs])

    def decode_row_ids_batch(self, hist: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """One decode step for a batch. hist [B, n-1] = the true last n-1 tokens (oldest first, eos-seeded at the
        start of a sequence), tok [B] -> [B, H]. An EOS ends ITS segment, so any EOS in hist cuts the history at
        and before it (GF/serve-port/vec_decode_hash.py)."""
        h = np.asarray(hist, dtype=np.int64)
        t = np.asarray(tok, dtype=np.int64).reshape(-1)
        n1 = self.context_len
        if h.shape != (t.shape[0], n1):
            raise ValueError(f"hist shape {h.shape} != {(t.shape[0], n1)}")
        xs = [t]
        cut = np.zeros(t.shape[0], dtype=bool)
        for k in range(1, n1 + 1):
            prev = h[:, n1 - k]  # token k positions back
            cut = cut | (prev == self.eos)
            xs.append(np.where(cut, self.eos, prev))
        return self._rows_from_mix(self._mix_table(xs))

    def next_context(self, hist: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """Decode-state update for the n-gram slot: append tok, keep the last n-1 tokens."""
        h = np.asarray(hist, dtype=np.int64)
        return np.concatenate([h, np.asarray(tok, dtype=np.int64).reshape(-1, 1)], axis=1)[:, -self.context_len:]


_HASHERS: dict[str, NGramHasher] = {}


def hasher_for(key: KeySpec) -> NGramHasher:
    fp = key.fingerprint()
    h = _HASHERS.get(fp)
    if h is None:
        h = _HASHERS[fp] = NGramHasher(key)
    return h


def rows_for(key: KeySpec, ids, *, context=None, mask=None) -> np.ndarray:
    return hasher_for(key).rows_for(ids, context=context, mask=mask)


def rows_for_ngram(key: KeySpec, ngram: Sequence[int]) -> np.ndarray:
    return hasher_for(key).rows_for_ngram(ngram)


def decode_row_ids_batch(key: KeySpec, hist, tok) -> np.ndarray:
    return hasher_for(key).decode_row_ids_batch(hist, tok)


def checkpoint_loc(table: TableSpec, key: KeySpec, row: int) -> tuple[int, int]:
    """(shard index, local row) of a global row in the checkpoint's shard tensors."""
    return table.shard_of(key, row)


__all__ = ["NGramHasher", "hasher_for", "rows_for", "rows_for_ngram", "decode_row_ids_batch", "checkpoint_loc"]
