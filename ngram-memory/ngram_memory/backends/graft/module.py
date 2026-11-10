"""Editor-init graft module (§2.4, §4.3): transformers' own `Qwen4ExpTextPLELayer`, built from a ModuleSpec
namespace (NOT `Qwen4ExpTextConfig`, which rejects hc_count=1), with

* G-lin (default): table 0, W_k 0 (gate 0.5), norms 0 (scale 1), conv 0, W_v = value_scale * seeded orthogonal
  -> out_t = 0.5 * W_v e_t, linear in the rows; min-norm rows for a residual change delta: e = 2 W_v^T delta / s^2.
* G-dir: as G-lin but conv1d[:, 0, j] = gdir_alpha * gdir_taps[j] (fixed, untrained direction branch
  silu(conv(N_conv(u)))); still an exact no-op on empty rows.
* rows: a SparseRowEmbedding over a SparseRowStore (default) or a dense nn.Embedding (tiny tests only).
* orders != (2, 3): UnigramNGramEmbedding (a Qwen4ExpTextNGramEmbedding subclass with an explicit head order).

Port of P/graft/graft_proto.py (NGramGraft); hooks/cache handling live in backends/graft/qwen35.py.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4

from ...hashing import hasher_for
from ...spec import FormatSpec, KeySpec
from ...tables import SparseRowEmbedding, SparseRowStore


def module_namespace(fmt: FormatSpec) -> SimpleNamespace:
    """The attribute set Qwen4ExpTextPLELayer / Qwen4ExpTextNGramEmbedding read from a config."""
    k, m = fmt.key, fmt.module
    if m.conv_dilation != k.ngram_size:
        raise ValueError("q4 uses conv dilation == ngram_size; ModuleSpec.conv_dilation must equal it")
    return SimpleNamespace(
        hidden_size=m.hidden_size, hc_count=m.hc_count, ple_embed_dim=m.ple_embed_dim,
        ple_conv_kernel_size=m.conv_kernel_size, rms_norm_eps=m.rms_norm_eps, vocab_size=k.vocab_size,
        ngram_size=k.ngram_size, heads_per_ngram=k.heads_per_ngram, ngram_vocab_size_base=k.base,
        make_ngram_vocab_size_divisible_by=fmt.table.pad_divisor, seed=k.seed, eos_token_id=k.eos_token_id)


class UnigramNGramEmbedding(q4.Qwen4ExpTextNGramEmbedding):
    """Qwen4ExpTextNGramEmbedding with an explicit head order from a KeySpec (graft-only format extension, §2.4).

    For orders (2, 3, 1): heads 0-7 = 2-gram, 8-15 = 3-gram (row ids identical to Flash-Next), 16-23 = 1-gram
    (mix1 = x0 * m0) on primes k = 17..24. The table and buffers are set by the caller (built on meta)."""

    def __init__(self, cfg: SimpleNamespace, key: KeySpec, row_width: int, layer_idx: int, table: nn.Module):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.ngram_size = key.ngram_size
        self.context_len = key.context_len
        self.heads_per_ngram = key.heads_per_ngram
        self.orders = tuple(key.orders)
        self.ngram_heads = key.n_heads
        self.ple_layer_index = key.ple_layer_index
        self.unigram_vocab_size = key.vocab_size
        self.ngram_vocab_size_base = key.base
        self.seed = key.seed
        self.eos_token_id = key.eos_token_id
        self.head_vocab_sizes = list(key.head_sizes)
        self.head_offsets = list(key.head_offsets)
        self.total_vocab_size = key.total_rows
        self.layer_multipliers = nn.Buffer(torch.tensor(key.multipliers, dtype=torch.long))
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))
        self.ngram_embedding = table

    def forward(self, input_ids: torch.Tensor, past_key_values) -> torch.Tensor:
        input_ids = input_ids.long()
        cached_ctx = past_key_values.get_conv_state(self.layer_idx, state_idx=2) if past_key_values is not None else None
        if cached_ctx is not None:
            previous_context = cached_ctx.clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        if past_key_values is not None:
            to_cache = input_ids
            if cached_ctx is None and input_ids.shape[1] < self.context_len:
                to_cache = torch.nn.functional.pad(to_cache, (self.context_len - input_ids.shape[1], 0),
                                                   value=self.eos_token_id)
            past_key_values.update_conv_state(to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len)
        hist = torch.cat([previous_context, input_ids], dim=-1)
        shifted = [self._shift_right_ignore_eos(hist, s) for s in range(self.ngram_size)]
        mixes = {1: shifted[0] * self.layer_multipliers[0]}
        for k in range(1, self.ngram_size):
            mixes[k + 1] = torch.bitwise_xor(mixes[k], shifted[k] * self.layer_multipliers[k])
        blocks = []
        for i, n in enumerate(self.orders):
            s, e = i * self.heads_per_ngram, (i + 1) * self.heads_per_ngram
            ids = torch.remainder(mixes[n].unsqueeze(-1), self.ngram_heads_vocab_sizes[s:e].view(1, 1, -1))
            blocks.append(ids + self.ngram_heads_offsets[s:e].view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
        dev = self.ngram_embedding.weight.device if self.ngram_embedding.weight.device.type != "meta" else None
        return self.ngram_embedding(ngram_ids.to(dev)).to(ngram_ids.device).flatten(-2)


def build_ple_layer(fmt: FormatSpec, table: str = "sparse", sparse_grad: bool = True,
                    rows: SparseRowStore | None = None) -> q4.Qwen4ExpTextPLELayer:
    """q4.Qwen4ExpTextPLELayer for `fmt`, built on meta (the logical table is never allocated) then materialised
    on CPU with the table replaced. Parameters are left uninitialised except buffers (call graft init)."""
    cfg = module_namespace(fmt)
    k = fmt.key
    w = fmt.row_width
    n_rows = fmt.padded_rows
    if table == "sparse":
        tab: nn.Module = SparseRowEmbedding(rows.copy() if rows is not None else SparseRowStore.empty(n_rows, w),
                                            sparse_grad=sparse_grad)
        if tab.num_embeddings != n_rows or tab.embedding_dim != w:
            raise ValueError("rows store shape does not match the format")
    elif table == "dense":
        if n_rows * w * 4 > (1 << 30):
            raise MemoryError("dense graft table > 1 GiB; use table='sparse'")
        tab = nn.Embedding(n_rows, w)
    else:
        raise ValueError(table)
    with torch.device("meta"):
        ple = q4.Qwen4ExpTextPLELayer(cfg, fmt.module.layer_idx, k.ple_layer_index)
    emb = ple.ple_embedding
    if tuple(k.orders) == (2, 3):
        emb.ngram_embedding = tab
    else:
        ple.ple_embedding = UnigramNGramEmbedding(cfg, k, w, fmt.module.layer_idx, tab)
        emb = ple.ple_embedding
    # materialise everything except the (already real) table
    ple.to_empty(device="cpu")
    if table == "sparse":  # to_empty replaced the values/ids storage with uninitialised memory: restore
        src = rows.copy() if rows is not None else SparseRowStore.empty(n_rows, w)
        emb.ngram_embedding._rebind(src)
    with torch.no_grad():
        emb.layer_multipliers.copy_(torch.tensor(k.multipliers, dtype=torch.long))
        emb.ngram_heads_vocab_sizes.copy_(torch.tensor(k.head_sizes, dtype=torch.long))
        emb.ngram_heads_offsets.copy_(torch.tensor(k.head_offsets, dtype=torch.long))
        # cross-check q4's own derivation where it applies
        if tuple(k.orders) == (2, 3):
            ref = q4._build_layer_multipliers(k.vocab_size, k.ngram_size, k.ple_layer_index, k.seed)
            assert torch.equal(ref, emb.layer_multipliers), "KeySpec multipliers disagree with q4"
            assert list(emb.head_vocab_sizes) == list(k.head_sizes), "KeySpec primes disagree with q4"
    assert emb.ngram_embedding.num_embeddings == n_rows, (emb.ngram_embedding.num_embeddings, n_rows)
    return ple


class NGramGraft(nn.Module):
    """A Flash-Next-format n-gram module for a host without a table. `forward(hidden, ple_ids, state, conv_mask)`
    returns the residual addition [B, T, d]. Install on a host with `install(host)` (backends/graft/qwen35.py)."""

    def __init__(self, fmt: FormatSpec, table: str = "sparse", init: str = "editor", sparse_grad: bool = True,
                 rows: SparseRowStore | None = None):
        super().__init__()
        if fmt.module.kind != "graft":
            raise ValueError("NGramGraft needs a graft ModuleSpec (kind='graft')")
        self.fmt = fmt
        self.table_kind = table
        self.ple = build_ple_layer(fmt, table=table, sparse_grad=sparse_grad)
        self.reset_parameters(init)
        if rows is not None:
            self.load_rows(rows)
        self.debug_last: dict = {}

    # ------------------------------------------------------------------------------------------------- props
    @property
    def key(self) -> KeySpec:
        return self.fmt.key

    @property
    def eos(self) -> int:
        return self.fmt.key.eos_token_id

    @property
    def layer_idx(self) -> int:
        return self.fmt.module.layer_idx

    @property
    def table(self) -> nn.Module:
        return self.ple.ple_embedding.ngram_embedding

    @property
    def value_proj(self) -> torch.Tensor:
        return self.ple.value_proj.weight

    # -------------------------------------------------------------------------------------------------- init
    @torch.no_grad()
    def reset_parameters(self, init: str = "editor") -> None:
        p, m = self.ple, self.fmt.module
        g = torch.Generator().manual_seed(m.value_seed)
        for n in (p.norm_key, p.norm_query, p.norm_conv):
            n.weight.zero_()  # Qwen4ExpTextRMSNorm scales by (1 + weight)
        tab = self.table
        if isinstance(tab, nn.Embedding):
            tab.weight.zero_()
        else:
            tab._rebind(SparseRowStore.empty(tab.num_embeddings, tab.embedding_dim))
        E, d = m.ple_embed_dim, m.hidden_size
        if init == "editor":
            p.key_proj.weight.zero_()
            q, _ = torch.linalg.qr(torch.randn(max(d, E), max(d, E), generator=g))
            p.value_proj.weight.copy_(q[:d, :E] * m.value_scale)  # orthonormal rows (d<=E) or columns (d>=E)
            p.conv1d.weight.zero_()
            if m.arm == "G-dir":
                taps = torch.tensor(m.gdir_taps, dtype=p.conv1d.weight.dtype) * m.gdir_alpha
                p.conv1d.weight.copy_(taps.view(1, 1, -1).expand_as(p.conv1d.weight))
        elif init == "random":  # stress test of every code path (gate, conv, norms)
            p.key_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            p.value_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            p.conv1d.weight.normal_(0, 0.5, generator=g)
            for n in (p.norm_key, p.norm_query, p.norm_conv):
                n.weight.normal_(0, 0.1, generator=g)
        else:
            raise ValueError(init)

    def freeze_all_but_rows(self) -> nn.Parameter:
        """requires_grad only on the row values (the design rule: gradients reach only rows)."""
        for prm in self.parameters():
            prm.requires_grad_(False)
        w = self.table.weight
        w.requires_grad_(True)
        return w

    # --------------------------------------------------------------------------------------------------- run
    def forward(self, hidden: torch.Tensor, ple_ids: torch.Tensor, state=None,
                conv_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.ple(hidden, ple_ids, state, conv_mask=conv_mask)

    # -------------------------------------------------------------------------------------------------- keys
    def row_ids(self, ids: torch.Tensor, mask: torch.Tensor | None = None,
                context: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        """Global row ids [B, T, H] the module reads for an uncached batch `ids` [B, T]: a fresh segment, or with
        `context` [B, n-1] the true preceding tokens (numpy port in ngram_memory.hashing; tests pin it to the
        module's own lookup)."""
        ctx = None if context is None else (context.cpu().numpy() if isinstance(context, torch.Tensor) else context)
        r = hasher_for(self.key).rows_for(ids.detach().cpu().numpy(), mask=None if mask is None else mask.cpu().numpy(),
                                          context=ctx)
        return torch.from_numpy(np.ascontiguousarray(r))

    @torch.no_grad()
    def captured_row_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Row ids as the module's own embedding computes them (stubbed table)."""
        cap: dict = {}
        emb = self.ple.ple_embedding
        orig = emb.ngram_embedding

        class _Cap(nn.Module):
            weight = torch.empty(0)

            def forward(self, x):
                cap["ids"] = x
                return torch.zeros(*x.shape, 1)

        emb.ngram_embedding = _Cap()
        try:
            emb(ids, None)
        finally:
            emb.ngram_embedding = orig
        return cap["ids"]

    def head_slices(self) -> list[slice]:
        w = self.fmt.row_width
        return [slice(h * w, (h + 1) * w) for h in range(self.key.n_heads)]

    # ------------------------------------------------------------------------------------------------ writes
    @torch.no_grad()
    def write_rows(self, ids, values: torch.Tensor, accumulate: bool = False) -> None:
        tab = self.table
        if isinstance(tab, SparseRowEmbedding):
            tab.write(ids, values, accumulate)
        else:
            ids = torch.as_tensor(np.asarray(ids), dtype=torch.long).reshape(-1)
            v = values.reshape(-1, tab.embedding_dim).to(tab.weight.dtype)
            if accumulate:
                tab.weight.index_add_(0, ids, v)
            else:
                tab.weight[ids] = v

    @torch.no_grad()
    def write_ngram(self, ngram: Sequence[int], e_concat: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        """Write the heads_per_ngram rows of one n-gram from a ple_embed_dim vector (only that order's slices)."""
        hs = self.key.heads_of_order(len(ngram))
        rid = torch.from_numpy(hasher_for(self.key).rows_for_ngram(ngram))
        sl = self.head_slices()
        vals = torch.stack([e_concat[sl[h]] for h in hs])
        self.write_rows(rid, vals, accumulate)
        return rid

    @torch.no_grad()
    def reserve_rows(self, ids) -> None:
        """Hold zero rows for ids (compact trick: only reachable rows exist). Sparse tables only."""
        if not isinstance(self.table, SparseRowEmbedding):
            raise TypeError("reserve_rows needs table='sparse'")
        self.table.reserve(ids)

    @torch.no_grad()
    def load_rows(self, rows: SparseRowStore) -> None:
        """Replace the table content with `rows` (a copy)."""
        tab = self.table
        if isinstance(tab, SparseRowEmbedding):
            tab.load_store(rows)
        else:
            tab.weight.zero_()
            tab.weight[rows.row_ids] = rows.values.to(tab.weight.dtype)

    def rows_store(self) -> SparseRowStore:
        tab = self.table
        if isinstance(tab, SparseRowEmbedding):
            return tab.store
        return SparseRowStore.from_dense(tab.weight)

    def min_norm_rows(self, delta: torch.Tensor) -> torch.Tensor:
        """G-lin: the min-norm concatenated row vector e with 0.5 * W_v e = delta (W_v = s * orthonormal)."""
        s = self.fmt.module.value_scale
        return 2.0 * self.value_proj.detach().T @ delta / (s * s)

    # ---------------------------------------------------------------------------------------------- install
    def install(self, host: nn.Module, **kw):
        from .qwen35 import install_graft
        return install_graft(host, self, **kw)


__all__ = ["NGramGraft", "UnigramNGramEmbedding", "build_ple_layer", "module_namespace"]
