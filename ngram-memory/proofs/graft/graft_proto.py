"""ngram-memory GRAFT prototype.

Installs a Flash-Next-format n-gram module (transformers' own `Qwen4ExpTextPLELayer`, imported, not re-implemented)
into host models that have no native n-gram table: qwen3_5 (Qwen3.8-27B), qwen3_5_moe (Qwen3.6-35B-A3B), gemma4
(gemma-4-26B-A4B-it, gemma-4-31B-it; PLE disabled).

Design (see report):
  * module   : Qwen4ExpTextPLELayer built from a SimpleNamespace config: host hidden_size, hc_count=1 (host has no
               hyper-connections; the documented simplification), Flash-Next KEY spec (key vocab 248320, ngram_size 3,
               heads_per_ngram 8, ngram_vocab_size_base 20M, divisor 128, seed 1234, eos 248044, ple_layer_index 0).
  * site     : residual stream at the INPUT of decoder layer 1 (0-indexed), i.e. exactly where Flash-Next adds its PLE
               (config ple_layer_ids=[2] is one-indexed -> layers.1.ple). Done with a forward-pre-hook on layers[1];
               token ids + 2-D padding mask are captured by a forward-pre-hook on the top-level module (the inner text
               model may only receive inputs_embeds in the ForConditionalGeneration path).
  * cache    : Qwen hosts: layer 1 is a GDN layer whose cache layer is LinearAttentionLayer -> we extend its state slots
               to 3 and the PLE uses state_idx 1 (short conv) / 2 (n-gram context) exactly as Flash-Next does natively.
               gemma4 hosts: layer 1 is an attention layer (no conv states) -> a side LinearAttentionLayer is attached
               to the Cache object and reorder/select/repeat/reset are forwarded to it.
  * init     : 'editor' = zero table, key_proj=0 (gate == sigmoid(0) == 0.5, query independent), conv1d=0 (no
               cross-position spread), norms 0 (== x1), value_proj = fixed random orthogonal (seeded). Then
               out(t) = 0.5 * W_v @ concat_h row[h, id_h(t)]  -- linear in the written rows.
  * table    : 'dense' (nn.Embedding, tiny tests) or 'sparse' (SparseRows: sorted written-row ids + compact values;
               mathematically identical to a zero dense table with those rows written; the real 320,001,536 x w table is
               never materialised).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import LinearAttentionCacheLayerMixin, LinearAttentionLayer
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4

FLASH_NEXT_KEY_SPEC = dict(
    vocab_size=248320,  # NB: the hash multipliers depend on vocab_size (modeling_qwen4_exp._build_layer_multipliers)
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    eos_token_id=248044,
    ple_layer_index=0,
)
PLE_STATE_CONV, PLE_STATE_NGRAM = 1, 2  # state ids used by Qwen4ExpTextPLELayer / Qwen4ExpTextNGramEmbedding


# ----------------------------------------------------------------------------------------------------------------- table
class SparseRows(nn.Module):
    """Drop-in for nn.Embedding(num_embeddings, dim) whose rows are all zero except `row_ids` (sorted, unique)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.register_buffer("row_ids", torch.empty(0, dtype=torch.long))
        self.rows = nn.Parameter(torch.empty(0, embedding_dim))

    @property
    def weight(self):  # Qwen4ExpTextNGramEmbedding.forward reads .weight.device
        return self.rows

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = self.row_ids.numel()
        if n == 0:
            return torch.zeros(*ids.shape, self.embedding_dim, dtype=self.rows.dtype, device=ids.device)
        pos = torch.searchsorted(self.row_ids, ids).clamp_max(n - 1)
        hit = self.row_ids[pos] == ids
        return self.rows[pos] * hit.unsqueeze(-1).to(self.rows.dtype)

    @torch.no_grad()
    def write(self, ids: torch.Tensor, values: torch.Tensor, accumulate: bool = False):
        ids = ids.reshape(-1).long().to(self.row_ids.device)
        values = values.reshape(-1, self.embedding_dim).to(self.rows)
        all_ids = torch.cat([self.row_ids, ids]).unique()
        new_rows = torch.zeros(all_ids.numel(), self.embedding_dim, dtype=self.rows.dtype, device=self.rows.device)
        if self.row_ids.numel():
            new_rows[torch.searchsorted(all_ids, self.row_ids)] = self.rows.data
        p = torch.searchsorted(all_ids, ids)
        if accumulate:
            new_rows.index_add_(0, p, values)
        else:
            new_rows[p] = values
        self.row_ids = all_ids
        self.rows = nn.Parameter(new_rows, requires_grad=self.rows.requires_grad)

    def dense(self) -> torch.Tensor:
        w = torch.zeros(self.num_embeddings, self.embedding_dim, dtype=self.rows.dtype)
        w[self.row_ids] = self.rows.data
        return w


# ------------------------------------------------------------------------------------------------------- cache adapters
def _ensure_ple_slots(layer: LinearAttentionCacheLayerMixin, n: int = 3):
    """Give a host GDN cache layer the extra conv-state slots Flash-Next's PLE uses (state 1 conv, state 2 n-gram ctx)."""
    if layer.number_of_states >= n:
        return
    for d, v in ((layer.conv_states, None), (layer.recurrent_states, None), (layer.is_conv_states_initialized, False),
                 (layer.is_recurrent_states_initialized, False), (layer.has_previous_state, False),
                 (layer.conv_kernel_size, None)):
        for i in range(layer.number_of_states, n):
            d.setdefault(i, v)
    layer.number_of_states = n


class _SideState:
    """Minimal get_conv_state/update_conv_state view used when the host's layer-1 cache has no conv states (gemma4)."""

    def __init__(self):
        self.layer = LinearAttentionLayer(number_of_states=3)

    def get_conv_state(self, layer_idx, state_idx=0):
        return self.layer.conv_states[state_idx] if self.layer.has_previous_state[state_idx] else None

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kw):
        return self.layer.update_conv_state(conv_states, state_idx, **kw)

    def _apply(self, fn):
        for i, t in self.layer.conv_states.items():
            if t is not None:
                self.layer.conv_states[i] = fn(t)


def _side_state_for(cache, key: str) -> _SideState:
    side = cache.__dict__.get(key)
    if side is not None:
        return side
    side = _SideState()
    cache.__dict__[key] = side
    # forward cache-wide batch ops to the side state (beam search / num_return_sequences / contrastive search / reset)
    orig_reorder, orig_reset = cache.reorder_cache, cache.reset
    orig_rep, orig_sel = cache.batch_repeat_interleave, cache.batch_select_indices

    def reorder_cache(beam_idx):
        orig_reorder(beam_idx)
        side._apply(lambda t: t.index_select(0, beam_idx.to(t.device)))

    def reset():
        orig_reset()
        side.layer.reset()

    def batch_repeat_interleave(repeats):
        orig_rep(repeats)
        side._apply(lambda t: t.repeat_interleave(repeats, dim=0))

    def batch_select_indices(indices):
        orig_sel(indices)
        side._apply(lambda t: t[indices, ...])

    cache.reorder_cache, cache.reset = reorder_cache, reset
    cache.batch_repeat_interleave, cache.batch_select_indices = batch_repeat_interleave, batch_select_indices
    return side


# ---------------------------------------------------------------------------------------------------------------- graft
def _decoder_layers(host):
    m = host
    for name in ("model", "language_model"):
        if hasattr(m, name) and not hasattr(m, "layers"):
            m = getattr(m, name)
    if not hasattr(m, "layers"):
        m = m.language_model
    return m.layers


class NGramGraft(nn.Module):
    def __init__(self, hidden_size: int, key_spec: dict = FLASH_NEXT_KEY_SPEC, layer_idx: int = 1,
                 ple_embed_dim: int | None = None, conv_kernel_size: int = 4, rms_norm_eps: float = 1e-6,
                 table: str = "sparse", init: str = "editor", value_seed: int = 0):
        super().__init__()
        ks = dict(key_spec)
        self.layer_idx = layer_idx
        self.cfg = SimpleNamespace(
            hidden_size=hidden_size, hc_count=1, ple_embed_dim=ple_embed_dim or hidden_size,
            ple_conv_kernel_size=conv_kernel_size, rms_norm_eps=rms_norm_eps,
            vocab_size=ks["vocab_size"], ngram_size=ks["ngram_size"], heads_per_ngram=ks["heads_per_ngram"],
            ngram_vocab_size_base=ks["ngram_vocab_size_base"],
            make_ngram_vocab_size_divisible_by=ks["make_ngram_vocab_size_divisible_by"],
            seed=ks["seed"], eos_token_id=ks["eos_token_id"],
        )
        n_heads = (self.cfg.ngram_size - 1) * self.cfg.heads_per_ngram
        assert self.cfg.ple_embed_dim % n_heads == 0, "ple_embed_dim must be divisible by the number of n-gram heads"
        # build on meta so the (possibly 320M-row) dense table is never allocated, then swap / materialise
        with torch.device("meta"):
            self.ple = q4.Qwen4ExpTextPLELayer(self.cfg, layer_idx, ks["ple_layer_index"])
        emb = self.ple.ple_embedding
        if table == "sparse":
            emb.ngram_embedding = SparseRows(emb.ngram_embedding.num_embeddings, emb.ngram_embedding.embedding_dim)
        self.ple.to_empty(device="cpu")
        self._reinit_buffers()
        self.reset_parameters(init, value_seed)
        self.eos = self.cfg.eos_token_id
        self._ids = self._mask2d = None
        self._handles = []
        self._side_key = f"_ngram_graft_side_{id(self)}"
        self.debug_last = {}

    # ---- init
    @torch.no_grad()
    def _reinit_buffers(self):
        e = self.ple.ple_embedding  # same as Qwen4ExpPreTrainedModel._init_weights for Qwen4ExpTextNGramEmbedding
        e.layer_multipliers.copy_(q4._build_layer_multipliers(e.unigram_vocab_size, e.ngram_size, e.ple_layer_index, e.seed))
        e.ngram_heads_vocab_sizes.copy_(torch.tensor(e.head_vocab_sizes, dtype=torch.long))
        e.ngram_heads_offsets.copy_(torch.tensor(e.head_offsets, dtype=torch.long))

    @torch.no_grad()
    def reset_parameters(self, init: str = "editor", value_seed: int = 0):
        p = self.ple
        g = torch.Generator().manual_seed(value_seed)
        for n in (p.norm_key, p.norm_query, p.norm_conv):
            n.weight.zero_()  # Qwen4ExpTextRMSNorm is (1 + weight)
        emb = p.ple_embedding.ngram_embedding
        if isinstance(emb, nn.Embedding):
            emb.weight.zero_()
        else:
            emb.row_ids = torch.empty(0, dtype=torch.long)
            emb.rows = nn.Parameter(torch.empty(0, emb.embedding_dim))
        E, d = self.cfg.ple_embed_dim, self.cfg.hidden_size
        if init == "editor":
            p.key_proj.weight.zero_()
            p.conv1d.weight.zero_()
            q, _ = torch.linalg.qr(torch.randn(max(d, E), max(d, E), generator=g))
            p.value_proj.weight.copy_(q[:d, :E])  # orthonormal rows (d<=E) or columns (d>=E)
        elif init == "random":  # stress test of every code path (gate, conv, norms)
            p.key_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            p.value_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            p.conv1d.weight.normal_(0, 0.5, generator=g)
            for n in (p.norm_key, p.norm_query, p.norm_conv):
                n.weight.normal_(0, 0.1, generator=g)
        else:
            raise ValueError(init)

    # ---- keys
    @torch.no_grad()
    def row_ids(self, token_history: torch.Tensor) -> torch.Tensor:
        """Global table row ids (B, T, 16) for each position of an (B, T) id tensor, no cache (fresh segment)."""
        cap = {}
        emb = self.ple.ple_embedding
        orig = emb.ngram_embedding

        class _Cap(nn.Module):
            weight = torch.empty(0)

            def forward(self, ids):
                cap["ids"] = ids
                return torch.zeros(*ids.shape, 1)

        emb.ngram_embedding = _Cap()
        try:
            emb(token_history, None)
        finally:
            emb.ngram_embedding = orig
        return cap["ids"]

    def head_slices(self):
        w = self.ple.ple_embedding.ngram_embedding.embedding_dim
        return [slice(h * w, (h + 1) * w) for h in range(self.ple.ple_embedding.ngram_heads)]

    @torch.no_grad()
    def write_ngram(self, ngram: list[int], e_concat: torch.Tensor, accumulate=False):
        """Write the 8 rows of n-gram `ngram` (len 2 or 3) from a ple_embed_dim vector (only that order's slices used)."""
        emb = self.ple.ple_embedding
        n = len(ngram)
        ids = self.row_ids(torch.tensor([ngram]))[0, -1]  # row ids at the last position of the n-gram
        hs = range((n - 2) * emb.heads_per_ngram, (n - 1) * emb.heads_per_ngram)
        sl = self.head_slices()
        vals = torch.stack([e_concat[sl[h]] for h in hs])
        tab = emb.ngram_embedding
        if isinstance(tab, SparseRows):
            tab.write(ids[list(hs)], vals, accumulate)
        else:
            if accumulate:
                tab.weight.data[ids[list(hs)]] += vals
            else:
                tab.weight.data[ids[list(hs)]] = vals
        return ids[list(hs)]

    # ---- install
    def install(self, host):
        layers = _decoder_layers(host)
        self.to(next(host.parameters()).device)
        self._handles = [
            host.register_forward_pre_hook(self._capture, with_kwargs=True),
            layers[self.layer_idx].register_forward_pre_hook(self._inject, with_kwargs=True),
        ]
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _capture(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise NotImplementedError("NGramGraft needs input_ids (inputs_embeds-only calls: pass input_ids too / "
                                      "reverse the embedding as modeling_qwen4_exp.reverse_embedding does)")
        m = kwargs.get("attention_mask")
        self._ids = ids
        self._mask2d = m if isinstance(m, torch.Tensor) and m.ndim == 2 else None

    def _inject(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        T = hidden.shape[1]
        ids = self._ids
        assert ids is not None and ids.shape[1] == T, (None if ids is None else ids.shape, hidden.shape)
        # replicate masking_utils.create_recurrent_attention_mask + Qwen4ExpTextModel.forward's eos substitution
        conv_mask = None
        m = self._mask2d
        if m is not None and T > 1 and not torch.all(m == 1):
            conv_mask = m[:, -T:].contiguous()
        ple_ids = torch.where(conv_mask.bool(), ids, self.eos) if conv_mask is not None else ids
        cache = kwargs.get("past_key_values")
        if cache is not None:
            hl = cache.layers[self.layer_idx] if self.layer_idx < len(cache.layers) else None
            if isinstance(hl, LinearAttentionCacheLayerMixin):
                _ensure_ple_slots(hl)
                state = cache
            else:
                state = _side_state_for(cache, self._side_key)
        else:
            state = None
        out = self.ple(hidden, ple_ids, state, conv_mask=conv_mask)
        self.debug_last = dict(out=out, conv_mask=conv_mask, layer_mask=kwargs.get("attention_mask"))
        hidden = hidden + out.to(hidden.dtype)
        if args:
            args = (hidden,) + tuple(args[1:])
        else:
            kwargs["hidden_states"] = hidden
        return args, kwargs
