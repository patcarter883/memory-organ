"""Cartridge — a trainable per-layer KV prefix (Cartridges, arXiv:2506.06266) for a FROZEN base.

A Cartridge Z is a set of trainable Key/Value vectors, one small block per attention layer, that is
PREPENDED to every layer's K/V (post-projection, post-RoPE). It is the amortization of the frozen
model's *in-context* behaviour into a persistent memory: train Z by KL-distilling the base-with-edit-
in-context (teacher) into the base+Z (student, NO edit text). See cam/cartridge_train.py.

Shape:  K, V  each [n_layers, p, n_kv_heads, head_dim]   (fp32 nn.Parameters; p in {128,512}).
The prefix is POSITION-0 / TIMELESS: no RoPE is applied to it (it is appended AFTER the base's RoPE,
exactly like gated_tap.KVInjector's memory columns) and its columns are ALWAYS attendable by every
query (a non-causal, sink-style block). Per-layer scalar openness gate `log_gate` (KVInjector-style):
sigmoid(log_gate) biases the prefix logits by log(g); init OPEN (g~=1, ~no bias). Closing the gate to
-inf makes the prefix an EXACT no-op regardless of K/V — the clean no-op / detach guarantee.

Only Z (K, V, log_gate) trains; the base is frozen / inference_mode. Mechanically a drop-in cousin of
gated_tap.KVInjector, but (a) EVERY attention layer, (b) a trainable per-layer prefix (no bank / no
projection), (c) faithful to standard-attention Qwen3 (no q_proj gate chunk).
"""
import os
import sys

import torch
import torch.nn as nn

try:
    from .gated_tap import decoder_layers
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from gated_tap import decoder_layers  # noqa: E402


def _attn_dims(base):
    """(n_layers, n_kv_heads, head_dim) from the base config, robust to head_dim absent."""
    cfg = base.config.get_text_config()
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return n_layers, n_kv, head_dim


class Cartridge(nn.Module):
    """Trainable per-layer KV prefix attached to every attention layer of a frozen base."""

    def __init__(self, base, p=128, gate_init=4.0):
        super().__init__()
        self.p = int(p)
        self.n_layers, self.n_kv, self.head_dim = _attn_dims(base)
        # fp32 params for stable distillation (the frozen base runs bf16; the prefix is cast per-forward).
        self.K = nn.Parameter(torch.zeros(self.n_layers, self.p, self.n_kv, self.head_dim))
        self.V = nn.Parameter(torch.zeros(self.n_layers, self.p, self.n_kv, self.head_dim))
        # per-layer openness gate: sigmoid(log_gate) in (0,1); init +4 -> ~0.98 (~fully attendable but
        # trainable). Closing to -inf masks the prefix (exact no-op). Kept for stability + the no-op smoke.
        self.log_gate = nn.Parameter(torch.full((self.n_layers,), float(gate_init)))
        self._layers = decoder_layers(base)
        self._handles_layers = []     # (layer_idx, attn_module) we patched
        self._active = True           # runtime toggle; False -> exact no-op (prefix skipped)

    # ---- runtime toggle (no-op verification / ablations) -----------------------------------------
    def set_active(self, flag: bool):
        self._active = bool(flag)
        return self

    # ---- the prepend (per layer L): position-0 K/V block + extended additive mask ----------------
    def _prepend(self, L, key_states, value_states, attention_mask, query_states):
        """key/value_states: [B,n_kv,Tk,hd] (post-RoPE). Returns the prefix-prepended (k,v,mask), or the
        inputs UNCHANGED when inactive (exact no-op)."""
        if not self._active:
            return key_states, value_states, attention_mask
        B, _, Tk, _ = key_states.shape
        p = self.p
        Km = self.K[L].permute(1, 0, 2).unsqueeze(0).expand(B, self.n_kv, p, self.head_dim)  # [B,n_kv,p,hd]
        Vm = self.V[L].permute(1, 0, 2).unsqueeze(0).expand(B, self.n_kv, p, self.head_dim)
        Km = Km.to(device=key_states.device, dtype=key_states.dtype)
        Vm = Vm.to(device=value_states.device, dtype=value_states.dtype)
        k2 = torch.cat([Km, key_states], dim=2)                        # prefix FIRST (position-0 block)
        v2 = torch.cat([Vm, value_states], dim=2)
        Tq = query_states.shape[2]
        g = torch.sigmoid(self.log_gate[L]).to(device=key_states.device, dtype=torch.float32)
        pb = torch.log(g).view(1, 1, 1, 1).expand(B, 1, Tq, p)         # prefix-column bias (0 when open)
        if attention_mask is None:
            # SDPA pure-causal fast path gave no mask: build additive causal over real keys + open prefix.
            off = Tk - Tq                                              # cache offset (0 without a cache)
            col = torch.arange(Tk, device=key_states.device).view(1, 1, 1, Tk)
            row = torch.arange(Tq, device=key_states.device).view(1, 1, Tq, 1) + off
            seq = torch.zeros(B, 1, Tq, Tk, dtype=torch.float32, device=key_states.device)
            seq = seq.masked_fill(col > row, float("-inf"))            # causal: key j > query i => masked
        else:
            am = attention_mask
            if am.dtype == torch.bool:                                # boolean (True=attend) -> additive
                am = torch.zeros_like(am, dtype=torch.float32).masked_fill(~am, float("-inf"))
            am = am.to(torch.float32)
            if am.shape[-1] != Tk:                                    # kv-dim must match real keys
                am = am[..., :Tk]
            seq = am
        full = torch.cat([pb, seq], dim=-1).to(dtype=query_states.dtype)   # prefix cols FIRST, then real
        return k2, v2, full

    # ---- attach / detach: monkey-patch each attention layer's self_attn.forward ------------------
    def attach(self, base):
        cart = self
        for L, layer in enumerate(self._layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "q_proj"):
                continue                                              # skip non-full-attention layers
            mod = sys.modules[type(attn).__module__]
            apply_rotary = mod.apply_rotary_pos_emb
            eager = mod.eager_attention_forward
            ALL = mod.ALL_ATTENTION_FUNCTIONS

            def make_patched(attn, L, apply_rotary, eager, ALL):
                def patched(hidden_states, position_embeddings, attention_mask=None,
                            past_key_values=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn.head_dim)
                    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary(query_states, key_states, cos, sin)
                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(
                            key_states, value_states, attn.layer_idx)
                    # ---- CARTRIDGE PREFIX: prepend timeless K/V after RoPE/cache ----
                    key_states, value_states, attention_mask = cart._prepend(
                        L, key_states, value_states, attention_mask, query_states)
                    attention_interface = ALL.get_interface(attn.config._attn_implementation, eager)
                    attn_output, attn_weights = attention_interface(
                        attn, query_states, key_states, value_states, attention_mask,
                        dropout=0.0 if not attn.training else attn.attention_dropout,
                        scaling=attn.scaling, sliding_window=getattr(attn, "sliding_window", None),
                        **kwargs)
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn.o_proj(attn_output)
                    return attn_output, attn_weights
                return patched

            if "forward" in attn.__dict__:                            # idempotent: restore before repatching
                del attn.__dict__["forward"]
            attn.forward = make_patched(attn, L, apply_rotary, eager, ALL)
            self._handles_layers.append((L, attn))
        return self

    def detach(self):
        for _L, attn in self._handles_layers:
            try:
                del attn.__dict__["forward"]                          # remove instance attr -> class method
            except KeyError:
                pass
        self._handles_layers = []
        return self

    # ---- init from the first p tokens of the edit text (random init is unstable per the paper) -----
    @torch.no_grad()
    def init_from_text(self, base, tok, edit_text, device=None):
        """Run the FROZEN base on the first p tokens of `edit_text`, capture each layer's post-RoPE K/V,
        and use them as Z's init. If the text is shorter than p, the block is left-padded with zeros
        (those columns start as pure sinks). No RoPE is stripped: we snapshot the K/V exactly as the base
        computed them (positions 0..len-1) — the prefix then lives as a timeless block."""
        device = device or next(base.parameters()).device
        ids = tok(edit_text, add_special_tokens=False).input_ids[: self.p]
        if tok.bos_token_id is not None:
            ids = [tok.bos_token_id] + ids
            ids = ids[: self.p]
        n = len(ids)
        cap_k = [None] * self.n_layers
        cap_v = [None] * self.n_layers
        saved = []
        for L, layer in enumerate(self._layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "q_proj"):
                continue
            mod = sys.modules[type(attn).__module__]
            apply_rotary = mod.apply_rotary_pos_emb

            def make_cap(attn, L, apply_rotary):
                def cap(hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn.head_dim)
                    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary(query_states, key_states, cos, sin)
                    cap_k[L] = key_states.detach()[0].transpose(0, 1).contiguous()   # [Tk,n_kv,hd]
                    cap_v[L] = value_states.detach()[0].transpose(0, 1).contiguous()
                    attention_interface = mod.ALL_ATTENTION_FUNCTIONS.get_interface(
                        attn.config._attn_implementation, mod.eager_attention_forward)
                    attn_output, attn_weights = attention_interface(
                        attn, query_states, key_states, value_states, attention_mask,
                        dropout=0.0, scaling=attn.scaling,
                        sliding_window=getattr(attn, "sliding_window", None), **kwargs)
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn.o_proj(attn_output)
                    return attn_output, attn_weights
                return cap

            if "forward" in attn.__dict__:
                del attn.__dict__["forward"]
            attn.forward = make_cap(attn, L, apply_rotary)
            saved.append(attn)
        try:
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            base(input_ids=inp)
        finally:
            for attn in saved:
                try:
                    del attn.__dict__["forward"]
                except KeyError:
                    pass
        for L in range(self.n_layers):
            if cap_k[L] is None:
                continue
            k = cap_k[L].to(self.K.dtype)      # [n, n_kv, hd]
            v = cap_v[L].to(self.V.dtype)
            m = min(n, self.p)
            self.K.data[L, self.p - m:].copy_(k[-m:].to(self.K.device))   # right-align: real tokens at the end
            self.V.data[L, self.p - m:].copy_(v[-m:].to(self.V.device))
        return self


class ComposedCartridge(Cartridge):
    """COMPOSABILITY (Cartridges §composition): concatenate N independently-trained Cartridges along the
    prefix (p) dim and serve them together WITHOUT retraining. Each member's per-layer K/V blocks are
    stacked so every member's timeless columns are attendable by every query simultaneously; each member
    keeps its OWN per-layer openness gate (broadcast over its own columns -> a per-column log-gate vector).

    Built via ``concat_cartridges([c1, c2, ...])``. Inference-only (params frozen); reuses the parent's
    attach/detach/_prepend contract, overriding only the gate to be per-column."""

    def __init__(self, carts):
        nn.Module.__init__(self)
        if not carts:
            raise ValueError("concat_cartridges: empty list")
        c0 = carts[0]
        self.n_layers, self.n_kv, self.head_dim = c0.n_layers, c0.n_kv, c0.head_dim
        for c in carts:
            if (c.n_layers, c.n_kv, c.head_dim) != (self.n_layers, self.n_kv, self.head_dim):
                raise ValueError("concat_cartridges: cartridges have mismatched attn dims")
        self.members = list(carts)
        self.p = int(sum(c.p for c in carts))
        # concat K/V along the prefix dim (dim=1 of [n_layers, p, n_kv, hd]); frozen (eval-only compose).
        self.K = nn.Parameter(torch.cat([c.K.data for c in carts], dim=1), requires_grad=False)
        self.V = nn.Parameter(torch.cat([c.V.data for c in carts], dim=1), requires_grad=False)
        # per-column log-gate: each member's per-layer scalar gate repeated over ITS own p columns.
        cols = [c.log_gate.data.unsqueeze(1).expand(self.n_layers, c.p) for c in carts]
        self._col_loggate = nn.Parameter(torch.cat(cols, dim=1), requires_grad=False)  # [n_layers, p_total]
        # keep a nominal per-layer scalar for API compatibility (unused by our _prepend).
        self.log_gate = nn.Parameter(torch.zeros(self.n_layers), requires_grad=False)
        self._layers = c0._layers      # same base -> same decoder layers
        self._handles_layers = []
        self._active = True

    def _prepend(self, L, key_states, value_states, attention_mask, query_states):
        """Identical to Cartridge._prepend but the prefix-column bias uses the PER-COLUMN log-gate so each
        composed member keeps its own openness."""
        if not self._active:
            return key_states, value_states, attention_mask
        B, _, Tk, _ = key_states.shape
        p = self.p
        Km = self.K[L].permute(1, 0, 2).unsqueeze(0).expand(B, self.n_kv, p, self.head_dim)
        Vm = self.V[L].permute(1, 0, 2).unsqueeze(0).expand(B, self.n_kv, p, self.head_dim)
        Km = Km.to(device=key_states.device, dtype=key_states.dtype)
        Vm = Vm.to(device=value_states.device, dtype=value_states.dtype)
        k2 = torch.cat([Km, key_states], dim=2)
        v2 = torch.cat([Vm, value_states], dim=2)
        Tq = query_states.shape[2]
        g = torch.sigmoid(self._col_loggate[L]).to(device=key_states.device, dtype=torch.float32)  # [p]
        pb = torch.log(g).view(1, 1, 1, p).expand(B, 1, Tq, p)                # per-column prefix bias
        if attention_mask is None:
            off = Tk - Tq
            col = torch.arange(Tk, device=key_states.device).view(1, 1, 1, Tk)
            row = torch.arange(Tq, device=key_states.device).view(1, 1, Tq, 1) + off
            seq = torch.zeros(B, 1, Tq, Tk, dtype=torch.float32, device=key_states.device)
            seq = seq.masked_fill(col > row, float("-inf"))
        else:
            am = attention_mask
            if am.dtype == torch.bool:
                am = torch.zeros_like(am, dtype=torch.float32).masked_fill(~am, float("-inf"))
            am = am.to(torch.float32)
            if am.shape[-1] != Tk:
                am = am[..., :Tk]
            seq = am
        full = torch.cat([pb, seq], dim=-1).to(dtype=query_states.dtype)
        return k2, v2, full


def concat_cartridges(carts):
    """Compose independently-trained Cartridges into one servable prefix (Cartridges paper: composition
    without retraining). Returns a ComposedCartridge whose per-layer K/V are the members' K/V stacked
    along the prefix dim. A single-element list is composed too (uniform code path)."""
    return ComposedCartridge(list(carts))
