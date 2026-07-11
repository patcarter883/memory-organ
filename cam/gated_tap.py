"""GatedMemoryTap — zero-init Memory-as-Gate (MAG) injection for the CAM v0 falsifier.

Replaces boltA's input-embeds prefix (MAC) with an additive, zero-initialised gated cross-attention
tap on one (or more) FROZEN decoder layer(s). At init the gate is exactly 0 -> the base is
bit-identical to baseline; training opens the gate only as far as LM loss rewards (the load-bearing
stability property, ref 2603.16413).

   q = Wq h_l ; k = Wk bank ; v = Wv bank
   a = softmax(q k^T / sqrt(d_head))         # cross-attention over the K memory slots
   y = Wo (a v)
   h_l' = h_l + tanh(gamma) ⊙ y              # gamma in R^H, init 0  => g=0 => exact no-op

Injected via a forward hook on base.model.layers[L] (HF decoder layers return a tuple; the post-hook
rewrites output[0]). The bank ([B,K,mem_dim], query-conditioned, leak-free) is stashed before each
base(...) call and read by the hook.
"""
import os
import torch
import torch.nn as nn


def decoder_layers(base):
    """Locate the ModuleList of decoder layers across HF causal/image-text-to-text wrappers."""
    for path in ("model.layers", "model.model.layers", "language_model.model.layers",
                 "model.language_model.layers", "transformer.h"):
        obj, ok = base, True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("could not locate decoder-layer ModuleList on the base model")


class GatedMemoryTap(nn.Module):
    """Zero-init gated cross-attention from the residual stream into the K-slot memory bank."""

    def __init__(self, base_hidden, mem_dim, n_heads=8, conf_gate=False, n_rel=1):
        super().__init__()
        assert base_hidden % n_heads == 0, "base_hidden must divide n_heads"
        self.H, self.n_heads, self.d_head = base_hidden, n_heads, base_hidden // n_heads
        self.to_q = nn.Linear(base_hidden, base_hidden, bias=False)
        self.to_k = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_v = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_o = nn.Linear(base_hidden, base_hidden, bias=False)
        self.gamma = nn.Parameter(torch.zeros(base_hidden))   # gate logit; tanh(0)=0 -> no-op at init
        # NORM-RELATIVE CALIBRATED GATE (Phase-R P-R1, #19): the single-site-injection ceiling (~0.7,
        # WISE/MEMIT) partly comes from a fixed/unnormalized scalar gate that can't hit a per-token,
        # norm-relative target (steering is non-monotonic in a raw coefficient). CAM_NORM_GATE=1 injects
        # the value DIRECTION at a learned fraction alpha=sigmoid(gate_alpha) of the local residual norm
        # ||h||, per token (Norm-Preserving Steering). gate_alpha init -6 -> alpha~0 -> ~no-op at init.
        self.gate_alpha = nn.Parameter(torch.tensor(-6.0))
        # TWO-SIDED injection suppression gate (R1-prior, #19): learned residual-damping fraction
        # sigmoid(supp), applied (conf-scaled) where memory fires. supp init -4 -> ~0.018 (starts as pure
        # addition). Opt-in CAM_TWOSIDED=1.
        self.supp = nn.Parameter(torch.tensor(-4.0))
        # NULL / sink slot: a learnable extra key with a ZERO value. softmax attention must sum to 1,
        # so without an escape the tap injects SOMETHING for every query (even an out-of-store neighbour)
        # -> collateral damage (the Track 1 locality leak). Attending to the null key (value 0) delivers
        # ~nothing, so the tap CAN be inert for a query that doesn't match the bank. The locality loss
        # (train_taps) teaches it to route non-matching queries here. Zero-init keeps the gate no-op.
        self.null_key = nn.Parameter(torch.zeros(1, n_heads, 1, self.d_head))
        # STORE-CONFIDENCE GATE (conf_gate=True): scale the WHOLE injection by a scalar c in (0,1) driven
        # by the store's retrieval strength (pk_store.read's factual-head pre-norm ‖ctx‖, passed via
        # set_bank). The learned null slot gates on PROMPT NOVELTY — a paraphrase of the edited subject
        # looks as unfamiliar as a neighbour, so it gets suppressed (the ~0.67 generalization ceiling).
        # The confidence scalar instead keys on ACTUAL retrieval: a paraphrase retrieves its own edit
        # (strong -> c~1 -> deliver) while a neighbour retrieves nothing (weak -> c~0 -> inert), decoupling
        # delivery from novelty. conf is standardized by a running EMA (absolute scale, NOT per-batch — the
        # strong/weak positives+negatives arrive in SEPARATE forward passes, so per-batch norm would erase
        # the very distinction we gate on), then c = sigmoid(conf_scale * (conf/EMA - conf_bias)).
        # PER-RELATION EMA (n_rel>1): the multi-relation editing case mixes relations with DIFFERENT ‖ctx‖
        # scales, so ONE global EMA can't separate strong-vs-weak across all of them (the locality leak). A
        # separate EMA per relation (indexed by the queried relation, batch-uniform per build) normalizes
        # each relation to its OWN scale. Shared scale/bias then shape the sigmoid uniformly. n_rel=1 is the
        # byte-identical single-EMA path.
        self.conf_gate = conf_gate
        self.n_rel = n_rel
        self.conf_scale = nn.Parameter(torch.tensor(4.0))     # sigmoid steepness (learned, shared)
        self.conf_bias = nn.Parameter(torch.tensor(1.0))      # threshold in EMA-normalized units (learned, shared)
        self.register_buffer("conf_ema", torch.full((n_rel,), -1.0))  # per-relation running mean; <0 = uninit
        self._conf = None                                     # [B] store-confidence scalar, set per-forward
        self._relidx = None                                   # queried relation index (batch-uniform), set per-forward
        self._bank = None                                     # [B,K,mem_dim], set per-forward
        self.last_gate = torch.tensor(0.0)
        self.last_attn_entropy = torch.tensor(0.0)
        self.last_null_attn = torch.tensor(0.0)               # mean softmax mass on the null slot
        self.last_conf = torch.tensor(0.0)                    # mean raw store-confidence
        self.last_cgate = torch.tensor(1.0)                   # mean confidence-gate multiplier c

    def set_bank(self, bank, conf=None, relidx=None):
        self._bank = bank
        self._conf = conf
        self._relidx = relidx

    def _split(self, t):
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)   # [B,nh,T,dh]

    def forward(self, h):
        """h: [B,T,H] residual hidden -> injected [B,T,H]. No-op when no bank is set.

        The tap params live in fp32 (stable gate training); the frozen base hidden is bf16. Compute
        the whole tap in the param dtype, then cast the additive update back to h.dtype so the base
        residual stream stays bf16 (and the gate=0 init is an exact no-op)."""
        bank = self._bank
        if bank is None:
            return h
        wdt = self.to_q.weight.dtype                           # tap compute dtype (fp32)
        h32 = h.to(wdt)
        bank = bank.to(device=h.device, dtype=wdt)             # MODEL-PARALLEL: bank (on cuda:0) -> layer's device
        B = h32.shape[0]
        q = self._split(self.to_q(h32))                        # [B,nh,T,dh]
        k = self._split(self.to_k(bank))                       # [B,nh,K,dh]
        v = self._split(self.to_v(bank))                       # [B,nh,K,dh]
        # append the null slot: learnable key, ZERO value (so attending to it injects nothing).
        nk = self.null_key.to(wdt).expand(B, self.n_heads, 1, self.d_head)          # [B,nh,1,dh]
        k = torch.cat([k, nk], dim=2)                          # [B,nh,K+1,dh]
        v = torch.cat([v, torch.zeros_like(nk)], dim=2)        # [B,nh,K+1,dh] (null value = 0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (self.d_head ** 0.5), dim=-1)   # [B,nh,T,K+1]
        ctx = (a @ v).transpose(1, 2).reshape(h32.shape)       # [B,T,H]  (null contributes 0)
        y = self.to_o(ctx)
        self.last_attn_entropy = (-(a.clamp_min(1e-9).log() * a).sum(-1)).mean().detach()
        self.last_null_attn = a[..., -1].mean().detach()       # softmax mass routed to the null slot
        if os.environ.get("CAM_NORM_GATE") == "1":             # P-R1: inject value DIRECTION at alpha*||h||
            ydir = y / (y.norm(dim=-1, keepdim=True) + 1e-6)   # [B,T,H] unit direction
            alpha = torch.sigmoid(self.gate_alpha)             # learned fraction in (0,1)
            upd = alpha * h32.norm(dim=-1, keepdim=True) * ydir  # per-token norm-relative magnitude
            self.last_gate = alpha.detach()
        else:
            g = torch.tanh(self.gamma)                         # [H]; 0 at init
            self.last_gate = g.abs().mean().detach()
            upd = g * y                                        # [B,T,H] gated injection
        if self.conf_gate and self._conf is not None:
            cf = self._conf.to(device=h.device, dtype=wdt).detach()   # [B] store frozen -> no grad; to layer's device
            ri = int(self._relidx) if self._relidx is not None else 0
            ri = max(0, min(ri, self.n_rel - 1))
            if self.training:                                  # track this relation's absolute conf scale
                m = cf.mean().detach()
                if float(self.conf_ema[ri]) < 0:
                    self.conf_ema[ri] = m
                else:
                    self.conf_ema[ri] = 0.99 * self.conf_ema[ri] + 0.01 * m
            scale = self.conf_ema[ri].clamp_min(1e-4)
            c = torch.sigmoid(self.conf_scale * (cf / scale - self.conf_bias))   # [B] in (0,1)
            self.last_conf = cf.mean().detach()
            self.last_cgate = c.mean().detach()
            upd = c.view(B, 1, 1) * upd                        # deliver in proportion to retrieval strength
        if os.environ.get("CAM_TWOSIDED") == "1":             # R1-prior: PROMOTE new + DAMP the prior.
            # R-mech: solo failures = strong-base-prior facts (the edit only pushes toward the new object,
            # never suppresses the original, so a confident prior wins). Two-sided: attenuate the residual
            # (the base's own next-token tendency) where memory fires, making room for the injected object.
            # Conf-scaled so non-memory positions are untouched; supp init -4 -> ~0 (starts as pure add).
            s = torch.sigmoid(self.supp)
            cs = c.view(B, 1, 1) if (self.conf_gate and self._conf is not None) else 1.0
            upd = upd - s * cs * h32
        return h + upd.to(h.dtype)


class MAGInjector:
    """Registers GatedMemoryTap forward-hooks on a set of frozen decoder layers, sharing one bank."""

    def __init__(self, base, tap_layers, mem_dim, n_heads=8, conf_gate=False, n_rel=1):
        H = base.config.get_text_config().hidden_size
        self.layers = decoder_layers(base)
        self.tap_layers = list(tap_layers)
        self.taps = nn.ModuleDict({str(L): GatedMemoryTap(H, mem_dim, n_heads, conf_gate=conf_gate, n_rel=n_rel)
                                   for L in self.tap_layers})
        self._handles = []

    def to(self, dev):
        self.taps.to(dev)
        return self

    def parameters(self):
        return self.taps.parameters()

    def train(self, mode=True):
        self.taps.train(mode)
        return self

    def eval(self):
        self.taps.eval()
        return self

    def set_bank(self, bank, conf=None, relidx=None):
        for t in self.taps.values():
            t.set_bank(bank, conf=conf, relidx=relidx)

    def gate_stats(self):
        return {L: float(self.taps[str(L)].last_gate) for L in self.tap_layers}

    def null_attn_stats(self):
        return {L: float(self.taps[str(L)].last_null_attn) for L in self.tap_layers}

    def cgate_stats(self):
        return {L: float(self.taps[str(L)].last_cgate) for L in self.tap_layers}

    def _hook(self, tap):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (tap(out[0]),) + tuple(out[1:])
            return tap(out)
        return fn

    def attach(self):
        for L in self.tap_layers:
            # MODEL-PARALLEL: place the tap on ITS layer's device (device_map may shard layers across
            # GPUs) so the injection h + g*y is same-device. Single-card = no-op (all on cuda:0).
            try:
                dev = next(self.layers[L].parameters()).device
                self.taps[str(L)].to(dev)
            except StopIteration:
                pass
            self._handles.append(self.layers[L].register_forward_hook(self._hook(self.taps[str(L)])))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self


# =================================================================================================
# KVInjector (1c) — memory the FROZEN base READS at a full-attention layer.
# -------------------------------------------------------------------------------------------------
# The residual tap ADDS a gated vector to the residual stream (over-steers / degenerates when
# sustained). 1c instead APPENDS position-free memory K/V columns at a mid-depth FULL-attention layer
# (Qwen3.5-4B: L11 in {3,7,11,15,19,23,27,31}); the base's own softmax decides, per-query/per-head,
# how much to read from memory (self-dosing — no residual over-steer). This is a drop-in for
# MAGInjector w.r.t. train_taps / eval_indist_ripple: same set_bank/eval/detach/parameters contract, so
# W_km,W_vm train by LM-loss through the frozen base exactly like the tap's projection.
#
# Mechanism (spec §Mechanism): from bank [B,K,mem_dim] compute K_mem=W_km(bank), V_mem=W_vm(bank),
# reshape [B,n_kv_heads,K,head_dim] (matches key/value_states; the q_proj doubled-dim + gate is Q-ONLY,
# applied post-attention, so K/V are standard head_dim). NO RoPE on memory (append AFTER the base's
# RoPE — memory tokens are timeless). Append along the key/value seq axis; extend the additive mask
# with K attendable (non-causal) columns, biased by log(gate) so a closed/weak gate masks memory
# (-inf => exact no-op). Byte-identical when NO bank is staged (the append is skipped).
class _KVMem(nn.Module):
    """The trainable bolt-on params for KV injection (held in an nn.Module so .parameters()/.to()/.train()
    behave; W_vm zero-init + log_gate near-closed so injection starts gentle, like the tap's zero gate)."""

    def __init__(self, mem_dim, n_kv, head_dim, conf_gate, n_rel):
        super().__init__()
        self.W_km = nn.Linear(mem_dim, n_kv * head_dim, bias=False)          # mem_dim -> n_kv*head_dim
        self.W_vm = nn.Linear(mem_dim, n_kv * head_dim, bias=False)
        nn.init.zeros_(self.W_vm.weight)                                     # start with ~zero memory value
        self.log_gate = nn.Parameter(torch.tensor(-6.0))                    # base openness (near-closed at init)
        self.conf_gate = conf_gate
        self.n_rel = n_rel
        self.conf_scale = nn.Parameter(torch.tensor(4.0))                   # conf-gate sigmoid steepness
        self.conf_bias = nn.Parameter(torch.tensor(1.0))                    # threshold in EMA-normalized units
        self.register_buffer("conf_ema", torch.full((n_rel,), -1.0))        # per-relation running conf mean; <0=uninit


class KVInjector:
    """Monkey-patches a full-attention layer's self_attn.forward to APPEND position-free memory K/V after
    RoPE, before the attention interface. Drop-in for MAGInjector (set_bank/attach/detach/eval/parameters)."""

    def __init__(self, base, kv_layer, mem_dim, conf_gate=False, n_rel=1):
        self.layers = decoder_layers(base)
        self.kv_layer = int(kv_layer)
        layer = self.layers[self.kv_layer]
        assert hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"), (
            f"layer {self.kv_layer} is not a full-attention layer (no self_attn.q_proj); pick a layer in "
            f"the model's full-attention set (Qwen3.5: 3,7,11,15,19,23,27,31)")
        self.attn = layer.self_attn
        cfg = base.config.get_text_config()
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = self.attn.head_dim
        self.mem = _KVMem(mem_dim, self.n_kv, self.head_dim, conf_gate, n_rel)
        self.conf_gate = conf_gate
        self.n_rel = n_rel
        self._bank = None
        self._conf = None
        self._relidx = None
        self._force_gate = None                     # test hook: override the gate scalar (None=normal path)
        self.last_gate = torch.tensor(0.0)
        self.last_cgate = torch.tensor(1.0)
        self.last_mem_attn = torch.tensor(0.0)      # (kept for null_attn_stats parity; not computed by default)

    # --- MAGInjector-compatible surface -----------------------------------------------------------
    def to(self, dev):
        self.mem.to(dev)
        return self

    def train(self, mode=True):
        self.mem.train(mode)
        return self

    def eval(self):
        self.mem.eval()
        return self

    def parameters(self):
        return self.mem.parameters()

    def set_bank(self, bank, conf=None, relidx=None):
        self._bank = bank
        self._conf = conf
        self._relidx = relidx

    def gate_stats(self):
        return {self.kv_layer: float(self.last_gate)}

    def null_attn_stats(self):
        return {self.kv_layer: float(self.last_mem_attn)}

    def cgate_stats(self):
        return {self.kv_layer: float(self.last_cgate)}

    # --- gate: base openness sigmoid(log_gate) * optional conf-gate (per-example, EMA-normalized) ---
    def _gate(self, B, device, dtype):
        if self._force_gate is not None:
            return torch.full((B,), float(self._force_gate), device=device, dtype=dtype)
        g0 = torch.sigmoid(self.mem.log_gate).to(device=device, dtype=dtype)       # scalar base openness
        if self.conf_gate and self._conf is not None:
            conf = self._conf.to(device=device, dtype=dtype).reshape(-1)
            ri = int(self._relidx) if self._relidx is not None else 0
            ri = max(0, min(ri, self.n_rel - 1))
            if self.mem.training:                                                   # track this relation's conf scale
                m = conf.mean().detach()
                if float(self.mem.conf_ema[ri]) < 0:
                    self.mem.conf_ema[ri] = m
                else:
                    self.mem.conf_ema[ri] = 0.99 * self.mem.conf_ema[ri] + 0.01 * m
            scale = self.mem.conf_ema[ri].clamp_min(1e-4)
            if os.environ.get("CAM_KV_GATE_HARD") == "1":
                cf = (conf / scale > self.mem.conf_bias).to(dtype)
            else:
                cf = torch.sigmoid(self.mem.conf_scale * (conf / scale - self.mem.conf_bias))
            cf = cf if cf.numel() == B else cf.expand(B)
        else:
            cf = torch.ones(B, device=device, dtype=dtype)
        return (g0 * cf).reshape(B)

    # --- the append (steps 1-5): key/value memory columns + extended additive mask ----------------
    def _append(self, key_states, value_states, attention_mask, query_states):
        """key/value_states: [B,n_kv,Tk,head_dim] (post-RoPE). Returns the memory-appended (k,v,mask) or
        the inputs UNCHANGED when no bank is staged (byte-identical no-op)."""
        bank = self._bank
        if bank is None:
            return key_states, value_states, attention_mask                        # EXACT no-op
        B, _, Tk, _ = key_states.shape
        K = bank.shape[1]
        wdt = self.mem.W_km.weight.dtype                                            # params fp32 (stable train)
        bank32 = bank.to(device=self.mem.W_km.weight.device, dtype=wdt)
        Km = self.mem.W_km(bank32).view(B, K, self.n_kv, self.head_dim).transpose(1, 2)   # [B,n_kv,K,hd]
        Vm = self.mem.W_vm(bank32).view(B, K, self.n_kv, self.head_dim).transpose(1, 2)   # NO RoPE (timeless)
        Km = Km.to(device=key_states.device, dtype=key_states.dtype)
        Vm = Vm.to(device=value_states.device, dtype=value_states.dtype)
        k2 = torch.cat([key_states, Km], dim=2)                                    # [B,n_kv,Tk+K,hd]
        v2 = torch.cat([value_states, Vm], dim=2)
        Tq = query_states.shape[2]
        c = self._gate(B, key_states.device, torch.float32).clamp_(0.0, 1.0)       # [B] in [0,1]
        self.last_cgate = c.mean().detach()
        self.last_gate = torch.sigmoid(self.mem.log_gate).detach()
        mb = torch.log(c).view(B, 1, 1, 1).expand(B, 1, Tq, K)                      # memory column bias; -inf when c==0
        if attention_mask is None:
            # pure-causal SDPA fast path gave no mask: build additive causal over the seq + OPEN memory cols.
            seq = torch.zeros(B, 1, Tq, Tk, dtype=torch.float32, device=key_states.device)
            off = Tk - Tq                                                          # cache offset (0 without a cache)
            col = torch.arange(Tk, device=key_states.device).view(1, 1, 1, Tk)
            row = torch.arange(Tq, device=key_states.device).view(1, 1, Tq, 1) + off
            seq = seq.masked_fill(col > row, float("-inf"))                        # causal: key j > query i => masked
            full = torch.cat([seq, mb], dim=-1)
        else:
            am = attention_mask
            if am.dtype == torch.bool:                                             # boolean mask (True=attend) -> additive
                am = torch.zeros_like(am, dtype=torch.float32).masked_fill(~am, float("-inf"))
            am = am.to(torch.float32)
            if am.shape[-1] != Tk:                                                 # guard: mask kv-dim must match seq keys
                am = am[..., :Tk]
            full = torch.cat([am, mb], dim=-1)
        full = full.to(dtype=query_states.dtype)                                   # eager/sdpa accept a float additive mask
        return k2, v2, full

    def attach(self):
        import sys as _sys
        attn = self.attn
        mod = _sys.modules[type(attn).__module__]                                  # the real modeling_qwen3_5 module
        apply_rotary = mod.apply_rotary_pos_emb
        eager = mod.eager_attention_forward
        ALL = mod.ALL_ATTENTION_FUNCTIONS
        inj = self

        def patched(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            # ---- faithful copy of Qwen3_5Attention.forward (q/k/v proj, norms, RoPE) ----
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)
            query_states, gate = torch.chunk(
                attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx)
            # ---- KV MEMORY INJECTION (bolt-on): append AFTER RoPE/cache, BEFORE the attention interface ----
            key_states, value_states, attention_mask = inj._append(
                key_states, value_states, attention_mask, query_states)
            # ---- rest of the original forward (GQA repeat_kv runs inside the interface, unchanged) ----
            attention_interface = ALL.get_interface(attn.config._attn_implementation, eager)
            attn_output, attn_weights = attention_interface(
                attn, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                scaling=attn.scaling, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)                        # Q-only gate (unchanged)
            attn_output = attn.o_proj(attn_output)
            return attn_output, attn_weights

        if "forward" in attn.__dict__:            # already patched -> restore first (idempotent attach)
            del attn.__dict__["forward"]
        attn.forward = patched                    # instance attribute shadows the class method
        return self

    def detach(self):
        try:
            del self.attn.__dict__["forward"]     # remove the instance attr -> class method restored (exact)
        except KeyError:
            pass
        return self
