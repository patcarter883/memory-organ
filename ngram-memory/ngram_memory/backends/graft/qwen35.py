"""Install an NGramGraft at the INPUT of decoder index `fmt.module.layer_idx` (1) of an HF host via hooks.

Hosts: qwen2 (Qwen2.5-0.5B proxy), qwen3_5 / qwen3_5_text (Qwen3.8-27B), qwen3_5_moe / qwen3_5_moe_text
(Qwen3.6-35B-A3B). Port of P/graft/graft_proto.py (install/_capture/_inject).

* A forward-pre-hook on the top-level module captures `input_ids` and the 2-D padding mask
  (ForConditionalGeneration calls the language model with input_ids=None, q35:1644, q35moe:1754).
* A forward-pre-hook on layers[layer_idx] adds the graft output to hidden_states before the sublayer norm.
* Padding: positions masked by the recurrent mask (masking_utils.create_recurrent_attention_mask) are hashed as eos
  and their gated values zeroed (q4:1465-1470), exactly as Flash-Next does natively.

Decode state (§2.3): slot 1 = the last (K-1)*dilation positions of N_conv(u); slot 2 = the last n-1 token ids,
eos-seeded.
* qwen3_5 hosts: layer 1 is a GDN layer whose cache layer is a LinearAttentionLayer; it is extended in place from 1 to
  3 state slots (P/graft/graft_proto.py:96), so HF's reorder_cache / reset / crop / activate_past_recording act on the
  graft state natively.
* qwen2 (attention at layer 1): a SideDecodeState hangs off the Cache object; reorder_cache, reset, crop,
  batch_repeat_interleave and batch_select_indices are forwarded to it. It keeps a bounded rollback window
  (`max_rollback` tokens), so `cache.crop(-k)` works for k <= max_rollback without activate_past_recording.
"""
from __future__ import annotations

from collections import OrderedDict

from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, LinearAttentionCacheLayerMixin

from .module import NGramGraft

PLE_STATE_CONV, PLE_STATE_NGRAM = 1, 2
SUPPORTED_MODEL_TYPES = ("qwen2", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")


# ------------------------------------------------------------------------------------------------ host helpers
def decoder_layers(host: nn.Module) -> nn.ModuleList:
    m: Any = host
    for _ in range(4):
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
            return m.layers
        for name in ("model", "language_model"):
            if hasattr(m, name):
                m = getattr(m, name)
                break
        else:
            break
    raise TypeError(f"cannot find decoder layers on {type(host).__name__}")


def host_model_type(host: nn.Module) -> str:
    cfg = getattr(host, "config", None)
    return getattr(cfg, "model_type", "") if cfg is not None else ""


def recurrent_conv_mask(mask2d: torch.Tensor | None, T: int) -> torch.Tensor | None:
    """masking_utils.create_recurrent_attention_mask: None for no/all-ones mask or a 1-token step, else the last T."""
    if mask2d is None or mask2d.ndim != 2 or T == 1 or bool(torch.all(mask2d == 1)):
        return None
    return mask2d[:, -T:].contiguous()


def ensure_ple_slots(layer: LinearAttentionCacheLayerMixin, n: int = 3) -> None:
    """Give a host GDN cache layer the conv-state slots Flash-Next's PLE uses (1: conv, 2: n-gram context)."""
    if layer.number_of_states >= n:
        return
    for d, v in ((layer.conv_states, None), (layer.recurrent_states, None), (layer.is_conv_states_initialized, False),
                 (layer.is_recurrent_states_initialized, False), (layer.has_previous_state, False),
                 (layer.conv_kernel_size, None)):
        for i in range(layer.number_of_states, n):
            d.setdefault(i, v)
    layer.number_of_states = n


# ------------------------------------------------------------------------------------------------- side state
class SideDecodeState:
    """get_conv_state / update_conv_state view (the subset q4's PLE layer calls) for hosts whose layer-1 cache
    has no conv slots. Keeps per slot a window of the last `kernel + max_rollback` entries, left-padded at the
    first update with the slot's pad value (0 for the conv slot, eos for the n-gram slot)."""

    def __init__(self, pad_values: dict[int, float | int], max_rollback: int = 16):
        self.pad_values = dict(pad_values)
        self.max_rollback = int(max_rollback)
        self.reset()

    def reset(self) -> None:
        self.store: dict[int, torch.Tensor] = {}
        self.kernel: dict[int, int] = {}
        self.n_tokens: dict[int, int] = {}

    # q4 protocol ------------------------------------------------------------------------------------------
    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
        s = self.store.get(state_idx)
        return None if s is None else s[..., -self.kernel[state_idx]:]

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0,
                          conv_kernel_size: int | None = None, **kw) -> torch.Tensor:
        x = conv_states
        s = self.store.get(state_idx)
        if s is None:
            ks = int(conv_kernel_size if conv_kernel_size is not None else x.shape[-1])
            self.kernel[state_idx] = ks
            # what HF's LinearAttentionLayer returns on prefill: x, zero-padded on the left to ks
            full = x if x.shape[-1] >= ks else torch.nn.functional.pad(x, (ks - x.shape[-1], 0), value=0)
            pad = ks + self.max_rollback
            hist = x if x.shape[-1] >= pad else torch.nn.functional.pad(
                x, (pad - x.shape[-1], 0), value=self.pad_values.get(state_idx, 0))
            self.n_tokens[state_idx] = x.shape[-1]
        else:
            ks = self.kernel[state_idx]
            full = torch.cat([s[..., -ks:], x], dim=-1)
            hist = torch.cat([s, x], dim=-1)
            self.n_tokens[state_idx] += x.shape[-1]
        self.store[state_idx] = hist[..., -(ks + self.max_rollback):].contiguous()
        return full

    # cache-wide ops ---------------------------------------------------------------------------------------
    def crop(self, tokens_to_remove: int) -> None:
        """HF convention: a negative int removes that many trailing tokens; 0 is a no-op."""
        if tokens_to_remove > 0:
            raise ValueError("crop takes a negative number of tokens to remove")
        k = -int(tokens_to_remove)
        if k == 0:
            return
        for i, s in list(self.store.items()):
            if k > self.n_tokens[i]:
                raise RuntimeError(f"cannot crop {k} tokens: only {self.n_tokens[i]} seen")
            if k > s.shape[-1] - self.kernel[i]:
                raise RuntimeError(f"cannot crop {k} tokens: rollback window is {self.max_rollback}")
            if k == self.n_tokens[i]:  # back to an empty sequence
                del self.store[i], self.kernel[i], self.n_tokens[i]
                continue
            self.store[i] = s[..., :-k].contiguous()
            self.n_tokens[i] -= k

    def _apply(self, fn) -> None:
        for i, t in self.store.items():
            self.store[i] = fn(t)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self._apply(lambda t: t.index_select(0, beam_idx.to(t.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._apply(lambda t: t.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._apply(lambda t: t[indices, ...])


class PrefixContext:
    """Cache-free state for one uncached forward whose n-gram history starts before the call: slot 2 reads the
    true preceding n-1 tokens `context` [B, n-1] (eos-seeded at a stream start); slot 1 (conv) has no history, as
    in a plain uncached forward. Used for training windows cut from a longer stream (§I.4 step 6)."""

    def __init__(self, context: torch.Tensor):
        self.context = context.long()

    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
        return self.context if state_idx == PLE_STATE_NGRAM else None

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0,
                          conv_kernel_size: int | None = None, **kw) -> torch.Tensor:
        return conv_states


def attach_side_state(cache: Cache, key: str, pad_values: dict[int, float | int],
                      max_rollback: int = 16) -> SideDecodeState:
    side = cache.__dict__.get(key)
    if side is not None:
        return side
    side = SideDecodeState(pad_values, max_rollback)
    cache.__dict__[key] = side
    orig = {n: getattr(cache, n) for n in ("reorder_cache", "reset", "crop", "batch_repeat_interleave",
                                           "batch_select_indices")}

    def reorder_cache(beam_idx):
        orig["reorder_cache"](beam_idx)
        side.reorder_cache(beam_idx)

    def reset():
        orig["reset"]()
        side.reset()

    def crop(tokens_to_remove):
        orig["crop"](tokens_to_remove)
        side.crop(tokens_to_remove)

    def batch_repeat_interleave(repeats):
        orig["batch_repeat_interleave"](repeats)
        side.batch_repeat_interleave(repeats)

    def batch_select_indices(indices):
        orig["batch_select_indices"](indices)
        side.batch_select_indices(indices)

    for n, f in (("reorder_cache", reorder_cache), ("reset", reset), ("crop", crop),
                 ("batch_repeat_interleave", batch_repeat_interleave), ("batch_select_indices", batch_select_indices)):
        setattr(cache, n, f)
    return side


def graft_state_for(cache: Cache | None, layer_idx: int, side_key: str, eos: int, max_rollback: int = 16):
    """The object the PLE layer reads/writes its decode state through (or None without a cache)."""
    if cache is None:
        return None
    layer = cache.layers[layer_idx] if layer_idx < len(cache.layers) else None
    if isinstance(layer, LinearAttentionCacheLayerMixin):
        ensure_ple_slots(layer)
        return cache
    return attach_side_state(cache, side_key, {PLE_STATE_CONV: 0.0, PLE_STATE_NGRAM: eos}, max_rollback)


# -------------------------------------------------------------------------------------------------- install
class InstalledGraft:
    """Live hooks of one graft on one host. `remove()` restores the host exactly (no-op afterwards)."""

    def __init__(self, host: nn.Module, graft: NGramGraft, layer_idx: int, max_rollback: int, record: bool):
        self.host, self.graft, self.layer_idx = host, graft, layer_idx
        self.max_rollback, self.record = max_rollback, record
        self.side_key = f"_ngram_graft_side_{id(self)}"
        self._ids: torch.Tensor | None = None
        self._mask2d: torch.Tensor | None = None
        self.last: dict = {}
        # true n-gram history [B, n-1] for the next UNCACHED calls (training windows); None = fresh segment
        self.context: torch.Tensor | None = None
        # forward-time (ids, mask, context) per layer-input storage, for gradient-checkpoint recomputes
        self._bound: "OrderedDict[tuple, tuple]" = OrderedDict()
        layers = decoder_layers(host)
        self._handles = [
            host.register_forward_pre_hook(self._capture, with_kwargs=True),
            layers[layer_idx].register_forward_pre_hook(self._inject, with_kwargs=True),
        ]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._bound.clear()

    def __enter__(self) -> "InstalledGraft":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def _capture(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise NotImplementedError("the graft needs input_ids (inputs_embeds-only calls are not supported)")
        m = kwargs.get("attention_mask")
        self._ids = ids
        self._mask2d = m if isinstance(m, torch.Tensor) and m.ndim == 2 else None

    @staticmethod
    def _bind_key(hidden: torch.Tensor) -> tuple:
        # the layer input is kept alive by the checkpoint (saved by reference, or detached sharing storage), so
        # its storage address identifies the forward call during the backward recompute
        return (hidden.data_ptr(), tuple(hidden.shape), tuple(hidden.stride()))

    def _inject(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        T = hidden.shape[1]
        if torch._C._current_graph_task_id() != -1 and torch.is_grad_enabled():
            # gradient-checkpoint RECOMPUTE inside backward: use the ids/mask/context bound at forward time, not
            # those of the most recent top-level call (a later forward may have replaced them)
            bound = self._bound.get(self._bind_key(hidden))
            if bound is None:
                raise RuntimeError("graft recompute: no forward-time binding for this layer input")
            ids, mask2d, context = bound
        else:
            ids, mask2d, context = self._ids, self._mask2d, self.context
            self._bound[self._bind_key(hidden)] = (ids, mask2d, context)
            while len(self._bound) > 256:
                self._bound.popitem(last=False)
        if ids is None or ids.shape[1] != T:
            raise RuntimeError(f"captured ids {None if ids is None else tuple(ids.shape)} do not match hidden "
                               f"{tuple(hidden.shape)}")
        conv_mask = recurrent_conv_mask(mask2d, T)
        ple_ids = torch.where(conv_mask.bool(), ids, self.graft.eos) if conv_mask is not None else ids
        cache = kwargs.get("past_key_values")
        if context is not None:
            if cache is not None:
                raise RuntimeError("InstalledGraft.context is for uncached calls; got past_key_values")
            ctx = context.to(ids.device)
            if ctx.shape != (ids.shape[0], self.graft.key.context_len):
                raise ValueError(f"context {tuple(ctx.shape)} != {(ids.shape[0], self.graft.key.context_len)}")
            state = PrefixContext(ctx)
        else:
            state = graft_state_for(cache, self.layer_idx, self.side_key, self.graft.eos, self.max_rollback)
        out = self.graft(hidden, ple_ids, state, conv_mask=conv_mask)
        if self.record:
            self.last = dict(out=out, conv_mask=conv_mask, layer_mask=kwargs.get("attention_mask"), ple_ids=ple_ids)
        hidden = hidden + out.to(hidden.dtype)
        if args:
            args = (hidden,) + tuple(args[1:])
        else:
            kwargs["hidden_states"] = hidden
        return args, kwargs


def install_graft(host: nn.Module, graft: NGramGraft, *, layer_idx: int | None = None, max_rollback: int = 16,
                  record: bool = True, check_model_type: bool = True) -> InstalledGraft:
    """Hook `graft` into `host`; moves the graft to the host's device. Returns the handle (call .remove())."""
    mt = host_model_type(host)
    if check_model_type and mt not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported host model_type {mt!r}; supported: {SUPPORTED_MODEL_TYPES}")
    d = getattr(getattr(host, "config", None), "hidden_size", None)
    tc = getattr(getattr(host, "config", None), "text_config", None)
    d = getattr(tc, "hidden_size", d) if d is None else d
    if d is not None and d != graft.fmt.module.hidden_size:
        raise ValueError(f"host hidden_size {d} != graft hidden_size {graft.fmt.module.hidden_size}")
    graft.to(next(host.parameters()).device)
    return InstalledGraft(host, graft, graft.layer_idx if layer_idx is None else layer_idx, max_rollback, record)


__all__ = ["install_graft", "InstalledGraft", "SideDecodeState", "PrefixContext", "attach_side_state", "graft_state_for",
           "ensure_ple_slots", "recurrent_conv_mask", "decoder_layers", "host_model_type", "PLE_STATE_CONV",
           "PLE_STATE_NGRAM", "SUPPORTED_MODEL_TYPES"]
