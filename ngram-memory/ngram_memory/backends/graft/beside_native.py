"""A graft installed next to Flash-Next's native PLE (the native-vs-graft arm of §6.A.6 / M6b "beside-graft G").

Placement: the graft reads exactly what the native PLE reads and adds to the same place.
* A forward hook on `layers[layer_idx].ple` (the native module) receives the native call's own arguments
  `(hidden_states [B, T, hc*d], ple_input_ids, past_key_values, conv_mask=...)` -- ids already eos-substituted under
  the recurrent mask by Qwen4ExpTextModel.forward (q4:1465-1470) -- and returns `native_out + graft_out` broadcast
  into ALL hc streams ("X_t += u_t + c_t into all streams", §2.2). The native query sees the un-grafted residual, so
  the native gate is unaffected by the graft.
* The graft is hc_count=1 (§2.4). Its query is the mean of the hc streams; with the editor init (W_k = 0) the gate is
  0.5 whatever the query, so the choice only matters for calibrated-gate arms.
* Decode state: the native PLE owns slots 1 and 2 of the layer-1 GDN cache layer, so the graft keeps its own
  SideDecodeState hung off the Cache (never the shared slots); crop/reorder/reset/batch_* are forwarded.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .module import NGramGraft
from .qwen35 import PLE_STATE_CONV, PLE_STATE_NGRAM, PrefixContext, attach_side_state, decoder_layers, host_model_type

QWEN4EXP_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")


class BesideNativeGraft:
    def __init__(self, host: nn.Module, graft: NGramGraft, layer_idx: int | None = None, max_rollback: int = 16,
                 record: bool = True):
        self.host, self.graft = host, graft
        self.layer_idx = graft.layer_idx if layer_idx is None else layer_idx
        self.max_rollback, self.record = max_rollback, record
        self.side_key = f"_ngram_beside_side_{id(self)}"
        self.last: dict = {}
        self.context: torch.Tensor | None = None  # true n-gram history for uncached calls (as InstalledGraft)
        ple = getattr(decoder_layers(host)[self.layer_idx], "ple", None)
        if ple is None:
            raise ValueError(f"decoder layer {self.layer_idx} has no native PLE to sit beside")
        self.hc = int(ple.hc_count)
        self.d = int(ple.hidden_size)
        if self.d != graft.fmt.module.hidden_size:
            raise ValueError(f"host hidden_size {self.d} != graft {graft.fmt.module.hidden_size}")
        self.native = ple
        self._handle = ple.register_forward_hook(self._add, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "BesideNativeGraft":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def _add(self, module, args, kwargs, native_out):
        hidden = args[0] if args else kwargs["hidden_states"]
        ids = args[1] if len(args) > 1 else kwargs["input_ids"]
        cache = args[2] if len(args) > 2 else kwargs.get("past_key_values")
        conv_mask = kwargs.get("conv_mask", args[3] if len(args) > 3 else None)
        B, T, _ = hidden.shape
        query = hidden.view(B, T, self.hc, self.d).mean(-2)
        if self.context is not None:
            if cache is not None:
                raise RuntimeError("context is for uncached calls")
            state = PrefixContext(self.context.to(ids.device))
        elif cache is not None:
            state = attach_side_state(cache, self.side_key, {PLE_STATE_CONV: 0.0, PLE_STATE_NGRAM: self.graft.eos},
                                      self.max_rollback)
        else:
            state = None
        out = self.graft(query, ids, state, conv_mask=conv_mask)
        if self.record:
            self.last = dict(out=out, native_out=native_out, conv_mask=conv_mask, ple_ids=ids)
        return native_out + out.to(native_out.dtype).repeat(1, 1, self.hc)


def install_beside_native(host: nn.Module, graft: NGramGraft, *, layer_idx: int | None = None,
                          max_rollback: int = 16, record: bool = True) -> BesideNativeGraft:
    mt = host_model_type(host)
    tc = getattr(host.config, "text_config", None)
    mt = getattr(tc, "model_type", mt) if tc is not None else mt
    if mt not in QWEN4EXP_MODEL_TYPES:
        raise ValueError(f"beside-native grafts need a qwen4_exp host, got {mt!r}")
    graft.to(next(host.parameters()).device)
    return BesideNativeGraft(host, graft, layer_idx, max_rollback, record)


__all__ = ["BesideNativeGraft", "install_beside_native", "QWEN4EXP_MODEL_TYPES"]
