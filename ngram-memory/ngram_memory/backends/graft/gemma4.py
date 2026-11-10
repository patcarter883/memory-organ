"""Install an NGramGraft on gemma4 hosts (gemma-4-26B-A4B-it, gemma-4-31B-it; DESIGN.md §4.4).

The Flash-Next format with a Gemma key instance (V = 262144, eos = 1 `<eos>`; the reset token could instead be
106 `<end_of_turn>`, still to confirm). Row width d/16: 176 (26B), 336 (31B).

Gemma-specific handling, all through the generic hooks of backends/graft/qwen35.py:
* `Gemma4Model` passes a dict mask to its layers (g4:2422), so the 2-D padding mask is captured at the top level
  (the host's forward-pre-hook), exactly as for qwen3_5 hosts.
* Layer 1 is a sliding-window KV layer, so the decode state is a SideDecodeState hung off the Cache; reorder_cache,
  reset, crop and batch_* are forwarded to it (crop within `max_rollback` tokens).
* `layer_scalar` (g4:1429) and `embed_scale = sqrt(d)` (g4:1565) are host properties the solver absorbs; in the 26B
  the router reads the injected residual, so edits also move routing (not handled here, by design).
"""
from __future__ import annotations

import torch.nn as nn

from ...spec import KeySpec, graft_key_spec
from .module import NGramGraft
from .qwen35 import InstalledGraft, host_model_type, install_graft

GEMMA4_MODEL_TYPES = ("gemma4", "gemma4_text")
GEMMA4_VOCAB = 262144
GEMMA4_EOS = 1
GEMMA4_END_OF_TURN = 106


def gemma4_text_config(host: nn.Module):
    cfg = host.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def gemma_graft_key_spec(vocab_size: int = GEMMA4_VOCAB, reset_token: int = GEMMA4_EOS,
                         orders=(2, 3), tokenizer_sha256: str | None = None) -> KeySpec:
    """The Gemma key instance: different V (and so multipliers) from Flash-Next; base 20M primes as Flash-Next."""
    return graft_key_spec(vocab_size, reset_token, orders=orders, tokenizer_sha256=tokenizer_sha256)


def install_graft_gemma4(host: nn.Module, graft: NGramGraft, *, layer_idx: int | None = None,
                         max_rollback: int = 16, record: bool = True) -> InstalledGraft:
    mt = host_model_type(host)
    if mt not in GEMMA4_MODEL_TYPES:
        raise ValueError(f"not a gemma4 host: model_type {mt!r}")
    t = gemma4_text_config(host)
    if t.hidden_size != graft.fmt.module.hidden_size:
        raise ValueError(f"host hidden_size {t.hidden_size} != graft {graft.fmt.module.hidden_size}")
    if graft.key.vocab_size < t.vocab_size:
        raise ValueError(f"graft key vocab {graft.key.vocab_size} < host vocab {t.vocab_size}")
    return install_graft(host, graft, layer_idx=layer_idx, max_rollback=max_rollback, record=record,
                         check_model_type=False)


__all__ = ["install_graft_gemma4", "gemma_graft_key_spec", "GEMMA4_MODEL_TYPES", "GEMMA4_VOCAB", "GEMMA4_EOS",
           "GEMMA4_END_OF_TURN"]
