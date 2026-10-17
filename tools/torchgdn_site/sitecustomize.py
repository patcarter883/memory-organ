"""sitecustomize: force transformers' Qwen3.5 gated-delta-net onto its PURE-TORCH chunked fallback.

Why: the fla fused kernels (chunk_gated_delta_rule / fused_recurrent_gated_delta_rule) and the
causal_conv1d package are the documented hazard class on gfx1201 — observed 2026-10-17 as a GPU
kernel hang mid-generation (thread wedged in memcpy_and_sync, GPU 100%, no progress). transformers
selects them at import time via `is_flash_linear_attention_available()` /
`is_causal_conv1d_available()`; patching those to False makes the modeling file import its own
`torch_chunk_gated_delta_rule` / `torch_causal_conv1d_update` implementations instead — plain
torch ops: deterministic, differentiable, no fused-kernel hazards. Slower per step; fine for the
research harness.

Point PYTHONPATH at this directory BEFORE importing transformers (it runs at interpreter start):
    PYTHONPATH=/engine/tools/torchgdn_site python -u /engine/cam/recall_mag.py ...
"""
import os

if os.environ.get("CAM_FORCE_TORCH_GDN", "1") != "0":
    try:
        import transformers.utils.import_utils as _iu

        _iu.is_flash_linear_attention_available = lambda: False
        _iu.is_causal_conv1d_available = lambda: False
    except Exception:  # transformers absent / layout changed — nothing to patch
        pass
