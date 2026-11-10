"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Verifier test for the gdnkit piece, defect VGK-1.

A plain training step ``host(ids, labels=ids)`` on a patched Qwen3.5 / Qwen3.5-MoE host never reaches our GDN
kernel. ``config.use_cache`` defaults to True, so HF builds a fresh DynamicCache for the call, and
``_patched_forward`` hands every call with ``cache_params is not None`` to HF's own forward. That happens even
though the cache holds no previous state, so the initial state is zero. The brief asked for the fallback only
on cached calls with a non-zero initial state.
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_host
from ngram_memory.backends.graft import gdn


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_verify_gdnkit_training_step_uses_our_kernel(kind):
    host = build_host(kind).float().train()
    assert host.config.use_cache is True
    cfg = gdn.patch_gdn(host, impl="ours")
    try:
        ids = torch.randint(3, 500, (2, 40), generator=torch.Generator().manual_seed(0))
        host(ids, labels=ids).loss.backward()
        assert cfg.calls == len(gdn.gdn_modules(host)) and cfg.fallbacks == 0, (cfg.calls, cfg.fallbacks)
    finally:
        gdn.unpatch_gdn(host)


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_verify_gdnkit_fresh_cache_prefill_fills_cache_like_hf(kind):
    """Our kernel on a fresh-cache prefill must leave the cache exactly usable for decode (conv + recurrent state)."""
    host = build_host(kind).float().eval()
    ids = torch.randint(3, 500, (2, 37), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = host.generate(ids, max_new_tokens=6, do_sample=False, output_logits=True, return_dict_in_generate=True)
        cfg = gdn.patch_gdn(host, impl="ours")
        try:
            got = host.generate(ids, max_new_tokens=6, do_sample=False, output_logits=True, return_dict_in_generate=True)
        finally:
            gdn.unpatch_gdn(host)
    assert cfg.calls == len(gdn.gdn_modules(host))          # the prefill went through our kernel
    assert torch.equal(ref.sequences, got.sequences)
    for a, b in zip(ref.logits, got.logits):
        assert (a - b).abs().max().item() < 1e-4
