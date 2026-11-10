"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Verifier test for the gdnkit piece, defect VGK-3 (plausible: the gdn_hip kernel source is not in the clone).

The fork always gives gdn_hip.gdn_prefill fp32 A_log / dt_bias. Its NativeGDNShim copies them with
``hf.A_log.float()`` (minisgl/gdn/hf_patch.py:49-50), its layer keeps them fp32 (minisgl/gdn/layer.py), and
every fork test passes fp32 (tools/vhip_patches/gdn/tests/test_gdn.py:55). Our ``_native_call`` passes the HF
module's parameters as they are. Those are bf16 when the layer is loaded in bf16, as
``host_quant.build_decoder_layer`` does for the M2-0 run, so the native op gets bf16 gate parameters.
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_host
from ngram_memory.backends.graft import gdn


def test_verify_gdnkit_native_op_gets_fp32_gate_params():
    host = build_host("qwen3_5_moe").to(torch.bfloat16)
    layer = gdn.gdn_modules(host)[0][1]
    seen = {}

    def fake_op(q, k, v, a, b, A_log, dt_bias, cu, state_idx, has_init, ssm_state, scale, use_l2norm):
        seen["A_log"], seen["dt_bias"] = A_log.dtype, dt_bias.dtype
        return torch.zeros(v.shape, dtype=torch.float32)

    gdn.patch_gdn(host, impl="native", native_op=fake_op)
    try:
        with torch.no_grad():
            layer(torch.randn(1, 8, host.config.hidden_size, dtype=torch.bfloat16))
    finally:
        gdn.unpatch_gdn(host)
    assert seen == {"A_log": torch.float32, "dt_bias": torch.float32}, seen
