"""Frozen 4-bit host layers (ngram_memory.backends.graft.host_quant): dequant bit-exact against an independent
reference and against the fork's code, dX-only linear and MoE experts vs autograd through HF, safetensors
readers, and a whole decoder layer built from a synthetic packed checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import fork_refs
import hq_reference as ref
from ngram_memory.backends.graft import host_quant as hq

HF = __import__("ngram_memory.paths", fromlist=["hf_hub_cache"]).hf_hub_cache()
CFG35 = HF / "models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/f1e8415c4f29508fe281da4e256df26573213bfd/config.json"
CFG27 = HF / "models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/93245f7c00f1a8964b66a97c6e02e2fd09080be4/config.json"


def _rand_awq(K, N, g, seed=0):
    r = np.random.default_rng(seed)
    q = r.integers(0, 16, (K, N))
    z = r.integers(0, 16, (K // g, N))
    s = (r.standard_normal((K // g, N)) * 0.01).astype(np.float16)
    return ref.awq_pack_ref(q), ref.awq_pack_ref(z), s


def _rand_mx(N, K, seed=0, extremes=False):
    r = np.random.default_rng(seed)
    w = r.integers(0, 256, (N, K // 2), dtype=np.uint8)
    s = r.integers(110, 130, (N, K // 32), dtype=np.uint8)
    if extremes:
        s[0, 0], s[0, 1] = 0, 254                  # subnormal 2^-127 and 2^127
    return w, s


@pytest.mark.parametrize("g", [32, 128])
def test_awq_dequant_bit_exact_vs_reference(g):
    qw, qz, s = _rand_awq(256, 96, g)
    want = ref.awq_dequant_ref(qw, qz, s, g)
    got32 = hq.awq_dequant(torch.from_numpy(qw), torch.from_numpy(qz), torch.from_numpy(s), g, torch.float32)
    assert torch.equal(got32, torch.from_numpy(want))
    got16 = hq.awq_dequant(torch.from_numpy(qw), torch.from_numpy(qz), torch.from_numpy(s), g)
    assert torch.equal(got16, torch.from_numpy(want).to(torch.bfloat16))


def test_awq_pack_unpack_roundtrip_and_reverse_order():
    q = np.random.default_rng(1).integers(0, 16, (8, 64))
    assert np.array_equal(hq.awq_unpack(torch.from_numpy(ref.awq_pack_ref(q))).numpy(), q)
    assert [hq.AWQ_ORDER[hq.AWQ_REVERSE[j]] for j in range(8)] == list(range(8))


def test_mxfp4_dequant_bit_exact_vs_reference():
    w, s = _rand_mx(64, 256, extremes=True)
    want = ref.mxfp4_dequant_ref(w, s)
    got = hq.mxfp4_dequant(torch.from_numpy(w), torch.from_numpy(s), torch.float32)
    assert torch.equal(got, torch.from_numpy(want))
    assert torch.equal(hq.mxfp4_dequant(torch.from_numpy(w), torch.from_numpy(s)), torch.from_numpy(want).to(torch.bfloat16))
    assert torch.isnan(hq.e8m0_to_fp32(torch.tensor([255], dtype=torch.uint8))).all()
    assert hq.e8m0_to_fp32(torch.tensor([0, 127, 254], dtype=torch.uint8)).tolist() == [2.0 ** -127, 1.0, 2.0 ** 127]


def test_fork_dequant_code_agrees_bit_exact():
    f_mx = fork_refs.fork_mxfp4_dequant()
    if f_mx is None or fork_refs.fork_awq_to_op_layout() is None:
        pytest.skip("minisgl clone absent")
    w, s = _rand_mx(32, 128, seed=3)
    assert torch.equal(f_mx(torch.from_numpy(w), torch.from_numpy(s)),
                       hq.mxfp4_dequant(torch.from_numpy(w), torch.from_numpy(s), torch.float32))
    qw, qz, sc = _rand_awq(128, 64, 32, seed=4)
    fk = fork_refs.fork_awq_dequant(torch.from_numpy(qw), torch.from_numpy(qz), torch.from_numpy(sc), 32)
    assert torch.equal(fk, hq.awq_dequant(torch.from_numpy(qw), torch.from_numpy(qz), torch.from_numpy(sc), 32,
                                          torch.float32))


@pytest.mark.parametrize("fmt", ["awq", "mxfp4"])
def test_quant_linear_forward_and_dx_only(fmt):
    K, N = 128, 64
    if fmt == "awq":
        qw, qz, s = _rand_awq(K, N, 32, seed=5)
        lin = hq.QuantLinear(K, N, "awq", {"qweight": torch.from_numpy(qw), "qzeros": torch.from_numpy(qz),
                                           "scales": torch.from_numpy(s)})
    else:
        w, s = _rand_mx(N, K, seed=5)
        lin = hq.QuantLinear(K, N, "mxfp4", {"weight": torch.from_numpy(w), "weight_scale": torch.from_numpy(s)})
    assert list(lin.parameters()) == []
    W = lin.dequant()
    lin2 = lin.to(torch.float32)                  # a dtype cast must not touch the packed / scale bits
    assert torch.equal(lin2.dequant(), W)
    x = torch.randn(3, 5, K, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    assert torch.equal(y, F.linear(x.detach(), W))
    dy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, dy)
    assert torch.equal(dx, dy @ W)


def test_fake_quant_fp8_straight_through():
    lin = hq.QuantLinear(64, 32, "mxfp4", {"weight": torch.from_numpy(_rand_mx(32, 64)[0]),
                                            "weight_scale": torch.full((32, 2), 120, dtype=torch.uint8)},
                         act_quant="fp8_e4m3")
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    assert torch.equal(y, F.linear(hq.fake_quant_fp8(x.detach()), lin.dequant()))
    (dx,) = torch.autograd.grad(y.sum(), x)
    assert torch.equal(dx, torch.ones_like(y) @ lin.dequant())


def _experts(E=6, d=64, I=32, g=32, seed=0):
    names = {}
    for e in range(E):
        for p, (K, N) in (("gate_proj", (d, I)), ("up_proj", (d, I)), ("down_proj", (I, d))):
            qw, qz, s = _rand_awq(K, N, g, seed=seed * 1000 + e * 10 + len(p))
            names[f"x.{e}.{p}.qweight"] = torch.from_numpy(qw)
            names[f"x.{e}.{p}.qzeros"] = torch.from_numpy(qz)
            names[f"x.{e}.{p}.scales"] = torch.from_numpy(s)
    return names


def test_moe_experts_match_hf_experts_and_grads():
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
    E, d, I = 6, 64, 32
    t = _experts(E, d, I)
    sch = hq.QuantScheme("awq", 32)
    mod = hq.load_awq_experts(None, "x", E, sch, hidden=d, intermediate=I, compute_dtype=torch.float32,
                              experts_per_group=4, tensors=t)
    hf = Qwen3_5MoeExperts(Qwen3_5MoeTextConfig(num_experts=E, hidden_size=d, moe_intermediate_size=I))
    gu, dn = mod.dequant_experts(torch.arange(E), torch.float32)
    with torch.no_grad():
        hf.gate_up_proj.copy_(gu)
        hf.down_proj.copy_(dn)
    for e in range(E):   # gate rows first, then up (HF chunk(2) order)
        want_gate = ref.awq_dequant_ref(t[f"x.{e}.gate_proj.qweight"].numpy(), t[f"x.{e}.gate_proj.qzeros"].numpy(),
                                        t[f"x.{e}.gate_proj.scales"].numpy(), 32)
        assert torch.equal(gu[e, :I], torch.from_numpy(want_gate))
    n, k = 40, 2
    x = torch.randn(n, d, requires_grad=True)
    idx = torch.randint(0, E, (n, k))
    idx[:, 1] = (idx[:, 0] + 1) % E
    w = torch.rand(n, k, requires_grad=True)
    y1 = mod(x, idx, w)
    y2 = hf(x, idx, w)
    assert ((y1 - y2).norm() / y2.norm()).item() < 1e-6
    dy = torch.randn_like(y1)
    a = torch.autograd.grad(y1, (x, w), dy)
    b = torch.autograd.grad(y2, (x, w), dy)
    for u, v in zip(a, b):
        assert ((u - v).norm() / v.norm()).item() < 1e-6
    assert list(mod.parameters()) == []


def test_safetensors_reader_local(tmp_path):
    from safetensors.torch import save_file
    ts = {"a": torch.arange(10, dtype=torch.int32), "b": torch.randn(3, 4).to(torch.bfloat16),
          "c": torch.randint(0, 255, (7,), dtype=torch.uint8), "d": torch.randn(2, 2, dtype=torch.float16)}
    save_file(ts, str(tmp_path / "model.safetensors"))
    r = hq.CheckpointReader.local(tmp_path)
    for k, v in ts.items():
        assert torch.equal(r.get(k), v)
    many = r.get_many(["d", "a", "c"], max_gap=0)
    assert all(torch.equal(many[k], ts[k]) for k in many)
    many = r.get_many(list(ts), piece=7)          # tiny pieces exercise the stitching
    assert all(torch.equal(many[k], ts[k]) for k in ts)


@pytest.mark.skipif(not (CFG35.exists() and CFG27.exists()), reason="cached HF configs absent")
def test_scheme_from_real_configs():
    s35 = hq.scheme_from_config(json.loads(CFG35.read_text()))
    s27 = hq.scheme_from_config(json.loads(CFG27.read_text()))
    assert (s35.method, s35.group_size) == ("awq", 32)
    assert (s27.method, s27.group_size) == ("mxfp4", 32)


def _tiny_cfg(kind):
    from conftest import _text_cfg
    t = _text_cfg("qwen3.6-35b-a3b.config.json" if kind == "moe" else "qwen3.8-27b.config.json")
    keep = {k: t[k] for k in ("attn_output_gate", "hidden_act", "rms_norm_eps", "full_attention_interval",
                              "partial_rotary_factor", "linear_conv_kernel_dim") if k in t}
    rope = dict(t["rope_parameters"], mrope_section=[2, 1, 1])
    tc = dict(keep, vocab_size=128, hidden_size=64, num_hidden_layers=4, layer_types=t["layer_types"][:4],
              num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=16,
              linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, rope_parameters=rope)
    if kind == "moe":
        tc.update(model_type="qwen3_5_moe_text", num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                  shared_expert_intermediate_size=32)
        return {"text_config": tc, "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 32,
                                                           "zero_point": True, "version": "gemm"}}
    tc.update(model_type="qwen3_5_text", intermediate_size=64)
    return {"text_config": tc, "quantization_config": json.loads(CFG27.read_text())["quantization_config"]
            if CFG27.exists() else None}


@pytest.mark.parametrize("kind,layer_idx", [("moe", 0), ("moe", 3), ("dense", 0), ("dense", 3)])
def test_build_decoder_layer_from_packed_checkpoint(tmp_path, kind, layer_idx):
    """Pack a random HF layer into the host's on-disk format with the independent quantisers, load it back
    with build_decoder_layer, and compare against the HF layer holding the dequantised weights."""
    if kind == "dense" and not CFG27.exists():
        pytest.skip("Quark config absent")
    from safetensors.torch import save_file
    cfg = _tiny_cfg(kind)
    tcfg, Layer = hq._text_config(cfg)
    torch.manual_seed(layer_idx)
    dense = Layer(tcfg, layer_idx).float()
    for p in dense.parameters():
        torch.nn.init.normal_(p, std=0.05)
    base = f"model.language_model.layers.{layer_idx}"
    sd, quant_mods = {}, set()
    for name, m in dense.named_modules():
        if isinstance(m, torch.nn.Linear) and (kind == "dense" and "mlp.gate" != name):
            pw, ps = ref.mxfp4_quantize_ref(m.weight.detach().numpy())
            sd[f"{base}.{name}.weight"], sd[f"{base}.{name}.weight_scale"] = torch.from_numpy(pw), torch.from_numpy(ps)
            with torch.no_grad():
                m.weight.copy_(torch.from_numpy(ref.mxfp4_dequant_ref(pw, ps)))
            quant_mods.add(name)
    if kind == "moe":
        ex = dense.mlp.experts
        I = ex.intermediate_dim
        for e in range(ex.num_experts):
            for p, W in (("gate_proj", ex.gate_up_proj[e, :I]), ("up_proj", ex.gate_up_proj[e, I:]),
                         ("down_proj", ex.down_proj[e])):
                qw, qz, s = ref.awq_quantize_ref(W.detach().numpy(), 32)
                sd[f"{base}.mlp.experts.{e}.{p}.qweight"] = torch.from_numpy(qw)
                sd[f"{base}.mlp.experts.{e}.{p}.qzeros"] = torch.from_numpy(qz)
                sd[f"{base}.mlp.experts.{e}.{p}.scales"] = torch.from_numpy(s)
                with torch.no_grad():
                    W.copy_(torch.from_numpy(ref.awq_dequant_ref(qw, qz, s, 32)))
    for n, p in dense.named_parameters():
        if n.startswith("mlp.experts.") or any(n == f"{q}.weight" for q in quant_mods):
            continue
        sd[f"{base}.{n}"] = p.detach().clone()
    save_file(sd, str(tmp_path / "model.safetensors"))
    layer, _ = hq.build_decoder_layer(cfg, layer_idx, hq.CheckpointReader.local(tmp_path),
                                      compute_dtype=torch.float32, experts_per_group=3)
    assert not any(p.requires_grad for p in layer.parameters())
    nq = sum(isinstance(m, (hq.QuantLinear, hq.QuantMoEExperts)) for m in layer.modules())
    assert nq == (1 if kind == "moe" else len(quant_mods))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    T = 24
    x = torch.randn(2, T, 64)
    pos = torch.arange(T)[None].expand(2, T)
    rot = Qwen3_5TextRotaryEmbedding(tcfg)
    pe = rot(x, pos[None].expand(3, 2, T))

    def run(L):
        xx = x.clone().requires_grad_(True)
        y = L(xx, position_embeddings=pe, position_ids=pos)
        (dx,) = torch.autograd.grad(y.pow(2).sum(), xx)
        return y.detach(), dx
    y1, d1 = run(layer)
    y0, d0 = run(dense.eval())
    assert ((y1 - y0).norm() / y0.norm()).item() < 1e-5
    assert F.cosine_similarity(d1.flatten(), d0.flatten(), dim=0).item() > 0.99999


def test_grouped_mm_probe_reports():
    r = hq.grouped_mm_probe("cpu")
    assert "available" in r
