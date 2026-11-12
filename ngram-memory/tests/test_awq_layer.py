"""M2-0 AWQ host layers (ngram_memory.backends.graft.awq35) on small synthetic checkpoints in the real 35B
on-disk format (per-expert ``mlp.experts.{e}.{gate,up,down}_proj.{qweight,qzeros,scales}`` packed with the
independent reference packer, F16 dense tensors, AWQ smoothing folded into post_attention_layernorm and the
expert / router / shared-expert input columns, two shards + index): the per-layer cache writer (exact bytes,
sha256, idempotent, byte caps, reuse), single-checkpoint provenance, fp32 A_log / dt_bias, memory accounting,
and whole decoder layers (GDN + MoE, attention + MoE) forward and dX against an fp64 HF reference built from
the numpy dequant. No network. ``NGM_AWQ35_CACHE=<cache>/cyankiwi_35b`` opts in to a real-weights check on a
layer cached by experiments/m2_0/fetch_layer_cache.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import fork_refs
import hq_reference as ref
from ngram_memory.backends.graft import awq35
from ngram_memory.backends.graft import host_quant as hq

LP = awq35.LP
G = 32


def _cfg():
    from conftest import _text_cfg
    t = _text_cfg("qwen3.6-35b-a3b.config.json")
    keep = {k: t[k] for k in ("attn_output_gate", "hidden_act", "rms_norm_eps", "full_attention_interval",
                              "partial_rotary_factor", "linear_conv_kernel_dim") if k in t}
    rope = dict(t["rope_parameters"], mrope_section=[2, 1, 1])
    tc = dict(keep, model_type="qwen3_5_moe_text", vocab_size=128, hidden_size=64, num_hidden_layers=4,
              layer_types=t["layer_types"][:4], num_attention_heads=4, num_key_value_heads=2, head_dim=32,
              linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
              rope_parameters=rope, num_experts=6, num_experts_per_tok=2, moe_intermediate_size=32,
              shared_expert_intermediate_size=32)
    assert tc["layer_types"][0] == "linear_attention" and tc["layer_types"][3] == "full_attention"
    return {"text_config": tc, "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": G,
                                                       "zero_point": True, "version": "gemm"}}


def _save_sharded(d: Path, shards: dict[str, dict[str, torch.Tensor]]):
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    wm = {}
    for fn, sd in shards.items():
        save_file(sd, str(d / fn))
        wm.update({k: fn for k in sd})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))


def _make_ckpts(root: Path, layers=(0, 3), seed=0):
    """Two functionally equivalent checkpoints of the same random layers: 'orig' (dense tensors unsmoothed)
    and 'awq' (AWQ smoothing folded in: post_attention_layernorm scaled by 1/s, router / shared expert / expert
    input columns by s; experts quantised after smoothing, in both). Returns cfg and the two dirs."""
    cfg = _cfg()
    tcfg, Layer = hq._text_config(cfg)
    g = torch.Generator().manual_seed(seed)
    sh = {"awq": ({}, {}), "orig": ({}, {})}           # (expert shard, dense shard)
    for L in layers:
        torch.manual_seed(seed + L)
        lay = Layer(tcfg, L).double()
        for n, p in lay.named_parameters():
            torch.nn.init.normal_(p, std=0.05 if "norm" not in n else 0.2)
        if L == 0:
            with torch.no_grad():
                lay.linear_attn.A_log.copy_(torch.empty(4, dtype=torch.float64).uniform_(0.01, 2, generator=g).log())
                lay.linear_attn.dt_bias.copy_(torch.randn(4, generator=g, dtype=torch.float64))
        base = f"{LP}.{L}"
        s = torch.exp(torch.randn(64, generator=g, dtype=torch.float64) * 0.7)     # smoothing, per input channel
        ex = lay.mlp.experts
        I = ex.intermediate_dim
        for e in range(ex.num_experts):
            for p, W, smooth in (("gate_proj", ex.gate_up_proj[e, :I], True), ("up_proj", ex.gate_up_proj[e, I:], True),
                                 ("down_proj", ex.down_proj[e], False)):
                Ws = (W * s) if smooth else W
                qw, qz, sc = ref.awq_quantize_ref(Ws.detach().numpy(), G)
                for k, v in (("qweight", qw), ("qzeros", qz), ("scales", sc)):
                    for kind in sh:
                        sh[kind][0][f"{base}.mlp.experts.{e}.{p}.{k}"] = torch.from_numpy(v)
        for n, p in lay.named_parameters():
            if n.startswith("mlp.experts."):
                continue
            v = p.detach()
            sm = v
            if n == "post_attention_layernorm.weight":
                sm = (1 + v) / s - 1
            elif n in ("mlp.gate.weight", "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                       "mlp.shared_expert_gate.weight"):
                sm = v * s
            sh["awq"][1][f"{base}.{n}"] = sm.to(torch.float16).contiguous()
            sh["orig"][1][f"{base}.{n}"] = v.to(torch.float16).contiguous()
    dirs = {}
    for kind, (exs, dense) in sh.items():
        dirs[kind] = root / kind
        _save_sharded(dirs[kind], {"model-00001-of-00002.safetensors": exs, "model-00002-of-00002.safetensors": dense})
    return cfg, dirs


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    return _make_ckpts(tmp_path_factory.mktemp("awq35"))


def _ref_dequant(qw, qz, sc, g):
    return torch.from_numpy(ref.awq_dequant_ref(qw.numpy(), qz.numpy(), sc.numpy(), g))


def _pe(tcfg, B, T):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    pos = torch.arange(T)[None].expand(B, T)
    x = torch.zeros(B, T, tcfg.hidden_size)
    return Qwen3_5MoeTextRotaryEmbedding(tcfg)(x, pos[None].expand(3, B, T)), pos


def _run(layer, x, dy, pe, pos, dtype):
    xx = x.to(dtype).clone().requires_grad_(True)
    y = layer(xx, position_embeddings=tuple(t.to(dtype) for t in pe), position_ids=pos)
    (dx,) = torch.autograd.grad(y, xx, dy.to(dtype))
    return y.detach().double(), dx.double()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _cache(src: Path, out: Path, names=None, **kw):
    rd = hq.CheckpointReader.local(src)
    names = names if names is not None else list(rd.keys())
    logs = []
    kw.setdefault("max_shard_fraction", 1.01)               # synthetic shards are tiny; the guard has its own test
    man = awq35.write_layer_cache(rd, names, out, repo="synthetic/awq", revision="r0",
                                  file_for=lambda n: f"L{n.split('.')[3]}.safetensors", log=logs.append, **kw)
    return man, rd


# ------------------------------------------------------------------------------------------------------------
# cache writer
# ------------------------------------------------------------------------------------------------------------

def test_cache_round_trip_exact_bytes_and_sha(ckpts, tmp_path):
    import hashlib
    cfg, dirs = ckpts
    out = tmp_path / "cache"
    man, src = _cache(dirs["awq"], out)
    assert set(man["files"]) == {"L0.safetensors", "L3.safetensors"}
    cache = hq.CheckpointReader.local(out)
    assert set(cache.keys()) == set(src.keys())
    for n in src.keys():
        a = awq35._raw(src.files[src.weight_map[n]], n)
        b = awq35._raw(cache.files[cache.weight_map[n]], n)
        assert a == b and hashlib.sha256(b).hexdigest() == man["tensors"][n]["sha256"]
        t = man["tensors"][n]
        assert (t["repo"], t["revision"], t["source"]) == ("synthetic/awq", "r0", "range")
        assert cache.dtype_of(n) == src.dtype_of(n)
        _, shape, x, y = src.files[src.weight_map[n]].info(n)
        assert t["src_offsets"] == [x, y] and t["shape"] == shape and t["nbytes"] == len(a)
    from safetensors import safe_open                     # the files are valid safetensors for the stock reader
    with safe_open(str(out / "L0.safetensors"), "pt") as f:
        n = f"{LP}.0.mlp.experts.2.up_proj.qweight"
        assert torch.equal(f.get_tensor(n), src.get(n))
    for f, meta in man["files"].items():
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == meta["sha256"]
    assert awq35.layer_provenance(cache, 0) == {("synthetic/awq", "r0"): len(awq35.layer_tensor_names(src, 0))}


def test_cache_idempotent_merge_caps_and_reuse(ckpts, tmp_path):
    cfg, dirs = ckpts
    out = tmp_path / "c"
    rd = hq.CheckpointReader.local(dirs["awq"])
    l0 = awq35.layer_tensor_names(rd, 0)
    l3_dense = awq35.layer_tensor_names(rd, 3, parts=("input_layernorm", "self_attn"))
    man, _ = _cache(dirs["awq"], out, l0)
    m0 = (out / "L0.safetensors").stat().st_mtime_ns
    man, _ = _cache(dirs["awq"], out, l0)                    # again: nothing fetched, file untouched
    assert man["history"][-1]["fetched"] == 0 and man["history"][-1]["present"] == len(l0)
    assert (out / "L0.safetensors").stat().st_mtime_ns == m0
    man, _ = _cache(dirs["awq"], out, l3_dense)              # merge another layer's subset
    assert man["history"][-1]["fetched"] == len(l3_dense) and len(man["tensors"]) == len(l0) + len(l3_dense)
    assert (out / "L0.safetensors").stat().st_mtime_ns == m0
    l3_more = awq35.layer_tensor_names(rd, 3, parts=("post_attention_layernorm",))
    m3 = (out / "L3.safetensors").stat().st_mtime_ns
    pl = awq35.plan_layer_cache(rd, l3_dense + l3_more, out, repo="synthetic/awq", revision="r0",
                                file_for=lambda n: f"L{n.split('.')[3]}.safetensors")
    nb = sum(pl["plan"][n]["nbytes"] for n in l3_more)
    assert pl["fetch_bytes"] == pl["new_disk_bytes"] == nb and set(pl["fetch"]) == set(l3_more)
    man, _ = _cache(dirs["awq"], out, l3_more)               # grows L3 append-only: a part file, L3 untouched
    assert set(awq35.layer_tensor_names(hq.CheckpointReader.local(out), 3)) == set(l3_dense + l3_more)
    assert (out / "L3.safetensors").stat().st_mtime_ns == m3 and (out / "L3.p1.safetensors").exists()
    assert {man["tensors"][n]["file"] for n in l3_more} == {"L3.p1.safetensors"}
    man, _ = _cache(dirs["awq"], out, l3_dense + l3_more)    # the part file counts as present
    assert man["history"][-1]["fetched"] == 0 and not (out / "L3.p2.safetensors").exists()
    with pytest.raises(ValueError, match="max_bytes"):
        _cache(dirs["awq"], tmp_path / "small", l0, max_bytes=1000)
    with pytest.raises(ValueError, match="whole shards"):    # both layers' experts = all of shard 1
        _cache(dirs["awq"], tmp_path / "frac", [n for n in rd.keys() if ".experts." in n], max_shard_fraction=0.5)
    man2, _ = _cache(dirs["awq"], tmp_path / "re", l0, reuse_dirs=[out])
    assert man2["history"][-1]["reused"] == len(l0) and man2["history"][-1]["fetched"] == 0
    assert all(t["source"].startswith("reuse:") for t in man2["tensors"].values())
    with pytest.raises(awq35.MixedCheckpointError):          # a cache dir holds one repo + revision only
        awq35.write_layer_cache(rd, l0, out, repo="other/repo", revision="r0", file_for=lambda n: "L0.safetensors",
                                log=lambda s: None)


# ------------------------------------------------------------------------------------------------------------
# provenance, dtype policy, memory
# ------------------------------------------------------------------------------------------------------------

def test_mixed_checkpoints_refused_and_actually_differ(ckpts):
    cfg, dirs = ckpts
    awq, orig = hq.CheckpointReader.local(dirs["awq"]), hq.CheckpointReader.local(dirs["orig"])
    # experts from the smoothed checkpoint, dense tensors from the unsmoothed one: the mistake to prevent
    fn_ex, fn_dn = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    mixed = hq.CheckpointReader({fn_ex: awq.files[fn_ex], fn_dn: orig.files[fn_dn]}, awq.weight_map)
    assert len(awq35.layer_provenance(mixed, 0)) == 2
    with pytest.raises(awq35.MixedCheckpointError):
        awq35.load_layer(cfg, 0, mixed)
    with pytest.raises(awq35.MixedCheckpointError):
        awq35.load_layer(cfg, 0, awq, expect_source=("cyankiwi/x", "y"))
    good, tcfg = awq35.load_layer(cfg, 0, awq, compute_dtype=torch.float32)
    bad, _ = hq.build_decoder_layer(cfg, 0, mixed, compute_dtype=torch.float32)
    x = torch.randn(2, 12, 64, dtype=torch.float64)
    with torch.no_grad():
        h = x.float()
        yg, yb = good.mlp(good.post_attention_layernorm(h)), bad.mlp(bad.post_attention_layernorm(h))
    assert _rel(yb.double(), yg.double()) > 0.05             # mixing changes the function, not just rounding


def test_a_log_dt_bias_kept_fp32_exact(ckpts):
    cfg, dirs = ckpts
    rd = hq.CheckpointReader.local(dirs["awq"])
    layer, _ = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.bfloat16)
    for n in ("A_log", "dt_bias"):
        p = getattr(layer.linear_attn, n)
        f16 = rd.get(f"{LP}.0.linear_attn.{n}")
        assert p.dtype == torch.float32 and torch.equal(p, f16.float()) and not p.requires_grad
        assert not torch.equal(f16.float().to(torch.bfloat16).float(), f16.float())   # bf16 would round them
    assert layer.linear_attn.in_proj_qkv.weight.dtype == torch.bfloat16
    plain, _ = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.bfloat16, keep_fp32=())
    assert plain.linear_attn.A_log.dtype == torch.bfloat16
    att, _ = awq35.load_layer(cfg, 3, rd, compute_dtype=torch.bfloat16)  # no GDN params: keep_fp32 is a no-op
    assert all(p.dtype == torch.bfloat16 for p in att.parameters())


def test_layer_memory_accounting(ckpts):
    cfg, dirs = ckpts
    rd = hq.CheckpointReader.local(dirs["awq"])
    layer, tcfg = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.bfloat16, experts_per_group=4)
    m = awq35.layer_memory(layer)
    disk_ex = sum(rd.files[rd.weight_map[n]].info(n)[3] - rd.files[rd.weight_map[n]].info(n)[2]
                  for n in awq35.layer_tensor_names(rd, 0) if ".experts." in n)
    assert m["packed_bytes"] == disk_ex and m["quant_modules"] == 1
    n_dense = sum(p.numel() for n, p in layer.named_parameters())
    n_fp32 = sum(getattr(layer.linear_attn, n).numel() for n in ("A_log", "dt_bias"))
    assert m["dense_param_bytes"] == 2 * (n_dense - n_fp32) + 4 * n_fp32
    assert m["dense_by_dtype"]["torch.float32"] == 4 * n_fp32
    assert m["dequant_transient_bytes"] == 4 * 3 * 32 * 64 * 2
    assert m["resident_bytes"] == m["packed_bytes"] + m["dense_param_bytes"] + m["other_buffer_bytes"]


# ------------------------------------------------------------------------------------------------------------
# whole layers vs the fp64 reference built from the numpy dequant
# ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("L", [0, 3])
def test_decoder_layer_vs_fp64_reference(ckpts, tmp_path, L):
    cfg, dirs = ckpts
    _cache(dirs["awq"], tmp_path / "c")
    rd = hq.CheckpointReader.local(tmp_path / "c")          # load through the cache, as the validators do
    seen = {}
    refl, tcfg = awq35.dense_reference_layer(cfg, L, rd, _ref_dequant, dtype=torch.float64,
                                             on_matrix=lambda e, p, w: seen.setdefault((e, p), w))
    assert len(seen) == 3 * tcfg.num_experts
    B, T = 2, 40
    pe, pos = _pe(tcfg, B, T)
    g = torch.Generator().manual_seed(L)
    x = torch.randn(B, T, 64, generator=g, dtype=torch.float64)
    dy = torch.randn(B, T, 64, generator=g, dtype=torch.float64)
    y0, d0 = _run(refl, x, dy, pe, pos, torch.float64)
    ours32, _ = awq35.load_layer(cfg, L, rd, compute_dtype=torch.float32, experts_per_group=4)
    gu, dn = ours32.mlp.experts.dequant_experts(torch.arange(tcfg.num_experts), torch.float32)
    I = tcfg.moe_intermediate_size
    for e in range(tcfg.num_experts):                        # stacked module == reference, bit for bit
        assert torch.equal(gu[e, :I], seen[(e, "gate_proj")]) and torch.equal(gu[e, I:], seen[(e, "up_proj")])
        assert torch.equal(dn[e], seen[(e, "down_proj")])
    y1, d1 = _run(ours32, x, dy, pe, pos, torch.float32)
    assert _rel(y1, y0) < 1e-5 and _cos(d1, d0) > 0.999999
    ours16, _ = awq35.load_layer(cfg, L, rd, compute_dtype=torch.bfloat16, experts_per_group=4)
    y2, d2 = _run(ours16, x, dy, pe, pos, torch.bfloat16)
    assert _rel(y2, y0) < 3e-2 and _cos(d2, d0) > 0.999
    assert not any(p.requires_grad for p in ours16.parameters())


def test_fork_dequant_agrees_on_synthetic_experts(ckpts):
    if fork_refs.fork_awq_to_op_layout() is None:
        pytest.skip("minisgl clone absent")
    cfg, dirs = ckpts
    rd = hq.CheckpointReader.local(dirs["awq"])
    for e in (0, 5):
        for p in ("gate_proj", "down_proj"):
            pre = f"{LP}.0.mlp.experts.{e}.{p}"
            qw, qz, sc = rd.get(pre + ".qweight"), rd.get(pre + ".qzeros"), rd.get(pre + ".scales")
            assert torch.equal(fork_refs.fork_awq_dequant(qw, qz, sc, G), hq.awq_dequant(qw, qz, sc, G, torch.float32))


# ------------------------------------------------------------------------------------------------------------
# opt-in: real 35B weights from the M2-0 cache (no network)
# ------------------------------------------------------------------------------------------------------------

REAL = os.environ.get("NGM_AWQ35_CACHE")


@pytest.mark.slow
@pytest.mark.skipif(not REAL, reason="set NGM_AWQ35_CACHE=<m2_0_cache>/cyankiwi_35b (fetch_layer_cache.py)")
def test_real_layer0_cached():
    d = Path(REAL)
    man = awq35.read_manifest(d)
    assert (man["repo"], man["revision"]) == (awq35.REPO, awq35.REVISION)
    cfg = json.loads((d / "config.json").read_text())
    rd = hq.CheckpointReader.local(d)
    assert awq35.layer_provenance(rd, 0) == {(awq35.REPO, awq35.REVISION): len(awq35.layer_tensor_names(rd, 0))}
    for e in (0, 97, 255):
        for p in ("gate_proj", "up_proj", "down_proj"):
            pre = f"{LP}.0.mlp.experts.{e}.{p}"
            qw, qz, sc = rd.get(pre + ".qweight"), rd.get(pre + ".qzeros"), rd.get(pre + ".scales")
            ours = hq.awq_dequant(qw, qz, sc, G, torch.float32)
            assert torch.equal(ours, _ref_dequant(qw, qz, sc, G))
            assert torch.equal(hq.awq_dequant(qw, qz, sc, G), ours.to(torch.bfloat16))
    parts = ("input_layernorm", "linear_attn")
    ours, tcfg = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.float32, parts=parts,
                                  expect_source=(awq35.REPO, awq35.REVISION))
    refl, _ = awq35.dense_reference_layer(cfg, 0, rd, _ref_dequant, dtype=torch.float64, parts=parts)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 64, tcfg.hidden_size, generator=g, dtype=torch.float64)
    with torch.no_grad():
        y0 = refl.linear_attn(refl.input_layernorm(x))
        y1 = ours.linear_attn(ours.input_layernorm(x.float())).double()
    assert _rel(y1, y0) < 1e-5
