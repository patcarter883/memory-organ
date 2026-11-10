"""GDN training kernel (ngram_memory.backends.graft.gdn): exactness of the hand-written chunked backward,
parity with HF's torch delta rule, layer patching for both HF GDN class names, cached-call fallback, and the
native-forward plumbing (with a CPU stand-in for gdn_hip)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import build_host
from ngram_memory.backends.graft import gdn

HFQ = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
HF_CHUNK = getattr(HFQ.torch_chunk_gated_delta_rule, "__wrapped__", HFQ.torch_chunk_gated_delta_rule)


def _inputs(B=2, T=150, H=3, dk=16, dv=8, dtype=torch.float64, seed=0, with_state=True):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    q, k, v = r(B, T, H, dk), r(B, T, H, dk), r(B, T, H, dv)
    gg = -torch.rand(B, T, H, generator=g, dtype=dtype) * 0.5
    be = torch.rand(B, T, H, generator=g, dtype=dtype)
    S0 = r(B, H, dk, dv) if with_state else None
    ts = [q, k, v, gg, be] + ([S0] if with_state else [])
    for t in ts:
        t.requires_grad_(True)
    return ts


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


@pytest.mark.parametrize("tri", ["solve", "doubling"])
@pytest.mark.parametrize("C", [16, 64])
def test_fp64_matches_recurrent_reference(tri, C):
    q, k, v, g, be, S0 = _inputs()
    o1, s1 = gdn.chunk_gated_delta_rule(q, k, v, g, be, chunk_size=C, initial_state=S0, output_final_state=True,
                                        use_qk_l2norm_in_kernel=True, tri=tri)
    o2, s2 = gdn.recurrent_gated_delta_rule_reference(q, k, v, g, be, initial_state=S0)
    assert (o1 - o2).abs().max() < 1e-12 and (s1 - s2).abs().max() < 1e-12
    go, gs = torch.randn_like(o1), torch.randn_like(s1)
    g1 = torch.autograd.grad((o1 * go).sum() + (s1 * gs).sum(), (q, k, v, g, be, S0))
    g2 = torch.autograd.grad((o2 * go).sum() + (s2 * gs).sum(), (q, k, v, g, be, S0))
    for a, b in zip(g1, g2):
        assert (a - b).abs().max() < 1e-11


def test_gradcheck_small():
    q, k, v, g, be, S0 = _inputs(B=1, T=11, H=2, dk=4, dv=3, seed=1)

    def f(q, k, v, g, be, S0):
        o, s = gdn.chunk_gated_delta_rule(q, k, v, g, be, chunk_size=4, initial_state=S0, output_final_state=True,
                                          use_qk_l2norm_in_kernel=True)
        return o, s
    assert torch.autograd.gradcheck(f, (q, k, v, g, be, S0), eps=1e-6, atol=1e-7)


def test_fp32_vs_hf_torch_chunk():
    q, k, v, g, be = _inputs(B=2, T=200, H=4, dk=32, dv=32, dtype=torch.float32, with_state=False)
    o1, _ = gdn.chunk_gated_delta_rule(q, k, v, g, be, use_qk_l2norm_in_kernel=True)
    o2, _ = HF_CHUNK(q, k, v, g=g, beta=be, use_qk_l2norm_in_kernel=True)
    assert ((o1 - o2).norm() / o2.norm()).item() < 1e-5
    go = torch.randn_like(o1)
    for a, b in zip(torch.autograd.grad(o1, (q, k, v, g, be), go), torch.autograd.grad(o2, (q, k, v, g, be), go)):
        assert _cos(a, b) > 0.99999


def test_save_states_and_groups_are_identical():
    q, k, v, g, be, S0 = _inputs(B=2, T=70, H=3, dtype=torch.float32)
    ref = gdn.chunk_gated_delta_rule(q, k, v, g, be, initial_state=S0, output_final_state=True,
                                     use_qk_l2norm_in_kernel=True, chunk_size=16)
    alt = gdn.chunk_gated_delta_rule(q, k, v, g, be, initial_state=S0, output_final_state=True,
                                     use_qk_l2norm_in_kernel=True, chunk_size=16, save_states=False, max_bh=4)
    assert torch.equal(ref[0], alt[0]) and torch.equal(ref[1], alt[1])
    go = torch.randn_like(ref[0])
    ga = torch.autograd.grad(ref[0], (q, k, v, g, be, S0), go)
    gb = torch.autograd.grad(alt[0], (q, k, v, g, be, S0), go)
    for a, b in zip(ga, gb):
        assert torch.equal(a, b)


def test_bf16_inputs_return_bf16_and_track_fp32():
    q, k, v, g, be = (t.detach().to(torch.bfloat16).requires_grad_() for t in
                      _inputs(dtype=torch.float32, with_state=False))
    o, _ = gdn.chunk_gated_delta_rule(q, k, v, g, be, use_qk_l2norm_in_kernel=True)
    o2, _ = HF_CHUNK(q, k, v, g=g, beta=be, use_qk_l2norm_in_kernel=True)
    assert o.dtype == torch.bfloat16
    assert ((o.float() - o2.float()).norm() / o2.float().norm()).item() < 1e-2


def _layer_run(model, x):
    x = x.clone().requires_grad_(True)
    y = model(x)
    gy = torch.Generator().manual_seed(7)
    w = torch.randn(y.shape, generator=gy)
    (dx,) = torch.autograd.grad((y * w).sum(), x)
    return y.detach(), dx


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_patch_layer_matches_hf_and_counts_both_class_names(kind):
    host = build_host(kind).float()
    mods = gdn.gdn_modules(host)
    assert len(mods) == 3                     # layer_types[:4] = 3 linear + 1 full attention
    assert type(mods[0][1]).__name__ == ("Qwen3_5MoeGatedDeltaNet" if kind.endswith("moe") else "Qwen3_5GatedDeltaNet")
    layer = mods[0][1]
    x = torch.randn(2, 77, host.config.hidden_size, generator=torch.Generator().manual_seed(3))
    y0, dx0 = _layer_run(layer, x)
    cfg = gdn.patch_gdn(host, impl="ours", chunk_size=16)
    try:
        y1, dx1 = _layer_run(layer, x)
        assert cfg.calls == 1 and cfg.fallbacks == 0
        assert ((y1 - y0).norm() / y0.norm()).item() < 1e-5
        assert _cos(dx1, dx0) > 0.99999
        sd_before = set(host.state_dict())
    finally:
        assert gdn.unpatch_gdn(host) == 3
    assert set(host.state_dict()) == sd_before
    y2, _ = _layer_run(layer, x)
    assert torch.equal(y2, y0)


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_whole_model_logits_grad_and_cached_fallback(kind):
    host = build_host(kind).float()
    ids = torch.randint(3, 500, (2, 40), generator=torch.Generator().manual_seed(5))
    emb = host.get_input_embeddings()

    def run():
        e = emb(ids).detach().requires_grad_(True)
        out = host(inputs_embeds=e, use_cache=False).logits
        (de,) = torch.autograd.grad(out.float().logsumexp(-1).sum(), e)
        return out.detach(), de
    l0, d0 = run()
    with torch.no_grad():
        c0 = host(ids, use_cache=True).logits
    cfg = gdn.patch_gdn(host, impl="ours")
    try:
        l1, d1 = run()
        assert cfg.calls == 3
        assert ((l1 - l0).norm() / l0.norm()).item() < 1e-5 and _cos(d1, d0) > 0.99999
        with torch.no_grad():
            o1 = host(ids, use_cache=True)
            c1 = o1.logits
            # a fresh-cache prefill (zero initial state) is served by our kernel and fills the cache (VGK-1) ...
            assert cfg.calls == 6 and cfg.fallbacks == 0
            assert ((c1 - c0).norm() / c0.norm()).item() < 1e-5
            # ... and the cached decode step that follows is handed to HF's own forward
            host(ids[:, -1:], past_key_values=o1.past_key_values, use_cache=True)
        assert cfg.fallbacks == 3
    finally:
        gdn.unpatch_gdn(host)


def _fake_native_op(q, k, v, a, b, A_log, dt_bias, cu, state_idx, has_init, ssm_state, scale, use_l2norm):
    """CPU stand-in with gdn_hip.gdn_prefill's signature: [B*T,...] varlen, zero state, HF math."""
    B = cu.numel() - 1
    T = q.shape[0] // B
    qq, kk = q.reshape(B, T, *q.shape[1:]), k.reshape(B, T, *k.shape[1:])
    vv = v.reshape(B, T, *v.shape[1:])
    aa, bb = a.reshape(B, T, -1), b.reshape(B, T, -1)
    rep = vv.shape[2] // qq.shape[2]
    g = -A_log.float().exp() * F.softplus(aa.float() + dt_bias)
    o, _ = HF_CHUNK(qq.repeat_interleave(rep, 2), kk.repeat_interleave(rep, 2), vv, g=g, beta=bb.sigmoid(),
                    use_qk_l2norm_in_kernel=bool(use_l2norm))
    return o.reshape(B * T, *o.shape[2:])


def test_native_forward_plumbing_matches_ours():
    torch.manual_seed(0)
    B, T, H, HV, dk, dv = 2, 90, 2, 4, 16, 8
    q, k = torch.randn(B, T, H, dk), torch.randn(B, T, H, dk)
    v, a, b = torch.randn(B, T, HV, dv), torch.randn(B, T, HV), torch.randn(B, T, HV)
    A_log, dt = torch.randn(HV), torch.randn(HV)
    ins = [t.requires_grad_(True) for t in (q, k, v, a, b, A_log, dt)]
    o1 = gdn.gated_delta_rule_from_raw(*ins, impl="hf_math", chunk_size=32)
    o2 = gdn.gated_delta_rule_from_raw(*ins, impl="native", native_op=_fake_native_op, chunk_size=32)
    assert ((o1 - o2).norm() / o1.norm()).item() < 1e-5
    go = torch.randn_like(o1)
    for x, y in zip(torch.autograd.grad(o1, ins, go), torch.autograd.grad(o2, ins, go)):
        assert _cos(x, y) > 0.99999


def test_patch_native_requires_op_and_rejects_empty_model():
    with pytest.raises(ValueError):
        gdn.patch_gdn(torch.nn.Linear(2, 2))
    host = build_host("qwen3_5_moe")
    if gdn.native_gdn_op() is None:
        with pytest.raises(RuntimeError):
            gdn.patch_gdn(host, impl="native")
    cfg = gdn.patch_gdn(host, impl="native", native_op=_fake_native_op)
    try:
        assert cfg.impl == "native"
    finally:
        gdn.unpatch_gdn(host)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("save_states", [True, False])
def test_raw_path_matches_hf_math_path(dtype, save_states):
    """RawGDN (saves raw tensors, recomputes the prep) == prep-with-autograd + chunk_gated_delta_rule."""
    g = torch.Generator().manual_seed(11)
    B, T, H, HV, dk, dv = 2, 75, 2, 4, 16, 8
    ins = [torch.randn(*s, generator=g).to(dtype) for s in
           ((B, T, H, dk), (B, T, H, dk), (B, T, HV, dv), (B, T, HV), (B, T, HV), (HV,), (HV,))]
    ins = [t.requires_grad_(True) for t in ins]
    o1 = gdn.gated_delta_rule_from_raw(*ins, impl="ours", chunk_size=16, save_states=save_states, max_bh=3)
    o2 = gdn.gated_delta_rule_from_raw(*ins, impl="hf_math", chunk_size=16)
    assert o1.dtype == dtype
    tol = 1e-6 if dtype == torch.float32 else 1e-2
    assert ((o1.float() - o2.float()).norm() / o2.float().norm()).item() < tol
    go = torch.randn(o1.shape, generator=g).to(dtype)
    for x, y in zip(torch.autograd.grad(o1, ins, go), torch.autograd.grad(o2, ins, go)):
        assert _cos(x, y) > (0.999999 if dtype == torch.float32 else 0.999)
