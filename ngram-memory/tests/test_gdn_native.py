"""gdn_native: native-forward + HF torch-chunk recompute backward (NativeFwdHFBwd), the arm dispatcher gdn_core, the
layer patch over it for both HF GDN class names, and the class-name widening of the fork's shim loop (N9).

The native gdn_hip op does not exist on CPU, so its interface is exercised with two stand-ins that have its exact
signature and state layout: hf_standin_op (HF's torch forward) and recurrent_standin_op (an independent per-token
recurrence in the native [HV, V, K] state layout). The fork is read-only and not importable here (msgpack), so
the shim adapter is tested with a fake ``minisgl.gdn.hf_patch`` module."""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from conftest import build_host
from ngram_memory.backends.graft import gdn, gdn_native as gn

HFM = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
FORK = Path("/home/user/patcarter883/minisglang-rdna4")
FORK_AUTOGRAD = FORK / "tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py"
FORK_HF_PATCH = FORK / "python/minisgl/gdn/hf_patch.py"


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _raw(B=2, T=70, H=2, HV=4, K=16, V=8, dtype=torch.float32, seed=0, grad=True):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    ts = [r(B, T, H, K), r(B, T, H, K), r(B, T, HV, V), r(B, T, HV), r(B, T, HV),
          torch.log(torch.rand(HV, generator=g, dtype=torch.float64) * 15 + 1), r(HV)]
    ts = [t.to(dtype) for t in ts]
    return [t.requires_grad_(grad) for t in ts]


def _op_call(op, q, k, v, a, b, A_log, dt, lens, init=None):
    """Call a native-signature op on packed varlen tensors (q etc. already [BT, ...])."""
    Bn = len(lens)
    HV, Vv, Kk = v.shape[1], v.shape[2], q.shape[2]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    st = torch.zeros(Bn + 1, HV, Vv, Kk) if init is None else init.clone()
    out = op(q, k, v, a, b, A_log.float(), dt.float(), cu, torch.arange(1, Bn + 1), (
        torch.zeros(Bn, dtype=torch.uint8) if init is None else torch.ones(Bn, dtype=torch.uint8)), st, Kk ** -0.5, 1)
    return out, st


# ------------------------------------------------------------------------------------------------------------
# HF torch function and the stand-ins
# ------------------------------------------------------------------------------------------------------------

def test_hf_chunk_torch_is_the_unwrapped_torch_function():
    f = gn.hf_chunk_torch()
    assert f is HFM.torch_chunk_gated_delta_rule.__wrapped__
    assert not hasattr(f, "__wrapped__") and f.__module__ == HFM.__name__
    assert gn.hf_torch_fn("causal_conv1d_fn") is HFM.causal_conv1d_fn.__wrapped__


@pytest.mark.parametrize("lens", [[37, 37], [37, 21, 50]])
def test_standins_agree_and_write_state_in_native_layout(lens):
    q, k, v, a, b, A, dt = (t.detach() for t in _raw(B=1, T=sum(lens), grad=False))
    flat = [t[0] for t in (q, k, v, a, b)]
    o_h, s_h = _op_call(gn.hf_standin_op, *flat, A, dt, lens)
    o_r, s_r = _op_call(gn.recurrent_standin_op, *flat, A, dt, lens)
    assert _rel(o_r, o_h) < 1e-5 and _rel(s_r[1:], s_h[1:]) < 1e-5
    assert s_h[0].abs().max() == 0                     # slot 0 is the null slot
    # the state is [HV, V, K]: compare with HF's own final state [HV, K, V] per sequence
    cu = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    for i in range(len(lens)):
        sl = slice(cu[i], cu[i + 1])
        _, sf = gn.hf_core(*(x[None, sl] for x in flat), A, dt, output_final_state=True)
        assert _rel(s_h[i + 1], sf[0].transpose(-1, -2)) < 1e-6


@pytest.mark.parametrize("op", [gn.hf_standin_op, gn.recurrent_standin_op])
def test_standin_initial_state_continues_a_sequence(op):
    ins = _raw(B=1, T=90, grad=False)
    q, k, v, a, b = (t[0] for t in ins[:5])
    A, dt = ins[5:]
    full, s_full = _op_call(op, q, k, v, a, b, A, dt, [90])
    o1, s1 = _op_call(op, *(x[:40] for x in (q, k, v, a, b)), A, dt, [40])
    o2, s2 = _op_call(op, *(x[40:] for x in (q, k, v, a, b)), A, dt, [50], init=s1)
    assert _rel(torch.cat([o1, o2]), full) < 1e-5 and _rel(s2[1], s_full[1]) < 1e-5


def test_hf_standin_rejects_other_scales():
    q, k, v, a, b, A, dt = (t.detach() for t in _raw(B=1, T=8, grad=False))
    with pytest.raises(ValueError):
        gn.hf_standin_op(q[0], k[0], v[0], a[0], b[0], A, dt, torch.tensor([0, 8], dtype=torch.int32),
                         torch.tensor([1]), torch.zeros(1, dtype=torch.uint8), torch.zeros(2, 4, 8, 16), 0.3, 1)


# ------------------------------------------------------------------------------------------------------------
# the autograd.Functions and the arm dispatcher
# ------------------------------------------------------------------------------------------------------------

ARMS = [("native", "hf"), ("hf", "hf_standin"), ("native", "ours"), ("ours", "ours"), ("native", "fork_ref")]


def _run_arm(ins, fwd, bwd, go, **kw):
    if bwd == "hf_standin":
        fwd, bwd = "hf", "hf"
        o = gn.gdn_core(*ins, fwd="native", bwd="hf", native_op=gn.hf_standin_op, **kw)
    else:
        o = gn.gdn_core(*ins, fwd=fwd, bwd=bwd, native_op=gn.recurrent_standin_op, **kw)
    return o, torch.autograd.grad(o, ins, go)


@pytest.mark.parametrize("fwd,bwd", ARMS)
def test_every_arm_matches_hf_autograd_fp32(fwd, bwd):
    if bwd == "fork_ref" and not FORK_AUTOGRAD.exists():
        pytest.skip("fork clone absent")
    ins = _raw(B=2, T=75)
    ref = gn.gdn_core(*ins, fwd="hf", bwd="hf")
    go = torch.randn(ref.shape, generator=torch.Generator().manual_seed(5))
    g_ref = torch.autograd.grad(ref, ins, go)
    o, gs = _run_arm(ins, fwd, bwd, go)
    assert o.dtype == ins[2].dtype and o.shape == ref.shape
    assert _rel(o, ref) < 1e-5
    for name, x, y in zip(("q", "k", "v", "a", "b", "A_log", "dt_bias"), gs, g_ref):
        assert _cos(x, y) > 0.99999, name


def test_hf_recompute_backward_equals_hf_autograd_exactly():
    """The backward of NativeFwdHFBwd is HF's autograd on the same raw tensors, whatever the forward was."""
    ins = _raw(B=3, T=64)
    ref = gn.gdn_core(*ins, fwd="hf", bwd="hf")
    go = torch.randn(ref.shape, generator=torch.Generator().manual_seed(6))
    g_ref = torch.autograd.grad(ref, ins, go)
    o = gn.NativeFwdHFBwd.apply(*ins, gn.recurrent_standin_op, 64, None)
    for x, y in zip(torch.autograd.grad(o, ins, go), g_ref):
        assert torch.equal(x, y)
    o = gn.NativeFwdHFBwd.apply(*ins, gn.recurrent_standin_op, 64, 1)          # one sequence per recompute
    for x, y in zip(torch.autograd.grad(o, ins, go), g_ref):
        assert _rel(x, y) < 1e-6


def test_bf16_saves_only_raw_tensors_and_returns_bf16_grads():
    ins = _raw(B=2, T=40, dtype=torch.bfloat16)
    o = gn.NativeFwdHFBwd.apply(*ins, gn.hf_standin_op, 64, None)
    saved = o.grad_fn.saved_tensors
    assert len(saved) == 7 and all(s.data_ptr() == t.data_ptr() for s, t in zip(saved, ins))
    assert o.dtype == torch.bfloat16
    ref = gn.hf_core(*ins)
    assert _rel(o.float(), ref.float()) < 1e-6   # same HF math in the stand-in
    gs = torch.autograd.grad(o, ins, torch.ones_like(o))
    assert all(g.dtype == torch.bfloat16 for g in gs)


def test_frozen_gates_get_no_gradient():
    q, k, v, a, b, A, dt = _raw(B=1, T=20)
    A.requires_grad_(False)
    dt.requires_grad_(False)
    o = gn.gdn_core(q, k, v, a, b, A, dt, fwd="hf", bwd="hf")
    o2 = gn.NativeFwdHFBwd.apply(q, k, v, a, b, A, dt, gn.hf_standin_op, 64, None)
    g1 = torch.autograd.grad(o, (q, k, v, a, b), torch.ones_like(o))
    g2 = torch.autograd.grad(o2, (q, k, v, a, b), torch.ones_like(o2))
    for x, y in zip(g1, g2):
        assert torch.equal(x, y)


def test_gradcheck_native_recurrent_forward_vs_hf_backward():
    """fp64 numerical Jacobian of the exact recurrent forward vs the HF-recompute analytic backward. HF's torch
    delta rule computes in fp32 whatever the input dtype, so the tolerances are fp32-level."""
    ins = [t.detach().double().requires_grad_(True) for t in _raw(B=1, T=6, H=1, HV=2, K=4, V=3, seed=2)]

    def f(*xs):
        return gn.NativeFwdHFBwd.apply(*xs, gn.recurrent_standin_op, 4, None)
    assert torch.autograd.gradcheck(f, tuple(ins), eps=1e-6, atol=2e-5, rtol=2e-3)


def test_dispatcher_errors():
    ins = _raw(B=1, T=8)
    with pytest.raises(ValueError):
        gn.gdn_core(*ins, fwd="ours", bwd="hf")
    with pytest.raises(ValueError):
        gn.gdn_core(*ins, fwd="xla", bwd="hf")
    if gn.resolve_native_op() is None:
        with pytest.raises(RuntimeError):
            gn.gdn_core(*ins, fwd="native", bwd="hf")


# ------------------------------------------------------------------------------------------------------------
# layer patch, both class names
# ------------------------------------------------------------------------------------------------------------

def _mixer_run(m, x, seed=7):
    x = x.clone().requires_grad_(True)
    y = m(x)
    w = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed))
    (dx,) = torch.autograd.grad((y * w).sum(), x)
    return y.detach(), dx


PATCH_ARMS = [dict(fwd="hf", bwd="hf"), dict(fwd="native", bwd="hf", native_op=gn.recurrent_standin_op),
              dict(fwd="native", bwd="hf", native_op=gn.hf_standin_op, bwd_group=1),
              dict(fwd="hf", bwd="ours"), dict(fwd="ours", bwd="ours"),
              dict(fwd="native", bwd="fork_ref", native_op=gn.hf_standin_op)]


@pytest.mark.parametrize("arm", PATCH_ARMS, ids=lambda d: f"{d['fwd']}-{d['bwd']}{'-g1' if d.get('bwd_group') else ''}")
@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_patch_matches_unpatched_hf_both_class_names(kind, arm):
    if arm["bwd"] == "fork_ref" and not FORK_AUTOGRAD.exists():
        pytest.skip("fork clone absent")
    host = build_host(kind).float()
    mods = gdn.gdn_modules(host)
    want = "Qwen3_5MoeGatedDeltaNet" if kind.endswith("moe") else "Qwen3_5GatedDeltaNet"
    assert len(mods) == 3 and {type(m).__name__ for _, m in mods} == {want}
    layer = mods[1][1]
    x = torch.randn(2, 70, host.config.hidden_size, generator=torch.Generator().manual_seed(3))
    y0, dx0 = _mixer_run(layer, x)
    sd = set(host.state_dict())
    cfg = gn.patch_gdn_native(host, chunk_size=16, **arm)
    try:
        assert len(cfg.layers) == 3
        y1, dx1 = _mixer_run(layer, x)
        assert cfg.calls == 1 and cfg.fallbacks == 0
        assert _rel(y1, y0) < 1e-5 and _cos(dx1, dx0) > 0.99999
        assert set(host.state_dict()) == sd
    finally:
        assert gn.unpatch_gdn_native(host) == 3
    y2, _ = _mixer_run(layer, x)
    assert torch.equal(y2, y0)


def test_whole_model_grad_and_cached_calls_fall_back():
    host = build_host("qwen3_5_moe").float()
    ids = torch.randint(3, 500, (2, 33), generator=torch.Generator().manual_seed(5))
    emb = host.get_input_embeddings()

    def run():
        e = emb(ids).detach().requires_grad_(True)
        out = host(inputs_embeds=e, use_cache=False).logits
        (de,) = torch.autograd.grad(out.float().logsumexp(-1).sum(), e)
        return out.detach(), de
    l0, d0 = run()
    with torch.no_grad():
        c0 = host(ids, use_cache=True).logits
    cfg = gn.patch_gdn_native(host, fwd="native", bwd="hf", native_op=gn.recurrent_standin_op)
    try:
        l1, d1 = run()
        assert cfg.calls == 3 and cfg.fallbacks == 0
        assert _rel(l1, l0) < 1e-5 and _cos(d1, d0) > 0.99999
        with torch.no_grad():
            out = host(ids, use_cache=True)                 # fresh cache: served natively, cache filled
            assert cfg.calls == 6 and cfg.fresh_cache_calls == 3 and cfg.fallbacks == 0
            c1 = out.logits
            nxt = host(ids[:, -1:], past_key_values=out.past_key_values, use_cache=True)   # decode -> HF
        assert cfg.fallbacks == 3 and _rel(c1, c0) < 1e-5
    finally:
        gn.unpatch_gdn_native(host)
    with torch.no_grad():
        ref = host(ids, use_cache=True)
        nref = host(ids[:, -1:], past_key_values=ref.past_key_values, use_cache=True)
    assert _rel(nxt.logits, nref.logits) < 1e-5          # the cache the patch filled continues exactly as HF's


@pytest.mark.parametrize("arm", PATCH_ARMS, ids=lambda d: f"{d['fwd']}-{d['bwd']}{'-g1' if d.get('bwd_group') else ''}")
def test_train_step_with_default_use_cache_runs_the_patch(arm):
    """Verifier D1: HF's config default use_cache=True builds a DynamicCache on a plain train-mode call; every GDN
    layer must still run gdn_core (calls == layers, fallbacks == 0), fill the cache like HF, and give HF's grads."""
    if arm["bwd"] == "fork_ref" and not FORK_AUTOGRAD.exists():
        pytest.skip("fork clone absent")
    host = build_host("qwen3_5_moe").float().train()
    assert host.config.use_cache is True
    ids = torch.randint(3, 500, (2, 40), generator=torch.Generator().manual_seed(9))

    def step():
        host.zero_grad(set_to_none=True)
        out = host(ids, labels=ids)                          # no use_cache argument: config default
        out.loss.backward()
        g = torch.cat([p.grad.flatten() for p in host.parameters() if p.grad is not None])
        return out, g
    o0, g0 = step()
    cfg = gn.patch_gdn_native(host, chunk_size=16, **arm)
    try:
        o1, g1 = step()
        assert cfg.calls == len(cfg.layers) == 3 and cfg.fallbacks == 0 and cfg.fresh_cache_calls == 3
        assert _rel(o1.logits.detach(), o0.logits.detach()) < 1e-5 and _cos(g1, g0) > 0.99999
        c0, c1 = o0.past_key_values, o1.past_key_values
        if c0 is not None and c1 is not None:
            for _, m in gdn.gdn_modules(host):
                i = m.layer_idx
                s0, s1 = c0.get_recurrent_state(i, state_idx=0), c1.get_recurrent_state(i, state_idx=0)
                assert s0 is not None and s1 is not None and _rel(s1, s0) < 1e-5
                v0, v1 = c0.get_conv_state(i, state_idx=0), c1.get_conv_state(i, state_idx=0)
                assert _rel(v1, v0) < 1e-5            # upstream layers differ at fp32 rounding
    finally:
        gn.unpatch_gdn_native(host)


def test_strict_refuses_a_silent_hf_fallback_while_grad_flows():
    host = build_host("qwen3_5_moe").float()
    ids = torch.randint(3, 500, (1, 12), generator=torch.Generator().manual_seed(2))
    cfg = gn.patch_gdn_native(host, fwd="native", bwd="hf", native_op=gn.recurrent_standin_op)
    try:
        with torch.no_grad():
            pkv = host(ids, use_cache=True).past_key_values
        with pytest.raises(gn.GDNFallbackError):
            host(ids[:, -1:], past_key_values=pkv, use_cache=True)     # cache with state + grad -> refuse
        assert cfg.fallbacks == 0
    finally:
        gn.unpatch_gdn_native(host)
    cfg = gn.patch_gdn_native(host, fwd="native", bwd="hf", native_op=gn.recurrent_standin_op, strict=False)
    try:
        with torch.no_grad():
            pkv = host(ids, use_cache=True).past_key_values
        host(ids[:, -1:], past_key_values=pkv, use_cache=True)
        assert cfg.fallbacks == 3
    finally:
        gn.unpatch_gdn_native(host)
    with pytest.raises(ValueError):
        gn.patch_gdn_native(host, fwd="hf", bwd="hf", strict="always")


@pytest.mark.parametrize("fwd,bwd", ARMS)
def test_state_sink_gives_hf_final_state(fwd, bwd):
    if bwd == "fork_ref" and not FORK_AUTOGRAD.exists():
        pytest.skip("fork clone absent")
    ins = _raw(B=2, T=45, grad=False)
    _, sf = gn.hf_core(*ins, output_final_state=True)
    sink: list = []
    if bwd == "hf_standin":
        gn.gdn_core(*ins, fwd="native", bwd="hf", native_op=gn.hf_standin_op, state_sink=sink)
    else:
        gn.gdn_core(*ins, fwd=fwd, bwd=bwd, native_op=gn.recurrent_standin_op, state_sink=sink)
    assert len(sink) == 1 and sink[0].shape == sf.shape and sink[0].dtype == torch.float32
    assert _rel(sink[0], sf) < 1e-5


def test_beta_fp32_default_follows_the_forward():
    """Verifier D3: with a native-math forward the recompute defaults to fp32 sigmoid(b); with the HF stand-in
    (bf16 sigmoid, like HF) it does not."""
    host = build_host("qwen3_5_moe")
    for op, want in ((gn.recurrent_standin_op, True), (gn.hf_standin_op, False)):
        cfg = gn.patch_gdn_native(host, fwd="native", bwd="hf", native_op=op)
        assert cfg.beta_fp32 is want
        gn.unpatch_gdn_native(host)
    assert gn.patch_gdn_native(host, fwd="hf", bwd="hf").beta_fp32 is None
    gn.unpatch_gdn_native(host)
    ins = _raw(B=1, T=30, dtype=torch.bfloat16, seed=3)
    go = torch.randn(1, 30, 4, 8, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    o = gn.gdn_core(*ins, fwd="native", bwd="hf", native_op=gn.recurrent_standin_op)
    g_def = torch.autograd.grad(o, ins, go)
    g_32 = torch.autograd.grad(gn.gdn_core(*ins, fwd="hf", bwd="hf", beta_fp32=True), ins, go)
    for x, y in zip(g_def, g_32):
        assert torch.equal(x, y)


def test_patch_resolution_and_errors():
    with pytest.raises(ValueError):
        gn.patch_gdn_native(torch.nn.Linear(2, 2))
    host = build_host("qwen3_5_moe")
    cfg = gn.patch_gdn_native(host)                       # auto on CPU -> HF torch forward
    assert (cfg.fwd, cfg.bwd, cfg.native_op) == ("hf", "hf", None)
    with pytest.raises(RuntimeError):                     # no double patching
        gn.patch_gdn_native(host)
    gn.unpatch_gdn_native(host)
    gdn.patch_gdn(host, impl="ours")
    with pytest.raises(RuntimeError):
        gn.patch_gdn_native(host, fwd="hf", bwd="hf")
    gdn.unpatch_gdn(host)
    if gn.resolve_native_op() is None:
        with pytest.raises(RuntimeError):
            gn.patch_gdn_native(host, fwd="native")
    with pytest.raises(ValueError):
        gn.patch_gdn_native(host, fwd="ours", bwd="hf")


# ------------------------------------------------------------------------------------------------------------
# the fork's shim loop, widened (N9)
# ------------------------------------------------------------------------------------------------------------

class _FakeShim(torch.nn.Module):
    built: list = []

    def __init__(self, hf):
        super().__init__()
        type(self).built.append(type(hf).__name__)
        self.layer_idx = getattr(hf, "layer_idx", None)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
        return hidden_states


@pytest.fixture
def fake_fork(monkeypatch):
    _FakeShim.built = []
    pkg = {n: types.ModuleType(n) for n in ("minisgl", "minisgl.gdn", "minisgl.gdn.hf_patch")}
    pkg["minisgl.gdn.hf_patch"].NativeGDNShim = _FakeShim
    for n, m in pkg.items():
        monkeypatch.setitem(sys.modules, n, m)
    return _FakeShim


def _fork_patch_fn():
    """The fork's own patch_qwen3_5_gdn, extracted by AST (its module imports serve-only deps)."""
    src = FORK_HF_PATCH.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "patch_qwen3_5_gdn")
    ns: dict = {"NativeGDNShim": _FakeShim}
    exec(compile(ast.get_source_segment(src, fn), str(FORK_HF_PATCH), "exec"), ns)
    return ns["patch_qwen3_5_gdn"]


@pytest.mark.skipif(not FORK_HF_PATCH.exists(), reason="fork clone absent")
def test_fork_loop_misses_moe_class_and_adapter_widens_it(fake_fork):
    fork_patch = _fork_patch_fn()
    assert fork_patch(build_host("qwen3_5_moe")) == 0          # N9: the exact-name test finds nothing on the MoE
    assert fork_patch(build_host("qwen3_5")) == 3
    fake_fork.built = []
    host = build_host("qwen3_5_moe")
    rec = gn.patch_fork_shims(host, keep_originals=True)       # default: imports minisgl.gdn.hf_patch lazily
    assert len(rec) == 3 and rec.class_names == {"Qwen3_5MoeGatedDeltaNet": 3}
    assert fake_fork.built == ["Qwen3_5MoeGatedDeltaNet"] * 3
    assert gdn.gdn_modules(host) == [] and sum(isinstance(m, _FakeShim) for m in host.modules()) == 3
    assert [m.layer_idx for m in host.modules() if isinstance(m, _FakeShim)] == [0, 1, 2]
    assert rec.restore() == 3 and len(gdn.gdn_modules(host)) == 3
    with pytest.raises(ValueError):
        gn.patch_fork_shims(torch.nn.Sequential(torch.nn.Linear(2, 2)), shim_cls=_FakeShim)
    rec = gn.patch_fork_shims(host, shim_cls=_FakeShim)
    with pytest.raises(RuntimeError):
        rec.restore()


@pytest.mark.skipif(not FORK_HF_PATCH.exists(), reason="fork clone absent")
def test_moe_mixer_has_every_attribute_the_fork_shim_reads():
    """_build_native_from_hf reads hf.<attr> (and hf.<attr>.weight); all must exist on the MoE class."""
    src = FORK_HF_PATCH.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_build_native_from_hf")
    chains = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute):
            parts, cur = [], node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "hf":
                chains.add(tuple(reversed(parts)))
    assert len(chains) >= 10
    for kind in ("qwen3_5_moe", "qwen3_5"):
        m = gdn.gdn_modules(build_host(kind))[0][1]
        for ch in chains:
            obj = m
            for p in ch:
                if p in ("weight", "dtype", "device") and isinstance(obj, torch.Tensor):
                    break
                assert hasattr(obj, p), f"{kind}: hf.{'.'.join(ch)}"
                obj = getattr(obj, p)


def test_beta_fp32_recompute_differentiates_the_native_kernel_math():
    """For bf16 b, HF rounds sigmoid(b) to bf16; the native kernel keeps it fp32. beta_fp32=True makes the
    recompute the exact gradient of the native math (and its forward agrees with the exact recurrence far better)."""
    ins = _raw(B=2, T=64, dtype=torch.bfloat16, seed=4)
    exact = gn.gdn_core(*ins, fwd="native", bwd="hf", native_op=gn.recurrent_standin_op)
    hf16 = gn.hf_core(*ins)
    hf32 = gn.hf_core(*ins, beta_fp32=True)
    assert _rel(exact.float(), hf32.float()) < 0.25 * _rel(exact.float(), hf16.float())
    go = torch.randn(exact.shape, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16)
    o = gn.NativeFwdHFBwd.apply(*ins, gn.recurrent_standin_op, 64, None, True)
    g_ref = torch.autograd.grad(gn.gdn_core(*ins, fwd="hf", bwd="hf", beta_fp32=True), ins, go)
    for x, y in zip(torch.autograd.grad(o, ins, go), g_ref):
        assert torch.equal(x, y)
    host = build_host("qwen3_5_moe")
    cfg = gn.patch_gdn_native(host, fwd="native", bwd="hf", native_op=gn.hf_standin_op, beta_fp32=True)
    assert cfg.beta_fp32
    gn.unpatch_gdn_native(host)
