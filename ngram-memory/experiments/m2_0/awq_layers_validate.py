"""M2-0 B2: real 35B AWQ layers (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit) loaded by ``awq35.load_layer`` from the
per-layer cache written by fetch_layer_cache.py (DESIGN §4.6, §7.5 #1, §8 M2-0 / M2 prerequisites).

Parts (``--parts``, default 1,2,3,4,5):
 1. Bit-exactness: every routed-expert matrix of the cached layer(s) (L0: 256 experts x gate/up/down = 768):
    ours (host_quant.awq_dequant) fp32 and bf16 vs the independent numpy reference (tests/hq_reference.py) vs
    the fork's awq_to_op_layout + op decode (tests/fork_refs.py, read-only); the stacked QuantMoEExperts module
    vs the per-matrix reference for all experts.
 2. Layers vs an fp32 reference (stock HF Qwen3_5MoeDecoderLayer, every tensor fp32, experts dequantised by the
    numpy reference): L0 whole decoder layer (GDN + MoE) forward and dX; the MoE block at 2k / 8k tokens; the
    GDN mixer at --gdn-B x --gdn-T; L3 attention (+ input norm) at --attn-B x --attn-T; L3 router logits and
    top-8 sets and the shared-expert branch. Ours in fp32 compute (bars: rel <= 1e-5, dX cos >= 0.999999) and
    bf16 compute as trained (provisional bars: rel <= 2e-2, dX cos >= 0.999), router top-k agreement.
 3. Report only: the AWQ checkpoint's dense tensors vs the bf16 original (Qwen/Qwen3.6-35B-A3B @ 345d467b) for
    layer 0's GDN inputs (input_layernorm, row slices of in_proj_qkv / in_proj_z / in_proj_a / in_proj_b) to see
    whether AWQ smoothing touched the GDN inputs; a few MB by range read.
 4. Memory (bf16 compute): resident bytes per layer (packed, dense, transient), peak live tensor bytes and peak
    RSS for forward and forward + backward of the GDN mixer / attention at 4k-token pieces and the MoE block at
    8k / 16k-token chunks, and a projection to 40 layers at the 57:43 card split (est.).
 5. All 30 GDN layers' A_log / dt_bias (AWQ F16, ~4 KB): exactly representable in bf16? vs the bf16 original.

CPU by default (contended box: wall times carry the load average). ``--gpu``: our layers on cuda (reference on
CPU), peak VRAM by torch.cuda.max_memory_allocated, and the grouped_mm probe on the card; run under
``gpu-lease -n 1``. The HF GDN path is pinned to its pure-torch functions (``__wrapped__``) so neither fla nor a
hub kernel is used (fla hangs on gfx1201).

  python experiments/m2_0/awq_layers_validate.py --cache-dir $SCRATCH/m2_0_cache --threads 3
  gpu-lease -n 1 -- python experiments/m2_0/awq_layers_validate.py --gpu --cache-dir <dir> --parts 1,2,4
  ... --ckpt <full local snapshot> reads the checkpoint directly (then L3's experts are there too: L3 bit-exactness
  and the whole L3 decoder layer run as well). The owner's one-command run is experiments/m2_0/run_owner.sh.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fork_refs  # noqa: E402
import hq_reference as ref  # noqa: E402
from ngram_memory.backends.graft import awq35  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

LP = awq35.LP
ORIG_REPO, ORIG_REV = "Qwen/Qwen3.6-35B-A3B", "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
ORIG_SNAP = Path(f"/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{ORIG_REV}")
AWQ_SNAP = Path(f"/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/{awq35.REVISION}")
BARS = dict(fp32_rel=1e-5, fp32_cos=0.999999, bf16_rel=2e-2, bf16_cos=0.999)
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------------------------------------------------
# measurement
# ------------------------------------------------------------------------------------------------------------

class LiveBytes(TorchDispatchMode):
    """Peak bytes of storages created by ops while active and still alive (weak refs, so tensors held by the
    autograd graph count until it frees them). Views of storages that already existed (weights, inputs) are not
    counted; allocator caching and kernel workspaces are not seen."""

    def __init__(self):
        super().__init__()
        self.live: dict[int, tuple] = {}
        self.ext: dict[int, object] = {}
        self.cur = self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        for t in tree_flatten((args, kwargs or {}))[0]:
            if isinstance(t, torch.Tensor):
                k = t.untyped_storage()._cdata
                if k not in self.live and (k not in self.ext or self.ext[k].expired()):
                    self.ext[k] = StorageWeakRef(t.untyped_storage())
        out = func(*args, **(kwargs or {}))
        for k in [k for k, (r, _) in self.live.items() if r.expired()]:
            self.cur -= self.live.pop(k)[1]
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                s = t.untyped_storage()
                k = s._cdata
                if k in self.live or (k in self.ext and not self.ext[k].expired()):
                    continue
                self.live[k] = (StorageWeakRef(s), s.nbytes())
                self.cur += s.nbytes()
        self.peak = max(self.peak, self.cur)
        return out


def _status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return -1


PEAK_HWM_KB = [0]


def _reset_hwm() -> bool:
    PEAK_HWM_KB[0] = max(PEAK_HWM_KB[0], _status_kb("VmHWM"))
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


class Meter:
    def __init__(self, dev: torch.device, live: bool):
        self.dev, self.cuda, self.live = dev, dev.type == "cuda", live

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.dev)

    def run(self, fn):
        """-> (result, stats): wall s, peak live bytes (CPU tracker), peak RSS delta, or cuda peak bytes."""
        gc.collect()
        self.sync()
        st: dict = {}
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.dev)
            base = torch.cuda.memory_allocated(self.dev)
        rss0 = _status_kb("VmRSS")
        hwm_ok = _reset_hwm()
        t0 = time.perf_counter()
        lb = LiveBytes() if (self.live and not self.cuda) else None
        with (lb if lb is not None else contextlib.nullcontext()):
            out = fn()
        self.sync()
        st["wall_s"] = round(time.perf_counter() - t0, 3)
        st["loadavg_1m"] = os.getloadavg()[0]
        if lb is not None:
            st["peak_live_bytes"] = lb.peak
        if hwm_ok:
            st["peak_rss_delta_bytes"] = (_status_kb("VmHWM") - rss0) * 1024
        if self.cuda:
            st["cuda_peak_bytes_over_base"] = torch.cuda.max_memory_allocated(self.dev) - base
            st["cuda_peak_bytes"] = torch.cuda.max_memory_allocated(self.dev)
        return out, st


@contextlib.contextmanager
def pure_torch_gdn():
    """Pin HF's GDN kernels to the pure-torch functions (no fla, no hub kernels)."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update")
    old = {n: getattr(m, n) for n in names}
    try:
        for n in names:
            setattr(m, n, getattr(old[n], "__wrapped__", old[n]))
        yield
    finally:
        for n, f in old.items():
            setattr(m, n, f)


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    return F.cosine_similarity(a.double().cpu().flatten(), b.double().cpu().flatten(), dim=0).item()


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {4: torch.int32, 2: torch.int16}[a.element_size()]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def compare(y, dx, y0, dx0, kind: str) -> dict:
    r = dict(fwd_rel=rel(y, y0), dx_cos=cos(dx, dx0), dx_rel=rel(dx, dx0))
    if kind in ("fp32", "bf16"):
        r["pass"] = r["fwd_rel"] <= BARS[f"{kind}_rel"] and r["dx_cos"] >= BARS[f"{kind}_cos"]
    return r


def router_agree(i_a: torch.Tensor, i_b: torch.Tensor) -> dict:
    a, b = i_a.cpu(), i_b.cpu()
    k = a.shape[-1]
    sa, sb = a.sort(-1).values, b.sort(-1).values
    ov = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).double() / k
    return dict(tokens=a.shape[0], top1_agree=(a[:, 0] == b[:, 0]).double().mean().item(),
                set_exact=(sa == sb).all(-1).double().mean().item(), mean_overlap=ov.mean().item(),
                min_overlap=ov.min().item())


class GateTap:
    """Capture the router's (logits, scores, indices) of a layer's mlp.gate."""

    def __init__(self, mod):
        self.out = None
        self.h = mod.register_forward_hook(lambda m, i, o: setattr(self, "out", tuple(t.detach() for t in o)))

    def close(self):
        self.h.remove()


def rope(tcfg, B, T, dev, dtype):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    pos = torch.arange(T, device=dev)[None].expand(B, T)
    x = torch.zeros(1, device=dev, dtype=dtype)
    cos_, sin_ = Qwen3_5MoeTextRotaryEmbedding(tcfg).to(dev)(x, pos[None].expand(3, B, T))
    return (cos_.to(dtype), sin_.to(dtype)), pos


# ------------------------------------------------------------------------------------------------------------
# parts
# ------------------------------------------------------------------------------------------------------------

def ref_dequant(qw, qz, sc, g):
    return torch.from_numpy(ref.awq_dequant_ref(qw.numpy(), qz.numpy(), sc.numpy(), g))


def build_ref_with_bitexact(cfg, L, rd, parts, do_bits: bool) -> tuple:
    """fp32 HF reference layer; while filling it, part 1's per-matrix checks (ours fp32 / bf16, fork)."""
    sch = hq.scheme_from_config(cfg)
    fork_ok = fork_refs.fork_awq_to_op_layout() is not None
    st = dict(matrices=0, ours_eq_ref_fp32=0, ours_eq_ref_bf16=0, ours_eq_fork_fp32=0 if fork_ok else None,
              fork_available=fork_ok, nan=0, zero_columns_down_proj=[])

    def on_matrix(e, p, w):
        if not do_bits:
            return
        pre = f"{LP}.{L}.mlp.experts.{e}.{p}"
        qw, qz, sc = rd.get(pre + ".qweight"), rd.get(pre + ".qzeros"), rd.get(pre + ".scales")
        ours = hq.awq_dequant(qw, qz, sc, sch.group_size, torch.float32)
        st["matrices"] += 1
        st["ours_eq_ref_fp32"] += same(ours, w)
        st["ours_eq_ref_bf16"] += same(hq.awq_dequant(qw, qz, sc, sch.group_size), w.to(torch.bfloat16))
        if fork_ok:
            st["ours_eq_fork_fp32"] += same(ours, fork_refs.fork_awq_dequant(qw, qz, sc, sch.group_size))
        st["nan"] += int(torch.isnan(w).sum())
        if p == "down_proj":
            st["zero_columns_down_proj"].append(int((w == 0).all(0).sum()))

    t0 = time.time()
    layer, tcfg = awq35.dense_reference_layer(cfg, L, rd, ref_dequant, dtype=torch.float32, parts=parts,
                                              on_matrix=on_matrix)
    st["ref_build_s"] = round(time.time() - t0, 1)
    z = st.pop("zero_columns_down_proj")
    if z:
        st["down_proj_all_zero_input_columns"] = dict(min=min(z), max=max(z), mean=sum(z) / len(z), of=tcfg.moe_intermediate_size)
    return layer, tcfg, st


def stacked_eq_ref(ours_layer, ref_layer) -> dict:
    ex, rx = ours_layer.mlp.experts, ref_layer.mlp.experts
    ok = 0
    for g0 in range(0, ex.num_experts, 32):
        ids = torch.arange(g0, min(g0 + 32, ex.num_experts))
        gu, dn = ex.dequant_experts(ids.to(ex.gu_qweight.device), torch.float32)
        for j, e in enumerate(ids.tolist()):
            ok += same(gu[j].cpu(), rx.gate_up_proj[e]) and same(dn[j].cpu(), rx.down_proj[e])
    return dict(experts=ex.num_experts, stacked_module_eq_ref=ok)


def fwd_bwd(fn, x, dy):
    xx = x.clone().requires_grad_(True)
    y = fn(xx)
    (dx,) = torch.autograd.grad(y, xx, dy)
    return y.detach(), dx


def run_cmp(name, meter, ref_fn, ours: dict, x, dy, dev, res: dict, taps=None):
    """ours: {'fp32': fn, 'bf16': fn}; ref on CPU fp32. Stores metrics + timings into res[name]."""
    out = {}
    (y0, d0), st0 = meter_cpu(meter).run(lambda: fwd_bwd(ref_fn, x, dy))
    out["ref_fp32"] = st0
    i0 = taps["ref"].out[2] if taps else None
    for kind, fn in ours.items():
        dt = torch.float32 if kind == "fp32" else torch.bfloat16
        (y, d), st = meter.run(lambda: fwd_bwd(fn, x.to(dev, dt), dy.to(dev, dt)))
        out[kind] = dict(compare(y, d, y0, d0, kind), **st)
        if taps:
            out[kind]["router_vs_ref_fp32"] = router_agree(taps[kind].out[2], i0)
            out[kind]["router_logits_rel"] = rel(taps[kind].out[0].float(), taps["ref"].out[0])
        del y, d
    res[name] = out
    log(name, json.dumps({k: {kk: (round(vv, 8) if isinstance(vv, float) else vv) for kk, vv in v.items()
                              if kk in ("fwd_rel", "dx_cos", "pass", "wall_s", "router_vs_ref_fp32")}
                          for k, v in out.items()}))


def meter_cpu(meter):
    return Meter(torch.device("cpu"), live=False)


def part_layers(cfg, rd, a, dev, meter, res, do_bits: bool, do_cmp: bool):
    g = torch.Generator().manual_seed(0)
    H = cfg["text_config"]["hidden_size"]
    # ---------------- L0: GDN + MoE ----------------
    reflayer, tcfg, bits = build_ref_with_bitexact(cfg, 0, rd, None, do_bits)
    ours32, _ = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.float32, device=dev, experts_per_group=a.epg,
                                 expect_source=(awq35.REPO, awq35.REVISION))
    if do_bits:
        bits.update(stacked_eq_ref(ours32, reflayer))
        res["1_bitexact"] = {"L0": bits, "L3": ("see part_l3_full" if has_experts(rd, 3) else
                                                "experts not cached (disk); run fetch_layer_cache.py --layer 3")}
        log("bitexact L0", json.dumps(bits))
    if not do_cmp:
        return
    ours16, _ = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.bfloat16, device=dev, experts_per_group=a.epg)
    cmp = res.setdefault("2_vs_fp32_ref", {})
    cmp["bars"] = BARS
    # whole decoder layer
    T = a.layer_T
    pe32c, pos_c = rope(tcfg, 1, T, "cpu", torch.float32)
    pe = {k: rope(tcfg, 1, T, dev, dt) for k, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    x = torch.randn(1, T, H, generator=g)
    dy = torch.randn(1, T, H, generator=g)
    taps = {"ref": GateTap(reflayer.mlp.gate), "fp32": GateTap(ours32.mlp.gate), "bf16": GateTap(ours16.mlp.gate)}
    with pure_torch_gdn():
        run_cmp(f"L0_decoder_layer_B1_T{T}", meter,
                lambda xx: reflayer(xx, position_embeddings=pe32c, position_ids=pos_c),
                {"fp32": lambda xx: ours32(xx, position_embeddings=pe["fp32"][0], position_ids=pe["fp32"][1]),
                 "bf16": lambda xx: ours16(xx, position_embeddings=pe["bf16"][0], position_ids=pe["bf16"][1])},
                x, dy, dev, cmp, taps)
    # MoE block alone (post_attention_layernorm + mlp)
    for n in a.moe_ref_tokens:
        xm = torch.randn(1, n, H, generator=g)
        dym = torch.randn(1, n, H, generator=g)
        run_cmp(f"L0_moe_block_{n}tok", meter, lambda xx: reflayer.mlp(reflayer.post_attention_layernorm(xx)),
                {"fp32": lambda xx: ours32.mlp(ours32.post_attention_layernorm(xx)),
                 "bf16": lambda xx: ours16.mlp(ours16.post_attention_layernorm(xx))}, xm, dym, dev, cmp, taps)
        # routed experts alone with the reference's routing (separates expert numerics from router flips)
        with torch.no_grad():
            h = reflayer.post_attention_layernorm(xm).view(-1, H)
            _, w_r, i_r = reflayer.mlp.gate(h)
        run_cmp(f"L0_routed_experts_ref_routing_{n}tok", meter, lambda hh: reflayer.mlp.experts(hh, i_r, w_r),
                {"fp32": lambda hh: ours32.mlp.experts(hh, i_r.to(dev), w_r.to(dev)),
                 "bf16": lambda hh: ours16.mlp.experts(hh, i_r.to(dev), w_r.to(dev, torch.bfloat16))},
                h, dym.view(-1, H), dev, cmp)
    for t in taps.values():
        t.close()
    del reflayer.mlp.experts, ours32.mlp.experts
    gc.collect()
    # GDN mixer at the 4k piece (fp32 dense weights are the same F16 values in both, so only bf16 is informative)
    B, T = a.gdn_B, a.gdn_T
    xg = torch.randn(B, T, H, generator=g)
    dyg = torch.randn(B, T, H, generator=g)
    with pure_torch_gdn():
        run_cmp(f"L0_gdn_mixer_B{B}_T{T}", meter,
                lambda xx: reflayer.linear_attn(reflayer.input_layernorm(xx)),
                {"bf16": lambda xx: ours16.linear_attn(ours16.input_layernorm(xx))}, xg, dyg, dev, cmp)
    res["L0_classes"] = dict(mixer=type(ours16.linear_attn).__name__, experts=type(ours16.mlp.experts).__name__,
                             A_log_dtype=str(ours16.linear_attn.A_log.dtype),
                             dt_bias_dtype=str(ours16.linear_attn.dt_bias.dtype))
    del reflayer, ours32, ours16
    gc.collect()
    # ---------------- L3: attention (+ dense MoE parts) ----------------
    parts3 = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp.gate", "mlp.shared_expert",
              "mlp.shared_expert_gate")
    ref3, tcfg, _ = build_ref_with_bitexact(cfg, 3, rd, parts3, False)
    o3 = {k: awq35.load_layer(cfg, 3, rd, compute_dtype=dt, device=dev, parts=parts3)[0]
          for k, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    B, T = a.attn_B, a.attn_T
    pe32c, _ = rope(tcfg, B, T, "cpu", torch.float32)
    pe = {k: rope(tcfg, B, T, dev, dt)[0] for k, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    xa = torch.randn(B, T, H, generator=g)
    dya = torch.randn(B, T, H, generator=g)
    attn = lambda L, p: (lambda xx: L.self_attn(L.input_layernorm(xx), position_embeddings=p,  # noqa: E731
                                                attention_mask=None)[0])
    run_cmp(f"L3_attention_B{B}_T{T}", meter, attn(ref3, pe32c), {k: attn(o3[k], pe[k]) for k in o3}, xa, dya,
            dev, cmp)
    n = a.moe_ref_tokens[0]
    xs = torch.randn(n, H, generator=g)
    dys = torch.randn(n, H, generator=g)
    shared = lambda L: (lambda xx: torch.sigmoid(L.mlp.shared_expert_gate(L.post_attention_layernorm(xx)))  # noqa
                        * L.mlp.shared_expert(L.post_attention_layernorm(xx)))
    run_cmp(f"L3_shared_expert_branch_{n}tok", meter, shared(ref3), {k: shared(o3[k]) for k in o3}, xs, dys, dev, cmp)
    with torch.no_grad():
        r0 = ref3.mlp.gate(ref3.post_attention_layernorm(xs))
        rr = {k: o3[k].mlp.gate(o3[k].post_attention_layernorm(xs.to(dev, dt)))
              for k, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    cmp[f"L3_router_{n}tok"] = {k: dict(logits_rel=rel(v[0].float(), r0[0]), **router_agree(v[2], r0[2]))
                                for k, v in rr.items()}
    log("L3 router", json.dumps(cmp[f"L3_router_{n}tok"]))
    del ref3, o3
    gc.collect()


def has_experts(rd, L: int) -> bool:
    return f"{LP}.{L}.mlp.experts.0.down_proj.qweight" in rd


def part_l3_full(cfg, rd, a, dev, meter, res, do_bits: bool, do_cmp: bool):
    """L3 (attention + MoE) as a whole decoder layer, when its routed experts are available (a full local
    checkpoint, or a cache that has them): bit-exactness of its 768 expert matrices and the layer vs fp32 ref."""
    g = torch.Generator().manual_seed(3)
    H = cfg["text_config"]["hidden_size"]
    ref3, tcfg, bits = build_ref_with_bitexact(cfg, 3, rd, None, do_bits)
    o32, _ = awq35.load_layer(cfg, 3, rd, compute_dtype=torch.float32, device=dev, experts_per_group=a.epg)
    if do_bits:
        bits.update(stacked_eq_ref(o32, ref3))
        res.setdefault("1_bitexact", {})["L3"] = bits
        log("bitexact L3", json.dumps(bits))
    if do_cmp:
        o16, _ = awq35.load_layer(cfg, 3, rd, compute_dtype=torch.bfloat16, device=dev, experts_per_group=a.epg)
        cmp = res.setdefault("2_vs_fp32_ref", {})
        T = a.layer_T
        pe32c, pos_c = rope(tcfg, 1, T, "cpu", torch.float32)
        pe = {k: rope(tcfg, 1, T, dev, dt) for k, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
        x = torch.randn(1, T, H, generator=g)
        dy = torch.randn(1, T, H, generator=g)
        taps = {"ref": GateTap(ref3.mlp.gate), "fp32": GateTap(o32.mlp.gate), "bf16": GateTap(o16.mlp.gate)}
        run_cmp(f"L3_decoder_layer_B1_T{T}", meter, lambda xx: ref3(xx, position_embeddings=pe32c, position_ids=pos_c),
                {"fp32": lambda xx: o32(xx, position_embeddings=pe["fp32"][0], position_ids=pe["fp32"][1]),
                 "bf16": lambda xx: o16(xx, position_embeddings=pe["bf16"][0], position_ids=pe["bf16"][1])},
                x, dy, dev, cmp, taps)
        for t in taps.values():
            t.close()
        del o16
    del ref3, o32
    gc.collect()


def part_memory(cfg, rd, a, dev, meter, res):
    meter = Meter(dev, live=True)
    out = res.setdefault("4_memory", {})
    g = torch.Generator().manual_seed(1)
    H = cfg["text_config"]["hidden_size"]
    l0, tcfg = awq35.load_layer(cfg, 0, rd, compute_dtype=torch.bfloat16, device=dev, experts_per_group=a.epg)
    parts3 = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp.gate", "mlp.shared_expert",
              "mlp.shared_expert_gate")
    l3, _ = awq35.load_layer(cfg, 3, rd, compute_dtype=torch.bfloat16, device=dev, parts=parts3)
    out["L3_routed_experts_cached"] = has_experts(rd, 3)
    m0, m3 = awq35.layer_memory(l0), awq35.layer_memory(l3)
    out["resident_L0_gdn_moe"] = m0
    out["resident_L3_attn_dense_parts"] = m3
    out["packed_layer_bytes_on_disk"] = {"L0": hq.packed_layer_bytes(rd, 0), "L3_cached_parts": hq.packed_layer_bytes(rd, 3)}
    log("resident", json.dumps({"L0": m0, "L3": m3}))

    def piece(name, fn, shape):
        x = torch.randn(*shape, generator=g).to(dev, torch.bfloat16)
        dy = torch.randn(*shape, generator=g).to(dev, torch.bfloat16)
        with torch.no_grad():
            _, sf = meter.run(lambda: fn(x))
        _, sb = meter.run(lambda: fwd_bwd(fn, x, dy))
        out[name] = dict(tokens=shape[0] * shape[1] if len(shape) == 3 else shape[0], forward=sf, forward_backward=sb)
        log("mem", name, json.dumps(out[name]))

    with pure_torch_gdn():
        piece(f"gdn_mixer_B1_T{a.piece_T}", lambda xx: l0.linear_attn(l0.input_layernorm(xx)), (1, a.piece_T, H))
    if dev.type == "cuda":                     # the training path: native forward + the fixed backward
        from ngram_memory.backends.graft import gdn_native as gn
        op = gn.resolve_native_op()
        if op is None:
            out["gdn_mixer_fix"] = "gdn_hip not importable"
        else:
            pc = gn.patch_gdn_native(l0.linear_attn, fwd="native", bwd="hf", native_op=op, bwd_group=1)
            piece(f"gdn_mixer_fix_g1_B1_T{a.piece_T}", lambda xx: l0.linear_attn(l0.input_layernorm(xx)),
                  (1, a.piece_T, H))
            out[f"gdn_mixer_fix_g1_B1_T{a.piece_T}"]["gdn_calls"] = dict(calls=pc.calls, fallbacks=pc.fallbacks)
            gn.unpatch_gdn_native(l0.linear_attn)
    pe, _ = rope(tcfg, 1, a.piece_T, dev, torch.bfloat16)
    piece(f"attention_B1_T{a.piece_T}", lambda xx: l3.self_attn(l3.input_layernorm(xx), position_embeddings=pe,
                                                                   attention_mask=None)[0], (1, a.piece_T, H))
    ex = l0.mlp.experts
    for epg in a.epg_mem:
        ex.experts_per_group = epg
        ids = torch.arange(epg, device=dev)
        _, sd = meter.run(lambda: ex.dequant_experts(ids))
        out[f"dequant_one_group_epg{epg}"] = dict(bf16_output_bytes=epg * 3 * ex.intermediate_dim * ex.hidden_dim * 2, **sd)

        def all_groups():
            for g0 in range(0, ex.num_experts, epg):
                ex.dequant_experts(torch.arange(g0, min(g0 + epg, ex.num_experts), device=dev))
        _, sa = meter.run(all_groups)
        out[f"dequant_all_256_experts_epg{epg}"] = sa
        log("mem dequant", epg, json.dumps(out[f"dequant_one_group_epg{epg}"]), json.dumps(sa))
        for n in a.moe_chunk_tokens:
            piece(f"moe_block_{n}tok_epg{epg}", lambda xx: l0.mlp(l0.post_attention_layernorm(xx)), (1, n, H))
    ex.experts_per_group = a.epg
    # projection to the 40-layer body at the 57:43 split (est.): GDN-type layers resident = L0; attention-type =
    # L3 dense parts + L0's routed experts (same shapes); lm_head bf16 on the cards, embed_tokens on the CPU.
    tc = cfg["text_config"]
    n_gdn = sum(t == "linear_attention" for t in tc["layer_types"])
    n_att = len(tc["layer_types"]) - n_gdn
    ex_packed = m0["packed_bytes"]
    att_layer = m3["resident_bytes"] + ex_packed
    lm_head = tc["vocab_size"] * H * 2
    body = n_gdn * m0["resident_bytes"] + n_att * att_layer
    tot = body + lm_head + H * 2
    pk = lambda v: v.get("cuda_peak_bytes_over_base", v.get("peak_live_bytes", 0))  # noqa: E731
    mixer_peak = max(pk(v["forward_backward"]) for k, v in out.items()
                     if isinstance(v, dict) and "forward_backward" in v and not k.startswith("moe_block"))
    ckpt = max(a.moe_chunk_tokens) * H * 2               # one bf16 layer-input checkpoint per layer at the MoE chunk
    nL = len(tc["layer_types"])
    proj = dict(n_gdn_layers=n_gdn, n_attention_layers=n_att, gdn_layer_resident=m0["resident_bytes"],
                attention_layer_resident=att_layer, lm_head_bf16=lm_head, body_resident=body, total_resident=tot,
                card0_resident=0.57 * tot, card1_resident=0.43 * tot, mixer_piece_fwd_bwd_peak=mixer_peak,
                layer_input_checkpoint_bytes_per_layer=ckpt, by_moe_chunk={},
                note="est.: CPU live-tensor peaks stand in for the GPU allocator's peak unless --gpu; one piece "
                     "(the largest of mixer / MoE fwd+bwd) live at a time under per-layer checkpointing; excludes "
                     "logits/CE chunks, row-table activations, allocator fragmentation and the HIP context")
    for epg in a.epg_mem:
        for n in a.moe_chunk_tokens:
            v = out.get(f"moe_block_{n}tok_epg{epg}")
            if not v:
                continue
            tr = max(mixer_peak, pk(v["forward_backward"]))
            c0 = 0.57 * tot + tr + round(0.57 * nL) * n * H * 2
            c1 = 0.43 * tot + tr + round(0.43 * nL) * n * H * 2
            proj["by_moe_chunk"][f"{n}tok_epg{epg}"] = dict(transient=tr, card0_est=c0, card1_est=c1,
                                                           bar_15_5GB=c0 <= 15.5e9 and c1 <= 15.5e9)
    out["projection_40_layers"] = proj
    log("projection", json.dumps(proj))
    del l0, l3
    gc.collect()


def _hub(repo, rev, snap, names, stats, key):
    idx = snap / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"]
    rd = hq.CheckpointReader.hub(repo, revision=rev, index_json=idx, files=sorted({wm[n] for n in names}))
    stats[key + "_header_bytes"] = stats.get(key + "_header_bytes", 0) + rd.bytes_read
    return rd


def part_original(cfg, rd, res, rows: int):
    """AWQ dense GDN-input tensors vs the bf16 original (layer 0): is AWQ smoothing folded into them?"""
    st: dict = {}
    b = f"{LP}.0."
    full = ["input_layernorm.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight", "linear_attn.norm.weight", "linear_attn.A_log", "linear_attn.dt_bias",
            "post_attention_layernorm.weight"]
    tc = cfg["text_config"]
    kd = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    segs = {"in_proj_qkv.q": (0, rows), "in_proj_qkv.k": (kd, kd + rows), "in_proj_qkv.v": (2 * kd, 2 * kd + rows),
            "in_proj_z": (0, rows), "out_proj": (0, rows)}
    ro = _hub(ORIG_REPO, ORIG_REV, ORIG_SNAP, [b + n for n in full] + [b + "linear_attn.in_proj_qkv.weight"], st, "orig")
    h0 = ro.bytes_read
    t = ro.get_many([b + n for n in full], max_gap=0)
    orig = {n: t[b + n] for n in full}
    awq = {n: rd.get(b + n) for n in full}
    for k, (s, e) in segs.items():
        nm = b + "linear_attn." + k.split(".")[0] + ".weight"
        orig[k] = ro.get_rows(nm, s, e)
        awq[k] = rd.get_rows(nm, s, e)
    st["orig_weight_bytes"] = ro.bytes_read - h0
    out = {}
    for k in awq:
        wa, wo = awq[k].double(), orig[k].double()
        d = dict(awq_dtype=str(awq[k].dtype), orig_dtype=str(orig[k].dtype),
                 shape=list(wa.shape), rel_diff=((wa - wo).norm() / wo.norm().clamp_min(1e-30)).item(),
                 max_abs_diff=(wa - wo).abs().max().item(),
                 equal_after_f16_rounding=bool(torch.equal(wo.to(torch.float16).double(), wa)))
        if wa.dim() == 2 and wa.shape[1] == tc["hidden_size"]:
            c = (wa * wo).sum(0) / (wo * wo).sum(0).clamp_min(1e-30)    # best per-input-column scale
            d["col_scale_median"] = c.median().item()
            d["col_scale_p05_p95"] = [c.quantile(0.05).item(), c.quantile(0.95).item()]
            d["rel_after_col_scale"] = ((wa - wo * c).norm() / wa.norm()).item()
        out[k] = d
    ln_a = 1 + awq["input_layernorm.weight"].double()
    ln_o = 1 + orig["input_layernorm.weight"].double()
    ratio = ln_a / ln_o
    out["input_layernorm_effective_scale_ratio"] = dict(median=ratio.median().item(), p01=ratio.quantile(0.01).item(),
                                                        p99=ratio.quantile(0.99).item(),
                                                        max_abs_log=ratio.log().abs().max().item())
    wo = orig["input_layernorm.weight"].float()
    wa = awq["input_layernorm.weight"]
    h16 = torch.float16
    cands = {"f16(w)": wo.to(h16),
             "f16(f16(1+w)-1)": ((1 + wo).to(h16).float() - 1).to(h16),
             "f16(bf16(1+w)-1)": ((1 + wo).to(torch.bfloat16).float() - 1).to(h16),
             "f16(f16(f16(1+w)/1)-1) [unit smoothing in f16]": (((1 + wo).to(h16) / torch.ones((), dtype=h16)).float() - 1).to(h16)}
    out["input_layernorm_rounding_hypotheses"] = {k: dict(frac_equal=(v == wa).float().mean().item(),
                                                          rel=((v.float() - wa.float()).norm() / wa.float().norm()).item())
                                                  for k, v in cands.items()}
    log("input_layernorm hypotheses", json.dumps(out["input_layernorm_rounding_hypotheses"]))
    res["3_vs_bf16_original_L0_gdn_inputs"] = dict(stats=st, tensors=out)
    log("vs original", json.dumps({k: {kk: v[kk] for kk in ("rel_diff", "equal_after_f16_rounding") if kk in v}
                                   for k, v in out.items() if isinstance(v, dict) and "rel_diff" in v}))
    log("input_layernorm (1+w) ratio", json.dumps(out["input_layernorm_effective_scale_ratio"]))


def part_alog(cfg, cache_dir: Path, res):
    tc = cfg["text_config"]
    gdn_layers = [i for i, t in enumerate(tc["layer_types"]) if t == "linear_attention"]
    names = [f"{LP}.{i}.linear_attn.{p}" for i in gdn_layers for p in ("A_log", "dt_bias")]
    wm = json.loads((AWQ_SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    st: dict = {}
    files = {}
    for s in sorted({wm[n] for n in names}):
        hp = cache_dir / "cyankiwi_35b" / "headers" / f"{s}.hdr.json"
        base = f"https://huggingface.co/{awq35.REPO}/resolve/{awq35.REVISION}/{s}"
        files[s] = awq35.CachedHeaderFile(base, hp.read_bytes()) if hp.exists() else hq.SafetensorsFile(base)
    ra = hq.CheckpointReader(files, {n: wm[n] for n in names})
    h0 = ra.bytes_read
    ta = ra.get_many(names, max_gap=0)
    st["awq_bytes"] = ra.bytes_read - h0
    ro = _hub(ORIG_REPO, ORIG_REV, ORIG_SNAP, names, st, "orig")
    h1 = ro.bytes_read
    to = ro.get_many(names, max_gap=0)
    st["orig_bytes"] = ro.bytes_read - h1
    rows = {}
    for n in names:
        a32 = ta[n].float()
        o = to[n]
        rows[n.split(".", 3)[-1]] = dict(
            awq_dtype=str(ta[n].dtype), orig_dtype=str(o.dtype),
            bf16_exact=bool(torch.equal(a32.to(torch.bfloat16).float(), a32)),
            bf16_max_rel_round=((a32.to(torch.bfloat16).float() - a32).abs() / a32.abs().clamp_min(1e-30)).max().item(),
            awq_eq_orig=bool(torch.equal(a32.double(), o.double())),
            awq_eq_f16_of_orig=bool(torch.equal(o.to(torch.float16).float(), a32)),
            max_abs_diff_vs_orig=(a32.double() - o.double()).abs().max().item())
    summ = dict(layers=len(gdn_layers), tensors=len(names),
                bf16_exact=sum(v["bf16_exact"] for v in rows.values()),
                awq_eq_orig=sum(v["awq_eq_orig"] for v in rows.values()),
                awq_eq_f16_of_orig=sum(v["awq_eq_f16_of_orig"] for v in rows.values()),
                worst_bf16_rel_round=max(v["bf16_max_rel_round"] for v in rows.values()),
                orig_dtypes=sorted({v["orig_dtype"] for v in rows.values()}))
    res["5_A_log_dt_bias_all_gdn_layers"] = dict(summary=summ, stats=st, per_tensor=rows)
    log("A_log/dt_bias", json.dumps(summ), json.dumps(st))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("NGM_M2_0_CACHE", "m2_0_cache")))
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="a full local checkpoint dir (e.g. the owner's HF snapshot) instead of <cache-dir>/cyankiwi_35b")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/m2_0/results/awq")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--parts", default="1,2,3,4,5")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epg", type=int, default=32, help="experts dequantised together")
    ap.add_argument("--layer-T", type=int, default=2048, help="tokens for the whole-L0 comparison (B=1)")
    ap.add_argument("--moe-ref-tokens", default="2048,8192")
    ap.add_argument("--gdn-B", type=int, default=1)
    ap.add_argument("--gdn-T", type=int, default=4096)
    ap.add_argument("--attn-B", type=int, default=1)
    ap.add_argument("--attn-T", type=int, default=4096)
    ap.add_argument("--piece-T", type=int, default=4096)
    ap.add_argument("--moe-chunk-tokens", default="8192,16384")
    ap.add_argument("--epg-mem", default="8,32", help="experts_per_group values for the memory part")
    ap.add_argument("--orig-rows", type=int, default=256)
    a = ap.parse_args()
    a.moe_ref_tokens = [int(x) for x in a.moe_ref_tokens.split(",") if x]
    a.moe_chunk_tokens = [int(x) for x in a.moe_chunk_tokens.split(",") if x]
    a.epg_mem = [int(x) for x in a.epg_mem.split(",") if x]
    torch.set_num_threads(a.threads)
    parts = set(a.parts.split(","))
    dev = torch.device("cuda" if a.gpu else "cpu")
    meter = Meter(dev, live=False)          # comparisons: timings only; part 4 tracks live bytes
    d = a.ckpt or (a.cache_dir / "cyankiwi_35b")
    man = awq35.read_manifest(d)
    if man is None and a.ckpt is None:
        raise SystemExit(f"no cache at {d}: run experiments/m2_0/fetch_layer_cache.py first")
    if man is None:
        if not (d / "model.safetensors.index.json").exists():
            raise SystemExit(f"{d}: no model.safetensors.index.json")
        man = dict(repo=awq35.REPO, revision=f"{awq35.REVISION} (local checkpoint, revision as given)",
                   files={"checkpoint_dir": str(d)})
    cfg = json.loads((d / "config.json").read_text())
    rd = hq.CheckpointReader.local(d)
    res: dict = dict(repo=man["repo"], revision=man["revision"], cache_files=man["files"],
                     device=str(dev), threads=a.threads, torch=torch.__version__, args={k: (str(v) if isinstance(v, Path)
                                                                                               else v) for k, v in vars(a).items()},
                     nproc=os.cpu_count(), loadavg_start=os.getloadavg())
    if a.gpu:
        res["gpu"] = dict(name=torch.cuda.get_device_name(dev), grouped_mm=hq.grouped_mm_probe(dev),
                          total_bytes=torch.cuda.get_device_properties(dev).total_memory)
    res["grouped_mm_cpu"] = hq.grouped_mm_probe("cpu")
    t0 = time.time()
    tag = a.tag or ("gpu" if a.gpu else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    outp = a.out / f"awq_validate_{tag}.json"

    def save():
        res["wall_s"] = round(time.time() - t0, 1)
        PEAK_HWM_KB[0] = max(PEAK_HWM_KB[0], _status_kb("VmHWM"))
        res["peak_rss_bytes"] = PEAK_HWM_KB[0] * 1024     # ru_maxrss is reset by the per-piece clear_refs
        outp.write_text(json.dumps(res, indent=1, default=str))
        (a.out / f"awq_validate_{tag}.log").write_text("\n".join(LOG) + "\n")

    try:
        if "1" in parts or "2" in parts:
            part_layers(cfg, rd, a, dev, meter, res, "1" in parts, "2" in parts)
            save()
            if has_experts(rd, 3):
                part_l3_full(cfg, rd, a, dev, meter, res, "1" in parts, "2" in parts)
                save()
        if "3" in parts:
            part_original(cfg, rd, res, a.orig_rows)
            save()
        if "4" in parts:
            part_memory(cfg, rd, a, dev, meter, res)
            save()
        if "5" in parts:
            part_alog(cfg, a.cache_dir, res)
    finally:
        save()
    log("wrote", outp, "wall", res["wall_s"], "s; peak RSS GB", res["peak_rss_bytes"] / 1e9)
    save()


if __name__ == "__main__":
    main()
