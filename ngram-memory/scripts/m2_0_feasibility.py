"""M2-0: one-layer feasibility of graft training on the 4-bit Qwen3.6-35B-A3B host (DESIGN §8 M2-0, §4.6, §7.5 #1-2).

One real 35B layer of each kind, loaded straight from safetensors with the custom AWQ loader (experts kept
packed, gptqmodel-free): a GDN mixer (layer --gdn-layer), a full-attention mixer and a MoE block (layer
--attn-layer). For each: forward + dX backward (weights frozen) with timings; per-card peak VRAM on GPU. GDN is
timed with HF's torch delta rule, with our chunked backward (gdn.py), with the native gdn_hip forward + our
backward (GPU, when importable) and with the fork's own path (NativeGDNShim: native forward + its reference
recompute backward). Also: whether torch._grouped_mm runs on the device. Prints PASS/FAIL against the
provisional M2-0 bars.

GPU (owner's machine, serve stopped; MUST be launched under the box's arbiter):
    gpu-lease -n 2 -- python scripts/m2_0_feasibility.py --ckpt-35b /path/to/Qwen3.6-35B-A3B-AWQ-4bit --out m2_0.json
CPU smoke run of the same code path (this container; weights by HTTP range read, small chunks):
    python scripts/m2_0_feasibility.py --cpu-small --out m2_0_cpu_small.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gdn_bench  # noqa: E402
from ngram_memory.backends.graft import gdn  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

REPO35 = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
SNAP35 = Path("/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/"
              "f1e8415c4f29508fe281da4e256df26573213bfd")
FORK_PY = Path("/home/user/patcarter883/minisglang-rdna4/python")
FORK_GDN_EXT = Path("/home/user/patcarter883/minisglang-rdna4/tools/vhip_patches/gdn/torch-ext")
LP = "model.language_model.layers"

# Provisional M2-0 bars (DESIGN §8 M2-0 row; kill criterion in the same row)
BAR_BWD_OVER_NATIVE_FWD = 1.5
BAR_DX_COS = 0.9999
BAR_CARD_PEAK_GB = 15.5
KILL_BWD_H_PER_1M = 0.2
CARD_CAP_GB = 16.82


class Dev:
    def __init__(self, device: torch.device):
        self.d = device
        self.cuda = device.type == "cuda"

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.d)

    def reset_peak(self):
        if self.cuda:
            torch.cuda.synchronize(self.d)
            torch.cuda.reset_peak_memory_stats(self.d)

    def peak_gb(self):
        return torch.cuda.max_memory_allocated(self.d) / 1e9 if self.cuda else None

    def alloc_gb(self):
        return torch.cuda.memory_allocated(self.d) / 1e9 if self.cuda else None


def timed_fwd_bwd(fn, x, dy, dev: Dev, reps: int = 1, warmup: int = 1):
    """(y, dx, fwd_s, bwd_s, peak_gb) for y = fn(x) and dx = d<y, dy>/dx; best of reps after warmup."""
    best = None
    out = None
    for i in range(warmup + reps):
        xx = x.detach().clone().requires_grad_(True)
        dev.reset_peak()
        dev.sync()
        t0 = time.perf_counter()
        y = fn(xx)
        dev.sync()
        t1 = time.perf_counter()
        (dx,) = torch.autograd.grad(y, xx, dy)
        dev.sync()
        t2 = time.perf_counter()
        pk = dev.peak_gb()
        if i >= warmup and (best is None or (t1 - t0) + (t2 - t1) < best[0] + best[1]):
            best = (t1 - t0, t2 - t1, pk)
            out = (y.detach(), dx.detach())
        del y, dx, xx
    return out[0], out[1], best[0], best[1], best[2]


def cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def reader_for(args):
    if args.ckpt_35b:
        return hq.CheckpointReader.local(args.ckpt_35b), json.loads((Path(args.ckpt_35b) / "config.json").read_text())
    idx = SNAP35 / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"]
    need = {v for k, v in wm.items() if k.startswith((f"{LP}.{args.gdn_layer}.", f"{LP}.{args.attn_layer}."))}
    return (hq.CheckpointReader.hub(REPO35, index_json=idx, files=sorted(need)),
            json.loads((SNAP35 / "config.json").read_text()))


def fork_shim_factory():
    """The fork's NativeGDNShim (widened to any GDN class by calling it directly), or None if not importable."""
    try:
        if str(FORK_PY) not in sys.path:
            sys.path.insert(0, str(FORK_PY))
        if str(FORK_GDN_EXT) not in sys.path:
            sys.path.insert(0, str(FORK_GDN_EXT))
        from minisgl.gdn.hf_patch import NativeGDNShim  # type: ignore[import-not-found]
        return NativeGDNShim
    except Exception as e:  # noqa: BLE001 - optional capability
        return f"{type(e).__name__}: {str(e)[:160]}"


def run_gdn(args, cfg, rd, dev: Dev, res: dict):
    L = args.gdn_layer
    t0 = time.time()
    layer, tcfg = hq.build_decoder_layer(cfg, L, rd, compute_dtype=torch.bfloat16, device=dev.d,
                                         parts=("input_layernorm", "linear_attn"))
    mixer = layer.linear_attn
    res["gdn"] = r = dict(layer=L, cls=type(mixer).__name__, load_s=time.time() - t0, B=args.gdn_B, T=args.gdn_T,
                          resident_gb=sum(p.numel() * p.element_size() for p in mixer.parameters()) / 1e9)
    d = tcfg.hidden_size
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        x = layer.input_layernorm(torch.randn(args.gdn_B, args.gdn_T, d, generator=g).to(dev.d, torch.bfloat16))
    dy = torch.randn(args.gdn_B, args.gdn_T, d, generator=g).to(dev.d, torch.bfloat16)
    ntok = args.gdn_B * args.gdn_T
    arms = {}

    def record(name, y, dx, tf, tb, pk):
        arms[name] = dict(fwd_s=tf, bwd_s=tb, fwd_us_per_tok=tf / ntok * 1e6, bwd_us_per_tok=tb / ntok * 1e6,
                          peak_gb=pk)
        print(f"  gdn {name:14s} fwd {tf:8.3f}s bwd {tb:8.3f}s  ({tf / ntok * 1e6:9.2f} / {tb / ntok * 1e6:9.2f} "
              f"us/tok) peak {pk if pk is None else round(pk, 3)} GB", flush=True)
        return y, dx

    y_hf, d_hf = record("hf_torch", *timed_fwd_bwd(mixer, x, dy, dev, reps=args.reps))
    gdn.patch_gdn(mixer, impl="ours", max_bh=args.max_bh)
    y_o, d_o = record("ours", *timed_fwd_bwd(mixer, x, dy, dev, reps=args.reps))
    gdn.unpatch_gdn(mixer)
    arms["ours"].update(fwd_rel_err_vs_hf=rel(y_o.float(), y_hf.float()), dx_cos_vs_hf=cos(d_o, d_hf))
    op = gdn.native_gdn_op() if dev.cuda else None
    if op is not None:
        gdn.patch_gdn(mixer, impl="native", native_op=op, max_bh=args.max_bh)
        y_n, d_n = record("native+ours", *timed_fwd_bwd(mixer, x, dy, dev, reps=args.reps))
        gdn.unpatch_gdn(mixer)
        with torch.no_grad():
            dev.sync()
            t1 = time.perf_counter()
            gdn.patch_gdn(mixer, impl="native", native_op=op)
            mixer(x)
            dev.sync()
            arms["native_fwd_only_s"] = time.perf_counter() - t1
            gdn.unpatch_gdn(mixer)
        arms["native+ours"].update(fwd_rel_err_vs_hf=rel(y_n.float(), y_hf.float()), dx_cos_vs_hf=cos(d_n, d_hf))
    else:
        arms["native+ours"] = "gdn_hip not importable here (CPU, or extension not built)"
    shim = fork_shim_factory() if dev.cuda else "fork NativeGDNShim needs the gdn_hip ops (GPU only)"
    if not isinstance(shim, str):
        try:
            fk = shim(mixer).to(dev.d)
            y_f, d_f = record("fork_shim", *timed_fwd_bwd(fk, x, dy, dev, reps=1))
            arms["fork_shim"].update(fwd_rel_err_vs_hf=rel(y_f.float(), y_hf.float()), dx_cos_vs_hf=cos(d_f, d_hf))
            del fk
        except Exception as e:  # noqa: BLE001
            arms["fork_shim"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
    else:
        arms["fork_shim"] = shim
    r["arms"] = arms
    # core-level op count / wall on this device for ours / HF / the fork's reference backward
    if args.core_compare:
        s = dict(H=tcfg.linear_num_key_heads, HV=tcfg.linear_num_value_heads, K=tcfg.linear_key_head_dim,
                 V=tcfg.linear_value_head_dim)
        raw, dout = gdn_bench.make_inputs(args.gdn_B, args.core_T, s)
        raw = {k: v.to(dev.d) for k, v in raw.items()}
        dout = dout.to(dev.d)
        ref = gdn_bench.load_fork_reference()
        core = {}
        for name, fn, extra in (("ours", gdn_bench.run_ours, ()), ("hf", gdn_bench.run_hf, ()),
                                ("fork_ref", gdn_bench.run_fork, (ref,))):
            (_, _, tf, tb), n, nv = gdn_bench.counted(fn, raw, dout, *extra)
            dev.sync()
            t1 = time.perf_counter()
            fn(raw, dout, *extra)
            dev.sync()
            wall = time.perf_counter() - t1
            core[name] = dict(ops=n, ops_per_tok=n / (args.gdn_B * args.core_T), wall_s=wall,
                              ms_per_tok=wall / (args.gdn_B * args.core_T) * 1e3)
            print(f"  core {name:9s} B={args.gdn_B} T={args.core_T}: {n:,} aten ops ({n / (args.gdn_B * args.core_T):.3f}"
                  f"/tok), wall {wall:.2f}s", flush=True)
        core["fork_ref_over_ours"] = dict(ops=core["fork_ref"]["ops"] / core["ours"]["ops"],
                                          wall=core["fork_ref"]["wall_s"] / core["ours"]["wall_s"])
        core["hf_over_ours"] = dict(ops=core["hf"]["ops"] / core["ours"]["ops"], wall=core["hf"]["wall_s"] / core["ours"]["wall_s"])
        r["core"] = core
    del layer, mixer


def run_attn_moe(args, cfg, rd, dev: Dev, res: dict):
    L = args.attn_layer
    t0 = time.time()
    layer, tcfg = hq.build_decoder_layer(cfg, L, rd, compute_dtype=torch.bfloat16, device=dev.d,
                                         experts_per_group=args.experts_per_group)
    load_s = time.time() - t0
    mod = sys.modules[type(layer).__module__]
    rot_cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("TextRotaryEmbedding"))
    rot = rot_cls(tcfg).to(dev.d)
    d = tcfg.hidden_size
    g = torch.Generator().manual_seed(1)
    # attention: one 4k piece (B=1)
    T = args.attn_T
    with torch.no_grad():
        x = layer.input_layernorm(torch.randn(1, T, d, generator=g).to(dev.d, torch.bfloat16))
        pos = torch.arange(T, device=dev.d)[None]
        pe = rot(x, pos[None].expand(3, 1, T))
    dy = torch.randn(1, T, d, generator=g).to(dev.d, torch.bfloat16)

    def attn(xx):
        return layer.self_attn(hidden_states=xx, position_embeddings=pe, attention_mask=None, position_ids=pos)[0]
    y, dx, tf, tb, pk = timed_fwd_bwd(attn, x, dy, dev, reps=args.reps)
    res["attention"] = dict(layer=L, T=T, fwd_s=tf, bwd_s=tb, fwd_us_per_tok=tf / T * 1e6, bwd_us_per_tok=tb / T * 1e6,
                            peak_gb=pk, load_s=load_s, attn_impl=tcfg._attn_implementation,
                            resident_gb=sum(p.numel() * p.element_size() for p in layer.self_attn.parameters()) / 1e9)
    print(f"  attention T={T}: fwd {tf:.3f}s bwd {tb:.3f}s peak {pk}", flush=True)
    # MoE block: the large chunk
    n = args.moe_tokens
    with torch.no_grad():
        xm = layer.post_attention_layernorm(torch.randn(1, n, d, generator=g).to(dev.d, torch.bfloat16))
    dym = torch.randn(1, n, d, generator=g).to(dev.d, torch.bfloat16)
    ex = layer.mlp.experts
    y, dxm, tf, tb, pk = timed_fwd_bwd(layer.mlp, xm, dym, dev, reps=args.reps)
    res["moe"] = m = dict(layer=L, tokens=n, fwd_s=tf, bwd_s=tb, fwd_us_per_tok=tf / n * 1e6, bwd_us_per_tok=tb / n * 1e6,
                          peak_gb=pk, experts_packed_gb=ex.packed_bytes() / 1e9, experts_per_group=ex.experts_per_group)
    print(f"  moe tokens={n}: fwd {tf:.3f}s bwd {tb:.3f}s peak {pk}", flush=True)
    if args.moe_reference:
        # dX against autograd through HF's own experts module holding the dequantised bf16 stacks
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
        with torch.device("meta"):
            hf_ex = Qwen3_5MoeExperts(tcfg)
        E_, I_ = ex.num_experts, ex.intermediate_dim
        hf_ex.gate_up_proj = torch.nn.Parameter(torch.empty(E_, 2 * I_, d, dtype=torch.bfloat16, device=dev.d))
        hf_ex.down_proj = torch.nn.Parameter(torch.empty(E_, d, I_, dtype=torch.bfloat16, device=dev.d))
        with torch.no_grad():
            for e0 in range(0, ex.num_experts, 32):
                ids = torch.arange(e0, min(ex.num_experts, e0 + 32), device=dev.d)
                gu, dn = ex.dequant_experts(ids)
                hf_ex.gate_up_proj[e0:e0 + len(ids)] = gu
                hf_ex.down_proj[e0:e0 + len(ids)] = dn
        for p in hf_ex.parameters():
            p.requires_grad_(False)
        layer.mlp.experts = hf_ex
        y0, dx0, tf0, tb0, _ = timed_fwd_bwd(layer.mlp, xm, dym, dev, reps=1, warmup=0)
        layer.mlp.experts = ex
        m.update(dense_ref_fwd_rel_err=rel(y.float(), y0.float()), dense_ref_dx_cos=cos(dxm, dx0),
                 dense_ref_fwd_s=tf0, dense_ref_bwd_s=tb0)
        print(f"  moe vs dense-bf16 HF experts: fwd rel {m['dense_ref_fwd_rel_err']:.2e}, dX cos {m['dense_ref_dx_cos']:.6f}",
              flush=True)
        del hf_ex
    if args.fp8_check:
        ex.act_quant = "fp8_e4m3"
        with torch.no_grad():
            y8 = layer.mlp(xm)
        ex.act_quant = None
        m["fp8_act_fakequant_rel_diff"] = rel(y8.float(), y.float())
    del layer


def projection(args, rd, res: dict, cfg: dict):
    """Per-card resident + transient projection at a memory split of the 40 layers (packed weights resident)."""
    t = cfg["text_config"]
    nl = t["num_hidden_layers"]
    if not args.ckpt_35b:
        # remote mode: headers of all shards are needed for the per-layer sizes; read them (KB each)
        idx = SNAP35 / "model.safetensors.index.json"
        rd = hq.CheckpointReader.hub(REPO35, index_json=idx)
    sizes = [hq.packed_layer_bytes(rd, i) / 1e9 for i in range(nl)]
    tot = sum(sizes)
    split, acc = 0, 0.0
    for i, s in enumerate(sizes):
        if acc + s > tot * args.split_frac:
            split = i
            break
        acc += s
    ntok = args.gdn_B * args.gdn_T
    ckpt_gb = ntok * t["hidden_size"] * 2 / 1e9                     # one bf16 layer-input checkpoint per layer
    trans = max([v for v in (res.get("gdn", {}).get("arms", {}).get("ours", {}).get("peak_gb"),
                             res.get("attention", {}).get("peak_gb"), res.get("moe", {}).get("peak_gb"))
                 if isinstance(v, (int, float))] or [0.0])
    cards = []
    for c, (a, b) in enumerate(((0, split), (split, nl))):
        res_gb = sum(sizes[a:b])
        cards.append(dict(layers=[a, b], resident_packed_gb=res_gb, checkpoints_gb=ckpt_gb * (b - a),
                          max_layer_transient_gb=trans, projected_peak_gb=res_gb + ckpt_gb * (b - a) + trans))
    res["vram_projection"] = dict(split_frac=args.split_frac, total_packed_body_gb=tot, cards=cards,
                                  note="peak_gb per layer already includes that layer's resident weights; the "
                                       "projection double-counts one layer's weights (conservative). Embeddings "
                                       "on CPU, lm_head + CE not included.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-small", action="store_true", help="CPU smoke run (small chunks, HTTP range reads)")
    ap.add_argument("--ckpt-35b", help="local dir of cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (GPU mode; optional on CPU)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gdn-layer", type=int, default=0)
    ap.add_argument("--attn-layer", type=int, default=3)
    ap.add_argument("--gdn-B", type=int)
    ap.add_argument("--gdn-T", type=int)
    ap.add_argument("--core-T", type=int)
    ap.add_argument("--attn-T", type=int)
    ap.add_argument("--moe-tokens", type=int)
    ap.add_argument("--experts-per-group", type=int, default=32)
    ap.add_argument("--max-bh", type=int, default=None, help="our GDN kernel: batch*heads per group (memory cap)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--split-frac", type=float, default=0.57, help="card-0 share of the packed body (57:43)")
    ap.add_argument("--no-core-compare", dest="core_compare", action="store_false")
    ap.add_argument("--no-moe-reference", dest="moe_reference", action="store_false")
    ap.add_argument("--no-fp8-check", dest="fp8_check", action="store_false")
    ap.add_argument("--no-lease-check", action="store_true")
    args = ap.parse_args()

    if args.cpu_small:
        torch.set_num_threads(args.threads)
        dev = Dev(torch.device("cpu"))
        defaults = dict(gdn_B=4, gdn_T=256, core_T=256, attn_T=512, moe_tokens=1024)
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise SystemExit("GPU mode needs 2 visible cards (run under: gpu-lease -n 2 -- ...) or use --cpu-small")
        if not args.no_lease_check and not os.environ.get("LEASE_ROCR_DEVICES"):
            raise SystemExit("not under gpu-lease (LEASE_ROCR_DEVICES unset): run 'gpu-lease -n 2 -- python ...'")
        if not args.ckpt_35b:
            raise SystemExit("GPU mode reads a local checkpoint: --ckpt-35b DIR")
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
        dev = Dev(torch.device("cuda:0"))
        defaults = dict(gdn_B=4, gdn_T=4096, core_T=1024, attn_T=4096, moe_tokens=16384)
        if args.moe_reference:
            args.moe_reference = False      # a dense bf16 copy of 256 experts (1.5 GB) is skipped on the card
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    res: dict = dict(mode="cpu-small" if args.cpu_small else "gpu", device=str(dev.d), torch=torch.__version__,
                     threads=torch.get_num_threads(), args=vars(args))
    if dev.cuda:
        res["cards"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    t0 = time.time()
    rd, cfg = reader_for(args)
    res["grouped_mm"] = hq.grouped_mm_probe(dev.d)
    print("grouped_mm:", res["grouped_mm"], flush=True)
    run_gdn(args, cfg, rd, dev, res)
    if dev.cuda:
        torch.cuda.empty_cache()
    run_attn_moe(args, cfg, rd, dev, res)
    projection(args, rd, res, cfg)
    res["bytes_read"] = rd.bytes_read
    res["wall_s"] = time.time() - t0
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    # ---- bars ----
    arms = res["gdn"]["arms"]
    ours = arms["ours"]
    bars = []
    if isinstance(arms.get("native+ours"), dict) and "native_fwd_only_s" in arms:
        base, label = arms["native_fwd_only_s"], "native gdn_hip forward"
        bwd = arms["native+ours"]["bwd_s"]
        proxy = False
    else:
        base, label = arms["hf_torch"]["fwd_s"], "PROXY: HF torch forward (native forward unavailable here)"
        bwd = ours["bwd_s"]
        proxy = True
    ratio = bwd / base
    bars.append(dict(bar=f"GDN fixed backward <= {BAR_BWD_OVER_NATIVE_FWD}x native forward per token", value=ratio,
                     baseline=label, passed=ratio <= BAR_BWD_OVER_NATIVE_FWD, proxy=proxy))
    dxc = arms["native+ours"]["dx_cos_vs_hf"] if isinstance(arms.get("native+ours"), dict) else ours["dx_cos_vs_hf"]
    bars.append(dict(bar=f"GDN dX cosine >= {BAR_DX_COS} vs HF torch delta rule (real layer)", value=dxc,
                     passed=dxc >= BAR_DX_COS, proxy=False))
    if dev.cuda:
        pk = max(c["projected_peak_gb"] for c in res["vram_projection"]["cards"])
        bars.append(dict(bar=f"per-card peak <= {BAR_CARD_PEAK_GB} GB at the planned split (projection)", value=pk,
                         passed=pk <= BAR_CARD_PEAK_GB, proxy=False))
    else:
        bars.append(dict(bar=f"per-card peak <= {BAR_CARD_PEAK_GB} GB", value=None, passed=None,
                         proxy=True, note="not measurable on CPU; see vram_projection for resident sizes"))
    n_gdn = sum(x == "linear_attention" for x in cfg["text_config"]["layer_types"])
    h_per_1m = bwd / (args.gdn_B * args.gdn_T) * 1e6 * n_gdn / 3600
    bars.append(dict(bar=f"kill check: GDN backward < {KILL_BWD_H_PER_1M} h per 1M tokens on the 35B ({n_gdn} GDN layers)",
                     value=h_per_1m, passed=h_per_1m < KILL_BWD_H_PER_1M, proxy=not dev.cuda,
                     note=None if dev.cuda else "CPU timing; the bar is defined for the owner's GPUs"))
    res["bars"] = bars
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print("\n==== M2-0 bars (" + res["mode"] + ") ====")
    for b in bars:
        st = "N/A " if b["passed"] is None else ("PASS" if b["passed"] else "FAIL")
        tag = " [proxy/CPU]" if b.get("proxy") else ""
        val = "-" if b["value"] is None else f"{b['value']:.4g}"
        print(f"{st} {b['bar']}: {val}{tag}" + (f"  ({b['baseline']})" if "baseline" in b else ""))
    hard = [b for b in bars if b["passed"] is not None and not b.get("proxy")]
    overall = all(b["passed"] for b in hard)
    print(("OVERALL (non-proxy bars): PASS" if overall else "OVERALL (non-proxy bars): FAIL")
          + ("" if dev.cuda else "  -- CPU smoke run: GPU bars must be measured on the owner's cards"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
