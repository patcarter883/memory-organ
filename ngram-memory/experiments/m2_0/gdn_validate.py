"""M2-0 GDN validation: the native-forward + HF-recompute backward fix (gdn_native.NativeFwdHFBwd) against HF's torch
delta rule at real Qwen3.6-35B-A3B shapes, beside our hand-written backward and the fork's reference backward
(DESIGN §4.6, §7.5 #1, §8 M2-0 / M2; review F1, N9).

(a) core   the delta-rule core alone at --core-cases (B:T; default 4:1024,4:4096) (35B GDN geometry H=16, HV=32, K=V=128). Arms:
             hf_ref        HF prep + torch_chunk_gated_delta_rule (``__wrapped__``), plain autograd: the reference
             fix           native forward + HF torch-chunk recompute backward (NativeFwdHFBwd)
             fix_g1        the same, recomputing one sequence at a time (bwd_group=1; bounds backward memory)
             native_ours   native forward + our chunked WY backward (gdn.RawGDN)
             ours          our torch forward + our backward
             fork_ref      native forward + the fork's reference recompute (B folded into heads, chunk max(4,64//B))
             fix_wmma      (GPU only) gdn_prefill_wmma forward + HF recompute backward
           Per arm: forward rel. error and dq/dk/dv/da/db/dA_log/ddt_bias cosine vs hf_ref; aten ops per token
           counted separately for forward and backward (TorchDispatchMode, as scripts/gdn_bench.py); wall time;
           peak live bytes of tensors allocated in the forward / backward (a storage tracker, see LiveBytes);
           peak RSS of the arm's own process (CPU) or max_memory_allocated (GPU).
(b) layer  the real layer-0 GDN mixer of cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (dense F16 weights, read from a local
           cache made with --fetch; A_log / dt_bias kept fp32), whole-mixer forward and dX vs HF's forward, at
           B x T = --layer-B x --layer-T, in fp32 and bf16 (dense weights cast to bf16, as served). Arms: hf (HF's
           unpatched module forward), hf_patched (our patch with fwd=hf, bwd=hf), fix, native_ours, ours, fork_ref,
           and on CPU fix_exact / fix_exact_b32 (the exact per-token recurrence stand-in, i.e. the native kernel's
           math, as the forward; _b32 recomputes with sigmoid(b) in fp32 as that kernel does), on GPU fix_b32,
           fork_shim (the fork's NativeGDNShim through gdn_native.patch_fork_shims) and fix_wmma. bf16 rows also
           compare every arm with 'truth' (fp32 mixer, bf16-rounded weights, same bf16 input).
           Bars (DESIGN §8 M2): forward rel. error <= 1e-3 and dX cosine >= 0.9999 vs HF's torch delta rule.

(c) exact  native-math accuracy against an fp64 ground truth (verifier D2): small shapes (B=2, T in --exact-T,
           GQA 2:4 heads with the real K=V=128), inputs drawn in fp64 and rounded to fp32 / bf16; the truth is an
           independent fp64 per-token recurrence (gdn.recurrent_gated_delta_rule_reference, fp64 sigmoid and gate)
           differentiated by fp64 autograd. Arms: hf_ref, fix (native op; on CPU the exact recurrence stand-in
           gdn_native.recurrent_standin_op, i.e. the native kernel's math), fix_hfbeta (recompute with HF's
           activation-dtype sigmoid), fix_g1, native_ours, ours, fork_ref (+ fix_wmma on GPU). This is the only
           CPU part whose 'native' forward is native math rather than HF's own forward.
(d) model  a model-level train step (verifier D1): a 4-layer Qwen3_5MoeForCausalLM with the 35B GDN geometry
           (random weights, 4 experts, vocab 1024), called as ``model(ids, labels=ids)`` with HF's default
           use_cache=True; asserts every GDN layer ran the patched kernel (cfg.calls == GDN layers, fallbacks 0)
           and that loss and grads match the unpatched model.

On CPU the native op does not exist: in parts (a) and (b) every 'native' forward runs gdn_native.hf_standin_op
(HF's torch forward behind gdn_hip.gdn_prefill's exact signature and state layout), so those numbers are WIRING
checks of the autograd.Functions and the backward, not native-math accuracy (that is part (c)); forward timings
of 'native' arms there are HF's torch forward, and the M2-0 timing bars are N/A.
--gpu selects the real gdn_prefill (and gdn_prefill_wmma) from the fork and evaluates the M2-0 bars:
backward <= 1.5x the native forward per token, and the GDN backward below 0.2 h per 1M tokens over 30 layers.

Every arm runs in its own child process (fresh allocator, own peak RSS, nothing left over between arms); the child
saves its output and gradients to --work, the parent compares them with the reference arm and deletes them.

Usage (CPU, this container):
    python experiments/m2_0/gdn_validate.py --fetch --cache-dir <dir>                 # 67.4 MB of layer-0 weights
    python experiments/m2_0/gdn_validate.py --cache-dir <dir> --core-cases 4:1024,1:4096 --layer-B 1 \
        --out experiments/m2_0/results/gdn/cpu.json      # B=1 at T=4096 keeps one arm under ~4 GB of host RAM
GPU (owner's machine, serve stopped, one card):
    gpu-lease -n 1 -- python experiments/m2_0/gdn_validate.py --gpu --cache-dir <dir> --out gdn_gpu.json
(the owner's one-command run is experiments/m2_0/run_owner.sh, which calls this with the M2-0 settings)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
REV = "f1e8415c4f29508fe281da4e256df26573213bfd"
SNAP = Path(f"/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/{REV}")
LP = "model.language_model.layers"
SHAPE35 = dict(H=16, HV=32, K=128, V=128, layers=30, hidden=2048)   # checked against config.json when present
BAR_FWD_REL = 1e-3
BAR_DX_COS = 0.9999
BAR_BWD_OVER_NATIVE_FWD = 1.5
KILL_BWD_H_PER_1M = 0.2
GRAD_NAMES = ("q", "k", "v", "a", "b", "A_log", "dt_bias")
CORE_ARMS = {
    "hf_ref": dict(fwd="hf", bwd="hf"),
    "fix": dict(fwd="native", bwd="hf"),
    "fix_g1": dict(fwd="native", bwd="hf", bwd_group=1),
    "native_ours": dict(fwd="native", bwd="ours"),
    "ours": dict(fwd="ours", bwd="ours"),
    "fork_ref": dict(fwd="native", bwd="fork_ref"),
    "fix_wmma": dict(fwd="native", bwd="hf", native_kind="wmma"),          # GPU only
    "fix_b32": dict(fwd="native", bwd="hf", beta_fp32=True),            # recompute with the kernel's fp32 sigmoid
    "fix_hfbeta": dict(fwd="native", bwd="hf", beta_fp32=False),        # recompute with HF's activation-dtype sigmoid
    "fix_exact": dict(fwd="native", bwd="hf", exact_standin=True),       # CPU: per-token recurrence forward
    "fix_exact_b32": dict(fwd="native", bwd="hf", exact_standin=True, beta_fp32=True),
}
# 'truth' (bf16 rows only): the whole mixer in fp32 with the bf16-rounded weights and the same bf16 input
# 'hf_b32': HF's math with sigmoid(b) in fp32 (what the native kernel computes), a second reference for bf16 rows
LAYER_ARMS_CPU = ("hf", "truth", "hf_b32", "hf_patched", "fix", "native_ours", "ours", "fork_ref", "fix_exact",
                  "fix_exact_b32")
LAYER_ARMS_GPU = ("hf_patched", "truth", "hf_b32", "fix", "fix_b32", "native_ours", "ours", "fork_ref", "fix_wmma",
                  "fork_shim")
EXTRA_REFS = ("truth", "hf_b32")
FIX_ARMS = ("fix", "fix_g1", "fix_b32")           # candidate GPU defaults for the fixed backward (bars per arm)
EXACT_ARMS_CPU = ("hf_ref", "fix", "fix_hfbeta", "fix_g1", "native_ours", "ours", "fork_ref")
EXACT_ARMS_GPU = EXACT_ARMS_CPU + ("fix_wmma",)


# ------------------------------------------------------------------------------------------------------------
# small utilities (no torch import at module level: the parent stays light)
# ------------------------------------------------------------------------------------------------------------

def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def shape35(cache_dir: Path | None) -> dict:
    for p in ([cache_dir / "b1_gdn_L0" / "config.json"] if cache_dir else []) + [SNAP / "config.json"]:
        if p.exists():
            t = json.loads(p.read_text())
            t = t.get("text_config", t)
            s = dict(H=t["linear_num_key_heads"], HV=t["linear_num_value_heads"], K=t["linear_key_head_dim"],
                     V=t["linear_value_head_dim"], layers=sum(x == "linear_attention" for x in t["layer_types"]),
                     hidden=t["hidden_size"])
            if s != SHAPE35:
                raise SystemExit(f"config {p} disagrees with the 35B GDN geometry {SHAPE35}: {s}")
            return dict(s, source=str(p))
    return dict(SHAPE35, source="constants (no config.json found)")


# ------------------------------------------------------------------------------------------------------------
# --fetch: layer-0 GDN tensors by HTTP range read, cached with sha256 + revision
# ------------------------------------------------------------------------------------------------------------

def fetch_layer0(cache_dir: Path, max_bytes: int) -> dict:
    sys.path.insert(0, str(ROOT))
    import torch  # noqa: F401
    from safetensors.torch import save_file

    from ngram_memory.backends.graft import host_quant as hq
    out = cache_dir / "b1_gdn_L0"
    man_p = out / "MANIFEST.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        print(f"cache present: {out} ({man['weight_bytes']:,} B, revision {man['revision']})")
        return man
    idx = SNAP / "model.safetensors.index.json"
    if not idx.exists():
        raise SystemExit(f"need the cached index {idx} (small json; hf_hub_download it first)")
    wm = json.loads(idx.read_text())["weight_map"]
    names = sorted(n for n in wm if n.startswith((f"{LP}.0.linear_attn.", f"{LP}.0.input_layernorm.")))
    shards = sorted({wm[n] for n in names})
    rd = hq.CheckpointReader.hub(REPO, revision=REV, index_json=idx, files=shards)
    info = {n: rd.files[wm[n]].info(n) for n in names}
    total = sum(b - a for _, _, a, b in info.values())
    if total > max_bytes:
        raise SystemExit(f"refusing: {total:,} B > --max-bytes {max_bytes:,}")
    hdr_bytes = rd.bytes_read
    t0 = time.time()
    tens = rd.get_many(names)
    fetched = rd.bytes_read - hdr_bytes
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    per = {}
    for n in names:
        dt, shape, a, b = info[n]
        x = tens[n].contiguous()
        raw = x.view(-1).view(torch.uint8).numpy().tobytes() if x.numel() else b""
        if len(raw) != b - a:
            raise SystemExit(f"{n}: {len(raw)} bytes != header span {b - a}")
        per[n] = dict(shard=wm[n], dtype=str(dt).replace("torch.", ""), shape=shape, src_offsets=[a, b],
                      nbytes=b - a, sha256=sha256_bytes(raw))
        tensors[n] = x
    save_file(tensors, str(out / "model.safetensors"), metadata={"repo": REPO, "revision": REV})
    shutil.copyfile(SNAP / "config.json", out / "config.json")
    man = dict(repo=REPO, revision=REV, purpose="M2-0 B1 GDN validation: layer-0 linear_attn + input_layernorm",
               created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), weight_bytes=total,
               bytes_downloaded=dict(headers=hdr_bytes, weights=fetched), fetch_s=time.time() - t0,
               config_sha256=sha256_bytes((out / "config.json").read_bytes()),
               file_sha256=sha256_bytes((out / "model.safetensors").read_bytes()), tensors=per)
    # verify the written file reproduces every tensor's bytes
    lf = hq.SafetensorsFile(str(out / "model.safetensors"))
    for n, e in per.items():
        _, _, a, b = lf.info(n)
        if sha256_bytes(lf.read(a, b)) != e["sha256"]:
            raise SystemExit(f"cache verification failed for {n}")
    lf.close()
    man_p.write_text(json.dumps(man, indent=1))
    print(f"fetched {len(names)} tensors, {total:,} weight bytes ({fetched:,} read) -> {out}")
    return man


# ------------------------------------------------------------------------------------------------------------
# child: one arm in its own process
# ------------------------------------------------------------------------------------------------------------

def _child(args) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import gdn_bench  # noqa: E402  (OpCount, make_inputs: the same counting method as S1/rt-feas)
    from ngram_memory.backends.graft import gdn_native as gn

    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if args.gpu else "cpu")
    cuda = dev.type == "cuda"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    class LiveBytes(TorchDispatchMode):
        """Bytes of storages created by ops while the mode is active and still alive (storage weak refs, so
        tensors saved for backward by the C++ autograd graph count until the graph frees them); views of storages
        that existed before the mode are not counted. Peak over the region; allocator-internal workspaces and
        caching are not seen (GPU: use max_memory_allocated)."""

        def __init__(self):
            super().__init__()
            self.live: dict[int, tuple] = {}
            self.ext: dict[int, object] = {}
            self.cur = self.peak = 0

        def _sweep(self):
            dead = [k for k, (r, _) in self.live.items() if r.expired()]
            for k in dead:
                self.cur -= self.live.pop(k)[1]

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            for t in tree_flatten((args, kwargs or {}))[0]:
                if isinstance(t, torch.Tensor):
                    k = t.untyped_storage()._cdata
                    if k not in self.live and (k not in self.ext or self.ext[k].expired()):
                        self.ext[k] = StorageWeakRef(t.untyped_storage())
            out = func(*args, **(kwargs or {}))
            self._sweep()
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

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)

    def reset_peak():
        if cuda:
            sync()
            torch.cuda.reset_peak_memory_stats(dev)

    def gpu_peak():
        return torch.cuda.max_memory_allocated(dev) if cuda else None

    def op_for(kw):
        if kw.get("fwd") != "native":
            return None
        if kw.get("exact_standin"):
            return gn.recurrent_standin_op
        if cuda:
            op = gn.resolve_native_op(kw.get("native_kind", "recurrent"))
            if op is None:
                raise SystemExit("gdn_hip not importable (build the fork's extension, or set NGM_MINISGL_FORK)")
            return op
        return gn.hf_standin_op

    def core_kw(kw):
        return {k: v for k, v in kw.items() if k not in ("exact_standin", "native_kind")} | (
            {"native_kind": kw["native_kind"]} if "native_kind" in kw else {})

    res: dict = dict(arm=args.arm, part=args.part, dtype=args.dtype, device=str(dev), threads=args.threads,
                     load_avg_start=os.getloadavg())
    if args.part == "core":
        kw = CORE_ARMS[args.arm]
        s = json.loads(args.shape)
        raw, dout = gdn_bench.make_inputs(args.B, args.T, s, seed=0)
        raw = {k: (v.to(dtype) if k in "qkvab" else v.float()).to(dev) for k, v in raw.items()}
        dout = dout.to(dtype).to(dev)
        op = op_for(kw)
        ck = core_kw(kw)

        def fwd(ins):
            return gn.gdn_core(*ins, native_op=op, **{k: v for k, v in ck.items()})

        def fresh():
            return [raw[k].detach().clone().requires_grad_(True) for k in ("q", "k", "v", "a", "b", "A_log", "dt_bias")]
        # 1) counted + live-bytes run (outputs kept for the comparison)
        ins = fresh()
        reset_peak()
        c1, m1 = gdn_bench.OpCount(), LiveBytes()
        with c1, m1:
            o = fwd(ins)
        fwd_retained = m1.cur
        c2, m2 = gdn_bench.OpCount(), LiveBytes()
        with c2, m2:
            grads = torch.autograd.grad(o, ins, dout)
        ntok = args.B * args.T
        res.update(ops=dict(fwd=c1.n, bwd=c2.n, fwd_per_tok=c1.n / ntok, bwd_per_tok=c2.n / ntok,
                            fwd_nonview_per_tok=(c1.n - c1.views) / ntok, bwd_nonview_per_tok=(c2.n - c2.views) / ntok),
                   live_bytes=dict(fwd_peak=m1.peak, fwd_retained_incl_output=fwd_retained, bwd_peak_extra=m2.peak),
                   gpu_peak_bytes_counted_run=gpu_peak())
        torch.save(dict(o=o.detach().cpu(), grads=[None if g is None else g.detach().cpu() for g in grads]),
                   args.save)
        del o, grads, ins, c1, c2, m1, m2
        # 2) timed runs, no dispatch modes
        best = None
        for i in range(args.warmup + args.reps):
            ins = fresh()
            reset_peak()
            sync()
            t0 = time.perf_counter()
            o = fwd(ins)
            sync()
            t1 = time.perf_counter()
            fwd_peak = gpu_peak()
            gr = torch.autograd.grad(o, ins, dout)
            sync()
            t2 = time.perf_counter()
            if i >= args.warmup and (best is None or (t1 - t0) + (t2 - t1) < sum(best[:2])):
                best = (t1 - t0, t2 - t1, fwd_peak, gpu_peak())
            del o, gr, ins
        res.update(fwd_s=best[0], bwd_s=best[1], fwd_us_per_tok=best[0] / ntok * 1e6,
                   bwd_us_per_tok=best[1] / ntok * 1e6, gpu_peak_fwd_bytes=best[2], gpu_peak_fwdbwd_bytes=best[3])
    else:
        res.update(_child_layer(args, torch, gn, dev, dtype, sync, reset_peak, gpu_peak, op_for, core_kw))
    res["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    res["load_avg_end"] = os.getloadavg()
    Path(args.result).write_text(json.dumps(res))


def _child_layer(args, torch, gn, dev, dtype, sync, reset_peak, gpu_peak, op_for, core_kw) -> dict:
    from torch import nn

    from ngram_memory.backends.graft import host_quant as hq
    ck = Path(args.layer_ckpt)
    cfg = json.loads((ck / "config.json").read_text())
    rd = hq.CheckpointReader.local(ck)
    layer, _ = hq.build_decoder_layer(cfg, 0, rd, compute_dtype=dtype, device=dev,
                                      parts=("input_layernorm", "linear_attn"))   # the rest stays on meta
    mixer = layer.linear_attn
    for leaf in ("A_log", "dt_bias"):     # exact F16 -> fp32, as the fork's native layer keeps them
        mixer._parameters[leaf] = nn.Parameter(rd.get(f"{LP}.0.linear_attn.{leaf}").float().to(dev),
                                               requires_grad=False)
    out = dict(cls=type(mixer).__name__, weight_dtype=str(mixer.in_proj_qkv.weight.dtype),
               A_log_dtype=str(mixer.A_log.dtype), B=args.B, T=args.T)
    g = torch.Generator().manual_seed(1234)
    d = layer.input_layernorm.weight.shape[0]
    with torch.no_grad():
        x = layer.input_layernorm(torch.randn(args.B, args.T, d, generator=g).to(dev, dtype))
    dy = torch.randn(args.B, args.T, d, generator=g).to(dev, dtype)
    holder = None
    if args.arm == "truth":
        if dtype != torch.bfloat16:
            raise SystemExit("truth arm is for bf16 rows")
        lf, _ = hq.build_decoder_layer(cfg, 0, rd, compute_dtype=torch.float32, device=dev,
                                       parts=("input_layernorm", "linear_attn"))
        with torch.no_grad():
            for n, p in lf.linear_attn.named_parameters():
                p.copy_(dict(mixer.named_parameters())[n].float())
        fn = lf.linear_attn
        x, dy = x.float(), dy.float()
    elif args.arm == "hf":
        fn = mixer
    elif args.arm in ("hf_patched", "hf_b32"):
        gn.patch_gdn_native(mixer, fwd="hf", bwd="hf", beta_fp32=args.arm == "hf_b32")
        fn = mixer
    elif args.arm == "fork_shim":
        os.environ.setdefault("GDN_HIP_WMMA_PREFILL", "0")     # exact recurrent prefill unless the caller set it
        holder = nn.Module()
        holder.linear_attn = mixer
        rec = gn.patch_fork_shims(holder)
        out.update(fork_shim=dict(patched=len(rec), classes=rec.class_names,
                                  GDN_HIP_WMMA_PREFILL=os.environ.get("GDN_HIP_WMMA_PREFILL")))
        fn = holder.linear_attn
    else:
        kw = CORE_ARMS[args.arm]
        gn.patch_gdn_native(mixer, native_op=op_for(kw), **{k: v for k, v in core_kw(kw).items()})
        fn = mixer
    best = None
    for i in range(args.warmup + args.reps):
        xx = x.detach().clone().requires_grad_(True)
        reset_peak()
        sync()
        t0 = time.perf_counter()
        y = fn(xx)
        sync()
        t1 = time.perf_counter()
        (dx,) = torch.autograd.grad(y, xx, dy.to(y.dtype))
        sync()
        t2 = time.perf_counter()
        if i >= args.warmup and (best is None or (t1 - t0) + (t2 - t1) < sum(best[:2])):
            best = (t1 - t0, t2 - t1, gpu_peak())
            keep = (y.detach().cpu(), dx.detach().cpu())
        del y, dx, xx
    ntok = args.B * args.T
    torch.save(dict(o=keep[0], grads=[keep[1]]), args.save)
    out.update(fwd_s=best[0], bwd_s=best[1], fwd_us_per_tok=best[0] / ntok * 1e6, bwd_us_per_tok=best[1] / ntok * 1e6,
               gpu_peak_fwdbwd_bytes=best[2])
    return out


# ------------------------------------------------------------------------------------------------------------
# (c) exact: native math vs an fp64 ground truth;  (d) model: a model-level train step
# ------------------------------------------------------------------------------------------------------------

def _truth_fp64(raw64: dict):
    """fp64 ground truth of the native kernel's math (independent of HF's chunked code): l2norm(eps 1e-6),
    scale K**-0.5, fp64 sigmoid / softplus gate, GQA by repeat_interleave, per-token recurrence; fp64 autograd."""
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from ngram_memory.backends.graft import gdn
    ins = [raw64[k].detach().clone().requires_grad_(True) for k in ("q", "k", "v", "a", "b", "A_log", "dt_bias")]
    q, k, v, a, b, A, dt = ins
    rep = v.shape[2] // q.shape[2]
    g = -A.exp() * F.softplus(a + dt)
    o, _ = gdn.recurrent_gated_delta_rule_reference(q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2), v, g,
                                                    b.sigmoid())
    return ins, o


def part_exact(args, res: dict) -> None:
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from ngram_memory.backends.graft import gdn_native as gn
    dev = torch.device("cuda" if args.gpu else "cpu")
    torch.set_num_threads(args.threads)
    arms = args.exact_arms.split(",") if args.exact_arms else list(EXACT_ARMS_GPU if args.gpu else EXACT_ARMS_CPU)
    rows = []
    worst: dict = {}

    def op_for(arm):
        kw = CORE_ARMS.get(arm, {})
        if kw.get("fwd") != "native":
            return None
        if args.gpu:
            op = gn.resolve_native_op(kw.get("native_kind", "recurrent"))
            if op is None:
                raise SystemExit("gdn_hip not importable (build the fork's extension, or set NGM_MINISGL_FORK)")
            return op
        return gn.recurrent_standin_op                  # CPU: the native kernel's exact math

    for T in (int(x) for x in args.exact_T.split(",")):
        g = torch.Generator().manual_seed(100 + T)
        B, H, HV, K, V = 2, 2, 4, SHAPE35["K"], SHAPE35["V"]
        r = lambda *sh: torch.randn(*sh, generator=g, dtype=torch.float64)  # noqa: E731
        raw64 = dict(q=r(B, T, H, K), k=r(B, T, H, K), v=r(B, T, HV, V), a=r(B, T, HV), b=r(B, T, HV),
                     A_log=torch.log(torch.rand(HV, generator=g, dtype=torch.float64) * 15 + 1), dt_bias=r(HV))
        dout64 = r(B, T, HV, V)
        for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            # every arm and the truth see the SAME rounded inputs (A_log / dt_bias fp32, as the native layer keeps)
            rnd = {k_: (x.to(dt) if k_ in "qkvab" else x.float()) for k_, x in raw64.items()}
            tins, to = _truth_fp64({k_: x.double() for k_, x in rnd.items()})
            tgrads = torch.autograd.grad(to, tins, dout64.to(dt).double())
            row = dict(T=T, B=B, dtype=dt_name, arms={})
            for arm in arms:
                kw = dict(CORE_ARMS[arm])
                kw.pop("exact_standin", None)
                try:
                    op = op_for(arm)
                    ins = [rnd[k_].detach().clone().to(dev).requires_grad_(True)
                           for k_ in ("q", "k", "v", "a", "b", "A_log", "dt_bias")]
                    o = gn.gdn_core(*ins, native_op=op, **kw)
                    grads = torch.autograd.grad(o, ins, dout64.to(dt).to(dev))
                except Exception as e:  # noqa: BLE001 - recorded per arm
                    row["arms"][arm] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
                    continue
                cosd, zero = {}, {}
                for n, x, y in zip(GRAD_NAMES, grads, tgrads):
                    if y.norm() == 0:          # e.g. T=1: the gates cannot act on a zero initial state
                        zero[n] = x.double().abs().max().item()
                        cosd[n] = None
                    else:
                        cosd[n] = F.cosine_similarity(x.double().cpu().flatten(), y.flatten(), dim=0).item()
                ent = dict(fwd_rel_vs_fp64=((o.double().cpu() - to).norm() / to.norm()).item(),
                           grad_cos_vs_fp64=cosd, min_grad_cos=min(c for c in cosd.values() if c is not None))
                if zero:
                    ent["zero_truth_grads_max_abs"] = zero
                row["arms"][arm] = ent
                w = worst.setdefault(f"{arm}/{dt_name}", dict(fwd_rel=0.0, min_grad_cos=1.0))
                w["fwd_rel"] = max(w["fwd_rel"], ent["fwd_rel_vs_fp64"])
                w["min_grad_cos"] = min(w["min_grad_cos"], ent["min_grad_cos"])
            rows.append(row)
            print(f"  exact T={T:4d} {dt_name}: " + "  ".join(
                f"{a_}: {e['fwd_rel_vs_fp64']:.1e}/{e['min_grad_cos']:.6f}" if "fwd_rel_vs_fp64" in e else f"{a_}: ERR"
                for a_, e in row["arms"].items()), flush=True)
    res["exact"] = dict(note=("fp64 ground truth of the native kernel's math; 'native' = "
                              + ("gdn_hip ops" if args.gpu else "gdn_native.recurrent_standin_op (exact recurrence)")
                              + "; per arm: forward rel. error / min gradient cosine over the 7 inputs"),
                        heads=dict(H=2, HV=4, K=SHAPE35["K"], V=SHAPE35["V"]), rows=rows, worst=worst)


class _PureTorchHF:
    """Pin HF's GDN kernel functions to their pure-torch bodies (no fla, no hub kernels; fla hangs on gfx1201),
    so the unpatched reference never leaves torch."""
    NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update")

    def __enter__(self):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        self.m, self.old = m, {n: getattr(m, n) for n in self.NAMES}
        for n, f in self.old.items():
            setattr(m, n, getattr(f, "__wrapped__", f))
        return self

    def __exit__(self, *exc):
        for n, f in self.old.items():
            setattr(self.m, n, f)


def part_model(args, res: dict) -> None:
    """A model-level train step through patch_gdn_native with HF's default use_cache=True (verifier D1)."""
    with _PureTorchHF():
        _part_model(args, res)


def _part_model(args, res: dict) -> None:
    import torch
    sys.path.insert(0, str(ROOT))
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    from ngram_memory.backends.graft import gdn_native as gn
    dev = torch.device("cuda" if args.gpu else "cpu")
    t = None
    for pth in [args.cache_dir / "cyankiwi_35b" / "config.json", args.cache_dir / "b1_gdn_L0" / "config.json",
                SNAP / "config.json"]:
        if pth.exists():
            t = json.loads(pth.read_text())["text_config"]
            break
    if t is None:
        res["model"] = dict(skipped="no 35B config.json found")
        return
    t = {k: v for k, v in t.items() if k not in ("model_type", "dtype")}
    t.update(num_hidden_layers=4, layer_types=t["layer_types"][:4], vocab_size=1024, num_experts=4,
             num_experts_per_tok=2, mtp_num_hidden_layers=0)
    cfg = Qwen3_5MoeTextConfig(**t)
    torch.manual_seed(0)
    dtype = torch.bfloat16 if args.gpu else torch.float32
    model = Qwen3_5MoeForCausalLM(cfg).to(dev, dtype).train()
    for n, p in model.named_parameters():      # as in graft training: frozen host, gradient flows to the inputs
        p.requires_grad_(n.endswith("embed_tokens.weight"))
    n_gdn = sum(x == "linear_attention" for x in cfg.layer_types)
    ids = torch.randint(3, 1000, (2, args.model_T), generator=torch.Generator().manual_seed(1)).to(dev)

    def step():
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)                    # HF default use_cache=True: a DynamicCache is built
        out.loss.backward()
        gr = torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])
        return out.loss.item(), gr
    l0, g0 = step()
    rows = {}
    arms = ["fix", "fix_g1", "native_ours", "ours"] + (["fork_ref"] if (fork_paths_ok()) else [])
    from ngram_memory.backends.graft.gdn_native import hf_standin_op, resolve_native_op
    for arm in arms:
        kw = dict(CORE_ARMS[arm])
        kw.pop("exact_standin", None)
        if kw.get("fwd") == "native":
            kw["native_op"] = (resolve_native_op(kw.pop("native_kind", "recurrent")) if args.gpu else hf_standin_op)
        try:
            pc = gn.patch_gdn_native(model, **kw)
            l1, g1 = step()
            import torch.nn.functional as F
            rows[arm] = dict(calls=pc.calls, fallbacks=pc.fallbacks, fresh_cache_calls=pc.fresh_cache_calls,
                             gdn_layers=n_gdn, loss_rel=abs(l1 - l0) / abs(l0),
                             grad_cos=F.cosine_similarity(g1.double(), g0.double(), dim=0).item(),
                             ok=pc.calls == n_gdn and pc.fallbacks == 0)
        except Exception as e:  # noqa: BLE001
            rows[arm] = dict(error=f"{type(e).__name__}: {str(e)[:300]}", ok=False)
        finally:
            gn.unpatch_gdn_native(model)
        print(f"  model step {arm:12s} {json.dumps(rows[arm])}", flush=True)
    res["model"] = dict(layers=cfg.layer_types, T=args.model_T, dtype=str(dtype), use_cache_default=cfg.use_cache,
                        grads="frozen host; gradient of the loss w.r.t. the input embedding table (dX path)",
                        arms=rows, all_ok=all(r.get("ok") for r in rows.values()))


def fork_paths_ok() -> bool:
    sys.path.insert(0, str(ROOT))
    from ngram_memory.backends.graft import gdn_native as gn
    return (gn.fork_paths()[1] / "gdn_hip" / "autograd.py").exists()


# ------------------------------------------------------------------------------------------------------------
# parent
# ------------------------------------------------------------------------------------------------------------

def run_child(args, part: str, arm: str, B: int, T: int, dtype: str, shape: dict, work: Path, need_gb: float,
              layer_ckpt: str | None = None) -> dict:
    av = mem_available_gb()
    if not args.gpu and av == av and av < need_gb:
        return dict(arm=arm, skipped=f"MemAvailable {av:.1f} GB < {need_gb:.1f} GB needed (est.)")
    save, result = work / f"{part}_{arm}_{dtype}_{T}.pt", work / f"{part}_{arm}_{dtype}_{T}.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_child", "--part", part, "--arm", arm, "--B", str(B),
           "--T", str(T), "--dtype", dtype, "--shape", json.dumps(shape), "--threads", str(args.threads),
           "--reps", str(args.reps), "--warmup", str(args.warmup), "--save", str(save), "--result", str(result)]
    if layer_ckpt:
        cmd += ["--layer-ckpt", layer_ckpt]
    if args.gpu:
        cmd.append("--gpu")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0 or not result.exists():
        return dict(arm=arm, error=f"exit {p.returncode}", stderr=p.stderr[-2000:], mem_available_gb=av)
    r = json.loads(result.read_text())
    r.update(wall_child_s=time.time() - t0, mem_available_gb_before=av, save=str(save))
    result.unlink()
    return r


def compare(r: dict, ref: dict, names) -> None:
    import torch
    import torch.nn.functional as F
    a, b = torch.load(r["save"]), torch.load(ref["save"])
    r["fwd_rel_err"] = ((a["o"].double() - b["o"].double()).norm() / b["o"].double().norm()).item()
    r["fwd_max_abs"] = (a["o"].double() - b["o"].double()).abs().max().item()
    r["grad_cos"] = {n: (None if x is None or y is None else
                         F.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0).item())
                     for n, x, y in zip(names, a["grads"], b["grads"])}
    r["grad_rel"] = {n: (None if x is None or y is None else
                         ((x.double() - y.double()).norm() / y.double().norm()).item())
                     for n, x, y in zip(names, a["grads"], b["grads"])}


def _fmt(r: dict) -> str:
    if "skipped" in r or "error" in r:
        return f"{r['arm']:12s} {r.get('skipped') or r.get('error')}"
    s = (f"{r['arm']:12s} fwd {r['fwd_s']:7.2f}s bwd {r['bwd_s']:7.2f}s "
         f"({r['fwd_us_per_tok']:8.1f}/{r['bwd_us_per_tok']:8.1f} us/tok)")
    if "ops" in r:
        s += f"  ops/tok fwd {r['ops']['fwd_per_tok']:7.3f} bwd {r['ops']['bwd_per_tok']:7.3f}"
        lb = r["live_bytes"]
        s += f"  live GB fwd {lb['fwd_retained_incl_output'] / 1e9:5.2f} bwd+ {lb['bwd_peak_extra'] / 1e9:5.2f}"
    s += f"  rss {r['peak_rss_bytes'] / 1e9:5.2f} GB"
    if r.get("gpu_peak_fwdbwd_bytes"):
        s += f"  vram {r['gpu_peak_fwdbwd_bytes'] / 1e9:5.2f} GB"
    if "fwd_rel_err" in r and "grad_cos" in r:
        mc = min(v for v in r["grad_cos"].values() if v is not None)
        s += f"  | fwd rel {r['fwd_rel_err']:.2e}  min grad cos {mc:.9f}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="fetch the layer-0 GDN tensors into --cache-dir and exit")
    ap.add_argument("--max-bytes", type=int, default=100_000_000)
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("NGM_M2_0_CACHE", "m2_0_cache")))
    ap.add_argument("--layer-ckpt", type=Path, default=None,
                    help="dir with config.json + model.safetensors(.index.json) holding layer 0 (default "
                         "<cache-dir>/b1_gdn_L0)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/m2_0/results/gdn/gdn_validate.json")
    ap.add_argument("--work", type=Path, default=None, help="scratch for per-arm tensors (deleted after compare)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-lease-check", action="store_true")
    ap.add_argument("--parts", default="core,layer,exact,model")
    ap.add_argument("--core-cases", default="4:1024,4:4096", help="B:T pairs for the core part")
    ap.add_argument("--B", type=int, default=4, help=argparse.SUPPRESS)
    ap.add_argument("--core-dtypes", default=None, help="default fp32 on CPU, bf16 on GPU")
    ap.add_argument("--core-arms", default=None)
    ap.add_argument("--layer-B", type=int, default=4)
    ap.add_argument("--layer-T", type=int, default=4096)
    ap.add_argument("--layer-dtypes", default="fp32,bf16")
    ap.add_argument("--layer-arms", default=None)
    ap.add_argument("--layer-cases", default=None,
                    help="dtype:B:T list for the layer part, overriding --layer-dtypes/-B/-T (e.g. bf16:4:4096,fp32:1:4096)")
    ap.add_argument("--exact-T", default="1,37,64,65,130,200", help="sequence lengths for the exact part")
    ap.add_argument("--exact-arms", default=None)
    ap.add_argument("--model-T", type=int, default=96, help="tokens per sequence for the model part")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=None, help="untimed runs per arm (default 1 on GPU, 0 on CPU)")
    ap.add_argument("--need-gb", type=float, default=None, help="skip an arm when MemAvailable is below this")
    # child-only
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--part", help=argparse.SUPPRESS)
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    ap.add_argument("--T", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--dtype", help=argparse.SUPPRESS)
    ap.add_argument("--shape", help=argparse.SUPPRESS)
    ap.add_argument("--save", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.warmup is None:
        args.warmup = 1 if args.gpu else 0
    if args._child:
        return _child(args)
    if args.fetch:
        fetch_layer0(args.cache_dir, args.max_bytes)
        return
    if args.gpu and not args.no_lease_check and not os.environ.get("LEASE_ROCR_DEVICES"):
        raise SystemExit("not under gpu-lease (LEASE_ROCR_DEVICES unset): run 'gpu-lease -n 1 -- python ...'")
    import torch
    sys.path.insert(0, str(ROOT))
    from ngram_memory.backends.graft import gdn_native as gn
    shape = shape35(args.cache_dir)
    work = Path(args.work or tempfile.mkdtemp(prefix="gdn_validate_"))
    work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import importlib.util as iu
    res: dict = dict(script="experiments/m2_0/gdn_validate.py", mode="gpu" if args.gpu else "cpu",
                     host=REPO, revision=REV, shape=shape, torch=torch.__version__,
                     transformers=__import__("transformers").__version__, python=platform.python_version(),
                     cpu=platform.processor() or platform.machine(), nproc=os.cpu_count(), threads=args.threads,
                     fla_installed=iu.find_spec("fla") is not None,
                     native_op_available=gn.resolve_native_op() is not None,
                     native_forward=("gdn_hip.gdn_prefill (recurrent)" if args.gpu else
                                     "CPU stand-in gdn_native.hf_standin_op (HF torch forward, native signature)"),
                     args={k: str(v) for k, v in vars(args).items() if not k.startswith("_")},
                     started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if args.gpu:
        res["gpu"] = dict(name=torch.cuda.get_device_name(0), env={k: os.environ.get(k) for k in (
            "LEASE_ROCR_DEVICES", "GDN_HIP_WMMA_PREFILL", "TORCH_BLAS_PREFER_HIPBLASLT")})

    def save():
        args.out.write_text(json.dumps(res, indent=1))

    parts = args.parts.split(",")
    if "core" in parts:
        arms = (args.core_arms.split(",") if args.core_arms else
                ["hf_ref", "fix", "fix_g1", "native_ours", "ours", "fork_ref"]
                + (["fix_hfbeta", "fix_wmma"] if args.gpu else []))
        dts = (args.core_dtypes or ("bf16" if args.gpu else "fp32")).split(",")
        res["core"] = []
        for case in args.core_cases.split(","):
            Bc, T = (int(x) for x in case.split(":"))
            for dt in dts:
                ntok = Bc * T
                row = dict(B=Bc, T=T, dtype=dt, arms={})
                # peak live ~0.55 MB/token for hf_ref / fix / ours at fp32 (smoke at B=4, T=256) + ~0.8 GB process
                need = args.need_gb if args.need_gb is not None else 0.8 + 0.6e-3 * ntok * 1.2
                print(f"== core B={Bc} T={T} {dt}  (MemAvailable {mem_available_gb():.1f} GB)", flush=True)
                ref = None
                for arm in arms:
                    r = run_child(args, "core", arm, Bc, T, dt, shape, work, need)
                    if arm == "hf_ref":
                        ref = r if "save" in r else None
                    elif ref is not None and "save" in r:
                        compare(r, ref, GRAD_NAMES)
                    row["arms"][arm] = r
                    print("  " + _fmt(r), flush=True)
                    if arm != "hf_ref" and "save" in r:
                        Path(r.pop("save")).unlink(missing_ok=True)
                    save()
                if ref is not None:
                    Path(ref.pop("save")).unlink(missing_ok=True)
                _core_summary(row, shape)
                res["core"].append(row)
                save()
    if "layer" in parts:
        ck = args.layer_ckpt or (args.cache_dir / "b1_gdn_L0")
        if not (ck / "config.json").exists():
            res["layer"] = dict(skipped=f"no layer checkpoint at {ck} (run --fetch)")
        else:
            man = ck / "MANIFEST.json"
            res["layer_ckpt"] = dict(path=str(ck), manifest_sha256=sha256_bytes(man.read_bytes()) if man.exists() else None,
                                     revision=json.loads(man.read_text()).get("revision") if man.exists() else None)
            arms = args.layer_arms.split(",") if args.layer_arms else list(LAYER_ARMS_GPU if args.gpu else LAYER_ARMS_CPU)
            ref_arm = "hf_patched" if args.gpu else "hf"
            res["layer"] = []
            cases = ([tuple(c.split(":")) for c in args.layer_cases.split(",")] if args.layer_cases else
                     [(dt, args.layer_B, args.layer_T) for dt in args.layer_dtypes.split(",")])
            for dt, LB, LT in cases:
                LB, LT = int(LB), int(LT)
                ntok = LB * LT
                # ~0.7 MB/token (smoke at B=2, T=256, fp32 whole mixer) + ~0.8 GB process
                need = args.need_gb if args.need_gb is not None else 0.8 + 0.7e-3 * ntok * 1.2
                row = dict(B=LB, T=LT, dtype=dt, reference=ref_arm, arms={})
                print(f"== layer 0 mixer B={LB} T={LT} {dt}  (MemAvailable "
                      f"{mem_available_gb():.1f} GB)", flush=True)
                ref = None
                extras: dict = {}
                xr = [a for a in EXTRA_REFS if a in arms and dt == "bf16"]
                order = [ref_arm] + xr + [a for a in arms if a not in (ref_arm, *EXTRA_REFS)]
                for arm in order:
                    r = run_child(args, "layer", arm, LB, LT, dt, shape, work, need, str(ck))
                    for en, e in extras.items():
                        if "save" in r:
                            t = dict(save=r["save"])
                            compare(t, e, ("x",))
                            r[f"vs_{en}"] = dict(fwd_rel=t["fwd_rel_err"], dx_cos=t["grad_cos"]["x"],
                                                 dx_rel=t["grad_rel"]["x"])
                    if arm == ref_arm:
                        ref = r if "save" in r else None
                    elif ref is not None and "save" in r:
                        compare(r, ref, ("x",))
                        r["dx_cos"] = r.pop("grad_cos")["x"]
                        r["dx_rel"] = r.pop("grad_rel")["x"]
                        if arm not in EXTRA_REFS:
                            r["pass"] = dict(fwd_rel=r["fwd_rel_err"] <= BAR_FWD_REL, dx_cos=r["dx_cos"] >= BAR_DX_COS)
                    if arm in EXTRA_REFS and "save" in r:
                        extras[arm] = r
                    row["arms"][arm] = r
                    msg = f"  | vs {ref_arm}: fwd rel {r['fwd_rel_err']:.2e} dX cos {r['dx_cos']:.9f}" if "dx_cos" in r else ""
                    for en in EXTRA_REFS:
                        if f"vs_{en}" in r:
                            msg += f"  | vs {en}: fwd rel {r[f'vs_{en}']['fwd_rel']:.2e} dX cos {r[f'vs_{en}']['dx_cos']:.9f}"
                    print("  " + _fmt(r) + msg, flush=True)
                    if arm not in (ref_arm, *EXTRA_REFS) and "save" in r:
                        Path(r.pop("save")).unlink(missing_ok=True)
                    save()
                for keep in [ref, *extras.values()]:
                    if keep is not None and "save" in keep:
                        Path(keep.pop("save")).unlink(missing_ok=True)
                res["layer"].append(row)
                save()
    if "exact" in parts:
        print("== exact: native math vs fp64 truth", flush=True)
        part_exact(args, res)
        save()
    if "model" in parts:
        print("== model-level train step (HF default use_cache=True)", flush=True)
        part_model(args, res)
        save()
    res["verdict"] = _verdict(res, args.gpu)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    shutil.rmtree(work, ignore_errors=True) if args.work is None else None
    print(json.dumps(res["verdict"], indent=1))
    print("wrote", args.out)


def _core_summary(row: dict, shape: dict) -> None:
    """Per candidate fixed-backward arm (fix = all B sequences per recompute, fix_g1 = one at a time, ...): the
    M2-0 timing quantities beside the same arm's op count and backward memory, so the GPU default is chosen on
    numbers measured the same way (verifier D4)."""
    a = row["arms"]
    ok = {k: v for k, v in a.items() if "fwd_s" in v}
    ntok = row["B"] * row["T"]
    native_fwd = ok.get("fix", {}).get("fwd_s")          # every fix arm runs the same native forward
    row["fixed_backward"] = {}
    for arm in FIX_ARMS:
        if arm not in ok:
            continue
        f = ok[arm]
        base = native_fwd or f["fwd_s"]
        row["fixed_backward"][arm] = dict(
            bwd_group=CORE_ARMS[arm].get("bwd_group"), beta_fp32=CORE_ARMS[arm].get("beta_fp32", "default"),
            bwd_over_native_fwd=f["bwd_s"] / base,
            bwd_h_per_1M_tok_30_layers=f["bwd_s"] / ntok * 1e6 * shape["layers"] / 3600,
            fwdbwd_h_per_1M_tok_30_layers=(f["fwd_s"] + f["bwd_s"]) / ntok * 1e6 * shape["layers"] / 3600,
            bwd_ops_per_tok=f["ops"]["bwd_per_tok"], bwd_nonview_ops_per_tok=f["ops"]["bwd_nonview_per_tok"],
            bwd_live_extra_bytes=f["live_bytes"]["bwd_peak_extra"],
            gpu_peak_fwdbwd_bytes=f.get("gpu_peak_fwdbwd_bytes"))
    if "fix" in row["fixed_backward"]:
        row["fix"] = row["fixed_backward"]["fix"]           # backwards-compatible key
    for ref_arm in ("fork_ref",):
        if ref_arm in ok:
            f = ok[ref_arm]
            row["reference_backward"] = dict(
                arm=ref_arm, bwd_over_native_fwd=f["bwd_s"] / (native_fwd or f["fwd_s"]),
                bwd_h_per_1M_tok_30_layers=f["bwd_s"] / ntok * 1e6 * shape["layers"] / 3600,
                bwd_ops_per_tok=f["ops"]["bwd_per_tok"], bwd_live_extra_bytes=f["live_bytes"]["bwd_peak_extra"],
                gpu_peak_fwdbwd_bytes=f.get("gpu_peak_fwdbwd_bytes"))
    for base in ("fork_ref", "ours", "native_ours", "fix_g1"):
        if base in ok and "fix" in ok:
            row.setdefault("ratios_vs_fix", {})[base] = dict(
                bwd_wall=ok[base]["bwd_s"] / ok["fix"]["bwd_s"],
                bwd_ops=ok[base]["ops"]["bwd"] / ok["fix"]["ops"]["bwd"],
                bwd_live_bytes=ok[base]["live_bytes"]["bwd_peak_extra"] / max(1, ok["fix"]["live_bytes"]["bwd_peak_extra"]))


def _verdict(res: dict, gpu: bool) -> dict:
    v: dict = {}
    for row in res.get("core", []):
        for arm, r in row["arms"].items():
            if "grad_cos" in r:
                v.setdefault("core_min_grad_cos", {})[f"{arm}@T{row['T']}/{row['dtype']}"] = min(
                    x for x in r["grad_cos"].values() if x is not None)
                v.setdefault("core_fwd_rel", {})[f"{arm}@T{row['T']}/{row['dtype']}"] = r["fwd_rel_err"]
        if gpu:
            for arm, fb in row.get("fixed_backward", {}).items():
                v.setdefault("m2_0_bars", {})[f"{arm}@B{row['B']}xT{row['T']}/{row['dtype']}"] = dict(
                    bwd_group=fb["bwd_group"], bwd_over_native_fwd=fb["bwd_over_native_fwd"],
                    pass_bwd_le_1p5x=fb["bwd_over_native_fwd"] <= BAR_BWD_OVER_NATIVE_FWD,
                    bwd_h_per_1M=fb["bwd_h_per_1M_tok_30_layers"],
                    pass_kill_lt_0p2h=fb["bwd_h_per_1M_tok_30_layers"] < KILL_BWD_H_PER_1M,
                    bwd_ops_per_tok=fb["bwd_ops_per_tok"], gpu_peak_fwdbwd_gb=(fb["gpu_peak_fwdbwd_bytes"] or 0) / 1e9)
            if "reference_backward" in row:
                v.setdefault("reference_backward", {})[f"B{row['B']}xT{row['T']}/{row['dtype']}"] = row["reference_backward"]
    if isinstance(res.get("layer"), list):
        for row in res["layer"]:
            for arm, r in row["arms"].items():
                if "pass" in r:
                    v.setdefault("layer_bars", {})[f"{arm}/{row['dtype']}"] = dict(
                        fwd_rel=r["fwd_rel_err"], dx_cos=r["dx_cos"], pass_=all(r["pass"].values()),
                        **{f"vs_{e}": r[f"vs_{e}"] for e in EXTRA_REFS if f"vs_{e}" in r})
                elif any(f"vs_{e}" in r for e in EXTRA_REFS):
                    v.setdefault("layer_bars", {})[f"{arm}/{row['dtype']}"] = {
                        f"vs_{e}": r[f"vs_{e}"] for e in EXTRA_REFS if f"vs_{e}" in r}
    if "exact" in res:
        v["exact_vs_fp64"] = res["exact"]["worst"]
    if "model" in res and "arms" in res["model"]:
        v["model_step_all_layers_patched"] = res["model"]["all_ok"]
    if not gpu:
        v["timing_bars"] = "N/A on CPU (native forward is a stand-in); run --gpu on the owner's card"
        v["note"] = ("CPU parts (a)/(b): the 'native' forward is HF's own torch forward behind the native signature, "
                     "so their accuracy numbers are wiring checks; part (c) is the native-math accuracy check")
    return v


if __name__ == "__main__":
    main()
