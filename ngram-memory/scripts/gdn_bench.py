"""GDN core cost per token: our chunked backward vs HF's torch_chunk_gated_delta_rule vs the fork's reference
recompute-backward (R/../../tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py), counted the same way as
S1/rt-feas/gdn_ref_ops.py (every aten op seen by a TorchDispatchMode, views counted separately) plus wall time.

Shapes come from the host configs: 35B-A3B (H=16 key heads, HV=32 value heads, K=V=128) and 27B (H=16, HV=48).
Only the delta-rule core is measured (inputs already conv'ed / split; q,k pre-l2norm, GQA expanded as in HF).

Usage:  python scripts/gdn_bench.py --threads 3 --out <json>  [--cases "35b:4:512,27b:4:512"] [--impls ours,hf,fork]
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
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory.backends.graft import gdn  # noqa: E402

FORK_AUTOGRAD = Path("/home/user/patcarter883/minisglang-rdna4/tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py")
HF_CFG = {
    "35b": "/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/"
           "f1e8415c4f29508fe281da4e256df26573213bfd/config.json",
    "27b": "/root/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/"
           "93245f7c00f1a8964b66a97c6e02e2fd09080be4/config.json",
}
VIEW_OPS = {"view", "_unsafe_view", "alias", "detach", "expand", "t", "transpose", "permute", "select", "slice",
            "unsqueeze", "squeeze", "as_strided", "split", "unbind", "reshape"}


def gdn_shape(host: str) -> dict:
    """GDN head geometry read from the cached HF config of the host."""
    c = json.loads(Path(HF_CFG[host]).read_text())
    t = c.get("text_config", c)
    return dict(H=t["linear_num_key_heads"], HV=t["linear_num_value_heads"], K=t["linear_key_head_dim"],
                V=t["linear_value_head_dim"], layers=sum(x == "linear_attention" for x in t["layer_types"]),
                hidden=t["hidden_size"])


def load_fork_reference():
    """The fork's ref_gdn_prefill_core, loaded read-only without its .so (as S1/rt-feas/gdn_ref_ops.py did)."""
    src = FORK_AUTOGRAD.read_text().replace("from ._ops import add_op_namespace_prefix, ops", "ops = None")
    ns: dict = {"__name__": "gdn_fork_ref"}
    exec(compile(src[: src.index("def _native_bwd_ok")], str(FORK_AUTOGRAD), "exec"), ns)
    return ns["ref_gdn_prefill_core"]


class OpCount(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.views = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        if func.is_view or func.__name__.split(".")[0] in VIEW_OPS:
            self.views += 1
        return func(*args, **(kwargs or {}))


def _hf_chunk():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    f = M.torch_chunk_gated_delta_rule
    return getattr(f, "__wrapped__", f)


def make_inputs(B, T, s, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, HV, K, V = s["H"], s["HV"], s["K"], s["V"]
    raw = dict(q=torch.randn(B, T, H, K, generator=g), k=torch.randn(B, T, H, K, generator=g),
               v=torch.randn(B, T, HV, V, generator=g), a=torch.randn(B, T, HV, generator=g),
               b=torch.randn(B, T, HV, generator=g), A_log=torch.log(torch.rand(HV, generator=g) * 15 + 1),
               dt_bias=torch.randn(HV, generator=g))
    dout = torch.randn(B, T, HV, V, generator=g)
    return raw, dout


def run_ours(raw, dout, save_states=True, tri="solve", max_bh=None, chunk=64):
    ins = {k: v.clone().requires_grad_(k in "qkvab") for k, v in raw.items()}
    t0 = time.perf_counter()
    o = gdn.gated_delta_rule_from_raw(ins["q"], ins["k"], ins["v"], ins["a"], ins["b"], ins["A_log"],
                                      ins["dt_bias"], impl="ours", chunk_size=chunk, tri=tri, max_bh=max_bh,
                                      save_states=save_states)
    t1 = time.perf_counter()
    grads = torch.autograd.grad(o, [ins[k] for k in "qkvab"], dout)
    t2 = time.perf_counter()
    return o.detach(), grads, t1 - t0, t2 - t1


def run_hf(raw, dout):
    f = _hf_chunk()
    ins = {k: v.clone().requires_grad_(k in "qkvab") for k, v in raw.items()}
    t0 = time.perf_counter()
    rep = ins["v"].shape[2] // ins["q"].shape[2]
    beta = ins["b"].sigmoid()
    g = -ins["A_log"].float().exp() * torch.nn.functional.softplus(ins["a"].float() + ins["dt_bias"])
    o, _ = f(ins["q"].repeat_interleave(rep, 2), ins["k"].repeat_interleave(rep, 2), ins["v"], g=g, beta=beta,
             initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    t1 = time.perf_counter()
    grads = torch.autograd.grad(o, [ins[k] for k in "qkvab"], dout)
    t2 = time.perf_counter()
    return o.detach(), grads, t1 - t0, t2 - t1


def run_fork(raw, dout, ref):
    """The fork's batched training backward (autograd.py:_GDNPrefillBatchFn.backward): fold B into heads,
    chunk = max(4, 64 // B), ref forward + autograd.grad. Its 'forward' here is the reference recompute
    that its backward performs; the native forward itself only exists on the GPU."""
    B, T, H, K = raw["q"].shape
    HV = raw["v"].shape[2]
    fold = lambda t: t.permute(1, 0, *range(2, t.dim())).reshape(T, B * t.shape[2], *t.shape[3:])  # noqa: E731
    ins = [fold(raw[k]).clone().requires_grad_(True) for k in "qkvab"]
    A = raw["A_log"].repeat(B).requires_grad_(True)
    d = raw["dt_bias"].repeat(B).requires_grad_(True)
    cw = max(4, 64 // max(1, B))
    t0 = time.perf_counter()
    o = ref(*ins, A, d, K ** -0.5, True, chunk=cw)
    t1 = time.perf_counter()
    grads = torch.autograd.grad(o, ins + [A, d], fold(dout))
    t2 = time.perf_counter()
    unf = o.detach().reshape(T, B, HV, -1).permute(1, 0, 2, 3)
    gq = grads[0].reshape(T, B, H, K).permute(1, 0, 2, 3)
    return unf, (gq,), t1 - t0, t2 - t1


def counted(fn, *a, **kw):
    m = OpCount()
    with m:
        r = fn(*a, **kw)
    return r, m.n, m.views


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--cases", default="35b:4:512,27b:4:512")
    ap.add_argument("--impls", default="ours,ours_nosave,hf,fork")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ref = load_fork_reference() if "fork" in args.impls else None
    results = []
    for case in args.cases.split(","):
        host, B, T = case.split(":")
        B, T = int(B), int(T)
        s = gdn_shape(host)
        raw, dout = make_inputs(B, T, s)
        row = dict(host=host, B=B, T=T, shape=s, threads=args.threads, reps=args.reps, load_avg=os.getloadavg())
        outs = {}
        for impl in args.impls.split(","):
            if impl == "ours":
                (o, gr, tf, tb), n, nv = counted(run_ours, raw, dout)
            elif impl == "ours_nosave":
                (o, gr, tf, tb), n, nv = counted(run_ours, raw, dout, save_states=False)
            elif impl == "hf":
                (o, gr, tf, tb), n, nv = counted(run_hf, raw, dout)
            elif impl == "fork":
                (o, gr, tf, tb), n, nv = counted(run_fork, raw, dout, ref)
            else:
                raise SystemExit(f"unknown impl {impl}")
            # wall time again WITHOUT the dispatch-mode overhead: best of --reps (the box is shared)
            best = None
            for _ in range(args.reps):
                if impl == "fork":
                    _, _, tf, tb = run_fork(raw, dout, ref)
                elif impl == "hf":
                    _, _, tf, tb = run_hf(raw, dout)
                else:
                    _, _, tf, tb = run_ours(raw, dout, save_states=(impl == "ours"))
                if best is None or tf + tb < sum(best):
                    best = (tf, tb)
            tf, tb = best
            outs[impl] = (o, gr)
            ntok = B * T
            row[impl] = dict(ops=n, ops_per_token=n / ntok, view_ops=nv, nonview_ops_per_token=(n - nv) / ntok,
                             fwd_s=tf, bwd_s=tb, ms_per_token_fwd_bwd=(tf + tb) / ntok * 1e3,
                             ms_per_token_bwd=tb / ntok * 1e3)
            print(f"{host} B={B} T={T} {impl:12s} ops {n:>8,} ({n / ntok:7.3f}/tok, non-view "
                  f"{(n - nv) / ntok:6.3f}/tok)  fwd {tf:7.2f}s bwd {tb:7.2f}s  "
                  f"({(tf + tb) / ntok * 1e3:.3f} ms/tok fwd+bwd, {tb / ntok * 1e3:.3f} ms/tok bwd)", flush=True)
        if "hf" in outs and "ours" in outs:
            o_h, g_h = outs["hf"]
            o_o, g_o = outs["ours"]
            row["ours_vs_hf"] = dict(fwd_rel_err=((o_o - o_h).norm() / o_h.norm()).item(),
                                     grad_cos={k: cos(a, b) for k, a, b in zip("qkvab", g_o, g_h)})
            print("  ours vs hf:", json.dumps(row["ours_vs_hf"]), flush=True)
        if "fork" in outs and "ours" in outs:
            o_f, g_f = outs["fork"]
            o_o, g_o = outs["ours"]
            row["ours_vs_fork"] = dict(fwd_rel_err=((o_o - o_f).norm() / o_f.norm()).item(),
                                       grad_q_cos=cos(g_o[0], g_f[0]))
            print("  ours vs fork:", json.dumps(row["ours_vs_fork"]), flush=True)
        for base in ("hf", "fork"):
            if base in row and "ours" in row:
                row[f"ratio_{base}_over_ours"] = dict(
                    ops=row[base]["ops"] / row["ours"]["ops"],
                    wall_fwd_bwd=(row[base]["fwd_s"] + row[base]["bwd_s"]) / (row["ours"]["fwd_s"] + row["ours"]["bwd_s"]))
                print(f"  {base}/ours: ops x{row[f'ratio_{base}_over_ours']['ops']:.1f}, wall "
                      f"x{row[f'ratio_{base}_over_ours']['wall_fwd_bwd']:.2f}", flush=True)
        row["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        results.append(row)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    main()
