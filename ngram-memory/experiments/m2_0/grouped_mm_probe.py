"""M2-0 / DESIGN §7.5 #2: does ``torch._grouped_mm`` (and ``torch.nn.functional.grouped_mm`` where present) run on
this device at the 35B-A3B MoE shapes, forward and input-gradient backward, and how does it compare with the
per-expert loop the AWQ experts use today (host_quant._MoEFn)?

Shapes (35B, one MoE chunk): N tokens x top-8 routed rows R = 8N sorted by expert, E = 256 experts,
gate_up [E, d=2048, 2I=1024] and down [E, I=512, d] in bf16 (dense, as after the just-in-time dequant); routing is
random top-8 (uneven counts, some experts empty, offsets not 16-aligned) and, as a second case, counts padded to a
multiple of 16. For each case: forward rel. error and dX rel. error / cosine against a per-expert fp32 loop, and
wall time of forward and forward + dX backward against the bf16 per-expert loop. Weights are frozen
(requires_grad False), so the backward is the dX-only path M2 uses.

The small 3-group probe in host_quant.grouped_mm_probe is also reported. CPU runs use small N by default.

  gpu-lease -n 1 -- python experiments/m2_0/grouped_mm_probe.py --device cuda:0 --tokens 8192,16384 --out gmm.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

E, D, I, TOPK = 256, 2048, 512, 8


def _fns() -> dict:
    out = {}
    if getattr(torch, "_grouped_mm", None) is not None:
        out["torch._grouped_mm"] = lambda x, w, offs: torch._grouped_mm(x, w, offs=offs)
    f = getattr(F, "grouped_mm", None)
    if f is not None:
        out["F.grouped_mm"] = lambda x, w, offs: f(x, w, offs=offs)
    return out


def _loop(x, w, counts):
    outs, s = [], 0
    for e, c in enumerate(counts):
        if c:
            outs.append(x[s:s + c] @ w[e])
        s += c
    return torch.cat(outs) if outs else x.new_zeros(0, w.shape[2])


def _moe(mm, x, wgu, wdn, offs):
    gu = mm(x, wgu, offs)
    g, u = gu.chunk(2, dim=-1)
    return mm(F.silu(g) * u, wdn, offs)


def _counts(n_rows_per_tok: int, N: int, gen: torch.Generator, pad16: bool) -> list[int]:
    idx = torch.stack([torch.randperm(E, generator=gen)[:n_rows_per_tok] for _ in range(N)]) if N <= 4096 else \
        torch.randint(0, E, (N, n_rows_per_tok), generator=gen)
    c = torch.bincount(idx.flatten(), minlength=E)
    if pad16:
        c = (c + 15) // 16 * 16
    return c.tolist()


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def probe(dev: torch.device, N: int, pad16: bool, reps: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed + N + int(pad16))
    counts = _counts(TOPK, N, gen, pad16)
    R = sum(counts)
    x = (torch.randn(R, D, generator=gen) * 0.5).to(dev, torch.bfloat16)
    wgu = (torch.randn(E, D, 2 * I, generator=gen) * D ** -0.5).to(dev, torch.bfloat16)
    wdn = (torch.randn(E, I, D, generator=gen) * I ** -0.5).to(dev, torch.bfloat16)
    dy = torch.randn(R, D, generator=gen).to(dev, torch.bfloat16)
    offs = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32).to(dev)
    out: dict = dict(tokens=N, rows=R, empty_experts=sum(c == 0 for c in counts), pad16=pad16,
                     min_count=min(counts), max_count=max(counts))
    # fp32 per-expert reference
    xr = x.float().requires_grad_(True)
    yr = _moe(lambda a, w, o: _loop(a, w.float(), counts), xr, wgu, wdn, offs)
    (dxr,) = torch.autograd.grad(yr, xr, dy.float())
    yr, dxr = yr.detach(), dxr.detach()

    def timed(mm):
        best = None
        for i in range(reps + 1):
            xx = x.detach().clone().requires_grad_(True)
            _sync(dev)
            t0 = time.perf_counter()
            y = _moe(mm, xx, wgu, wdn, offs)
            _sync(dev)
            t1 = time.perf_counter()
            (dx,) = torch.autograd.grad(y, xx, dy)
            _sync(dev)
            t2 = time.perf_counter()
            if i > 0 or reps == 0:
                if best is None or (t2 - t0) < best[0] + best[1]:
                    best = (t1 - t0, t2 - t1, y.detach(), dx.detach())
        return best
    arms = {"loop_bf16": lambda a, w, o: _loop(a, w, counts), **_fns()}
    for name, mm in arms.items():
        try:
            tf, tb, y, dx = timed(mm)
            out[name] = dict(ok=True, fwd_s=tf, bwd_s=tb,
                             fwd_rel=((y.float() - yr).norm() / yr.norm()).item(),
                             dx_rel=((dx.float() - dxr).norm() / dxr.norm()).item(),
                             dx_cos=F.cosine_similarity(dx.double().flatten(), dxr.double().flatten(), dim=0).item())
        except Exception as e:  # noqa: BLE001 - this is a capability probe
            out[name] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    for name in _fns():
        r, lp = out.get(name, {}), out.get("loop_bf16", {})
        if r.get("ok") and lp.get("ok"):
            r["speedup_vs_loop_fwdbwd"] = (lp["fwd_s"] + lp["bwd_s"]) / (r["fwd_s"] + r["bwd_s"])
            r["pass"] = r["fwd_rel"] < 1e-2 and r["dx_cos"] >= 0.9999
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", default=None, help="tokens per chunk (default 8192,16384 on GPU, 256 on CPU)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    toks = [int(t) for t in (a.tokens or ("8192,16384" if dev.type == "cuda" else "256")).split(",")]
    res: dict = dict(script="experiments/m2_0/grouped_mm_probe.py", device=str(dev), torch=torch.__version__,
                     available=list(_fns()), small_probe=hq.grouped_mm_probe(dev), cases=[])
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        res["card"] = dict(name=p.name, arch=getattr(p, "gcnArchName", None))
    for N in toks:
        for pad in (False, True):
            r = probe(dev, N, pad, a.reps)
            res["cases"].append(r)
            print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                                                     for kk, vv in v.items()})
                              for k, v in r.items()}), flush=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    names = list(_fns())
    res["verdict"] = {n: dict(runs_all_cases=all(c.get(n, {}).get("ok") for c in res["cases"]),
                              accurate_all_cases=all(c.get(n, {}).get("pass") for c in res["cases"]),
                              runs_unaligned=all(c.get(n, {}).get("ok") for c in res["cases"] if not c["pad16"]))
                      for n in names}
    res["grouped_mm_runs"] = any(v["runs_all_cases"] and v["accurate_all_cases"] for v in res["verdict"].values())
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    return res


if __name__ == "__main__":
    main()
