"""M2-0 on the owner's machine, in one command (DESIGN §8 M2-0, §7.5 #1-2, §4.6). Use experiments/m2_0/run_owner.sh.

What it does (about 1-2 h, est.; the serve must be stopped):
  0. preflight, on the host: refuses if the serve is up (TCP port 1919 or --serve-port answers; a running process
     or container that looks like a minisgl / sglang / vllm serve), if any process holds a GPU (/sys/class/kfd),
     or if the cards are leased (gpu-lease --nowait fails). Then it re-runs itself under `gpu-lease -n 2`.
     Inside the lease: exactly two gfx1201 cards visible, each with < 1.5 GB in use, gdn_hip importable.
  1. weights: the local HF snapshot of the pinned revision (or --ckpt). Without one, it range-reads only layers 0
     and 3 (~1.07 GB) into --cache-dir; a whole-checkpoint download happens only with --allow-full-download.
  2. grouped_mm at the real MoE shapes on each card (§7.5 #2)                      [grouped_mm_probe.py]
  3. GDN on card 0: the delta-rule core at B=4, T=1024/4096 (bf16) with HF's reference, the fixed backward
     (native forward + HF torch-chunk recompute; all sequences / one at a time), our backward, the fork's
     reference backward and WMMA; layer 0's real GDN mixer at B=4 T=4096 bf16 (and B=1 fp32) against HF incl.
     the fork's own NativeGDNShim widened to Qwen3_5MoeGatedDeltaNet; native-math accuracy vs an fp64 truth;
     a model-level train step with HF's default use_cache=True                      [gdn_validate.py --gpu]
  4. AWQ on card 0: L0 (and L3) expert bit-exactness, whole layers vs an fp32 reference, per-piece memory at the
     real chunk sizes (4k mixer pieces, 8k / 16k MoE chunks)                         [awq_layers_validate.py --gpu]
  5. the planned split on both cards: all 40 layers resident (packed), one training step per chunk size (8k, 16k)
     with per-layer checkpointing, mixed chunking and the fixed GDN backward; per-card peak VRAM, step time,
     hours per 1M tokens; and once with the fork's reference backward at 8k             [split_peak.py]
  6. prints every M2-0 pass / kill criterion of DESIGN §8 (and the M2 prerequisites M2-0 measures), and writes
     <out>/M2_0_SUMMARY.txt, <out>/M2_0_SUMMARY.json and <out>.tar.gz (JSON + logs, no weights) to send back.

  experiments/m2_0/run_owner.sh                       # everything, defaults
  experiments/m2_0/run_owner.sh --preflight-only      # only the checks
  experiments/m2_0/run_owner.sh --quick               # ~10-15 min smoke at small sizes
  NGM_IMAGE=<torch-rocm image> experiments/m2_0/run_owner.sh   # run the GPU steps in a container
  python experiments/m2_0/run_owner.py --selftest-cpu --cache-dir <cache>   # CPU self-test of the plumbing

The outer process uses only the standard library; torch is imported by the steps.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
REV = "f1e8415c4f29508fe281da4e256df26573213bfd"
CONFIG_SHA256 = "dcc210da7562ecc8900dc25ac350333d0c6ae992a310d1c0f387014f0083f281"   # config.json @ REV
SERVE_PORTS = (1919,)
SERVE_ENGINES = ("minisgl", "sglang", "vllm")
SERVE_WORDS = ("serve", "server", "launch", "entrypoints")
MONITOR_WORDS = ("gpu-lease", "gpu-status", "amd-smi", "rocm-smi", "lease ")    # may touch KFD, are not jobs
MAX_USED_GB_PER_CARD = 1.5
RANGE_FETCH_MAX = 1.2e9

# DESIGN §8 M2-0 row (provisional) and the M2 prerequisites M2-0 measures
BAR_BWD_OVER_NATIVE_FWD = 1.5
BAR_DX_COS = 0.9999
BAR_FWD_REL = 1e-3
BAR_CARD_PEAK_GB = 15.5
KILL_BWD_H_PER_1M = 0.2


# ------------------------------------------------------------------------------------------------------------
# preflight (host side, standard library only)
# ------------------------------------------------------------------------------------------------------------

def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _proc_cmdlines() -> dict[int, str]:
    out = {}
    for d in Path("/proc").glob("[0-9]*"):
        try:
            out[int(d.name)] = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
    return out


def _kfd_pids(exclude: set[int]) -> list[int]:
    """PIDs with the ROCm KFD device open (any GPU user), from /sys/class/kfd/kfd/proc."""
    d = Path("/sys/class/kfd/kfd/proc")
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit() and int(p.name) not in exclude)


def _docker_ps() -> list[str] | None:
    if not shutil.which("docker"):
        return None
    try:
        p = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Command}}"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [ln for ln in p.stdout.splitlines() if ln.strip()] if p.returncode == 0 else None


def _ancestors() -> set[int]:
    out, pid = set(), os.getpid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def is_serve_cmdline(c: str) -> bool:
    lc = c.lower()
    return "tools/serve.sh" in lc or (any(e in lc for e in SERVE_ENGINES) and any(w in lc for w in SERVE_WORDS))


def preflight(ports=SERVE_PORTS, *, me: set[int] | None = None, cmdlines: dict[int, str] | None = None,
              kfd: list[int] | None = None, docker: list[str] | None | bool = True,
              port_check=_port_open, ignore_pids: tuple[int, ...] = ()) -> dict:
    """Host checks. Returns {'ok': bool, 'refusals': [...], 'checks': {...}}; every input is injectable (tests).
    Processes of this run (ancestors, anything whose command line names this script) and lease / monitoring
    tools are not counted."""
    me = (me if me is not None else _ancestors()) | set(ignore_pids)
    refusals, checks = [], {}
    up = [p for p in ports if port_check(p)]
    checks["serve_ports"] = dict(checked=list(ports), answering=up)
    if up:
        refusals.append(f"the serve looks up: port(s) {up} answer on 127.0.0.1 (stop it: docker compose "
                        "--profile serve down, or stop tools/serve.sh)")
    cl = cmdlines if cmdlines is not None else _proc_cmdlines()
    own = str(Path(__file__).name)
    mine = {pid for pid, c in cl.items() if own in c} | me
    hits = {pid: c for pid, c in cl.items() if pid not in mine and c and is_serve_cmdline(c)
            and not c.split(" ")[0].endswith("grep")}
    checks["serve_processes"] = {str(k): v[:200] for k, v in hits.items()}
    if hits:
        refusals.append(f"serve-like processes are running: {sorted(hits)[:8]} ({next(iter(hits.values()))[:120]})")
    dps = _docker_ps() if docker is True else (docker or None)
    if dps is not None:
        serve_c = [c for c in dps if any(s in c.lower() for s in ("serve", "minisgl", "sglang", "vllm", "1919"))]
        checks["docker_serve_containers"] = serve_c
        if serve_c:
            refusals.append(f"serve-like containers are running: {[c.split(chr(9))[0] for c in serve_c]}")
    else:
        checks["docker_serve_containers"] = "docker not available (not checked)"
    holders = kfd if kfd is not None else _kfd_pids(me)
    holders = [p for p in holders if p not in mine and not any(w in cl.get(p, "") for w in MONITOR_WORDS)]
    checks["kfd_gpu_processes"] = {str(p): cl.get(p, "?")[:160] for p in holders}
    if holders:
        refusals.append(f"processes hold the GPU (/sys/class/kfd): {holders[:8]}")
    return dict(ok=not refusals, refusals=refusals, checks=checks)


def gpu_status_text() -> str:
    exe = shutil.which("gpu-status")
    if not exe:
        return "gpu-status not on PATH"
    try:
        return subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout[-3000:]
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"gpu-status failed: {e}"


# ------------------------------------------------------------------------------------------------------------
# checkpoint resolution
# ------------------------------------------------------------------------------------------------------------

def hf_snapshot_candidates() -> list[Path]:
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    name = "models--" + REPO.replace("/", "--")
    return [r / name / "snapshots" / REV for r in roots]


def checkpoint_status(d: Path) -> dict:
    """Is ``d`` a complete local checkpoint (config.json, index, every shard present)?"""
    st: dict = dict(path=str(d), exists=d.is_dir())
    if not d.is_dir():
        return dict(st, complete=False)
    idx, cfg = d / "model.safetensors.index.json", d / "config.json"
    if not (idx.exists() and cfg.exists()):
        return dict(st, complete=False, reason="no index or config.json")
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    missing = [s for s in shards if not (d / s).exists()]
    import hashlib
    sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    return dict(st, complete=not missing, shards=len(shards), missing=missing, config_sha256=sha,
                config_matches_pinned_revision=sha == CONFIG_SHA256)


# ------------------------------------------------------------------------------------------------------------
# steps (inner, under the lease)
# ------------------------------------------------------------------------------------------------------------

def plan_steps(a, py: str, out: Path, ckpt: Path | None, cache: Path, mode: str) -> list[dict]:
    """mode: 'full' (owner's cards), 'quick' (small sizes on the cards), 'cpu' (self-test here)."""
    e = HERE
    S = []
    gpu = mode != "cpu"
    if gpu:
        for i in (0, 1):
            S.append(dict(name=f"grouped_mm_card{i}", timeout=1200, out=out / f"grouped_mm_card{i}.json",
                          cmd=[py, str(e / "grouped_mm_probe.py"), "--device", f"cuda:{i}", "--threads", str(a.threads),
                               "--tokens", "8192,16384" if mode == "full" else "4096",
                               "--out", str(out / f"grouped_mm_card{i}.json")]))
    else:
        S.append(dict(name="grouped_mm_cpu", timeout=900, out=out / "grouped_mm_cpu.json",
                      cmd=[py, str(e / "grouped_mm_probe.py"), "--device", "cpu", "--tokens", "128", "--reps", "1",
                           "--threads", str(a.threads), "--out", str(out / "grouped_mm_cpu.json")]))
    layer_ck = str(ckpt) if ckpt else str(cache / "cyankiwi_35b")
    g = [py, str(e / "gdn_validate.py"), "--cache-dir", str(cache), "--layer-ckpt", layer_ck,
         "--threads", str(a.threads), "--out", str(out / "gdn.json"), "--work", str(out / "gdn_work")]
    if mode == "full":
        g += ["--gpu", "--parts", "core,layer,exact,model", "--core-cases", "4:1024,4:4096", "--core-dtypes", "bf16",
              "--layer-cases", "bf16:4:4096,fp32:1:4096", "--reps", "3", "--warmup", "1"]
    elif mode == "quick":
        g += ["--gpu", "--parts", "core,layer,exact,model", "--core-cases", "4:1024", "--core-dtypes", "bf16",
              "--layer-cases", "bf16:4:1024", "--exact-T", "1,65,130", "--reps", "1", "--warmup", "1"]
    else:
        g += ["--parts", "core,layer,exact,model", "--core-cases", "2:128", "--core-arms",
              "hf_ref,fix,fix_g1,native_ours,ours,fork_ref", "--layer-cases", "fp32:1:256,bf16:1:256",
              "--layer-arms", "hf,truth,hf_b32,hf_patched,fix,fix_exact,fix_exact_b32", "--exact-T", "1,65",
              "--model-T", "48"]
    S.append(dict(name="gdn", timeout=3 * 3600, out=out / "gdn.json", cmd=g))
    w = [py, str(e / "awq_layers_validate.py"), "--out", str(out / "awq"), "--tag", "owner" if gpu else "selftest",
         "--threads", str(a.threads)]
    w += ["--ckpt", str(ckpt)] if ckpt else ["--cache-dir", str(cache)]
    if mode == "full":
        w += ["--gpu", "--parts", "1,2,4"]
    elif mode == "quick":
        w += ["--gpu", "--parts", "1,4", "--piece-T", "2048", "--moe-chunk-tokens", "8192", "--epg-mem", "8"]
    else:
        w += ["--parts", "1"]
    S.append(dict(name="awq", timeout=3 * 3600, out=out / "awq" / f"awq_validate_{'owner' if gpu else 'selftest'}.json",
                  cmd=w))
    src = ["--ckpt", str(ckpt)] if ckpt and checkpoint_status(ckpt).get("complete") else ["--cache-dir", str(cache)]
    if mode == "full":
        S.append(dict(name="split_fix", timeout=2 * 3600, out=out / "split_fix.json",
                      cmd=[py, str(e / "split_peak.py"), *src, "--devices", "cuda:0,cuda:1", "--T", "4096",
                           "--chunks", "8192,16384", "--gdn", "fix_g1", "--steps", "2", "--threads", str(a.threads),
                           "--out", str(out / "split_fix.json")]))
        S.append(dict(name="split_fork_ref", timeout=2 * 3600, out=out / "split_fork_ref.json", optional=True,
                      cmd=[py, str(e / "split_peak.py"), *src, "--devices", "cuda:0,cuda:1", "--T", "4096",
                           "--chunks", "8192", "--gdn", "fork_ref", "--steps", "2", "--threads", str(a.threads),
                           "--out", str(out / "split_fork_ref.json")]))
    elif mode == "quick":
        S.append(dict(name="split_fix", timeout=3600, out=out / "split_fix.json",
                      cmd=[py, str(e / "split_peak.py"), *src, "--devices", "cuda:0,cuda:1", "--T", "4096",
                           "--chunks", "8192", "--gdn", "fix_g1", "--steps", "1", "--threads", str(a.threads),
                           "--out", str(out / "split_fix.json")]))
    else:
        S.append(dict(name="split_fix", timeout=3600, out=out / "split_fix.json",
                      cmd=[py, str(e / "split_peak.py"), "--cache-dir", str(cache), "--devices", "cpu,cpu",
                           "--layers", "4", "--split", "2", "--T", "64", "--chunks", "128", "--gdn", "fix_g1",
                           "--steps", "1", "--threads", str(a.threads), "--out", str(out / "split_fix.json")]))
    skip = set((a.skip or "").split(","))
    return [s for s in S if s["name"] not in skip]


def inner_checks(py: str, gpu: bool) -> dict:
    """Under the lease: what torch sees (two gfx1201 cards, little memory in use, gdn_hip importable)."""
    code = r"""
import json, os, sys
sys.path.insert(0, %r)
import torch, transformers
out = dict(torch=torch.__version__, transformers=transformers.__version__, hip=getattr(torch.version, "hip", None),
           cuda_available=torch.cuda.is_available(), cards=[])
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out["cards"].append(dict(index=i, name=p.name, arch=getattr(p, "gcnArchName", "?"), total_gb=total / 1e9,
                                 used_gb=(total - free) / 1e9))
from ngram_memory.backends.graft import gdn_native as gn
out["gdn_hip_recurrent"] = gn.resolve_native_op("recurrent") is not None
out["gdn_hip_wmma"] = gn.resolve_native_op("wmma") is not None
out["fork_autograd_present"] = (gn.fork_paths()[1] / "gdn_hip" / "autograd.py").exists()
import importlib.util as iu
out["fla_installed"] = iu.find_spec("fla") is not None
out["env"] = {k: os.environ.get(k) for k in ("LEASE_ROCR_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                             "TORCH_BLAS_PREFER_HIPBLASLT", "GDN_HIP_WMMA_PREFILL", "NGM_MINISGL_FORK")}
print(json.dumps(out))
""" % str(ROOT)
    p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return dict(ok=False, refusals=[f"cannot import torch / transformers / ngram_memory with {py}: "
                                        f"{p.stderr.strip()[-600:]}"])
    info = json.loads(p.stdout.strip().splitlines()[-1])
    ref = []
    if gpu:
        cards = info["cards"]
        if len(cards) != 2:
            ref.append(f"expected exactly 2 visible cards under the lease, torch sees {len(cards)}")
        for c in cards:
            if not str(c["arch"]).startswith("gfx1201"):
                ref.append(f"card {c['index']} is {c['name']} ({c['arch']}), not gfx1201")
            if c["used_gb"] > MAX_USED_GB_PER_CARD:
                ref.append(f"card {c['index']} already has {c['used_gb']:.1f} GB in use (serve or another job?)")
        if not info["gdn_hip_recurrent"]:
            ref.append("gdn_hip is not importable: build the fork's gdn_hip extension (tools/vhip_patches/gdn) in "
                       "this environment, or point NGM_MINISGL_FORK at a fork clone that has it")
        if info["fla_installed"]:
            info["warning_fla"] = "fla is installed; every HF path used here is pinned to pure torch, fla is not called"
    if not info["transformers"].startswith("5.22"):
        info["warning_transformers"] = (f"transformers {info['transformers']}: the GDN patch copies 5.22.2's forward; "
                                        "the layer / model parts check it against HF either way")
    return dict(ok=not ref, refusals=ref, info=info)


def run_steps(steps: list[dict], out: Path, env: dict) -> list[dict]:
    rec = []
    for s in steps:
        log = out / "logs" / f"{s['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== step {s['name']}: {' '.join(shlex.quote(c) for c in s['cmd'][1:])}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            f.write("$ " + " ".join(shlex.quote(c) for c in s["cmd"]) + "\n")
            f.flush()
            p = subprocess.Popen(s["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                 cwd=str(ROOT))
            timed_out = threading.Event()

            def _kill(p=p, ev=timed_out):
                ev.set()
                p.kill()
            wd = threading.Timer(s["timeout"], _kill)
            wd.start()
            try:
                for line in p.stdout:                                  # tee to the log and the terminal
                    f.write(line)
                    f.flush()
                    sys.stdout.write("  | " + line)
                rc = p.wait()
            finally:
                wd.cancel()
            if timed_out.is_set():
                rc = "timeout"
        r = dict(name=s["name"], rc=rc, wall_s=round(time.time() - t0, 1), log=str(log.relative_to(out)),
                 out=str(Path(s["out"]).relative_to(out)) if Path(s["out"]).exists() else None)
        rec.append(r)
        print(f"=== step {s['name']}: exit {rc} in {r['wall_s']:.0f}s", flush=True)
        (out / "steps.json").write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------------------------------------------------
# criteria (DESIGN §8 M2-0 row + the M2 prerequisites M2-0 measures)
# ------------------------------------------------------------------------------------------------------------

def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _crit(cid, kind, text, status, value, detail=""):
    return dict(id=cid, kind=kind, criterion=text, status=status, value=value, detail=detail)


def evaluate(out: Path, gpu: bool) -> list[dict]:
    """Every M2-0 criterion from the step outputs. status: PASS / FAIL / N/A (not measured or CPU)."""
    C = []
    gdn = _load(out / "gdn.json") or {}
    na = "N/A"
    # ---- GDN timing bars (core, B=4 T=4096 bf16 if run, else the largest core case)
    rows = [r for r in gdn.get("core", []) if r.get("dtype") == "bf16"] or gdn.get("core", [])
    rows = sorted(rows, key=lambda r: (r["B"] == 4, r["T"]))
    row = rows[-1] if rows else None
    fb = (row or {}).get("fixed_backward", {})
    if row and fb and gpu:
        best = min(fb.items(), key=lambda kv: kv[1]["bwd_over_native_fwd"])
        txt = "; ".join(f"{k} (bwd_group={v['bwd_group']}): {v['bwd_over_native_fwd']:.2f}x, "
                        f"{v['bwd_ops_per_tok']:.2f} ops/tok, peak {(v['gpu_peak_fwdbwd_bytes'] or 0) / 1e9:.2f} GB"
                        for k, v in fb.items())
        C.append(_crit("M2-0.1", "pass bar", f"fixed GDN backward <= {BAR_BWD_OVER_NATIVE_FWD}x the native forward "
                       f"per token (B={row['B']}, T={row['T']}, {row['dtype']})",
                       "PASS" if best[1]["bwd_over_native_fwd"] <= BAR_BWD_OVER_NATIVE_FWD else "FAIL",
                       round(best[1]["bwd_over_native_fwd"], 3), f"best arm {best[0]}; {txt}"))
        hb = min(v["bwd_h_per_1M_tok_30_layers"] for v in fb.values())
        C.append(_crit("M2-0.K", "kill", f"GDN backward < {KILL_BWD_H_PER_1M} h per 1M tokens on the 35B "
                       "(30 GDN layers); else re-cost M2 before starting it",
                       "PASS" if hb < KILL_BWD_H_PER_1M else "FAIL (kill: re-cost M2)", round(hb, 4),
                       "; ".join(f"{k}: {v['bwd_h_per_1M_tok_30_layers']:.4f} h" for k, v in fb.items())))
        rb = row.get("reference_backward")
        if rb:
            C.append(_crit("info.ref_bwd", "info", "the fork's reference backward at the same shape (for §7.2's "
                           "brackets)", "INFO", round(rb["bwd_h_per_1M_tok_30_layers"], 4),
                           f"{rb['bwd_over_native_fwd']:.2f}x native fwd, {rb['bwd_ops_per_tok']:.1f} ops/tok, "
                           f"h per 1M tokens over 30 layers"))
    else:
        C.append(_crit("M2-0.1", "pass bar", f"fixed GDN backward <= {BAR_BWD_OVER_NATIVE_FWD}x native forward", na,
                       None, "needs the native forward (owner's cards)"))
        C.append(_crit("M2-0.K", "kill", f"GDN backward < {KILL_BWD_H_PER_1M} h per 1M tokens on the 35B", na, None,
                       "needs the owner's cards"))
    # ---- dX cosine / forward rel on the real layer
    lay = [r for r in gdn.get("layer", []) if isinstance(r, dict) and r.get("dtype") == "bf16"]
    lay = sorted(lay, key=lambda r: r["B"] * r["T"])
    fix_names = ("fix_b32", "fix") if gpu else ("fix_exact_b32", "fix_exact", "fix")
    if lay:
        L = lay[-1]
        arms = L["arms"]
        arm = next((n for n in fix_names if n in arms and "dx_cos" in arms[n]), None)
        if arm:
            r = arms[arm]
            C.append(_crit("M2-0.2", "pass bar", f"dX cosine >= {BAR_DX_COS} vs HF's torch delta rule (layer-0 "
                           f"mixer, B={L['B']} T={L['T']} bf16, arm {arm})",
                           ("PASS" if r["dx_cos"] >= BAR_DX_COS else "FAIL") + ("" if gpu else " (CPU stand-in)"),
                           round(r["dx_cos"], 8)))
            vt = r.get("vs_truth", {})
            vb = r.get("vs_hf_b32", {})
            C.append(_crit("M2.fwd", "M2 prereq", f"forward rel. err <= {BAR_FWD_REL} vs HF's torch delta rule "
                           "(same layer)", ("PASS" if r["fwd_rel_err"] <= BAR_FWD_REL else "FAIL")
                           + ("" if gpu else " (CPU stand-in)"), float(f"{r['fwd_rel_err']:.3g}"),
                           f"vs HF with fp32 sigmoid(b): {vb.get('fwd_rel', float('nan')):.3g}; vs fp32 truth: "
                           f"{vt.get('fwd_rel', float('nan')):.3g} (HF bf16 itself vs truth: "
                           f"{arms.get('hf_b32', {}).get('vs_truth', {}).get('fwd_rel', float('nan')):.3g}); an exact "
                           "kernel cannot meet 1e-3 against HF's bf16-rounded sigmoid, see README"))
        sh = arms.get("fork_shim")
        if sh and "fork_shim" in sh:
            ok = sh["fork_shim"].get("patched") == 1 and "Qwen3_5MoeGatedDeltaNet" in sh["fork_shim"].get("classes", {})
            C.append(_crit("M2.shim", "M2 prereq", "the fork's gdn_hip patch widened to Qwen3_5MoeGatedDeltaNet "
                           "(NativeGDNShim, reference backward) matches HF", "PASS" if ok and sh.get("dx_cos", 0) >=
                           BAR_DX_COS else "FAIL", dict(patched=sh["fork_shim"].get("patched"),
                                                        dx_cos=sh.get("dx_cos"), fwd_rel=sh.get("fwd_rel_err"))))
        elif gpu:
            C.append(_crit("M2.shim", "M2 prereq", "fork NativeGDNShim widened to the MoE class", "FAIL",
                           None, (sh or {}).get("error", "fork_shim arm missing")[:300]))
    else:
        C.append(_crit("M2-0.2", "pass bar", f"dX cosine >= {BAR_DX_COS} vs HF (real layer)", na, None, "layer part not run"))
    ex = gdn.get("exact", {}).get("worst", {})
    if ex:
        key = "fix/bf16"
        C.append(_crit("info.exact", "info", "native math vs fp64 truth (worst over T; fwd rel / min grad cos)",
                       "INFO", {k: [float(f"{v['fwd_rel']:.3g}"), round(v["min_grad_cos"], 7)] for k, v in ex.items()
                                if k.startswith(("fix/", "hf_ref/", "fix_wmma/", "fork_ref/"))},
                       f"fix/bf16 vs fp64: {ex.get(key, {})}"))
    md = gdn.get("model", {})
    if "arms" in md:
        C.append(_crit("M2.model", "M2 prereq", "model-level train step with HF's default use_cache=True: every GDN "
                       "layer runs the patched kernel (no silent HF fallback)", "PASS" if md.get("all_ok") else "FAIL",
                       {k: [v.get("calls"), v.get("fallbacks")] for k, v in md["arms"].items()}))
    # ---- AWQ bit-exact
    awq = None
    for p in sorted((out / "awq").glob("awq_validate_*.json")):
        awq = _load(p)
    if awq and "1_bitexact" in awq:
        b = awq["1_bitexact"]
        ok, txt = True, []
        for L, v in b.items():
            if isinstance(v, dict):
                good = v["ours_eq_ref_fp32"] == v["matrices"] == v["ours_eq_ref_bf16"] and v.get("nan", 0) == 0 \
                    and v.get("stacked_module_eq_ref") == v.get("experts")
                ok &= good
                txt.append(f"{L}: {v['ours_eq_ref_fp32']}/{v['matrices']} fp32, {v['ours_eq_ref_bf16']} bf16, fork "
                           f"{v.get('ours_eq_fork_fp32')}, stacked {v.get('stacked_module_eq_ref')}/{v.get('experts')}")
        C.append(_crit("M2.awq", "M2 prereq", "AWQ dequant bit-exact vs a reference unpack", "PASS" if ok else "FAIL",
                       "; ".join(txt)))
    # ---- grouped_mm
    gm = [(_load(p), p.stem) for p in sorted(out.glob("grouped_mm_*.json"))]
    gm = [(g, n) for g, n in gm if g]
    if gm:
        runs = all(g.get("grouped_mm_runs") for g, _ in gm)
        det = "; ".join(f"{n}: " + ", ".join(f"{k} runs={v['runs_all_cases']} accurate={v['accurate_all_cases']}"
                                             for k, v in g.get("verdict", {}).items()) for g, n in gm)
        sp = [c.get("torch._grouped_mm", {}).get("speedup_vs_loop_fwdbwd") for g, _ in gm for c in g["cases"]]
        C.append(_crit("7.5#2", "measure", "torch grouped_mm runs on " + ("gfx12 (both cards)" if gpu else "CPU")
                       + " at the 35B MoE shapes, fwd + dX", "PASS" if runs else "FAIL (use the chunked bmm fallback)",
                       [round(x, 2) for x in sp if x], det + " (value: speed-up vs the per-expert loop)"))
    # ---- per-card peak at the planned split
    sf = _load(out / "split_fix.json")
    if sf and gpu:
        for run in sf.get("runs", []):
            if "per_card" in run:
                pk = {k: round(v["peak_allocated_gb"], 2) for k, v in run["per_card"].items()}
                rs = {k: round(v["peak_reserved_gb"], 2) for k, v in run["per_card"].items()}
                st = run["steps"][-1]
                C.append(_crit(f"M2-0.3@{run['chunk_tokens']}", "pass bar", f"per-card peak <= {BAR_CARD_PEAK_GB} GB at "
                               f"the planned split (layers [0,{sf['split']}) | [{sf['split']},{sf['n_layers']}), "
                               f"{run['chunk_tokens']}-token MoE chunk, 4k mixer pieces)",
                               "PASS" if run["pass_15_5GB_all_cards"] else "FAIL", pk,
                               f"reserved {rs}; step {st['fwd_s'] + st['bwd_s']:.1f}s = {st['h_per_1M_tok']:.3f} h per 1M "
                               f"tokens (§7.2: 0.1-0.45 h); weights: {sf['weights'][:40]}"))
            elif "oom" in run:
                C.append(_crit(f"M2-0.3@{run['chunk_tokens']}", "pass bar", f"per-card peak <= {BAR_CARD_PEAK_GB} GB "
                               f"({run['chunk_tokens']}-token chunk)", "FAIL (OOM)", None, run["oom"][:200]))
        if not sf.get("gdn_all_calls_patched", True):
            C.append(_crit("M2.split_gdn", "M2 prereq", "every GDN call in the split step ran the patched kernel",
                           "FAIL", None))
    elif not gpu:
        C.append(_crit("M2-0.3", "pass bar", f"per-card peak <= {BAR_CARD_PEAK_GB} GB at the planned split", na,
                       None, "needs the owner's cards" + (" (CPU plumbing check ran: checkpointing on/off dX rel "
                                                          f"{sf['runs'][0].get('checkpoint_check')})" if sf else "")))
    sr = _load(out / "split_fork_ref.json")
    if sr and gpu:
        for run in sr.get("runs", []):
            pk = {k: round(v["peak_allocated_gb"], 2) for k, v in run.get("per_card", {}).items()}
            C.append(_crit(f"info.split_ref@{run['chunk_tokens']}", "info", "same step with the fork's reference "
                           "backward (DESIGN's 16.6 GB est. at 8k)", "INFO" if "oom" not in run else "OOM", pk,
                           f"{run['steps'][-1]['h_per_1M_tok']:.3f} h per 1M tokens" if run.get("steps") else ""))
    return C


def write_summary(out: Path, crits: list[dict], meta: dict) -> str:
    lines = [f"M2-0 summary  ({meta.get('mode')}, {time.strftime('%Y-%m-%d %H:%M:%S')})",
             f"host {REPO} @ {REV}", f"weights: {meta.get('weights')}", ""]
    for c in crits:
        v = c["value"]
        vs = json.dumps(v) if isinstance(v, (dict, list)) else ("-" if v is None else str(v))
        lines.append(f"[{c['status']:<22s}] {c['id']:<14s} ({c['kind']}) {c['criterion']}")
        lines.append(f"{'':27s}value: {vs}")
        if c["detail"]:
            lines.append(f"{'':27s}{c['detail']}")
    bars = [c for c in crits if c["kind"] in ("pass bar", "kill")]
    failed = [c["id"] for c in bars if c["status"].startswith("FAIL")]
    unmeasured = [c["id"] for c in bars if c["status"].startswith("N/A")]
    cpu_only = [c["id"] for c in bars if "CPU" in c["status"]]
    overall = ("NOT PASSED (" + ", ".join(failed) + ")" if failed else
               f"INCOMPLETE ({len(unmeasured)} of {len(bars)} bars not measured: {', '.join(unmeasured)})" if unmeasured
               else f"INCOMPLETE (CPU stand-in only: {', '.join(cpu_only)})" if cpu_only else
               "PASS" if bars else "N/A")
    lines += ["", "M2-0 overall: " + overall]
    lines += ["steps: " + ", ".join(f"{s['name']}={s['rc']}" for s in meta.get("steps", []))]
    txt = "\n".join(lines)
    (out / "M2_0_SUMMARY.txt").write_text(txt + "\n")
    (out / "M2_0_SUMMARY.json").write_text(json.dumps(dict(meta, criteria=crits), indent=1, default=str))
    return txt


def pack(out: Path) -> Path:
    tgz = out.with_suffix(".tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".txt", ".log") and p.stat().st_size < 50e6:
                t.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
    return tgz


# ------------------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------------------

def _resolve_ckpt(a) -> tuple[Path | None, dict]:
    if a.ckpt:
        st = checkpoint_status(a.ckpt)
        return a.ckpt, st
    for c in hf_snapshot_candidates():
        st = checkpoint_status(c)
        if st.get("complete"):
            return c, st
    return None, dict(candidates=[str(c) for c in hf_snapshot_candidates()], complete=False)


def ensure_weights(a, py: str) -> tuple[Path | None, str]:
    ckpt, st = _resolve_ckpt(a)
    if ckpt and st.get("complete"):
        if not st.get("config_matches_pinned_revision", True):
            print(f"WARNING: {ckpt}/config.json differs from revision {REV}'s; continuing (recorded)")
        return ckpt, f"local checkpoint {ckpt}"
    if a.allow_full_download:
        print(f"downloading the whole {REPO}@{REV} (~25 GB) because --allow-full-download was given")
        code = ("from huggingface_hub import snapshot_download as s; "
                f"print(s({REPO!r}, revision={REV!r}, allow_patterns=['*.json', '*.safetensors']))")
        p = subprocess.run([py, "-c", code], capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"snapshot_download failed: {p.stderr[-800:]}")
        return Path(p.stdout.strip().splitlines()[-1]), "downloaded checkpoint (--allow-full-download)"
    man = a.cache_dir / "cyankiwi_35b" / "MANIFEST.json"
    t = json.loads(man.read_text())["tensors"] if man.exists() else {}
    lp = "model.language_model.layers"
    has = lambda L, part: any(n.startswith(f"{lp}.{L}.{part}") for n in t)  # noqa: E731
    want = []                                  # minimal: L0 whole + L3 dense; L3's experts are a bonus
    if not (has(0, "mlp.experts.") and has(0, "linear_attn.") and has(0, "input_layernorm")):
        want.append("0")
    if not (has(3, "self_attn.") and has(3, "mlp.gate.")):
        want.append("3:dense")
    if want and a.no_download:
        raise SystemExit(f"no local checkpoint ({st}) and the layer cache lacks {want}; --no-download given")
    if a.gpu_mode and not a.no_download and not has(3, "mlp.experts."):
        want.append("3")                       # experts too (whole-L3 checks); skipped by the fetcher's guards
    if want:
        print(f"no complete local checkpoint found ({st.get('candidates') or st.get('path')}); range-reading "
              f"{want} only (<= {RANGE_FETCH_MAX / 1e9:.1f} GB per call) into {a.cache_dir}", flush=True)
        for spec in want:
            cmd = [py, str(HERE / "fetch_layer_cache.py"), "--cache-dir", str(a.cache_dir), "--max-bytes",
                   str(RANGE_FETCH_MAX), "--track-budget", "1e12", "--layer", spec]
            if (a.cache_dir / "b1_gdn_L0" / "MANIFEST.json").exists():
                cmd += ["--reuse-from", str(a.cache_dir / "b1_gdn_L0")]
            if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
                if spec == "3":
                    print("  (L3 experts not fetched; whole-L3 checks are skipped)")
                    continue
                raise SystemExit(f"range fetch of {spec} failed")
    return None, f"layer cache {a.cache_dir / 'cyankiwi_35b'} (layers 0, 3; the split step replicates them)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="results dir (default experiments/m2_0/results/owner_<time>)")
    ap.add_argument("--ckpt", type=Path, default=None, help="local checkpoint dir (default: the HF snapshot of the pinned revision)")
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("NGM_M2_0_CACHE",
                                                                          Path.home() / ".cache" / "ngram-memory" / "m2_0_cache")))
    ap.add_argument("--no-download", action="store_true", help="never touch the network")
    ap.add_argument("--allow-full-download", action="store_true",
                    help="allow downloading the whole checkpoint (~25 GB) when no local snapshot exists")
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    ap.add_argument("--image", default=os.environ.get("NGM_IMAGE"), help="run the steps inside this docker image")
    ap.add_argument("--serve-port", type=int, action="append", default=None)
    ap.add_argument("--ignore-pid", type=int, action="append", default=[],
                    help="a PID the preflight should not count (only if you know it does not use the GPU)")
    ap.add_argument("--lease-timeout", type=int, default=0, help="seconds to wait for the lease (0: refuse if leased)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="small sizes on the cards (~10-15 min)")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--selftest-cpu", action="store_true", help="CPU self-test of every step (no GPU, no lease)")
    ap.add_argument("--skip", default="", help="comma list of step names to skip")
    ap.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    mode = "cpu" if a.selftest_cpu else ("quick" if a.quick else "full")
    out = (a.out or ROOT / "experiments" / "m2_0" / "results" / f"owner_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    a.cache_dir = a.cache_dir.resolve()
    ports = tuple(a.serve_port) if a.serve_port else SERVE_PORTS
    gpu = mode != "cpu"
    a.gpu_mode = gpu

    if not a._inner:
        pf = preflight(ports, ignore_pids=tuple(a.ignore_pid))
        print("preflight:", json.dumps(pf, indent=1))
        if gpu:
            print("gpu-status:\n" + gpu_status_text())
        if not pf["ok"] and gpu:
            print("\nREFUSED:\n  - " + "\n  - ".join(pf["refusals"]))
            return 3
        if a.preflight_only:
            print("preflight only: OK" if pf["ok"] else "preflight only: would refuse")
            return 0 if pf["ok"] else 3
        if gpu and not os.environ.get("LEASE_ROCR_DEVICES"):
            lease = shutil.which("gpu-lease")
            if not lease:
                print("REFUSED: gpu-lease is not on PATH (every GPU job on this box runs under it; `lease install`)")
                return 3
            out.mkdir(parents=True, exist_ok=True)
            marker = out / ".inner_started"
            marker.unlink(missing_ok=True)
            wait = ["--nowait"] if a.lease_timeout <= 0 else ["--timeout", str(a.lease_timeout)]
            inner = _inner_cmd(a, out, argv)
            print(f"acquiring the lease: gpu-lease -n 2 {' '.join(wait)} -- ...", flush=True)
            rc = subprocess.run([lease, "-n", "2", *wait, "--", *inner]).returncode
            if not marker.exists():
                print(f"REFUSED: could not get both cards (gpu-lease exit {rc}); they are leased or busy "
                      "(see gpu-status). Re-run when the box is free, or pass --lease-timeout S to wait.")
                return 3
            if rc == 76:
                print("gpu-lease exit 76: a card wedged during the run (not a bug in the command); re-run")
            return rc
        if gpu and len(os.environ["LEASE_ROCR_DEVICES"].split(",")) != 2:
            print(f"REFUSED: the current lease holds {os.environ['LEASE_ROCR_DEVICES']!r}; M2-0 needs -n 2")
            return 3

    # ---- inner: under the lease (or the CPU self-test)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".inner_started").write_text(str(os.getpid()))
    if gpu:
        pf = preflight(ports, ignore_pids=tuple(a.ignore_pid))   # again, inside the lease
        if not pf["ok"]:
            print("REFUSED (inside the lease):\n  - " + "\n  - ".join(pf["refusals"]))
            return 3
    ic = inner_checks(a.python, gpu)
    print("environment:", json.dumps(ic, indent=1))
    if not ic["ok"]:
        print("REFUSED:\n  - " + "\n  - ".join(ic["refusals"]))
        return 3
    ckpt, weights = ensure_weights(a, a.python)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), PYTHONUNBUFFERED="1")
    if gpu:
        env.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
        env.setdefault("HF_HUB_OFFLINE", "1")
    steps = plan_steps(a, a.python, out, ckpt, a.cache_dir, mode)
    meta = dict(mode=mode, repo=REPO, revision=REV, weights=weights, environment=ic.get("info"),
                started=time.strftime("%Y-%m-%dT%H:%M:%S"), git=_git_rev())
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    meta["steps"] = run_steps(steps, out, env)
    meta["wall_s"] = round(time.time() - t0, 1)
    crits = evaluate(out, gpu)
    txt = write_summary(out, crits, meta)
    print("\n" + txt)
    tgz = pack(out)
    print(f"\nsend back: {out / 'M2_0_SUMMARY.txt'} and {tgz} ({tgz.stat().st_size / 1e6:.1f} MB)")
    return 0 if all(s["rc"] == 0 for s in meta["steps"]) else 1


def _git_rev() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _inner_cmd(a, out: Path, argv) -> list[str]:
    args = list(argv if argv is not None else sys.argv[1:])
    args = [x for x in args if x != "--_inner"] + ["--_inner", "--out", str(out), "--cache-dir", str(a.cache_dir)]
    if not a.image:
        return [a.python, str(Path(__file__).resolve()), *args]
    # docker: the canonical ROCm passthrough (fork CLAUDE.md), the lease's device pair forwarded verbatim
    fork = os.environ.get("NGM_MINISGL_FORK", "")
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    mounts = [f"{ROOT}:{ROOT}", f"{out}:{out}", f"{a.cache_dir}:{a.cache_dir}", f"{hf}:/root/.cache/huggingface"]
    if fork:
        mounts.append(f"{fork}:{fork}")
    if a.ckpt:
        mounts.append(f"{a.ckpt.resolve()}:{a.ckpt.resolve()}")
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
           "--security-opt", "seccomp=unconfined", "--security-opt", "label=disable", "--ipc", "host",
           "--shm-size", "16gb", "--network", "host", "-e", "HIP_VISIBLE_DEVICES", "-e", "ROCR_VISIBLE_DEVICES",
           "-e", "LEASE_ROCR_DEVICES", "-e", "HF_HUB_OFFLINE=1", "-e", f"NGM_MINISGL_FORK={fork}"]
    for m in mounts:
        cmd += ["-v", m]
    inner = " ".join(shlex.quote(x) for x in ["python", str(Path(__file__).resolve()), *args, "--python", "python"])
    return cmd + ["--entrypoint", "bash", a.image, "-lc",
                  f"source /app/.venv/bin/activate 2>/dev/null; cd {shlex.quote(str(ROOT))} && {inner}"]


if __name__ == "__main__":
    sys.exit(main())
