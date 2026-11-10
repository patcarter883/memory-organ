#!/usr/bin/env python
"""Thread / concurrency scaling probe for the M1b stages on the current machine (not part of the pre-registered run).

    python experiments/m1b/scaling_probe.py --out DIR [--steps 4] [--configs 1x8,1x16,1x32,2x16,4x8]
                                            [--eval-windows 16] [--skip-eval]

Training: for each KxT config, K training workers of arm A1_base (the full-budget recipe, 4,096-token windows) run
side by side with T threads each, for --steps steps. Each worker gets its own directory; the KL-target cache is
shared (the first config builds it). Reported: median step time over steps after the first, and throughput
(K / step time, steps per second across the workers), with each worker's peak RSS.

Scoring: evaluate.py with the base condition on the first Split-A dev windows (--eval-windows), alone at each T,
then K copies side by side for each KxT with K > 1. Reported: wall time and windows per second.

Nothing here touches a run directory of experiments/m1b/run.py.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402  (experiments/m1b/run.py: grid, resolve_arm, run_guarded, QWEN25)


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_configs(s: str) -> list[tuple[int, int]]:
    out = []
    for c in s.split(","):
        k, t = c.lower().split("x")
        out.append((int(k), int(t)))
    return out


def train_probe(out: Path, k: int, t: int, steps: int, rss_gb: float) -> dict:
    grid = R.load_grid(False)
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(k):
        d = out / f"train_{k}x{t}_{i}"
        (d / "A1_base").mkdir(parents=True, exist_ok=True)
        if not (d / "_cache").exists():
            (d / "_cache").symlink_to(cache, target_is_directory=True)
        for f in ("checkpoint.pt", "metrics.jsonl", "train.json"):
            (d / "A1_base" / f).unlink(missing_ok=True)
        R.jdump(d / "A1_base" / "arm.json", R.resolve_arm(grid, "A1_base", d))
        cmd = [R.PY, str(HERE / "run.py"), "--worker", "train", "--stage", "A1_base", "--out", str(d),
               "--threads", str(t), "--max-steps", str(steps)]
        jobs.append((d, cmd))
    t0 = time.time()
    with ThreadPoolExecutor(k) as ex:
        runs = list(ex.map(lambda j: R.run_guarded(j[1], j[0] / "A1_base" / "train.log", rss_gb * 1024, t), jobs))
    res = []
    for (d, _), r in zip(jobs, runs):
        recs = [json.loads(x) for x in (d / "A1_base" / "metrics.jsonl").read_text().splitlines() if x.strip()] \
            if (d / "A1_base" / "metrics.jsonl").exists() else []
        st = [x["step_s"] for x in recs]
        res.append(dict(returncode=r["returncode"], aborted=r["aborted"], wall_s=r["wall_s"],
                        peak_rss_mb=r["peak_rss_mb"], step_s=st,
                        step_s_median=statistics.median(st[1:]) if len(st) > 1 else (st[0] if st else None)))
    med = [x["step_s_median"] for x in res if x["step_s_median"]]
    step = statistics.median(med) if med else None
    return dict(k=k, threads=t, workers=res, step_s_median=step,
                steps_per_s=(k / step) if step else None, wall_s=time.time() - t0)


def eval_probe(out: Path, k: int, t: int, windows: int, rss_gb: float) -> dict:
    from ngram_memory.eval.stage1.splits import Splits
    from ngram_memory.ingest.corpus import SPLITS_DIR
    sp = out / "_cache" / "splits.json"
    if not sp.exists():
        sp.parent.mkdir(parents=True, exist_ok=True)
        Splits.from_ingest(SPLITS_DIR / "split_a.json", SPLITS_DIR / "split_b.json").save(sp)
    E = R.load_grid(False)["eval_flags"]
    jobs = []
    for i in range(k):
        d = out / f"eval_{k}x{t}_{i}"
        cmd = [R.PY, str(ROOT / "scripts/evaluate.py"), "--splits", str(sp), "--host", R.QWEN25, "--out", str(d),
               "--sets", "dev", "--conditions", "base", "--threads", str(t), "--context", str(E["context"]),
               "--stride", str(E["stride"]), "--min-context", str(E["min_context"]),
               "--max-windows", str(windows), "--no-classes"]
        jobs.append((d, cmd))
    t0 = time.time()
    with ThreadPoolExecutor(k) as ex:
        runs = list(ex.map(lambda j: R.run_guarded(j[1], j[0].with_suffix(".log"), rss_gb * 1024, t), jobs))
    wall = time.time() - t0
    ok = all(r["returncode"] == 0 and not r["aborted"] for r in runs)
    return dict(k=k, threads=t, windows_each=windows, ok=ok, wall_s=wall,
                windows_per_s=(k * windows / wall) if ok else None,
                peak_rss_mb=[r["peak_rss_mb"] for r in runs])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--configs", default="1x8,1x16,1x32,2x16,4x8")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--rss-limit-gb", type=float, default=11.0)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfgs = parse_configs(a.configs)
    log("machine", dict(cpus=os.cpu_count(), affinity=len(os.sched_getaffinity(0)), configs=cfgs))
    report: dict = dict(cpus=len(os.sched_getaffinity(0)), steps=a.steps, train=[], eval=[])
    for k, t in cfgs:
        log("train", f"{k}x{t}")
        r = train_probe(out, k, t, a.steps, a.rss_limit_gb)
        report["train"].append(r)
        log("train", f"{k}x{t}", f"step {r['step_s_median']}", f"steps/s {r['steps_per_s']}",
            "peak MB", [round(w["peak_rss_mb"]) for w in r["workers"]])
        R.jdump(out / "scaling.json", report)
    if not a.skip_eval:
        for k, t in cfgs:
            log("eval", f"{k}x{t}")
            r = eval_probe(out, k, t, a.eval_windows, a.rss_limit_gb)
            report["eval"].append(r)
            log("eval", f"{k}x{t}", f"wall {r['wall_s']:.0f}s", f"windows/s {r['windows_per_s']}")
            R.jdump(out / "scaling.json", report)
    print("\n| config | train step (s) | train steps/s | eval windows/s |")
    print("|---|---|---|---|")
    ev = {(r["k"], r["threads"]): r for r in report["eval"]}
    for r in report["train"]:
        e = ev.get((r["k"], r["threads"]), {})
        f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
        print(f"| {r['k']}x{r['threads']} | {f(r['step_s_median'], 1)} | {f(r['steps_per_s'])} | "
              f"{f(e.get('windows_per_s'))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
