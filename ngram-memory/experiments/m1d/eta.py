#!/usr/bin/env python
"""Wall-time estimate of the M1d grid on the self-hosted runner (32 cores), from M1c's measured stage times
(experiments/m1c/results/runner-2026-11-11, PREREG §10 of M1d). No model is run.

    python experiments/m1d/eta.py [--grid experiments/m1d/grid.json] [--parallel 2 3 4] [--threads 32] [--cap 16]

The scheduler is simulated with run.py's own policy (run_parallel): stages start in grid order as soon as their
dependencies are settled, at most --parallel at once, each with max(1, min(cap, free // remaining slots)) threads
fixed at its start. Every stage is a list of phases with a work amount in "seconds alone at 16 threads"; a phase
runs at speed 1 / f(t, busy) with

    f(t, busy) = (16 / t) * (1 + beta * (busy - 16) / 32)          busy = threads in use by all running stages

With countgrid_kl_sets 'r1' in the grid (M1d's default) the count-grid sets other than R1 cost what `lite` says
(from F1's measured per-set forward and mixing times); without it, M1c's measured evaluation times are used.
beta is calibrated on M1c's two measured operating points (alone at 16 threads, and 4 jobs x 8 threads):
training steps 23.0 s vs 49.5 s -> beta_train = 0.152; arm evaluations 3,446 s vs 8,850 s -> beta_eval = 0.568.
--eval-thread-exp < 1 makes evaluations scale worse than linearly with threads (sensitivity).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------------- measured unit costs
STEP_NLL = 23.02                 # s per 4k step, alone at 16 threads (M1c F1 train.json step_s_mean)
DISTILL = 54.83 / 49.49          # distill / nll step at 4x8 (M1c D1 vs N1)
TEACHER_WIN = 20.90 * 23.02 / 49.49   # teacher forward per window: 20.9 s at 4x8 (D1), scaled to alone@16
SETUP = 70.0                     # per training process: corpus, rows, KL targets, host load (F1: 7,194 s - 310 x 23)
EVAL_ARM_M1C = 3446.0            # rows + countgrid:kn+rows on dev, r1, r2, r2ml, alone at 16 (M1c F1 eval)
EVAL_C0_M1C = 3555.0 * 3446.0 / 8850.0   # C0 at 4x8 (3,555 s) scaled like an arm evaluation to alone@16


def lite(wall: float, mix: float, comp: float, two: bool) -> float:
    """A count-grid set without KL (countgrid_kl_sets): one forward instead of two when the inner condition has
    hooks, and per weight 3 full-vocabulary passes (mix, argmax) instead of 8 (mix, log, argmax, KL)."""
    fwd = wall - mix - comp
    return (fwd / 2 if two else fwd) + mix * 3 / 8 + comp


# M1c F1 eval at 16 threads, per set: rows condition wall; countgrid+rows wall, mix_s, components_s
_F1 = dict(dev=(281.1, 1309.9, 787.7, 51.3), r1=(188.4, 423.5, 247.0, 16.4), r2=(185.4, 451.4, 273.4, 17.9),
           r2ml=(175.4, 413.0, 232.9, 16.1))
_C0 = dict(dev=(1887.7, 1341.6, 63.7), r1=(524.1, 362.7, 17.7), r2=(556.3, 390.9, 19.3), r2ml=(560.4, 396.2, 20.1))
EVAL_ARM = sum(r + (cw if s == "r1" else lite(cw, m, c, True)) for s, (r, cw, m, c) in _F1.items())
EVAL_C0 = sum(w if s == "r1" else lite(w, m, c, False) for s, (w, m, c) in _C0.items()) * 3446.0 / 8850.0
B0 = 560.0                       # M1c B0, alone at 16
# PC: 512-token windows, no anchors: forward/backward tokens 512 vs 4,096 + 2 x 1,024 per step, plus per-step
# overhead (row gather, optimiser); 1.5 x the token-proportional cost
PC_STEP = STEP_NLL * 512 / 6144 * 1.5
# REG v2 per condition: M1c's measured regurgitation (171 s for 200 x 128 tokens at 16 threads) + the prompt-
# excluded recompute (~30 s) + canary probes: 8 canaries x (48 greedy tokens + 1,001 reference tails of ~28 tokens
# on a cached 256-token prompt, one prompt forward per 16 tails) ~ 8 x 46k forward tokens; M1c's probe forward ran
# ~1,800 tokens/s at 16 threads (8 x 101 x ~280 tokens in ~125 s)
REG_COND = 171.0 + 30.0 + 8 * 46_000 / 1800.0
REG_SETUP = 90.0
BETA_TRAIN = (49.49 / 23.02 / 2 - 1) / 0.5
BETA_EVAL = (8850.0 / 3446.0 / 2 - 1) / 0.5


def stage_phases(grid: dict, n_windows: int = 310, pc_windows: int = 60) -> dict[str, list[tuple[float, str]]]:
    arms = grid["arms"]
    kl_r1_only = grid["eval"]["full"].get("countgrid_kl_sets") == "r1"
    EVAL_ARM_ = EVAL_ARM if kl_r1_only else EVAL_ARM_M1C
    EVAL_C0_ = EVAL_C0 if kl_r1_only else EVAL_C0_M1C
    out: dict[str, list[tuple[float, str]]] = {}
    for st in grid["stages"]:
        n, k = st["name"], st["kind"]
        if k == "base":
            out[n] = [(B0, "eval")]
        elif k == "countgrid":
            out[n] = [(EVAL_C0_, "eval")]
        elif k in ("train", "train_eval"):
            a = dict(grid["common"], **arms[n])
            per = pc_windows if a.get("train_windows") == "canary" else n_windows
            steps = per * int(a.get("epochs", 1))
            step = PC_STEP if a.get("train_windows") == "canary" else STEP_NLL
            ph = [(SETUP, "train")]
            if a.get("objective", "nll") != "nll":
                ph.append((n_windows * TEACHER_WIN, "train"))
                step *= DISTILL
            ph.append((steps * step, "train"))
            if k == "train_eval":
                ph.append((EVAL_ARM_, "eval"))
            out[n] = ph
        elif k == "rows_eval":
            out[n] = [(EVAL_ARM_, "eval")]
        elif k == "reg2":
            out[n] = [(REG_SETUP + REG_COND * (1 + len(st["conditions"])), "train")]
        else:
            out[n] = [(5.0, "train")]
    return out


def simulate(grid: dict, parallel: int, threads: int, cap: int, eval_exp: float = 1.0,
             beta_train: float = BETA_TRAIN, beta_eval: float = BETA_EVAL) -> dict:
    phases = stage_phases(grid)
    order = [s["name"] for s in grid["stages"]]
    after = {s["name"]: set(s.get("after", [])) for s in grid["stages"]}
    done: set[str] = set()
    pending = list(order)
    running: dict[str, dict] = {}
    t, log = 0.0, []

    def f(th: int, busy: int, kind: str) -> float:
        if kind == "eval":
            return (16 / th) ** eval_exp * (1 + beta_eval * (busy - 16) / 32)
        return (16 / th) * (1 + beta_train * (busy - 16) / 32)
    while pending or running:
        ready = [n for n in pending if after[n] <= done]
        slots = min(parallel - len(running), len(ready))
        free = threads - sum(r["t"] for r in running.values())
        for i, n in enumerate(ready[:slots]):
            if free < 1:
                break
            th = max(1, min(cap, free // (slots - i)))
            free -= th
            pending.remove(n)
            running[n] = dict(t=th, ph=list(phases[n]), start=t)
        if not running:
            raise RuntimeError(f"deadlock: {pending}")
        busy = sum(r["t"] for r in running.values())
        # time until the first running phase completes
        dt = min(r["ph"][0][0] * f(r["t"], busy, r["ph"][0][1]) for r in running.values())
        t += dt
        fin = []
        for n, r in running.items():
            w, kind = r["ph"][0]
            left = w - dt / f(r["t"], busy, kind)
            if left <= 1e-6:
                r["ph"].pop(0)
                if not r["ph"]:
                    fin.append(n)
            else:
                r["ph"][0] = (left, kind)
        for n in fin:
            r = running.pop(n)
            done.add(n)
            log.append(dict(stage=n, threads=r["t"], start_h=r["start"] / 3600, end_h=t / 3600))
    return dict(parallel=parallel, threads=threads, cap=cap, hours=t / 3600, stages=log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(ROOT / "experiments/m1d/grid.json"))
    ap.add_argument("--parallel", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--eval-thread-exp", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--detail", type=int, default=None, help="print the stage timeline of this --parallel value")
    a = ap.parse_args(argv)
    grid = json.loads(Path(a.grid).read_text())
    res = []
    for p in a.parallel:
        for exp in sorted({a.eval_thread_exp, 1.0, 0.8}, reverse=True):
            for bt, be, tag in ((BETA_TRAIN, BETA_EVAL, "calibrated"), (BETA_TRAIN * 1.5, BETA_EVAL * 1.5, "beta x1.5")):
                r = simulate(grid, p, a.threads, a.cap, exp, bt, be)
                r.update(eval_thread_exp=exp, contention=tag)
                res.append(r)
                if not a.json:
                    print(f"--parallel {p} --threads {a.threads} --max-stage-threads {a.cap}  eval-thread-exp {exp}  "
                          f"{tag:10s}: {r['hours']:.2f} h")
    if a.detail is not None:
        r = simulate(grid, a.detail, a.threads, a.cap, a.eval_thread_exp)
        for s in r["stages"]:
            print(f"  {s['stage']:14s} t={s['threads']:2d}  {s['start_h']:5.2f} h -> {s['end_h']:5.2f} h")
    if a.json:
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
