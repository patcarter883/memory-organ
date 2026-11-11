"""M1c bar arithmetic (experiments/m1c/PREREG.md §4-§6, §12): the counts-only frontier, lambda*, the retention
budget share, counts@matched(X), lambda_X (X's own best counts weight within the bars), Bar A (adds to counts,
frontier to frontier), Bar B (beats counts), and the paired-difference file-cluster CIs they need.

Inputs are scripts/evaluate.py outputs: summary.json (sets.<set>.<label>.paired / paired_nondup for single
conditions, sets.<set>.<label>.countgrid.grid[i].paired / paired_nondup for count grids) and the per-target arrays
(<set>__<label>.npz ScoreResult, <set>__<label>__grid.npz GridResult, <set>__nondup.npy mask in base target order).

Conventions. E1 = -rel_pct (relative NLL reduction, %) on the scored set; "points" are percentage points of E1.
R1 = WikiText rel_pct, KL = mean KL(base || cond) on R1, R2 = token-weighted rel_pct of r2 (stdlib) and r2ml (ML
code). primary 'nondup': E1 is read on the targets whose 16-gram is not in train (paired_nondup); 'all': on every
target. A paired difference of two conditions A, B scored on the same targets against the same base is
E1(A) - c*E1(B) = -100 * sum((a - base) - c (b - base)) / sum(base), bootstrapped over files exactly like
metrics.paired (cluster_bootstrap, 10k resamples, seed 0).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..metrics import cluster_bootstrap, per_group_sums

BARS = dict(r1_max_pct=0.2, r2_max_pct=0.5, kl_max=0.005)
S_FLOOR = 0.01             # counts@matched: share floor (1% of the retention budget), PREREG §4 / §12
BAR_A = dict(min_pts=1.0, ci_lo_pts=0.5)
BAR_B = dict(ratio=1.5, ci_lo_pts=0.0)
C1_MAX = 1.0 / 3.0


# --------------------------------------------------------------------------------------------- summary access
def _paired(S: Mapping, set_name: str, label: str, weight: float | None, nondup: bool) -> dict | None:
    e = S["sets"].get(set_name, {}).get(label)
    if e is None:
        return None
    key = "paired_nondup" if nondup else "paired"
    if weight is None:
        return e.get(key)
    for row in e["countgrid"]["grid"]:
        if math.isclose(row["weight"], weight, rel_tol=0, abs_tol=1e-12):
            return row.get(key)
    raise KeyError(f"{label}: weight {weight} not in its grid")


def grid_weights(S: Mapping, label: str, set_name: str = "dev") -> list[float]:
    return [float(r["weight"]) for r in S["sets"][set_name][label]["countgrid"]["grid"]]


def point(S: Mapping, label: str, weight: float | None = None, primary: str = "nondup",
          retention: Mapping | None = None, ret_label: str | None = None) -> dict:
    """Dev-stage numbers of one condition (or one grid weight): E1 (primary), E1 all targets, R1, KL, R2.
    `retention`: another summary holding the r* sets (default S); `ret_label`: its label there (default label)."""
    R = retention if retention is not None else S
    rl = ret_label or label
    pa = _paired(S, "dev", label, weight, nondup=False)
    pp = _paired(S, "dev", label, weight, nondup=primary == "nondup") if primary == "nondup" else pa
    if pp is None or pa is None:
        raise KeyError(f"dev {label} (weight {weight}): no {'paired_nondup' if primary == 'nondup' else 'paired'}")
    r1 = _paired(R, "r1", rl, weight, nondup=False)
    r2s = {x: _paired(R, x, rl, weight, nondup=False) for x in ("r2", "r2ml") if x in R["sets"]}
    r2s = {k: v for k, v in r2s.items() if v is not None}
    n2 = sum(v["n_tokens"] for v in r2s.values())
    r2 = sum(v["rel_pct"] * v["n_tokens"] for v in r2s.values()) / max(n2, 1)
    lo, hi = pp["rel_pct_ci"]
    alo, ahi = pa["rel_pct_ci"]
    return dict(label=label, weight=weight, primary=primary, e1=-pp["rel_pct"], e1_ci=[-hi, -lo],
                e1_n_tokens=pp["n_tokens"], e1_all=-pa["rel_pct"], e1_all_ci=[-ahi, -alo],
                r1=r1["rel_pct"], kl=r1.get("mean_kl", 0.0), r2=r2, r2_parts={k: v["rel_pct"] for k, v in r2s.items()})


def within(p: Mapping, bars: Mapping = BARS) -> bool:
    return bool(p["r1"] <= bars["r1_max_pct"] and p["r2"] <= bars["r2_max_pct"] and p["kl"] <= bars["kl_max"])


# ------------------------------------------------------------------------------------------ counts frontier
def frontier(S: Mapping, label: str = "countgrid[kn]", primary: str = "nondup") -> list[dict]:
    return [point(S, label, w, primary) for w in grid_weights(S, label)]


def lambda_star(front: Sequence[Mapping], bars: Mapping = BARS) -> dict:
    """The counts-only weight with the largest dev E1 among weights within R1, R2 and KL (ties: smaller weight)."""
    ok = [p for p in front if within(p, bars)]
    if not ok:
        raise ValueError("no counts-only weight is within the bars (weight 0 should always be)")
    return max(ok, key=lambda p: (p["e1"], -p["weight"]))


def share(p: Mapping, bars: Mapping = BARS) -> float:
    """The share of the retention budget a condition uses: max(R1/0.2, R2/0.5, KL/0.005), each term floored at 0
    (a negative retention cost uses none of the budget). 1.0 = at the edge of the bars."""
    return float(max(max(p["r1"], 0.0) / bars["r1_max_pct"], max(p["r2"], 0.0) / bars["r2_max_pct"],
                     max(p["kl"], 0.0) / bars["kl_max"]))


def matched(front: Sequence[Mapping], x: Mapping, s_floor: float = S_FLOOR, bars: Mapping = BARS) -> dict:
    """counts@matched(X): the largest counts-only grid weight w with share(counts@w) <= max(share(X), s_floor).
    One scalar budget share, so the retention measure X spends most of its budget on sets the match (a per-measure
    match would let the measure where output mixing is relatively expensive, KL, push the counts to a tiny weight).
    The floor keeps a non-zero weight feasible when X's cost is about 0; weight 0 (share 0) always qualifies."""
    sx = share(x, bars)
    t = max(sx, float(s_floor))
    ok = [p for p in front if share(p, bars) <= t]
    if not ok:
        raise ValueError("no counts-only weight matches X's retention cost (weight 0 should always)")
    best = max(ok, key=lambda p: p["weight"])
    return dict(best, share=share(best, bars), x_share=sx, threshold=t, s_floor=float(s_floor))


def lambda_x(plus: Sequence[Mapping], x_alone_within: bool, bars: Mapping = BARS) -> dict:
    """lambda_X: the weight of X's own counts grid (X + counts@w) with the largest dev E1 among the weights where
    X + counts is within the bars (ties -> the smaller weight). Weight 0 is X alone; it qualifies only if X alone is
    within the bars (x_alone_within). Returns dict(weight=None, reason=...) when no weight qualifies."""
    ok = [p for p in plus if (x_alone_within if p["weight"] == 0 else within(p, bars))]
    table = [dict(weight=p["weight"], e1=p["e1"], share=share(p, bars),
                  within=bool(x_alone_within if p["weight"] == 0 else within(p, bars))) for p in plus]
    if not ok:
        return dict(weight=None, point=None, table=table,
                    reason="no weight of X + counts (X alone included) is within R1 <= 0.2%, R2 <= 0.5%, KL <= 0.005")
    best = max(ok, key=lambda p: (p["e1"], -p["weight"]))
    return dict(weight=best["weight"], point=best, table=table, reason=None)


# ------------------------------------------------------------------------------------------ per-target data
@dataclass
class Targets:
    """One eval directory's per-target arrays for one set: base NLL, file groups, primary mask."""
    base: np.ndarray
    groups: np.ndarray
    key: np.ndarray
    mask: np.ndarray            # primary-metric targets (all True for primary 'all')
    eval_dir: Path
    set_name: str

    @classmethod
    def load(cls, eval_dir: str | Path, set_name: str = "dev", primary: str = "nondup") -> "Targets":
        from .scoring import ScoreResult
        d = Path(eval_dir)
        b = ScoreResult.load(d / f"{set_name}__base.npz")
        groups = np.asarray([b.groups[i] for i in b.doc])
        if primary == "nondup":
            mask = np.load(d / f"{set_name}__nondup.npy").astype(bool)
            if mask.shape != b.nll.shape:
                raise ValueError(f"{d}: nondup mask does not match the base targets")
        else:
            mask = np.ones(b.nll.shape, bool)
        return cls(b.nll.astype(np.float64), groups, b.key(), mask, d, set_name)

    def nll(self, label: str, weight: float | None = None, eval_dir: str | Path | None = None) -> np.ndarray:
        """Per-target NLL of `label` (a ScoreResult npz, or one weight of a __grid.npz) aligned to these targets."""
        d = Path(eval_dir) if eval_dir is not None else self.eval_dir
        if weight is None:
            from .scoring import ScoreResult
            r = ScoreResult.load(d / f"{self.set_name}__{label}.npz")
            k, v = r.key(), r.nll
        else:
            from .countgrid import GridResult
            g = GridResult.load(d / f"{self.set_name}__{label}__grid.npz")
            i = int(np.flatnonzero(np.isclose(g.weights, weight, rtol=0, atol=1e-12))[0])
            k, v = g.key(), g.nll[i]
        if not np.array_equal(k, self.key):
            raise ValueError(f"{d}/{label}: targets differ from the base targets")
        return np.asarray(v, np.float64)


def gain(t: Targets, a: np.ndarray, n_boot: int = 10_000) -> dict:
    """E1 (points) of one condition on t's primary targets, with its file-cluster CI."""
    return diff(t, a, None, 0.0, n_boot)


def diff(t: Targets, a: np.ndarray, b: np.ndarray | None, coef: float = 1.0, n_boot: int = 10_000) -> dict:
    """E1(a) - coef * E1(b) in points on t's primary targets, with the file-cluster bootstrap CI of that paired
    difference (same resampling as metrics.paired)."""
    m = t.mask
    d = (a - t.base) - (coef * (b - t.base) if b is not None else 0.0)
    u, (ds, bs), n = per_group_sums(t.groups[m], [d[m], t.base[m]])
    rel = 100.0 * ds.sum() / bs.sum()
    ci = cluster_bootstrap(ds, n, bs, n_boot, 0)["rel_pct_ci"]
    return dict(pts=-rel, ci=[-ci[1], -ci[0]], n_tokens=int(m.sum()), n_files=int(len(u)), coef=coef)


# ----------------------------------------------------------------------------------------------------- bars
def bar_a(t: Targets, x_plus: np.ndarray | None, counts: np.ndarray, retention_ok: bool | None,
          reason: str | None = None) -> dict:
    """Bar A (frontier to frontier): E1(X + counts@lambda_X) - E1(counts@lambda*) >= 1.0 point with CI lower bound
    > 0.5, X + counts@lambda_X within the retention bars (retention_ok, from dev; None when not evaluated on this
    set). x_plus None: no weight of X + counts is within the bars, so Bar A fails (reason recorded)."""
    if x_plus is None:
        return dict(diff=None, margin_pts=None, gain_ok=False, retention_ok=False, holds=False,
                    reason=reason or "no X + counts weight within the bars")
    d = diff(t, x_plus, counts, 1.0)
    ok = d["pts"] >= BAR_A["min_pts"] and d["ci"][0] > BAR_A["ci_lo_pts"]
    return dict(diff=d, margin_pts=d["pts"] - BAR_A["min_pts"], gain_ok=bool(ok), retention_ok=retention_ok,
                holds=bool(ok and (retention_ok is not False)), reason=reason)


def bar_b(t: Targets, x: np.ndarray, counts_matched: np.ndarray) -> dict:
    """Bar B: E1(X) > 0 and E1(X) >= 1.5 x max(E1(counts@matched(X)), 0), and the paired difference
    E1(X) - E1(counts@matched(X)) has a file-cluster CI lower bound > 0. Also reported: the CI of
    E1(X) - 1.5 E1(counts@matched(X))."""
    gx, gc = gain(t, x), gain(t, counts_matched)
    d = diff(t, x, counts_matched, 1.0)
    d15 = diff(t, x, counts_matched, BAR_B["ratio"])
    ratio_ok = gx["pts"] > 0 and gx["pts"] >= BAR_B["ratio"] * max(gc["pts"], 0.0)
    ok = ratio_ok and d["ci"][0] > BAR_B["ci_lo_pts"]
    return dict(e1_x=gx, e1_counts=gc, ratio=(gx["pts"] / gc["pts"]) if gc["pts"] > 0 else None,
                ratio_ok=bool(ratio_ok), diff=d, diff_1p5=d15, holds=bool(ok))


def per_file(t: Targets, a: np.ndarray, b: np.ndarray | None = None) -> dict:
    """Per-file E1(a) - E1(b) (points, each file against its own base NLL) on t's primary targets, with each file's
    share of the primary targets (files by decreasing share), and n_eff = 1 / sum(share^2), the effective number of
    files the pooled number rests on. Diagnostics only (Z_report)."""
    m = t.mask
    d = (a - t.base) - ((b - t.base) if b is not None else 0.0)
    u, (ds, bs), n = per_group_sums(t.groups[m], [d[m], t.base[m]])
    tot = float(n.sum())
    rows = [dict(file=str(f), pts=float(-100.0 * x / y) if y else None, n_targets=int(k),
                 target_share=float(k / tot) if tot else None) for f, x, y, k in zip(u, ds, bs, n)]
    rows.sort(key=lambda r: (-r["n_targets"], r["file"]))
    w = np.asarray([r["target_share"] or 0.0 for r in rows], float)
    return dict(files=rows, n_eff=float(1.0 / (w ** 2).sum()) if w.size and w.any() else None)


def c1(t: Targets, op: np.ndarray, perm: np.ndarray) -> dict:
    g, p = gain(t, op), gain(t, perm)
    ratio = (p["pts"] / g["pts"]) if g["pts"] > 0 else None
    return dict(op=g, perm=p, ratio=ratio, holds=bool(ratio is not None and ratio <= C1_MAX))


def weight_tag(w: float) -> str:
    """0.0075 -> '0p0075' (evaluate.py's fixed-weight countlm spec)."""
    s = f"{w:.10f}".rstrip("0").rstrip(".")
    return (s if "." in s else s).replace(".", "p")


def load_summary(eval_dir: str | Path) -> dict:
    return json.loads((Path(eval_dir) / "summary.json").read_text())


__all__ = ["BARS", "S_FLOOR", "BAR_A", "BAR_B", "C1_MAX", "point", "within", "share", "frontier", "lambda_star",
           "matched", "lambda_x", "per_file", "Targets", "gain", "diff", "bar_a", "bar_b", "c1", "weight_tag", "grid_weights", "load_summary"]
