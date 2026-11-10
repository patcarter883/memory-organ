"""Paired metrics for teacher-forced arms: per-file aggregation and file-cluster bootstrap CIs (§6.A.2, §6.A.7).

The bootstrap follows S1/rt-science/cluster_ci.py: resample files with replacement and recompute ratio-of-sums
statistics, delta = sum(d) / sum(n) and rel = sum(d) / sum(base). numpy only.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def per_group_sums(groups: np.ndarray, values: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """groups [N] (any hashable codes) -> (unique groups, [sum of each value array per group], count per group)."""
    u, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    sums = [np.bincount(inv, weights=np.asarray(v, np.float64), minlength=len(u)) for v in values]
    return u, sums, n


def cluster_bootstrap(d_sum: np.ndarray, n: np.ndarray, b_sum: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                      q: tuple[float, float] = (2.5, 97.5)) -> dict:
    """File-cluster bootstrap of delta/token and relative delta (ratio of sums)."""
    d_sum, n, b_sum = (np.asarray(x, np.float64) for x in (d_sum, n, b_sum))
    k = len(n)
    rng = np.random.default_rng(seed)
    D, N, B = (np.empty(n_boot) for _ in range(3))
    blk = max(1, min(n_boot, 4_000_000 // max(k, 1)))  # resamples per block (bounded memory)
    for b0 in range(0, n_boot, blk):
        m = min(blk, n_boot - b0)
        idx = rng.integers(0, k, size=(m, k))
        D[b0:b0 + m], N[b0:b0 + m], B[b0:b0 + m] = d_sum[idx].sum(1), n[idx].sum(1), b_sum[idx].sum(1)
    delta = D / np.maximum(N, 1e-12)
    rel = D / np.where(B == 0, np.nan, B)
    return dict(delta_ci=[float(x) for x in np.nanpercentile(delta, q)],
                rel_pct_ci=[float(100 * x) for x in np.nanpercentile(rel, q)],
                n_boot=n_boot, seed=seed)


def mde_file_cluster(per_file_delta: np.ndarray, n_files: int | None = None, z: float = 2.8) -> float:
    """Minimum detectable effect (80% power, two-sided 5%: z = 1.96 + 0.84) for an unweighted file-mean,
    from the between-file SD (S1/rt-science/cluster_ci.py)."""
    x = np.asarray(per_file_delta, np.float64)
    if len(x) < 2:
        return float("nan")
    return float(z * x.std(ddof=1) / math.sqrt(n_files or len(x)))


def align(base_key: np.ndarray, cond_key: np.ndarray) -> None:
    if base_key.shape != cond_key.shape or not np.array_equal(base_key, cond_key):
        raise ValueError("base and condition were not scored on the same targets")


def paired(base_nll: np.ndarray, cond_nll: np.ndarray, groups: np.ndarray, *, base_hit: np.ndarray | None = None,
           cond_hit: np.ndarray | None = None, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison on the same targets. groups = bootstrap cluster per target (the file)."""
    b = np.asarray(base_nll, np.float64)
    c = np.asarray(cond_nll, np.float64)
    if b.shape != c.shape or b.shape[0] != len(groups):
        raise ValueError("shape mismatch")
    if len(b) == 0:
        return dict(n_tokens=0)
    d = c - b
    u, (ds, bs), n = per_group_sums(np.asarray(groups), [d, b])
    out = dict(n_tokens=int(len(b)), n_files=int(len(u)), base_nll=float(b.mean()), cond_nll=float(c.mean()),
               delta=float(d.mean()), rel_pct=float(100 * d.mean() / b.mean()) if b.mean() else float("nan"),
               base_ppl=float(math.exp(b.mean())), cond_ppl=float(math.exp(c.mean())))
    out.update(cluster_bootstrap(ds, n, bs, n_boot, seed))
    per_file = ds / n
    out["per_file_delta_sd"] = float(per_file.std(ddof=1)) if len(u) > 1 else float("nan")
    out["mde_nats_80pct"] = mde_file_cluster(per_file)
    out["files_improved"] = int((per_file < 0).sum())
    if base_hit is not None and cond_hit is not None:
        bh, ch = np.asarray(base_hit, np.float64), np.asarray(cond_hit, np.float64)
        out["base_em"] = float(bh.mean())
        out["cond_em"] = float(ch.mean())
        _, (dh,), _ = per_group_sums(np.asarray(groups), [ch - bh])
        ci = cluster_bootstrap(dh, n, n, n_boot, seed)
        out["delta_em_pts"] = float(100 * (ch - bh).mean())
        out["delta_em_pts_ci"] = [100 * x for x in ci["delta_ci"]]
        out["mcnemar"] = mcnemar(bh.astype(bool), ch.astype(bool))
    return out


def summary(nll: np.ndarray, groups: np.ndarray, hit: np.ndarray | None = None) -> dict:
    x = np.asarray(nll, np.float64)
    out = dict(n_tokens=int(len(x)), n_files=int(len(np.unique(groups))), nll=float(x.mean()) if len(x) else float("nan"))
    if len(x):
        out["ppl"] = float(math.exp(x.mean()))
    if hit is not None and len(x):
        out["em"] = float(np.asarray(hit).mean())
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired booleans (a = base correct, b = arm correct)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        # two-sided exact binomial tail, computed in log space
        logc = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
        p = min(1.0, 2 * sum(math.exp(x) for x in logc))
    return dict(n01=n01, n10=n10, p=p)


def class_breakdown(base_nll: np.ndarray, cond_nll: np.ndarray, groups: np.ndarray,
                    classes: Mapping[str, np.ndarray], n_boot: int = 2_000, seed: int = 0) -> dict:
    """paired() per token class (boolean masks over the same targets), plus each class's share of the gain."""
    b, c = np.asarray(base_nll, np.float64), np.asarray(cond_nll, np.float64)
    tot = (c - b).sum()
    out = {}
    for name, m in classes.items():
        m = np.asarray(m, bool)
        if not m.any():
            continue
        r = paired(b[m], c[m], np.asarray(groups)[m], n_boot=n_boot, seed=seed)
        r["frac"] = float(m.mean())
        r["share_of_gain"] = float((c[m] - b[m]).sum() / tot) if tot else float("nan")
        out[name] = r
    return out


def retention(base_nll: np.ndarray, cond_nll: np.ndarray, groups: np.ndarray, kl: np.ndarray | None = None,
              agree: np.ndarray | None = None, n_boot: int = 10_000, seed: int = 0) -> dict:
    """R1/R2: paired ΔNLL plus mean KL(base || ON) and top-1 agreement when available."""
    r = paired(base_nll, cond_nll, groups, n_boot=n_boot, seed=seed)
    if kl is not None:
        r["mean_kl"] = float(np.mean(kl))
    if agree is not None:
        r["top1_agreement"] = float(np.mean(agree))
    return r


__all__ = ["paired", "summary", "cluster_bootstrap", "mde_file_cluster", "mcnemar", "class_breakdown", "retention",
           "per_group_sums", "align"]
