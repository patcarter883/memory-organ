"""§D experiment G: statistics of Flash-Next's n-gram table rows with no model forward (numpy only).

Definitions (pilot G0, `GF/dissect/row_pilot.py`):

* For an n-gram g of order n with heads H_n (8 heads), its rows e_h (h in H_n) are centred on the head mean row mu_h
  and mapped through the real value projection slice: c_h = W_v[:, h*w:(h+1)*w] (e_h - mu_h)  (d = 2560).
* **coherence(g)** = mean pairwise cosine of {c_h}. A bucket is shared by many n-grams (co-tenants); co-tenant noise
  is independent across the 8 hashes, the n-gram's own signal is common to them, so coherence ~ signal share rho.
  The summed 8-head value then has SNR ~ 8 rho / (1 - rho).
* **f\\***: the proxy-corpus frequency at which the coherence-vs-frequency curve crosses a threshold (0.05, §D.3),
  by linear interpolation of per-bin means in log frequency; CI by bootstrap over n-grams within bins.

N-gram keys: an n-gram (x_1..x_n) is packed as sum x_i V^(n-i) (int64; V^3 < 2^63 for V = 248320).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ------------------------------------------------------------------------------------------------ counting
def pack(ngrams: np.ndarray, vocab: int) -> np.ndarray:
    """[N, n] token ids (oldest first) -> int64 keys."""
    g = np.asarray(ngrams, dtype=np.int64)
    k = np.zeros(g.shape[0], dtype=np.int64)
    for i in range(g.shape[1]):
        k = k * vocab + g[:, i]
    return k


def unpack(keys: np.ndarray, n: int, vocab: int) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64).copy()
    out = np.empty((k.shape[0], n), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[:, i] = k % vocab
        k //= vocab
    return out


def doc_ngram_keys(ids: np.ndarray, n: int, vocab: int) -> np.ndarray:
    """Keys of every n-gram fully inside one document (no eos-padded history)."""
    a = np.asarray(ids, dtype=np.int64)
    if a.shape[0] < n:
        return np.zeros(0, dtype=np.int64)
    k = np.zeros(a.shape[0] - n + 1, dtype=np.int64)
    for i in range(n):
        k = k * vocab + a[i:a.shape[0] - n + 1 + i]
    return k


def count_keys(docs: Iterable[np.ndarray], n: int, vocab: int, chunk_tokens: int = 20_000_000
               ) -> tuple[np.ndarray, np.ndarray]:
    """(sorted unique keys, counts) of the n-grams in `docs`, merged chunk by chunk to bound memory."""
    acc_k = np.zeros(0, dtype=np.int64)
    acc_c = np.zeros(0, dtype=np.int64)
    buf: list[np.ndarray] = []
    nb = 0

    def flush() -> None:
        nonlocal acc_k, acc_c, buf, nb
        if not buf:
            return
        k, c = np.unique(np.concatenate(buf), return_counts=True)
        acc_k, acc_c = merge_counts(acc_k, acc_c, k, c.astype(np.int64))
        buf, nb = [], 0

    for d in docs:
        k = doc_ngram_keys(d, n, vocab)
        buf.append(k)
        nb += k.shape[0]
        if nb >= chunk_tokens:
            flush()
    flush()
    return acc_k, acc_c


def merge_counts(k1: np.ndarray, c1: np.ndarray, k2: np.ndarray, c2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if k1.size == 0:
        return k2, c2
    if k2.size == 0:
        return k1, c1
    k = np.concatenate([k1, k2])
    c = np.concatenate([c1, c2])
    order = np.argsort(k, kind="stable")
    k, c = k[order], c[order]
    first = np.ones(k.shape[0], dtype=bool)
    first[1:] = k[1:] != k[:-1]
    idx = np.flatnonzero(first)
    return k[idx], np.add.reduceat(c, idx)


def lookup_counts(keys: np.ndarray, counts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Counts of query keys in a sorted (keys, counts) table (0 when absent)."""
    q = np.asarray(q, dtype=np.int64)
    if keys.size == 0:
        return np.zeros(q.shape, dtype=np.int64)
    pos = np.searchsorted(keys, q)
    pos_c = np.minimum(pos, keys.shape[0] - 1)
    hit = keys[pos_c] == q
    return np.where(hit, counts[pos_c], 0)


# ------------------------------------------------------------------------------------------------ sampling
def log2_bins(max_count: int) -> list[tuple[int, int]]:
    """[1,1], [2,3], [4,7], ... up to max_count (inclusive bounds)."""
    out, lo = [], 1
    while lo <= max_count:
        out.append((lo, 2 * lo - 1))
        lo *= 2
    return out


def allocate(pool_sizes: Sequence[int], total: int) -> list[int]:
    """Split `total` samples as evenly as possible over bins, capping each at its pool size (water-filling)."""
    pools = [int(p) for p in pool_sizes]
    alloc = [0] * len(pools)
    left = total
    open_ = [i for i, p in enumerate(pools) if p > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for i in open_:
            take = min(share, pools[i] - alloc[i], left)
            alloc[i] += take
            left -= take
            if alloc[i] < pools[i]:
                nxt.append(i)
            if left == 0:
                break
        open_ = nxt
    return alloc


def stratified_sample(keys: np.ndarray, counts: np.ndarray, bins: Sequence[tuple[int, int]], per_bin: int | None,
                      rng: np.random.Generator, exclude: np.ndarray | None = None,
                      total: int | None = None) -> list[tuple[int, int, np.ndarray]]:
    """[(lo, hi, sampled keys)]: uniformly within each count bin, either up to per_bin keys per bin or (with total)
    `total` keys spread evenly over the bins with any shortfall of small bins redistributed (allocate())."""
    pools = []
    for lo, hi in bins:
        m = (counts >= lo) & (counts <= hi)
        if exclude is not None and exclude.size:
            m &= ~np.isin(keys, exclude)
        pools.append(keys[m])
    sizes = allocate([p.shape[0] for p in pools], total) if total is not None else [
        min(int(per_bin), p.shape[0]) for p in pools]
    out = []
    for (lo, hi), pool, k in zip(bins, pools, sizes):
        pick = rng.choice(pool, size=k, replace=False) if k else pool[:0]
        out.append((lo, hi, np.sort(pick)))
    return out


# ------------------------------------------------------------------------------------------------- metrics
def head_slices(w_v: np.ndarray, n_heads: int) -> list[np.ndarray]:
    """W_v [d, H*w] -> per-head [d, w] views."""
    w = w_v.shape[1] // n_heads
    return [w_v[:, h * w:(h + 1) * w] for h in range(n_heads)]


def contributions(rows: np.ndarray, heads: Sequence[int], wh: Sequence[np.ndarray],
                  mu: np.ndarray | None = None) -> np.ndarray:
    """rows [k, w] of heads `heads` -> [k, d] = W_v^h (e_h - mu_h) (mu None: uncentred)."""
    out = []
    for i, h in enumerate(heads):
        e = rows[i] - (mu[h] if mu is not None else 0.0)
        out.append(wh[h] @ e)
    return np.stack(out)


def coherence(c: np.ndarray) -> float:
    """Mean pairwise cosine of the rows of c [k, d]."""
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    u = c / np.maximum(nrm, 1e-30)
    s = u @ u.T
    k = c.shape[0]
    return float((s.sum() - np.trace(s)) / (k * (k - 1)))


def snr_from_coherence(rho: np.ndarray | float) -> np.ndarray | float:
    """SNR of the k=8-head summed value if each head = shared signal (share rho) + independent noise."""
    r = np.asarray(rho, dtype=np.float64)
    return 8 * r / np.maximum(1 - r, 1e-9)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                      rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def crossing(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """First x (in the given, increasing order) at which the piecewise-linear curve y(x) reaches thr from below and
    stays at/above it for the remaining points ("the curve has crossed"); nan if it never does.
    x is typically log10(frequency)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return float("nan")
    above = y >= thr
    if above.all():
        return float(x[0])  # already above at the lowest frequency: report the lowest bin (a bound)
    if not above[-1]:
        return float("nan")
    # last index below thr; curve stays above afterwards
    j = int(np.flatnonzero(~above)[-1])
    x0, x1, y0, y1 = x[j], x[j + 1], y[j], y[j + 1]
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


@dataclass
class BinCurve:
    lo: np.ndarray      # bin lower count
    hi: np.ndarray
    x: np.ndarray       # log10 of the bin's geometric-centre frequency (per token)
    groups: list[np.ndarray]   # per-bin metric values


def bootstrap_crossing(curve: BinCurve, thr: float, n_boot: int = 2000, rng: np.random.Generator | None = None,
                       alpha: float = 0.05) -> tuple[float, float, float, float]:
    """(x*, lo, hi, share of bootstrap replicates with a crossing) for the per-bin-mean curve."""
    rng = rng or np.random.default_rng(0)
    means = np.array([g.mean() if g.size else np.nan for g in curve.groups])
    est = crossing(curve.x, means, thr)
    reps = []
    for _ in range(n_boot):
        m = np.array([g[rng.integers(0, g.size, g.size)].mean() if g.size else np.nan for g in curve.groups])
        reps.append(crossing(curve.x, m, thr))
    reps = np.asarray(reps)
    ok = np.isfinite(reps)
    if not ok.any():
        return est, float("nan"), float("nan"), 0.0
    return est, float(np.quantile(reps[ok], alpha / 2)), float(np.quantile(reps[ok], 1 - alpha / 2)), float(ok.mean())


def value_share(v2: np.ndarray, v3: np.ndarray) -> dict[str, np.ndarray]:
    """Per-position shares of v = v2 + v3 ([N, d] each): projection share <v, v_n>/|v|^2 (sums to 1), norm ratio."""
    v = v2 + v3
    vv = np.maximum((v * v).sum(1), 1e-30)
    return dict(proj2=(v * v2).sum(1) / vv, proj3=(v * v3).sum(1) / vv,
                norm2=np.linalg.norm(v2, axis=1), norm3=np.linalg.norm(v3, axis=1),
                ratio32=np.linalg.norm(v3, axis=1) / np.maximum(np.linalg.norm(v2, axis=1), 1e-30))


def load_fstar(results_json) -> dict[int, float]:
    """{order: f* per token} from a G results file (proofs/dtrack/G/g2_results.json)."""
    import json
    from pathlib import Path
    d = json.loads(Path(results_json).read_text())
    return {int(n): float(o["f_star_per_token"]) for n, o in d["orders"].items() if o.get("f_star_per_token")}


def below_fstar(freq_per_token: np.ndarray, order: int, fstar: dict[int, float]) -> np.ndarray:
    """Row-policy helper: True where an n-gram's general-text frequency (per token, from the same kind of proxy) is
    below that order's f* (Flash-Next's rows are mostly co-tenant noise there)."""
    return np.asarray(freq_per_token, dtype=np.float64) < fstar[order]


# ------------------------------------------------------------------------------------------- text helpers
def wikitext_detokenize(s: str) -> str:
    """Undo WikiText-103's tokenisation artefacts (the detokenizer used by lm-evaluation-harness for wikitext)."""
    import re
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = s.replace(" : ", ": ").replace(" ; ", "; ").replace(" . ", ". ").replace(" ! ", "! ").replace(" ? ", "? ")
    s = s.replace(" , ", ", ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176))
    s = s.replace(" \n", "\n").replace("\n ", "\n")
    s = s.replace(" 's", "'s")
    return s


def wikitext_articles(lines: Iterable[str]) -> Iterable[str]:
    """Group WikiText lines into articles (a top-level ' = Title = ' heading starts one)."""
    import re
    head = re.compile(r"^ = [^=].* = \n?$")
    cur: list[str] = []
    for ln in lines:
        if head.match(ln) and cur:
            yield "".join(cur)
            cur = []
        cur.append(ln)
    if cur:
        yield "".join(cur)


__all__ = ["pack", "unpack", "doc_ngram_keys", "count_keys", "merge_counts", "lookup_counts", "log2_bins", "allocate",
           "stratified_sample", "head_slices", "contributions", "coherence", "snr_from_coherence",
           "bootstrap_mean_ci", "crossing", "BinCurve", "bootstrap_crossing", "value_share",
           "wikitext_detokenize", "wikitext_articles", "load_fstar", "below_fstar"]
