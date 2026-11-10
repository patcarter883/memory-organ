"""Zero-training baseline (§6.A.4 "N-gram / cache-LM interpolation"): the host interpolated with a count-based
project LM = interpolated modified Kneser-Ney trigram (Chen & Goodman) on the train files + an in-file cache
(Witten-Bell bigram over the file prefix; Tu et al. 2014, Hellendoorn & Devanbu 2017):

    p(y) = (1 - lam_n - lam_c) p_host(y) + lam_n p_KN(y | x_{t-1}, x_t) + lam_c p_cache(y | x_t; file prefix)

lam_n, lam_c are tuned on dev targets only (`tune_lambdas`, a grid over the stored target probabilities; no extra
host forwards). Contexts before a document start are EOS (as the hasher). "Gated" use = only on project text
(retention cost 0 by construction); `CountLMCondition` applied to retention sets is the always-on variant.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np
import torch

from ..scoring import Condition, Doc, ScoreResult, Window


def _discounts(counts: np.ndarray) -> tuple[float, float, float]:
    n = [int((counts == k).sum()) for k in (1, 2, 3, 4)]
    if n[0] == 0 or n[1] == 0:
        return 0.5, 0.75, 0.9
    Y = n[0] / (n[0] + 2 * n[1])
    d1 = 1 - 2 * Y * n[1] / n[0]
    d2 = 2 - 3 * Y * n[2] / n[1] if n[1] else 1.0
    d3 = 3 - 4 * Y * n[3] / n[2] if n[2] else d2
    return tuple(float(min(max(d, 0.05), k)) for d, k in zip((d1, d2, d3), (1.0, 2.0, 3.0)))  # type: ignore


def _disc(c: np.ndarray, D: tuple[float, float, float]) -> np.ndarray:
    return np.where(c >= 3, D[2], np.where(c == 2, D[1], np.where(c == 1, D[0], 0.0)))


class _Level:
    """Sparse counts of one order grouped by context: sorted context keys and CSR (w, count) lists."""

    def __init__(self, ctx: np.ndarray, w: np.ndarray, c: np.ndarray):
        self.D = _discounts(c)
        self.ctx_keys, start, cnt = np.unique(ctx, return_index=True, return_counts=True)
        self.start, self.end = start, start + cnt
        self.w, self.c = w, c.astype(np.float64)
        self.num = np.maximum(self.c - _disc(c, self.D), 0.0)
        tot = np.add.reduceat(self.c, start) if len(c) else np.zeros(0)
        n1 = np.add.reduceat((c == 1).astype(np.float64), start) if len(c) else np.zeros(0)
        n2 = np.add.reduceat((c == 2).astype(np.float64), start) if len(c) else np.zeros(0)
        n3 = np.add.reduceat((c >= 3).astype(np.float64), start) if len(c) else np.zeros(0)
        self.total = tot
        self.gamma = (self.D[0] * n1 + self.D[1] * n2 + self.D[2] * n3) / np.maximum(tot, 1e-12)

    def find(self, ctx: np.ndarray) -> np.ndarray:
        """index into ctx_keys, or -1."""
        if self.ctx_keys.size == 0:
            return np.full(ctx.shape, -1, np.int64)
        i = np.searchsorted(self.ctx_keys, ctx).clip(0, self.ctx_keys.size - 1)
        return np.where(self.ctx_keys[i] == ctx, i, -1)

    def target(self, ci: np.ndarray, w: np.ndarray) -> np.ndarray:
        """discounted numerator / total for (context index, w); 0 where absent."""
        out = np.zeros(len(ci))
        for k, (i, y) in enumerate(zip(ci.tolist(), w.tolist())):
            if i < 0:
                continue
            a, b = self.start[i], self.end[i]
            j = a + np.searchsorted(self.w[a:b], y)
            if j < b and self.w[j] == y:
                out[k] = self.num[j] / self.total[i]
        return out


class KNTrigram:
    """Interpolated modified Kneser-Ney 3-gram over token ids (vocab V; contexts before a doc start = eos)."""

    def __init__(self, seqs: Iterable[np.ndarray], vocab: int, eos: int):
        V = self.V = int(vocab)
        self.eos = int(eos)
        if V ** 3 >= 2 ** 63:
            raise ValueError("vocab too large for int64 trigram keys")
        keys = []
        for s in seqs:
            s = np.asarray(s, np.int64)
            if len(s) == 0:
                continue
            p = np.concatenate([[eos, eos], s])
            keys.append((p[:-2] * V + p[1:-1]) * V + p[2:])
        k3, c3 = np.unique(np.concatenate(keys), return_counts=True) if keys else (np.zeros(0, np.int64),) * 2
        self.n_tokens = int(c3.sum()) if len(c3) else 0
        self.l3 = _Level(k3 // V, k3 % V, c3)
        k2, c2 = np.unique(k3 % (V * V), return_counts=True)   # continuation counts N1+(. v w)
        self.l2 = _Level(k2 // V, k2 % V, c2)
        w1, c1 = np.unique(k2 % V, return_counts=True)          # N1+(. w)
        D1 = _discounts(c1)
        tot1 = float(c1.sum())
        self.p1 = np.zeros(V)
        self.p1[w1] = np.maximum(c1 - _disc(c1, D1), 0) / tot1
        g1 = float((D1[0] * (c1 == 1).sum() + D1[1] * (c1 == 2).sum() + D1[2] * (c1 >= 3).sum()) / tot1)
        self.p1 += g1 / V
        # raw trigram counts for the "confident context" headroom statistic
        self.c3_raw = self.l3

    def _ctx(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.l3.find(u * self.V + v), self.l2.find(v)

    def prob(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """p(w | u, v) for arrays of equal length."""
        u, v, w = (np.asarray(x, np.int64) for x in (u, v, w))
        i3, i2 = self._ctx(u, v)
        g2 = np.where(i2 >= 0, self.l2.gamma[i2.clip(0)] if len(self.l2.gamma) else 1.0, 1.0)
        p2 = self.l2.target(i2, w) + g2 * self.p1[w]
        g3 = np.where(i3 >= 0, self.l3.gamma[i3.clip(0)] if len(self.l3.gamma) else 1.0, 1.0)
        return self.l3.target(i3, w) + g3 * p2

    def dist(self, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
        """Full distributions [n, V] (float64 -> float32 tensor)."""
        u, v = np.asarray(u, np.int64), np.asarray(v, np.int64)
        i3, i2 = self._ctx(u, v)
        out = np.empty((len(u), self.V))
        for k in range(len(u)):
            g3 = self.l3.gamma[i3[k]] if i3[k] >= 0 else 1.0
            g2 = self.l2.gamma[i2[k]] if i2[k] >= 0 else 1.0
            row = (g3 * g2) * self.p1
            if i2[k] >= 0:
                a, b = self.l2.start[i2[k]], self.l2.end[i2[k]]
                row[self.l2.w[a:b]] += g3 * self.l2.num[a:b] / self.l2.total[i2[k]]
            if i3[k] >= 0:
                a, b = self.l3.start[i3[k]], self.l3.end[i3[k]]
                row[self.l3.w[a:b]] += self.l3.num[a:b] / self.l3.total[i3[k]]
            out[k] = row
        return torch.from_numpy(out).float()

    def top_continuation(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(context count, top raw-count next token, top share) per context; count 0 -> token -1."""
        i3, _ = self._ctx(np.asarray(u, np.int64), np.asarray(v, np.int64))
        cnt, top, share = np.zeros(len(i3)), np.full(len(i3), -1, np.int64), np.zeros(len(i3))
        for k, i in enumerate(i3.tolist()):
            if i < 0:
                continue
            a, b = self.l3.start[i], self.l3.end[i]
            j = a + int(np.argmax(self.l3.c[a:b]))
            cnt[k], top[k], share[k] = self.l3.total[i], self.l3.w[j], self.l3.c[j] / self.l3.total[i]
        return cnt, top, share


def doc_contexts(ids: np.ndarray, targets: np.ndarray, eos: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) = (x_{j-2}, x_{j-1}) for targets j, eos before the doc start."""
    p = np.concatenate([[eos, eos], np.asarray(ids, np.int64)])
    t = np.asarray(targets, np.int64)
    return p[t], p[t + 1]


class InFileCache:
    """Witten-Bell bigram cache over doc[:j] for each target j (unigram fallback, uniform when empty)."""

    def __init__(self, vocab: int):
        self.V = int(vocab)

    def _state(self, ids: np.ndarray, upto: int):
        pre = np.asarray(ids[:upto], np.int64)
        uni = np.bincount(pre, minlength=self.V).astype(np.float64)
        big: dict[int, dict[int, int]] = {}
        for a, b in zip(pre[:-1].tolist(), pre[1:].tolist()):
            big.setdefault(a, {}).setdefault(b, 0)
            big[a][b] += 1
        return uni, big

    @staticmethod
    def _push(uni, big, ids, j):
        uni[ids[j]] += 1
        if j > 0:
            d = big.setdefault(int(ids[j - 1]), {})
            d[int(ids[j])] = d.get(int(ids[j]), 0) + 1

    def _row(self, uni, big, v: int, n: int, w: int | None):
        """p(. | v) (w=None: full row as numpy) or p(w | v)."""
        if n == 0:
            return np.full(self.V, 1.0 / self.V) if w is None else 1.0 / self.V
        fol = big.get(v)
        lam, cv = 0.0, 0
        if fol:
            cv = sum(fol.values())
            lam = cv / (cv + len(fol))
        # unigram cache smoothed with a uniform floor so p > 0 everywhere
        if w is None:
            row = (1 - lam) * (uni + 1.0 / self.V) / (n + 1.0)
            if fol:
                ks = np.fromiter(fol.keys(), np.int64)
                row[ks] += lam * np.fromiter(fol.values(), np.float64) / cv
            return row
        p = (1 - lam) * (uni[w] + 1.0 / self.V) / (n + 1.0)
        if fol and w in fol:
            p += lam * fol[w] / cv
        return p

    def probs(self, ids: np.ndarray, targets: np.ndarray, full: bool = False):
        """p_cache(y_j) for sorted targets j (or full rows [n, V] if full)."""
        ids = np.asarray(ids, np.int64)
        t = np.asarray(targets, np.int64)
        if len(t) == 0:
            return np.zeros((0, self.V)) if full else np.zeros(0)
        uni, big = self._state(ids, int(t[0]))
        n, cur = int(t[0]), int(t[0])
        out = np.zeros((len(t), self.V)) if full else np.zeros(len(t))
        for k, j in enumerate(t.tolist()):
            while cur < j:
                self._push(uni, big, ids, cur)
                cur += 1
                n += 1
            v = int(ids[j - 1])
            out[k] = self._row(uni, big, v, n, None if full else int(ids[j]))
        return out


class CountLMCondition(Condition):
    """Host interpolated with KN trigram + in-file cache. lam_n / lam_c from `tune_lambdas` (dev only)."""

    def __init__(self, kn: KNTrigram, lam_n: float, lam_c: float, name: str = "countlm"):
        if lam_n < 0 or lam_c < 0 or lam_n + lam_c >= 1:
            raise ValueError("need lam_n, lam_c >= 0 and lam_n + lam_c < 1")
        self.kn, self.lam_n, self.lam_c, self.name = kn, lam_n, lam_c, name
        self.cache = InFileCache(kn.V)

    def adjust(self, logp, *, hidden, doc: Doc, win: Window, targets: np.ndarray):
        V = logp.shape[1]
        if V != self.kn.V:
            raise ValueError(f"host vocab {V} != count-LM vocab {self.kn.V}")
        p = (1 - self.lam_n - self.lam_c) * logp.exp()
        for c0 in range(0, len(targets), 64):  # 64-row blocks keep the float64 count-LM rows small
            sl = slice(c0, c0 + 64)
            if self.lam_n:
                u, v = doc_contexts(doc.ids, targets[sl], self.kn.eos)
                p[sl] += self.lam_n * self.kn.dist(u, v)
            if self.lam_c:
                p[sl] += self.lam_c * torch.from_numpy(self.cache.probs(doc.ids, targets[sl], full=True)).float()
        return p.clamp_min(1e-30).log()

    def describe(self) -> dict:
        return dict(name=self.name, lam_n=self.lam_n, lam_c=self.lam_c, train_tokens=self.kn.n_tokens)


def component_target_probs(kn: KNTrigram, docs: Sequence[Doc], res: ScoreResult) -> tuple[np.ndarray, np.ndarray]:
    """(p_KN(y), p_cache(y)) for every target of a ScoreResult (no model needed)."""
    pk, pc = np.zeros(len(res)), np.zeros(len(res))
    cache = InFileCache(kn.V)
    for i, d in enumerate(docs):
        m = np.where(res.doc == i)[0]
        if not len(m):
            continue
        order = m[np.argsort(res.pos[m], kind="stable")]
        t = res.pos[order].astype(np.int64)
        u, v = doc_contexts(d.ids, t, kn.eos)
        pk[order] = kn.prob(u, v, d.ids[t])
        pc[order] = cache.probs(d.ids, t)
    return pk, pc


def tune_lambdas(base: ScoreResult, pk: np.ndarray, pc: np.ndarray,
                 grid: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)) -> dict:
    """Grid search of (lam_n, lam_c) minimising mean NLL of the mixture on `base` targets (use dev only)."""
    ph = np.exp(-base.nll.astype(np.float64))
    best = None
    table = []
    for ln, lc in itertools.product(grid, grid):
        if ln + lc >= 1:
            continue
        p = (1 - ln - lc) * ph + ln * pk + lc * pc
        nll = float(-np.log(np.maximum(p, 1e-300)).mean())
        table.append((ln, lc, nll))
        if best is None or nll < best[2]:
            best = (ln, lc, nll)
    return dict(lam_n=best[0], lam_c=best[1], dev_nll=best[2], base_dev_nll=float(base.nll.mean()), grid=table)


def headroom(kn: KNTrigram, docs: Sequence[Doc], base: ScoreResult, min_count: int = 2, min_share: float = 0.5) -> dict:
    """§I.3 headroom: share of targets whose confident train 3-gram context (count >= 2, top share >= 0.5) predicts
    the target while the base's top-1 is wrong."""
    conf = np.zeros(len(base), bool)
    right = np.zeros(len(base), bool)
    for i, d in enumerate(docs):
        m = np.where(base.doc == i)[0]
        if not len(m):
            continue
        t = base.pos[m].astype(np.int64)
        u, v = doc_contexts(d.ids, t, kn.eos)
        cnt, top, share = kn.top_continuation(u, v)
        c = (cnt >= min_count) & (share >= min_share)
        conf[m], right[m] = c, c & (top == d.ids[t])
    hit = np.asarray(base.hit, bool)
    n = max(1, len(base))
    return dict(confident=float(conf.sum() / n), confident_top1=float(right.sum() / max(1, conf.sum())),
                headroom=float((right & ~hit).sum() / n), base_top1=float(hit.mean()) if len(hit) else math.nan)


__all__ = ["KNTrigram", "InFileCache", "CountLMCondition", "component_target_probs", "tune_lambdas", "headroom",
           "doc_contexts"]
