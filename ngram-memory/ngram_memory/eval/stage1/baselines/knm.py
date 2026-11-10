"""kNM-lite (§6.A.4 "kNM-LM datastore"): a forward-only datastore of in-project tokens the base mispredicts.

Keys = the host's final (post-norm) hidden state at position t, value = the next token x_{t+1}, kept only where the
base's argmax is wrong (kNM-LM's decoupled domain database, Tang et al. 2023). At eval the k nearest keys (exact
L2, brute force in fp32 blocks) give p_knn(w) proportional to sum exp(-d_i / T) over neighbours with value w, and

    p = (1 - lam) p_host + lam p_knn

lam and T are tuned on dev (`tune`), "lite" = no Bayesian lam. Keys are stored fp16; the build uses the same
windows / token budget as a cartridge arm.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch

from ..scoring import Condition, Doc, Window, chunked_logprobs, final_hidden, lm_head_of


class KNMDatastore:
    def __init__(self, keys: torch.Tensor, vals: torch.Tensor, meta: dict | None = None):
        self.keys = keys.to(torch.float16).contiguous()
        self.vals = vals.long().contiguous()
        self.meta = meta or {}
        self._sq = self.keys.float().pow(2).sum(-1)

    def __len__(self) -> int:
        return int(self.vals.numel())

    @classmethod
    @torch.no_grad()
    def build(cls, model, docs: Sequence[Doc], windows: Sequence[Window], *, mispredicted_only: bool = True,
              max_tokens: int | None = None, chunk: int = 256) -> "KNMDatastore":
        head = lm_head_of(model)
        ks, vs, seen = [], [], 0
        for w in windows:
            d = docs[w.doc]
            tg = np.arange(w.s, w.e)
            if max_tokens is not None:
                tg = tg[: max(0, max_tokens - seen)]
            if len(tg) == 0:
                break
            ids = torch.from_numpy(d.ids[w.a:w.e])[None]
            h = final_hidden(model, ids)[0, torch.from_numpy(tg - 1 - w.a)]
            y = torch.from_numpy(d.ids[tg])
            for c0, lp in chunked_logprobs(head, h, chunk):
                yy = y[c0:c0 + lp.shape[0]]
                keep = lp.argmax(-1) != yy if mispredicted_only else torch.ones_like(yy, dtype=torch.bool)
                ks.append(h[c0:c0 + lp.shape[0]][keep].to(torch.float16))
                vs.append(yy[keep])
            seen += len(tg)
        keys = torch.cat(ks) if ks else torch.zeros(0, model.config.hidden_size, dtype=torch.float16)
        vals = torch.cat(vs) if vs else torch.zeros(0, dtype=torch.long)
        return cls(keys, vals, dict(scanned_tokens=seen, kept=int(vals.numel()), mispredicted_only=mispredicted_only))

    @torch.no_grad()
    def search(self, q: torch.Tensor, k: int = 16, block: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
        """(squared L2 distances [n, k] ascending, values [n, k]) of the k nearest keys to q [n, d]."""
        q = q.float()
        n, k = q.shape[0], min(k, len(self))
        if k == 0:
            return torch.zeros(n, 0), torch.zeros(n, 0, dtype=torch.long)
        qq = q.pow(2).sum(-1, keepdim=True)
        best_d = torch.full((n, k), float("inf"))
        best_i = torch.zeros(n, k, dtype=torch.long)
        for b0 in range(0, len(self), block):
            kb = self.keys[b0:b0 + block].float()
            d = (qq - 2 * q @ kb.T + self._sq[b0:b0 + block][None]).clamp_min(0)
            dd, ii = torch.topk(d, min(k, d.shape[1]), dim=1, largest=False)
            cd = torch.cat([best_d, dd], 1)
            ci = torch.cat([best_i, ii + b0], 1)
            best_d, o = torch.topk(cd, k, dim=1, largest=False)
            best_i = ci.gather(1, o)
        return best_d, self.vals[best_i]

    def state_dict(self) -> dict:
        return dict(keys=self.keys, vals=self.vals)


def knn_probs(dist: torch.Tensor, vals: torch.Tensor, V: int, temp: float) -> torch.Tensor:
    """p_knn [n, V] from neighbour distances / values."""
    w = torch.softmax(-dist / temp, dim=-1)
    out = torch.zeros(dist.shape[0], V)
    out.scatter_add_(1, vals, w)
    return out


class KNMCondition(Condition):
    uses_hidden = True

    def __init__(self, store: KNMDatastore, lam: float, temp: float, k: int = 16, name: str = "knm_lite"):
        self.store, self.lam, self.temp, self.k, self.name = store, lam, temp, k, name

    def adjust(self, logp, *, hidden, doc, win, targets):
        if self.lam == 0 or len(self.store) == 0:
            return logp
        d, v = self.store.search(hidden, self.k)
        p = (1 - self.lam) * logp.exp() + self.lam * knn_probs(d, v, logp.shape[1], self.temp)
        return p.clamp_min(1e-30).log()

    def describe(self) -> dict:
        return dict(name=self.name, lam=self.lam, temp=self.temp, k=self.k, datastore=len(self.store), **self.store.meta)


class _Collect(Condition):
    """Records per-target host target prob and neighbours (for tuning); returns the host log-probs unchanged."""
    uses_hidden = True
    name = "knm_collect"

    def __init__(self, store: KNMDatastore, k: int):
        self.store, self.k = store, k
        self.rows: list = []

    def adjust(self, logp, *, hidden, doc, win, targets):
        d, v = self.store.search(hidden, self.k)
        y = torch.from_numpy(doc.ids[targets])
        self.rows.append((logp.gather(1, y[:, None])[:, 0].exp(), d, v, y))
        return logp


def tune(model, store: KNMDatastore, docs: Sequence[Doc], windows: Sequence[Window], k: int = 16,
         lams: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
         temp_mults: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)) -> dict:
    """Pick (lam, temp) on dev windows; temp is a multiple of the median nearest-neighbour distance."""
    from ..scoring import score
    col = _Collect(store, k)
    score(model, docs, windows, col)
    ph = torch.cat([r[0] for r in col.rows]).double()
    D = torch.cat([r[1] for r in col.rows])
    Vv = torch.cat([r[2] for r in col.rows])
    Y = torch.cat([r[3] for r in col.rows])
    med = float(D[:, 0].median()) if D.numel() else 1.0
    best, table = None, []
    for lam, tm in itertools.product(lams, temp_mults):
        T = max(med * tm, 1e-6)
        w = torch.softmax(-D / T, -1).double()
        pk = (w * (Vv == Y[:, None])).sum(-1)
        nll = float(-torch.log(((1 - lam) * ph + lam * pk).clamp_min(1e-300)).mean())
        table.append((lam, tm, nll))
        if best is None or nll < best[2]:
            best = (lam, T, nll, tm)
    return dict(lam=best[0], temp=best[1], temp_mult=best[3], dev_nll=best[2],
                base_dev_nll=float(-torch.log(ph).mean()), median_nn_dist=med, grid=table)


__all__ = ["KNMDatastore", "KNMCondition", "knn_probs", "tune"]
