"""Count-LM frontier in one model pass (M1c): many mixing weights of the KN trigram (or the in-file cache) scored
from the same forward, with per-target KL against the BASE host.

For every target the host log-probs are computed once under the inner condition (none = the plain host; e.g. a
rows cartridge for "table + counts"), and the count component's full distribution is computed once. Then for every
weight w of the grid

    p_w(y) = (1 - w - lam_o) p_cond(y) + w p_comp(y) + lam_o p_other(y)

(comp = KN trigram or in-file cache; other = an optional FIXED in-file cache weight on top of a KN grid) gives the
per-target NLL, top-1 hit, KL(p_base || p_w) and top-1 agreement with the base. The KL is always against the plain
host distribution, not against the inner condition, so "table + counts" KL is comparable with "counts only" KL.

The mixture arithmetic is the same float32 sequence as `countlm.CountLMCondition.adjust`, so a weight of the grid
reproduces the single-weight count-LM condition's NLL (tests/test_countgrid.py). Weight 0 with no fixed other
weight is the inner condition itself (its log-probs unchanged, so KL = the inner condition's KL; 0 for counts only).

Specs (scripts/evaluate.py --conditions): countgrid:kn (KN grid), countgrid:cache (in-file cache grid),
countgrid:kn-c0p005 (KN grid plus a fixed in-file cache weight 0.005; 'p' is the decimal point because condition
labels use Path(arg).stem), and any of these combined with other parts, e.g. countgrid:kn+rows:P.pt.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..metrics import paired, retention
from .baselines.countlm import InFileCache, KNTrigram, doc_contexts, parse_weight
from .scoring import (Composite, Condition, Doc, ScoreResult, Window, _target_masks, final_hidden, lm_head_of)

DEFAULT_WEIGHTS = (0.0, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 6e-3, 7e-3, 7.5e-3, 1e-2, 1.5e-2, 2e-2,
                   3e-2)                # M1c C0 grid (PREREG §3; 3e-4, 6e-3, 7e-3 added before launch, §12)
NONDUP_CLASS = "dup:16gram_not_in_train"
BLOCK = 64  # rows per mixing block (as CountLMCondition): bounds the [rows, V] float temporaries


def parse_spec(arg: str) -> dict:
    """countgrid:<arg> -> dict(component='kn'|'cache', other_weight=float). '' = 'kn'."""
    toks = [t for t in (arg or "kn").split("-") if t]
    comp = [t for t in toks if t in ("kn", "cache")]
    rest = [t for t in toks if t not in ("kn", "cache")]
    if len(comp) != 1:
        raise ValueError(f"countgrid variant {arg!r}: give exactly one of kn, cache (e.g. kn, cache, kn-c0p005)")
    other, seen_c = 0.0, False
    for t in rest:
        if t.startswith("c") and comp[0] == "kn":
            if seen_c:
                raise ValueError(f"countgrid variant {arg!r}: more than one fixed cache weight c<w>")
            seen_c = True
            other = parse_weight(t[1:])
            if not math.isfinite(other):
                raise ValueError(f"countgrid variant {arg!r}: weight must be finite")
        else:
            raise ValueError(f"countgrid variant {arg!r}: unknown part {t!r} (only kn-c<weight> adds a fixed cache)")
    return dict(component=comp[0], other_weight=other)


def parse_weights(s: str | None) -> tuple[float, ...]:
    """'0,1e-4,0.0005' -> weights (sorted, unique). None/'' -> DEFAULT_WEIGHTS."""
    if not s:
        return DEFAULT_WEIGHTS
    ws = {float(x) for x in s.split(",") if x.strip()}
    if not ws or not all(math.isfinite(w) for w in ws):
        raise ValueError(f"--countgrid-weights {s!r}: need finite numbers")
    return tuple(sorted(ws))


def _has_hooks(c: Condition | None) -> bool:
    if c is None:
        return False
    if isinstance(c, Composite):
        return any(_has_hooks(p) for p in c.parts)
    return type(c).activate is not Condition.activate


class CountGridCondition(Condition):
    """Marker condition for `score_grid`: the inner condition (or none) plus the count component and its weights.
    It cannot be scored by `scoring.score` (adjust raises): use `score_grid`."""

    def __init__(self, kn: KNTrigram, component: str = "kn", weights: Sequence[float] = DEFAULT_WEIGHTS,
                 other_weight: float = 0.0, inner: Condition | None = None, name: str = "countgrid"):
        if component not in ("kn", "cache"):
            raise ValueError(component)
        ws = tuple(float(w) for w in weights)
        if not ws or not all(math.isfinite(w) for w in ws + (float(other_weight),)):
            raise ValueError("countgrid weights must be finite")
        if component == "cache" and other_weight:
            raise ValueError("other_weight (a fixed in-file cache weight) applies to a KN grid only")
        if any(w < 0 for w in ws) or other_weight < 0 or max(ws) + other_weight >= 1:
            raise ValueError("need weights >= 0, other_weight >= 0 and max(weight) + other_weight < 1")
        if inner is not None and inner.provides_prefix:
            raise ValueError("countgrid computes KL against the base: the inner condition must not supply a prefix")
        self.kn, self.component, self.weights, self.other_weight = kn, component, ws, float(other_weight)
        self.inner, self.name = inner, name
        self.uses_hidden = bool(inner and inner.uses_hidden)
        self.cache = InFileCache(kn.V)

    def activate(self, model):
        return self.inner.activate(model) if self.inner is not None else contextlib.nullcontext()

    def adjust(self, logp, **kw):
        raise TypeError("CountGridCondition is scored by countgrid.score_grid, not scoring.score")

    def describe(self) -> dict:
        return dict(name=self.name, kind="countgrid", component=self.component, weights=list(self.weights),
                    other_weight=self.other_weight, train_tokens=self.kn.n_tokens,
                    inner=self.inner.describe() if self.inner is not None else None)

    # ------------------------------------------------------------------------------------------------ mixing
    def components(self, doc: Doc, targets: np.ndarray) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(p_comp, p_other) float32 [n, V] for targets (p_other None when other_weight == 0)."""
        kn_p = ca_p = None
        if self.component == "kn" or self.other_weight:
            u, v = doc_contexts(doc.ids, targets, self.kn.eos)
            kn_p = self.kn.dist(u, v) if self.component == "kn" else None
        if self.component == "cache" or self.other_weight:
            ca_p = torch.from_numpy(self.cache.probs(doc.ids, targets, full=True)).float()
        if self.component == "kn":
            return kn_p, ca_p if self.other_weight else None
        return ca_p, None

    def mix(self, lp: torch.Tensor, ph: torch.Tensor, comp: torch.Tensor, other: torch.Tensor | None,
            w: float) -> torch.Tensor:
        """log p_w, the same float32 operations as CountLMCondition.adjust (lam = w for the grid component)."""
        if w == 0 and not self.other_weight:
            return lp
        if self.component == "kn":
            p = (1 - w - self.other_weight) * ph
            if w:
                p += w * comp
            if self.other_weight:
                p += self.other_weight * other
        else:
            p = (1 - 0.0 - w) * ph
            if w:
                p += w * comp
        return p.clamp_min(1e-30).log()


@dataclass
class GridResult:
    """Per-weight per-target arrays [W, N] on the same targets as `scoring.score` (rows = weights)."""
    doc_names: list[str]
    groups: list[str]
    doc: np.ndarray
    pos: np.ndarray
    ctx: np.ndarray
    weights: np.ndarray        # float64 [W]
    nll: np.ndarray            # float32 [W, N]
    hit: np.ndarray            # bool [W, N]
    kl: np.ndarray             # float32 [W, N] KL(base host || mixture)
    agree: np.ndarray          # bool [W, N] argmax(base host) == argmax(mixture)
    host_nll: np.ndarray       # float32 [N] the plain host's NLL from this pass (check against the base run)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.doc.shape[0])

    def key(self) -> np.ndarray:
        return self.doc.astype(np.int64) * (1 << 32) + self.pos.astype(np.int64)

    def at(self, i: int) -> ScoreResult:
        """weight i as a ScoreResult (kl/agree filled)."""
        return ScoreResult(self.doc_names, self.groups, self.doc, self.pos, self.ctx, self.nll[i], self.hit[i],
                           self.kl[i], self.agree[i], dict(self.meta, weight=float(self.weights[i])))

    def select(self, mask: np.ndarray) -> "GridResult":
        return GridResult(self.doc_names, self.groups, self.doc[mask], self.pos[mask], self.ctx[mask], self.weights,
                          self.nll[:, mask], self.hit[:, mask], self.kl[:, mask], self.agree[:, mask],
                          self.host_nll[mask], dict(self.meta))

    def save(self, path: str | Path) -> None:
        import json
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(dict(
            meta=self.meta, doc_names=self.doc_names, groups=self.groups), default=str).encode(), dtype=np.uint8),
            doc=self.doc, pos=self.pos, ctx=self.ctx, weights=self.weights, nll=self.nll, hit=self.hit, kl=self.kl,
            agree=self.agree, host_nll=self.host_nll)

    @classmethod
    def load(cls, path: str | Path) -> "GridResult":
        import json
        z = np.load(path)
        m = json.loads(bytes(z["meta"]).decode())
        return cls(m["doc_names"], m["groups"], z["doc"], z["pos"], z["ctx"], z["weights"], z["nll"], z["hit"],
                   z["kl"], z["agree"], z["host_nll"], m["meta"])


@torch.no_grad()
def score_grid(model: nn.Module, docs: Sequence[Doc], windows: Sequence[Window], cond: CountGridCondition, *,
               chunk: int = 256, progress: Callable[[int, int, float], None] | None = None) -> GridResult:
    """Teacher-forced scoring of every grid weight in one pass (two forwards per window only when the inner
    condition installs hooks: one under it, one for the base host's distribution)."""
    head = lm_head_of(model)
    W = len(cond.weights)
    keys = ("doc", "pos", "ctx", "nll", "hit", "kl", "agree", "host")
    out: dict[str, list] = {k: [] for k in keys}
    two = _has_hooks(cond.inner)
    t0 = time.time()
    fwd_tokens = 0
    t_mix = t_comp = 0.0
    for wi, w in enumerate(windows):
        d = docs[w.doc]
        if not (0 <= w.a < w.s < w.e <= len(d)):
            raise ValueError(f"bad window {w} for doc of {len(d)} tokens")
        ids = torch.from_numpy(d.ids[w.a:w.e])[None]
        fwd_tokens += ids.shape[1] * (2 if two else 1)
        targets = np.arange(w.s, w.e)
        rows = torch.from_numpy(targets - 1 - w.a)
        with cond.activate(model):
            h = final_hidden(model, ids)[0, rows]
        hb = final_hidden(model, ids)[0, rows] if two else h
        y = torch.from_numpy(d.ids[targets])
        for c0 in range(0, h.shape[0], chunk):
            raw = torch.log_softmax(head(h[c0:c0 + chunk]).float(), dim=-1)
            tg_all = targets[c0:c0 + raw.shape[0]]
            lp_all = raw if cond.inner is None else cond.inner.adjust(
                raw, hidden=h[c0:c0 + raw.shape[0]], doc=d, win=w, targets=tg_all)  # adjust returns a new tensor
            lb_all = torch.log_softmax(head(hb[c0:c0 + chunk]).float(), dim=-1) if two else raw
            tm = time.time()
            for b0 in range(0, lp_all.shape[0], BLOCK):
                sl = slice(b0, b0 + BLOCK)
                lp, lb, tg = lp_all[sl], lb_all[sl], tg_all[sl]
                yy = y[c0 + b0:c0 + b0 + lp.shape[0]]
                tc = time.time()
                comp, other = cond.components(d, tg)
                t_comp += time.time() - tc
                ph, pb = lp.exp(), lb.exp()
                bam = lb.argmax(-1)
                out["host"].append((-lb.gather(1, yy[:, None])[:, 0]).numpy())
                nll, hit, kl, agree = (np.empty((W, len(tg)), dt) for dt in (np.float32, bool, np.float32, bool))
                for k, wt in enumerate(cond.weights):
                    lm = cond.mix(lp, ph, comp, other, wt)
                    nll[k] = (-lm.gather(1, yy[:, None])[:, 0]).numpy()
                    am = lm.argmax(-1)
                    hit[k] = (am == yy).numpy()
                    kl[k] = (pb * (lb - lm)).sum(-1).clamp_min(0).numpy()
                    agree[k] = (bam == am).numpy()
                for k_, v_ in (("nll", nll), ("hit", hit), ("kl", kl), ("agree", agree)):
                    out[k_].append(v_)
            t_mix += time.time() - tm
        out["doc"].append(np.full(len(targets), w.doc, np.int32))
        out["pos"].append(targets.astype(np.int32))
        out["ctx"].append((targets - w.a).astype(np.int32))
        if progress:
            progress(wi + 1, len(windows), time.time() - t0)
    cat1 = lambda k, dt: np.concatenate(out[k]).astype(dt) if out[k] else np.zeros(0, dt)  # noqa: E731
    cat2 = lambda k, dt: np.concatenate(out[k], axis=1).astype(dt) if out[k] else np.zeros((W, 0), dt)  # noqa: E731
    wall = time.time() - t0
    meta = dict(condition=cond.describe(), wall_s=wall, mix_s=t_mix, components_s=t_comp, n_windows=len(windows),
                forward_tokens=int(fwd_tokens), threads=torch.get_num_threads(), two_forwards=two)
    res = GridResult([d.name for d in docs], [d.group for d in docs], cat1("doc", np.int32), cat1("pos", np.int32),
                     cat1("ctx", np.int32), np.asarray(cond.weights, np.float64), cat2("nll", np.float32),
                     cat2("hit", bool), cat2("kl", np.float32), cat2("agree", bool), cat1("host", np.float32), meta)
    probe = ScoreResult(res.doc_names, res.groups, res.doc, res.pos, res.ctx, res.host_nll,
                        np.zeros(len(res), bool))
    keep = _target_masks(docs, probe)
    if keep is not None:
        res = res.select(keep)
        res.meta = meta
    return res


def nondup_mask(masks: dict | None) -> np.ndarray | None:
    """The primary-metric target mask (targets whose 16-gram is not in train), or None when classes are off."""
    if not masks or NONDUP_CLASS not in masks:
        return None
    return np.asarray(masks[NONDUP_CLASS], bool)


def paired_masked(base: ScoreResult, res: ScoreResult, groups: np.ndarray, mask: np.ndarray, **kw) -> dict:
    """paired() restricted to `mask` (targets aligned between base and res), with EM; KL/agreement if present."""
    m = np.asarray(mask, bool)
    f = (lambda x: None if x is None else np.asarray(x)[m])  # noqa: E731
    if res.kl is not None:
        return retention(base.nll[m], res.nll[m], np.asarray(groups)[m], f(res.kl), f(res.agree),
                         base_hit=base.hit[m], cond_hit=res.hit[m], **kw)
    return paired(base.nll[m], res.nll[m], np.asarray(groups)[m], base_hit=base.hit[m], cond_hit=res.hit[m], **kw)


def grid_metrics(base: ScoreResult, g: GridResult, groups: np.ndarray, *, nondup: np.ndarray | None = None,
                 n_boot: int = 10_000) -> dict:
    """Per-weight paired metrics vs the base run (rel_pct + file-cluster CI, EM, mean KL, top-1 agreement), and the
    same on the non-duplicated targets when `nondup` is given."""
    if not np.array_equal(base.key(), g.key()):
        raise RuntimeError("countgrid targets differ from base")
    hd = np.abs(g.host_nll.astype(np.float64) - base.nll.astype(np.float64))
    rows = []
    for i, w in enumerate(g.weights.tolist()):
        r = g.at(i)
        row = dict(weight=w, paired=retention(base.nll, r.nll, groups, r.kl, r.agree, base_hit=base.hit,
                                              cond_hit=r.hit, n_boot=n_boot))
        if nondup is not None:
            row["paired_nondup"] = paired_masked(base, r, groups, nondup, n_boot=n_boot)
        rows.append(row)
    d = g.meta["condition"]
    return dict(component=d["component"], other_weight=d["other_weight"],
                inner=(d["inner"] or {}).get("name"), weights=g.weights.tolist(), grid=rows,
                host_nll_max_abs_diff_vs_base=float(hd.max()) if len(hd) else 0.0,
                nondup_fraction=float(np.mean(nondup)) if nondup is not None else None)


__all__ = ["DEFAULT_WEIGHTS", "NONDUP_CLASS", "CountGridCondition", "GridResult", "score_grid", "grid_metrics",
           "parse_spec", "parse_weights", "nondup_mask", "paired_masked"]
