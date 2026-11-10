"""Sparse-row Stage-1 trainer (§I.1, §I.2) for an NGramGraft on a frozen HF causal LM.

Objective per step (one training window + `anchors_per_step` windows of each anchor set):

    NLL(train window; per-position n-gram weight cap)                                  mean over scored targets
  + sum_a lambda_a * mean_t KL(p0 || p_delta) on anchor set a (top-k targets, kltargets.py)
  + lambda_rho * mean_t relu(r_t / rho - 1)^2,  r_t = ||graft_out_t|| / ||h_t|| on anchor batches (rho budget)
  + sum_j omega_j [lambda_2 ||d_j||^2 / s_j^2]                    (smooth; lazily on the rows read this step)
  then proximal row group lasso  d_j <- max(0, 1 - lr*lambda_1*omega_j / (s_j ||d_j|| sqrt(a_j))) d_j
      with lambda_1 = lambda_l1_rel * G0 (or an absolute lambda_l1, legacy). G0 = median over the first
      `acc_init_steps` steps of the per-step median of ||g_j|| * s_j over the rows read with a nonzero gradient,
      i.e. the typical per-row loss gradient in row-scale units. For a row starting at zero the Adagrad metric
      cancels and the prox keeps it at exactly zero iff ||g_j|| * s_j < lambda_1 * omega_j, so lambda_l1_rel
      is dimensionless: 1.0 keeps a median-gradient row (omega = 1) at zero; 0.1 only blocks rows whose
      gradient is 10x below the median. An absolute lambda_1 is not portable: per-row gradients of a
      per-token-mean loss scale with reads/window length (about 1e-5 on the 0.5B proxy at 4k windows).
  and trust region               d_j <- d_j * min(1, kappa s_j / ||d_j||)

Optimiser: row-wise Adagrad on the fp32 SparseRowStore values (S1/ingest-algo/optim_bench.py):
    a_j += mean(g_j^2);  d_j -= lr g_j / sqrt(a_j + eps)
A row's accumulator starts at a0 = median over the first `acc_init_steps` steps of the per-step median of
mean(g_j^2) over the rows read in that step (so once-seen rows do not take full-size first steps).

Rows: only the reserved rows of the SparseRowStore (the row policy's selection) can change; every other row reads
0 and gets no gradient. Arms: 'G-lin', 'G-dir' (fmt.module.arm must agree), 'rows+Wv' (W_v trained with Adam,
an ablation), 'key-shuffle' (C1 control, §6.A.4: at train time n-gram A's positions read and train the slot of
sigma(A), a fixed random per-head permutation of the reserved rows; `cartridge()` returns the trained slots AS
STORED, so served through the true keys n-gram A reads the row trained by sigma^-1(A)'s positions: content-free
rows with real training. omega / row_scale follow the n-gram that trains each slot. Note: for a bijective sigma
this is equivalent, up to float noise, to the eval-time permuted-keys control applied to the base arm with
sigma^-1; `relabelled_store()` gives the rows that WOULD reproduce the trained function through true keys).

Activation checkpointing: HF gradient checkpointing (non-reentrant) on the decoder layers plus checkpointed
chunks of the output head, so a 4,096-token window never holds [T, V] logits. Determinism: data order is a pure
function of (seed, step); resume from a checkpoint is bit-exact on a fixed thread count.
"""
from __future__ import annotations

import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..backends.graft.module import NGramGraft
from ..backends.graft.qwen35 import install_graft
from ..hashing import hasher_for
from ..spec import FormatSpec, KeySpec
from ..tables import SparseRowEmbedding, SparseRowStore
from .corpus import IGNORE, Window
from .kltargets import KLTargets, WindowTargets, final_hidden, head_logits, logit_softcap, output_head, topk_kl

ARMS = ("G-lin", "G-dir", "rows+Wv", "key-shuffle")


# ---------------------------------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    arm: str = "G-lin"
    lr: float = 1e-2
    eps: float = 1e-30                  # accumulators start at a0 > 0; on the 0.5B proxy a0 ~ 1e-10, so 1e-10 was not negligible
    acc_init_steps: int = 50
    lr_wv: float = 5e-5
    anchor_lambdas: dict = field(default_factory=lambda: {"near": 0.3, "chat": 0.3})
    anchors_per_step: int = 1
    lambda_l2: float = 0.0
    lambda_l1: float = 0.0              # absolute lambda_1 (legacy; not portable across hosts/window lengths)
    lambda_l1_rel: float = 0.0          # lambda_1 = lambda_l1_rel * G0 (see module doc); use this one
    kappa: float | None = None
    rho: float = 0.01
    lambda_rho: float = 0.1
    rho_on_train: bool = False
    loss_chunk: int = 512
    grad_checkpointing: bool = True
    seed: int = 0
    shuffle_seed: int = 1
    epochs: int = 1
    max_steps: int = 0                  # 0: epochs * len(train windows)
    ckpt_every: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**d)


# ------------------------------------------------------------------------------------------------- key shuffle
class KeyShuffle:
    """Fixed per-head permutation of the reserved rows (C1 control, §6.A.4 'train-time key shuffle'): a position
    whose key reads row src[i] reads (and trains) slot dst[i] instead; dst is a permutation of src within each
    head, so n-gram -> row is a bijection onto another n-gram's row. Rows outside the reserved set are unchanged."""

    def __init__(self, key: KeySpec, row_ids: torch.Tensor, seed: int):
        src = row_ids.clone().long()
        g = torch.Generator().manual_seed(int(seed))
        pos = torch.arange(src.numel())
        bounds = torch.searchsorted(src, torch.tensor(list(key.head_offsets) + [key.total_rows], dtype=torch.long))
        for h in range(key.n_heads):
            a, b = int(bounds[h]), int(bounds[h + 1])
            if b - a > 1:
                pos[a:b] = a + torch.randperm(b - a, generator=g)
        self.key, self.seed = key, int(seed)
        self.src, self.dst = src, src[pos]
        self.perm = pos             # dst[i] = src[perm[i]]: compact slot perm[i] is trained by src[i]'s positions

    def remap(self, ids: torch.Tensor) -> torch.Tensor:
        n = self.src.numel()
        if n == 0:
            return ids
        p = torch.searchsorted(self.src, ids.contiguous()).clamp_max(n - 1)
        return torch.where(self.src[p] == ids, self.dst[p], ids)

    def remap_np(self, ids: np.ndarray) -> np.ndarray:
        return self.remap(torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64))).numpy()

    def to_slots(self, x: torch.Tensor) -> torch.Tensor:
        """Re-align a per-row quantity given in src order (x[i] belongs to n-gram row src[i]) to the slots that
        n-gram trains: out[perm[i]] = x[i]."""
        out = torch.empty_like(x)
        out[self.perm] = x
        return out

    def served_store(self, trained: SparseRowStore) -> SparseRowStore:
        """The RELABELLED rows (served[src[i]] = trained[dst[i]]): through the plain keys they reproduce the
        trained function exactly, i.e. they are the base arm again. Diagnostic only; never the C1 cartridge."""
        return SparseRowStore(trained.num_rows, trained.row_width, self.src.clone(), trained.gather(self.dst))


class PermutedRowTable(nn.Module):
    """A SparseRowEmbedding read through a KeyShuffle."""

    def __init__(self, inner: SparseRowEmbedding, shuffle: KeyShuffle):
        super().__init__()
        self.inner, self.shuffle = inner, shuffle
        self.num_embeddings, self.embedding_dim = inner.num_embeddings, inner.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.values

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.inner(self.shuffle.remap(ids))


# --------------------------------------------------------------------------------------------- the optimiser
class SparseRowAdagrad:
    """Row-wise Adagrad over a compact [U, w] value tensor with lazily initialised accumulators (see module doc)."""

    def __init__(self, n_rows: int, init_steps: int = 50, eps: float = 1e-30):
        self.acc = torch.full((n_rows,), float("nan"), dtype=torch.float32)
        self.hist: list[float] = []
        self.ghist: list[float] = []     # per-step median ||g_j|| * s_j (for G0)
        self.init_steps, self.eps = int(init_steps), float(eps)

    @property
    def a0(self) -> float | None:
        return float(np.median(self.hist)) if self.hist else None

    @property
    def g0(self) -> float | None:
        return float(np.median(self.ghist)) if self.ghist else None

    @torch.no_grad()
    def observe_gscale(self, g: torch.Tensor, scale: torch.Tensor) -> float | None:
        """Record this step's median ||g_j|| * s_j (rows with a nonzero gradient) during the warm-up; return G0."""
        if len(self.ghist) < self.init_steps:
            u = g.norm(dim=1) * scale
            u = u[u > 0]
            if u.numel():
                self.ghist.append(float(u.median()))
        return self.g0

    @torch.no_grad()
    def step(self, values: torch.Tensor, pos: torch.Tensor, g: torch.Tensor, lr: float,
             l1: torch.Tensor | None = None, radius: torch.Tensor | None = None) -> dict:
        """values[pos] update. g [m, w] (smooth gradient incl. L2), l1 [m] = lambda_1 omega / s (prox),
        radius [m] = kappa * s (trust region)."""
        if pos.numel() == 0:
            return dict(rows=0)
        gsq = (g * g).mean(1)
        if len(self.hist) < self.init_steps:
            self.hist.append(float(gsq.median()))
        a = self.acc[pos]
        fresh = torch.isnan(a)
        a = torch.where(fresh, torch.full_like(a, self.a0), a) + gsq
        self.acc[pos] = a
        den = torch.sqrt(a + self.eps)
        d = values[pos] - lr * g / den[:, None]
        zeroed = 0
        if l1 is not None:
            nrm = d.norm(dim=1)
            shrink = (1.0 - lr * l1 / (nrm.clamp_min(1e-30) * den)).clamp_min(0.0)   # same metric as the step
            zeroed = int((shrink == 0).sum())
            d = d * shrink[:, None]
        clipped = 0
        if radius is not None:
            nrm = d.norm(dim=1)
            fac = (radius / nrm.clamp_min(1e-30)).clamp_max(1.0)
            clipped = int((fac < 1).sum())
            d = d * fac[:, None]
        values[pos] = d
        return dict(rows=int(pos.numel()), fresh=int(fresh.sum()), lasso_zeroed=zeroed, trust_clipped=clipped)

    def state_dict(self) -> dict:
        return dict(acc=self.acc.clone(), hist=list(self.hist), ghist=list(self.ghist), init_steps=self.init_steps,
                    eps=self.eps)

    def load_state_dict(self, sd: dict) -> None:
        if sd["acc"].shape != self.acc.shape:
            raise ValueError("accumulator shape mismatch")
        self.acc.copy_(sd["acc"])
        self.hist = list(sd["hist"])
        self.ghist = list(sd.get("ghist", []))
        self.init_steps, self.eps = int(sd["init_steps"]), float(sd["eps"])


# ------------------------------------------------------------------------------------------------- utilities
def rss_mb() -> tuple[float, float]:
    """(current RSS, peak RSS) in MB."""
    try:
        with open("/proc/self/statm") as f:
            cur = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        cur = float("nan")
    return cur, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _q(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q)) if x.numel() else float("nan")


def host_fingerprint(host: nn.Module) -> str:
    """sha256 over all host parameters and buffers (frozen-host check)."""
    import hashlib
    hsh = hashlib.sha256()
    for n, t in sorted(host.state_dict().items()):
        hsh.update(n.encode())
        if t.numel():  # hash the buffer in place (no 2x copy of the host)
            hsh.update(memoryview(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
    return hsh.hexdigest()


# --------------------------------------------------------------------------------------------------- trainer
class RowTrainer:
    """See module docstring. `rows` = the trainable (reserved) rows, usually zero; `omega` [U] and `row_scale`
    (scalar or [U]) are aligned with rows.row_ids."""

    def __init__(self, host: nn.Module, fmt: FormatSpec, rows: SparseRowStore, train: Sequence[Window],
                 cfg: TrainConfig, anchors: dict[str, tuple[Sequence[Window], KLTargets]] | None = None,
                 omega: np.ndarray | torch.Tensor | None = None, row_scale: float | np.ndarray | torch.Tensor = 1.0,
                 loss_weights: Sequence[np.ndarray] | None = None, log_path: str | Path | None = None):
        if cfg.arm not in ARMS:
            raise ValueError(f"arm {cfg.arm!r} not in {ARMS}")
        if cfg.lambda_l1 > 0 and cfg.lambda_l1_rel > 0:
            raise ValueError("set lambda_l1_rel or lambda_l1, not both")
        if cfg.arm == "G-dir" and fmt.module.arm != "G-dir":
            raise ValueError("arm G-dir needs a G-dir FormatSpec")
        if cfg.arm != "G-dir" and fmt.module.arm == "G-dir":
            raise ValueError(f"arm {cfg.arm} with a G-dir FormatSpec")
        self.host, self.fmt, self.cfg = host, fmt, cfg
        self.key = fmt.key
        self.train_windows = list(train)
        self.anchors = dict(anchors or {})
        for name, (ws, tg) in self.anchors.items():
            if len(ws) != len(tg):
                raise ValueError(f"anchor set {name}: {len(ws)} windows vs {len(tg)} targets")
        self.loss_weights = None if loss_weights is None else list(loss_weights)
        self.log_path = Path(log_path) if log_path is not None else None
        self.step_idx = 0
        for p in host.parameters():
            p.requires_grad_(False)

        self.graft = NGramGraft(fmt, table="sparse", sparse_grad=True)
        self.graft.load_rows(rows)
        self.graft.freeze_all_but_rows()
        self.inner: SparseRowEmbedding = self.graft.table  # type: ignore[assignment]
        self.shuffle: KeyShuffle | None = None
        if cfg.arm == "key-shuffle":
            self.shuffle = KeyShuffle(self.key, self.inner.row_ids, cfg.shuffle_seed)
            self.graft.ple.ple_embedding.ngram_embedding = PermutedRowTable(self.inner, self.shuffle)
        self.wv_opt: torch.optim.Optimizer | None = None
        if cfg.arm == "rows+Wv":
            w = self.graft.ple.value_proj.weight
            w.requires_grad_(True)
            self.wv_opt = torch.optim.Adam([w], lr=cfg.lr_wv)
        U = self.inner.row_ids.numel()
        self.opt = SparseRowAdagrad(U, cfg.acc_init_steps, cfg.eps)
        self.omega = torch.ones(U) if omega is None else torch.as_tensor(np.asarray(omega), dtype=torch.float32)
        rs = torch.as_tensor(np.asarray(row_scale), dtype=torch.float32)
        self.row_scale = rs.expand(U).clone() if rs.ndim == 0 else rs
        if self.omega.shape != (U,) or self.row_scale.shape != (U,):
            raise ValueError("omega / row_scale must align with the reserved rows")
        if self.shuffle is not None:   # slot perm[i] is trained by n-gram src[i]: give it that n-gram's weights
            self.omega = self.shuffle.to_slots(self.omega)
            self.row_scale = self.shuffle.to_slots(self.row_scale)
        self.hasher = hasher_for(self.key)
        self.head = output_head(host)
        self.softcap = logit_softcap(host)

        self.installed = install_graft(host, self.graft, record=False)
        self._rho_buf: list[torch.Tensor] = []
        self._recording = False
        self._meter = self.graft.register_forward_hook(self._rho_hook)
        self._gc = False
        if cfg.grad_checkpointing and getattr(host, "supports_gradient_checkpointing", False):
            host.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self._gc = True
        host.train(self._gc)  # HF only checkpoints in train mode; the proxy hosts have no active dropout
        drop = [m.p for m in host.modules() if isinstance(m, nn.Dropout) and m.p > 0]
        cfg_drop = float(getattr(getattr(host, "config", None), "attention_dropout", 0.0) or 0.0)
        if self._gc and (drop or cfg_drop > 0):
            raise RuntimeError("host has active dropout; training mode would make the frozen host stochastic")

    # ----------------------------------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._meter.remove()
        self.installed.remove()
        if self._gc:
            self.host.gradient_checkpointing_disable()
        self.host.eval()

    @property
    def total_steps(self) -> int:
        return self.cfg.max_steps or self.cfg.epochs * len(self.train_windows)

    def _rho_hook(self, mod, args, out):
        # computed on every call (also during checkpoint recomputation, so both passes save the same tensors);
        # kept only in the original forward
        r = (out.norm(dim=-1) / args[0].norm(dim=-1).clamp_min(1e-12)).reshape(-1)
        if self._recording:
            self._rho_buf.append(r)

    # ------------------------------------------------------------------------------------------- data order
    def train_index(self, k: int) -> tuple[int, int]:
        n = len(self.train_windows)
        e, i = divmod(k, n)
        return e, int(np.random.default_rng([self.cfg.seed, e]).permutation(n)[i])

    def anchor_indices(self, set_no: int, n: int, k: int) -> list[int]:
        out = []
        for j in range(self.cfg.anchors_per_step):
            e, i = divmod(k * self.cfg.anchors_per_step + j, n)
            out.append(int(np.random.default_rng([self.cfg.seed, 7919, set_no, e]).permutation(n)[i]))
        return out

    # -------------------------------------------------------------------------------------------- reading
    def read_rows(self, w: Window) -> np.ndarray:
        """Unique compact positions (into the reserved rows) that window `w` reads (after the key shuffle)."""
        r = self.hasher.rows_for(w.ids, context=w.context).reshape(-1)
        if self.shuffle is not None:
            r = self.shuffle.remap_np(r)
        r = np.unique(r)
        ids = self.inner.row_ids.numpy()
        if ids.size == 0:
            return np.zeros(0, np.int64)
        p = np.clip(np.searchsorted(ids, r), 0, ids.size - 1)
        return p[ids[p] == r]

    # ------------------------------------------------------------------------------------------------ loss
    def _chunked(self, h: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None,
                 targets: WindowTargets | None) -> torch.Tensor:
        """Sum over scored positions of weighted NLL (targets None) or KL (targets given); h [T, d]."""
        head, cap, C = self.head, self.softcap, self.cfg.loss_chunk
        total = h.new_zeros(())
        for s in range(0, h.shape[0], C):
            lab = labels[s:s + C]
            m = lab != IGNORE
            if not bool(m.any()):
                continue
            if targets is None:
                wt = (weights[s:s + C] if weights is not None else torch.ones_like(lab, dtype=torch.float32)) * m

                def f(hc, lab=lab, wt=wt):
                    z = head_logits(head, hc, cap)
                    return (F.cross_entropy(z, lab.clamp_min(0), reduction="none") * wt).sum()
            else:
                tg = (targets.idx[s:s + C], targets.logp[s:s + C], targets.rest[s:s + C])

                def f(hc, tg=tg, m=m):
                    return (topk_kl(head_logits(head, hc, cap), tg) * m).sum()
            hc = h[s:s + C]
            total = total + (checkpoint(f, hc, use_reentrant=False) if torch.is_grad_enabled() else f(hc))
        return total

    def _pass(self, w: Window, targets: WindowTargets | None, lam: float, weights: np.ndarray | None,
              rho_penalty: bool) -> dict:
        ids = torch.from_numpy(np.ascontiguousarray(w.ids))[None]
        labels = torch.from_numpy(np.ascontiguousarray(w.labels))
        n = int((labels != IGNORE).sum())
        self.installed.context = torch.from_numpy(np.ascontiguousarray(w.context))[None]
        self._rho_buf, self._recording = [], True
        try:
            h = final_hidden(self.host, ids)[0]
            self._recording = False
            r = torch.cat(self._rho_buf) if self._rho_buf else torch.zeros(0)
            wt = None if weights is None else torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))
            s = self._chunked(h, labels, wt, targets)
            main = s / max(n, 1)
            loss = lam * main
            pen = torch.zeros(())
            if rho_penalty and self.cfg.lambda_rho > 0 and r.numel():
                pen = self.cfg.lambda_rho * F.relu(r / self.cfg.rho - 1.0).pow(2).mean()
                loss = loss + pen
            if loss.requires_grad:
                loss.backward()
        finally:
            self._recording = False
            self.installed.context = None
        rd = r.detach()
        return dict(value=float(main.detach()), n=n, rho_pen=float(pen.detach()),
                    rho_median=_q(rd, 0.5), rho_mean=float(rd.mean()) if rd.numel() else float("nan"),
                    rho_p99=_q(rd, 0.99), rho_frac_over=float((rd > self.cfg.rho).float().mean()) if rd.numel() else 0.0)

    # ------------------------------------------------------------------------------------------------ step
    def step(self) -> dict:
        t0 = time.perf_counter()
        cfg, k = self.cfg, self.step_idx
        vals = self.inner.values
        vals.grad = None
        if self.wv_opt is not None:
            self.wv_opt.zero_grad(set_to_none=True)
        epoch, wi = self.train_index(k)
        win = self.train_windows[wi]
        rec: dict = dict(step=k, epoch=epoch, window=wi, window_start=int(win.start))
        read = [self.read_rows(win)]
        m = self._pass(win, None, 1.0, None if self.loss_weights is None else self.loss_weights[wi], cfg.rho_on_train)
        rec.update(nll=m["value"], n_targets=m["n"], rho_train_median=m["rho_median"], rho_train_mean=m["rho_mean"],
                   rho_train_p99=m["rho_p99"], rho_train_frac_over=m["rho_frac_over"])
        for set_no, name in enumerate(sorted(self.anchors)):
            ws, tg = self.anchors[name]
            lam = float(cfg.anchor_lambdas.get(name, 1.0))
            ms = []
            for j in self.anchor_indices(set_no, len(ws), k):
                read.append(self.read_rows(ws[j]))
                ms.append(self._pass(ws[j], tg[j], lam, None, True))
            rec[f"kl_{name}"] = float(np.mean([x["value"] for x in ms]))
            for q in ("rho_median", "rho_mean", "rho_p99", "rho_frac_over", "rho_pen"):
                rec[f"{q}_{name}"] = float(np.mean([x[q] for x in ms]))
        t_fb = time.perf_counter() - t0

        pos = torch.from_numpy(np.unique(np.concatenate(read)))
        g = torch.zeros(pos.numel(), self.inner.embedding_dim)
        if vals.grad is not None and pos.numel():
            gr = vals.grad.coalesce() if vals.grad.is_sparse else vals.grad.to_sparse(1).coalesce()
            gi, gv = gr.indices()[0], gr.values()
            p = torch.searchsorted(gi, pos).clamp_max(max(gi.numel() - 1, 0))
            hit = gi[p] == pos if gi.numel() else torch.zeros_like(pos, dtype=torch.bool)
            g[hit] = gv[p[hit]]
        with torch.no_grad():
            om, s = self.omega[pos], self.row_scale[pos]
            g0 = self.opt.observe_gscale(g, s)   # loss gradient only, before the smooth L2 term is added
            if cfg.lambda_l2 > 0:
                g = g + 2.0 * cfg.lambda_l2 * (om / s.pow(2))[:, None] * vals.data[pos]
            gnorm = float(g.norm())
            lam1 = cfg.lambda_l1 if cfg.lambda_l1 > 0 else (cfg.lambda_l1_rel * g0 if cfg.lambda_l1_rel > 0 and g0
                                                           else 0.0)
            st = self.opt.step(vals.data, pos, g, cfg.lr,
                               l1=(lam1 * om / s) if lam1 > 0 else None,
                               radius=(cfg.kappa * s) if cfg.kappa is not None else None)
            vals.grad = None
            if self.wv_opt is not None:
                self.wv_opt.step()
            rn = vals.data[pos].norm(dim=1)
        cur, peak = rss_mb()
        self.step_idx += 1
        rec.update(rows_read=int(pos.numel()), grad_norm=gnorm, a0=self.opt.a0, g0=self.opt.g0, lambda_1=lam1, row_norm_mean_read=float(rn.mean()) if
                   rn.numel() else 0.0, rows_nonzero=int((vals.data != 0).any(1).sum()), lr=cfg.lr,
                   fwd_bwd_s=t_fb, step_s=time.perf_counter() - t0, rss_mb=cur, peak_rss_mb=peak, **st)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def train(self, n_steps: int | None = None, ckpt_path: str | Path | None = None, callback=None) -> list[dict]:
        end = self.total_steps if n_steps is None else min(self.total_steps, self.step_idx + n_steps)
        out = []
        while self.step_idx < end:
            rec = self.step()
            out.append(rec)
            if callback is not None:
                callback(rec)
            if ckpt_path is not None and self.cfg.ckpt_every and self.step_idx % self.cfg.ckpt_every == 0:
                self.save_checkpoint(ckpt_path)
        if ckpt_path is not None:
            self.save_checkpoint(ckpt_path)
        return out

    # ------------------------------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def eval_nll(self, windows: Sequence[Window], graft_on: bool = True) -> dict:
        """Mean NLL over the scored targets of `windows` with the graft on or off (no grad)."""
        tot, n = 0.0, 0
        was = self.host.training
        self.host.eval()
        try:
            for w in windows:
                ids = torch.from_numpy(np.ascontiguousarray(w.ids))[None]
                labels = torch.from_numpy(np.ascontiguousarray(w.labels))
                if graft_on:
                    self.installed.context = torch.from_numpy(np.ascontiguousarray(w.context))[None]
                    h = final_hidden(self.host, ids)[0]
                    self.installed.context = None
                else:
                    self.installed.remove()
                    try:
                        h = final_hidden(self.host, ids)[0]
                    finally:
                        self.installed = install_graft(self.host, self.graft, record=False)
                tot += float(self._chunked(h, labels, None, None))
                n += int((labels != IGNORE).sum())
        finally:
            self.host.train(was)
        return dict(nll=tot / max(n, 1), n=n)

    # ------------------------------------------------------------------------------------------- artifacts
    def relabelled_store(self) -> SparseRowStore:
        """key-shuffle only: the rows that reproduce the TRAINED function through the true keys (diagnostic)."""
        if self.shuffle is None:
            raise ValueError("relabelled_store is for the key-shuffle arm")
        return self.shuffle.served_store(self.trained_store())

    def trained_store(self) -> SparseRowStore:
        """The trained slots (compact store; for key-shuffle these are indexed by the permuted slot)."""
        return self.inner.store.copy()

    def cartridge(self, prune_zero: bool = True) -> SparseRowStore:
        """Rows to serve through the plain graft. key-shuffle: the trained slots as stored (read through the TRUE
        keys, the C1 control; see module doc). Zero rows pruned by default."""
        st = self.trained_store()
        if prune_zero:
            st.prune_zero()
        return st

    # ------------------------------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        sd = dict(step=self.step_idx, row_ids=self.inner.row_ids.clone(), values=self.inner.values.data.clone(),
                  opt=self.opt.state_dict(), cfg=self.cfg.to_dict(), fmt=self.fmt.to_json(),
                  omega=self.omega.clone(), row_scale=self.row_scale.clone())
        if self.wv_opt is not None:
            sd["wv"] = self.graft.ple.value_proj.weight.data.clone()
            sd["wv_opt"] = self.wv_opt.state_dict()
        if self.shuffle is not None:
            sd["shuffle_dst"] = self.shuffle.dst.clone()
        return sd

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str | Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if FormatSpec.from_json(sd["fmt"]).fingerprint() != self.fmt.fingerprint():
            raise ValueError("checkpoint format fingerprint differs")
        a, b = dict(sd["cfg"]), self.cfg.to_dict()
        for d in (a, b):
            for k in ("max_steps", "epochs", "ckpt_every"):
                d.pop(k, None)
        if a != b:
            raise ValueError(f"checkpoint config differs: {a} vs {b}")
        if not torch.equal(sd["row_ids"], self.inner.row_ids):
            raise ValueError("checkpoint reserved rows differ")
        with torch.no_grad():
            self.inner.values.data.copy_(sd["values"])
            self.omega.copy_(sd["omega"])
            self.row_scale.copy_(sd["row_scale"])
            if self.wv_opt is not None:
                self.graft.ple.value_proj.weight.data.copy_(sd["wv"])
        self.opt.load_state_dict(sd["opt"])
        if self.wv_opt is not None:
            self.wv_opt.load_state_dict(sd["wv_opt"])
        if self.shuffle is not None and not torch.equal(sd["shuffle_dst"], self.shuffle.dst):
            raise ValueError("checkpoint key-shuffle permutation differs")
        self.step_idx = int(sd["step"])


# ------------------------------------------------------------------------------------------ setup helpers
@torch.no_grad()
def layer_input_norm(host: nn.Module, windows: Sequence[Window], layer_idx: int, max_tokens: int = 4096) -> float:
    """Median per-token norm of the host's layer-`layer_idx` input hidden state (no graft), for row scales."""
    from ..backends.graft.qwen35 import decoder_layers
    norms: list[torch.Tensor] = []
    hk = decoder_layers(host)[layer_idx].register_forward_pre_hook(
        lambda m, a, kw: norms.append((a[0] if a else kw["hidden_states"])[0].norm(dim=-1)), with_kwargs=True)
    was = host.training
    host.eval()
    try:
        n = 0
        for w in windows:
            ids = w.ids[: max_tokens - n]
            host(input_ids=torch.from_numpy(np.ascontiguousarray(ids))[None], use_cache=False, logits_to_keep=1)
            n += ids.size
            if n >= max_tokens:
                break
    finally:
        hk.remove()
        host.train(was)
    return float(torch.cat(norms).median())


def glin_row_scale(fmt: FormatSpec, h_norm: float) -> float:
    """G-lin s_j (§I.1): with W_v = value_scale * orthogonal, ||0.5 W_v,h d_j|| = 0.5 value_scale ||d_j|| exactly,
    so s_j = 2 ||h|| / value_scale makes ||d_j|| / s_j the row's output norm relative to the hidden norm."""
    return 2.0 * h_norm / fmt.module.value_scale


__all__ = ["ARMS", "TrainConfig", "KeyShuffle", "PermutedRowTable", "SparseRowAdagrad", "RowTrainer", "rss_mb",
           "host_fingerprint", "layer_input_norm", "glin_row_scale"]
