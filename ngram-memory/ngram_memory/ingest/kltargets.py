"""Base-model KL targets for anchor batches (§I.1 "KL targets"): top-k log-probs plus the log of the remaining
mass, computed once with the cartridge/graft OFF, so the trainer's KL(p_0 || p_delta) needs no second host
forward. k = 64 at int32 index + fp32 log-prob is 512 B per token (+4 B remainder).

KL with a top-k teacher (the remainder is one bucket):
    KL = sum_{i in topk} P_i (log P_i - log q_i) + R (log R - log q_R),   R = 1 - sum P_i,  q_R = 1 - sum q_i
which is exact when k = V and otherwise a lower bound of the full KL (merging outcomes cannot increase KL).

Also holds the host-side helpers shared with train.py: final_hidden() (decoder output without materialising the
[T, V] logits) and output_head().
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .corpus import Window


# -------------------------------------------------------------------------------------------- host helpers
def output_head(host: nn.Module) -> nn.Module:
    head = host.get_output_embeddings()
    if head is None:
        raise TypeError(f"{type(host).__name__} has no output embeddings")
    return head


def logit_softcap(host: nn.Module) -> float | None:
    cfg = getattr(host, "config", None)
    cap = getattr(cfg, "final_logit_softcapping", None)
    tc = getattr(cfg, "text_config", None)
    return getattr(tc, "final_logit_softcapping", cap) if cap is None and tc is not None else cap


def head_logits(head: nn.Module, h: torch.Tensor, softcap: float | None = None) -> torch.Tensor:
    z = head(h).float()
    if softcap:
        z = torch.tanh(z / softcap) * softcap
    return z


def final_hidden(host: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Last hidden state [B, T, d] of an HF causal LM (after its final norm), through the host's own forward so that
    top-level hooks (the graft's input_ids capture) fire; only the last position's logits are computed."""
    dec = host.get_decoder()
    box: dict = {}

    def grab(_m, _a, out):
        box["h"] = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    hk = dec.register_forward_hook(grab)
    try:
        host(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
    finally:
        hk.remove()
    return box["h"]


# ---------------------------------------------------------------------------------------------- targets
@dataclass
class WindowTargets:
    idx: torch.Tensor      # int32 [T, k]
    logp: torch.Tensor     # fp32  [T, k]
    rest: torch.Tensor     # fp32  [T]  log of the mass outside the top-k (-inf when k = V)


@dataclass
class KLTargets:
    k: int
    windows: list[WindowTargets]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> WindowTargets:
        return self.windows[i]

    def nbytes(self) -> int:
        return sum(w.idx.numel() * 4 + w.logp.numel() * 4 + w.rest.numel() * 4 for w in self.windows)

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file
        t = {}
        for i, w in enumerate(self.windows):
            t[f"{i}.idx"], t[f"{i}.logp"], t[f"{i}.rest"] = w.idx.contiguous(), w.logp.contiguous(), w.rest.contiguous()
        save_file(t, str(path), metadata={"k": str(self.k), "n": str(len(self.windows)),
                                          "meta": json.dumps(self.meta, default=str)})

    @classmethod
    def load(cls, path: str | Path) -> "KLTargets":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            md = f.metadata()
            n = int(md["n"])
            ws = [WindowTargets(f.get_tensor(f"{i}.idx"), f.get_tensor(f"{i}.logp"), f.get_tensor(f"{i}.rest"))
                  for i in range(n)]
        return cls(int(md["k"]), ws, json.loads(md.get("meta", "{}")))


def topk_targets_from_logits(z: torch.Tensor, k: int) -> WindowTargets:
    """z [T, V] fp32 logits -> top-k log-probs and the log of the remaining mass (exact logsumexp of the rest)."""
    lse = torch.logsumexp(z, dim=-1)
    top, idx = torch.topk(z, k, dim=-1)
    if k >= z.shape[-1]:
        rest = torch.full_like(lse, float("-inf"))
    else:
        zr = z.scatter(-1, idx, float("-inf"))
        rest = torch.logsumexp(zr, dim=-1) - lse
    return WindowTargets(idx.to(torch.int32), (top - lse[:, None]).float(), rest.float())


def topk_kl(z: torch.Tensor, t: WindowTargets | tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Per-position KL(p0 || q) [T] for student logits z [T, V] against top-k teacher targets."""
    idx, logp, rest = (t.idx, t.logp, t.rest) if isinstance(t, WindowTargets) else t
    idx = idx.long()
    lse = torch.logsumexp(z, dim=-1)
    lq_top = z.gather(-1, idx) - lse[:, None]
    kl = (logp.exp() * (logp - lq_top)).sum(-1)
    finite = torch.isfinite(rest)
    if bool(finite.any()):
        lq_rest = torch.logsumexp(z.scatter(-1, idx, float("-inf")), dim=-1) - lse
        r = torch.where(finite, rest, torch.zeros_like(rest))
        term = r.exp() * (r - lq_rest)
        kl = kl + torch.where(finite, term, torch.zeros_like(term))
    return kl


@torch.no_grad()
def compute_kl_targets(host: nn.Module, windows: Sequence[Window], k: int = 64, chunk: int = 512,
                       allow_hooks: bool = False, meta: dict | None = None) -> KLTargets:
    """Teacher targets for each window with the graft OFF. Refuses to run while forward pre-hooks are attached to
    the host (an installed graft registers one) unless allow_hooks=True."""
    if host._forward_pre_hooks and not allow_hooks:
        raise RuntimeError("host has forward pre-hooks (graft installed?); KL targets must be computed graft-off")
    head, cap = output_head(host), logit_softcap(host)
    was_training = host.training
    host.eval()
    out = []
    try:
        for w in windows:
            h = final_hidden(host, torch.from_numpy(np.ascontiguousarray(w.ids))[None])[0]
            parts = [topk_targets_from_logits(head_logits(head, h[s:s + chunk], cap), k)
                     for s in range(0, h.shape[0], chunk)]
            out.append(WindowTargets(torch.cat([p.idx for p in parts]), torch.cat([p.logp for p in parts]),
                                     torch.cat([p.rest for p in parts])))
    finally:
        host.train(was_training)
    return KLTargets(k, out, dict(meta or {}, n_windows=len(windows), tokens=int(sum(w.ids.size for w in windows))))


def window_digest(w: Window) -> str:
    """Short sha256 of a window's ids and labels (aligning stored targets with the windows they were built for)."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w.ids, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(w.labels, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


__all__ = ["output_head", "logit_softcap", "head_logits", "final_hidden", "WindowTargets", "KLTargets",
           "topk_targets_from_logits", "topk_kl", "compute_kl_targets", "window_digest"]
