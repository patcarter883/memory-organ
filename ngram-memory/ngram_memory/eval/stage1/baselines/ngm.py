"""NGM control (§6.A.4): untrained causal 2/3-token averages of the host's own input embeddings, injected at the
graft site (input of decoder `layer_idx`) with no corpus:

    u_t = mean_{n in orders} mean_{k < n} E[x_{t-k}]      (only tokens inside the current EOS segment)
    inject_t = ratio * ||h_t|| * u_t / ||u_t||            (fixed injection-to-hidden ratio; tune `ratio` on dev)

The fixed ratio makes the arm's size comparable to a cartridge's measured ||out|| / ||h|| (§I.8: median 3.8% on
held-out project text for the proxy).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..conditions import LayerInjection


def ngram_embedding_average(ids: torch.Tensor, emb: torch.Tensor, orders: Sequence[int] = (2, 3),
                            eos: int | None = None) -> torch.Tensor:
    """u [B, T, d] for ids [B, T] (causal; positions before the sequence or an EOS segment start are skipped)."""
    B, T = ids.shape
    e = emb[ids]  # [B, T, d]
    pos = torch.arange(T).expand(B, T)
    if eos is not None:
        is_eos = ids == eos
        last = torch.where(is_eos, pos, torch.full_like(pos, -1)).cummax(1).values
        seg0 = torch.cat([torch.full((B, 1), -1), last[:, :-1]], 1) + 1  # first index of the token's segment
    else:
        seg0 = torch.zeros(B, T, dtype=torch.long)
    outs = []
    for n in orders:
        acc = torch.zeros_like(e)
        cnt = torch.zeros(B, T, 1, dtype=e.dtype)
        for k in range(n):
            sh = torch.zeros_like(e)
            if k:
                sh[:, k:] = e[:, :-k]
            else:
                sh = e
            ok = ((pos - k) >= seg0).unsqueeze(-1).to(e.dtype)
            acc = acc + sh * ok
            cnt = cnt + ok
        outs.append(acc / cnt.clamp_min(1))
    return torch.stack(outs).mean(0)


class NGMCondition(LayerInjection):
    def __init__(self, model: nn.Module, layer_idx: int = 1, ratio: float = 0.038, orders: Sequence[int] = (2, 3),
                 eos: int | None = None, name: str | None = None):
        emb = model.get_input_embeddings().weight.detach()
        self.ratio, self.orders = ratio, tuple(orders)

        def inject(ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            u = ngram_embedding_average(ids, emb, self.orders, eos).to(hidden.dtype)
            return self.ratio * hidden.norm(dim=-1, keepdim=True) * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        super().__init__(layer_idx, inject, name or f"ngm_r{ratio:g}")

    def describe(self) -> dict:
        d = super().describe()
        d.update(ratio=self.ratio, orders=list(self.orders))
        return d


__all__ = ["NGMCondition", "ngram_embedding_average"]
