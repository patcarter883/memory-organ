"""LoRA baseline (§6.A.4 "LoRA all-linear r=16"): a small own implementation (peft is not in the venv).

`add_lora` wraps every nn.Linear inside the decoder layers (attention / GDN projections and the dense MLP; on MoE
hosts the shared-expert projections) except the MoE router (`mlp.gate`) and `shared_expert_gate`; routed experts
stored as fused 3-D parameters (transformers 5 MoE blocks) are NOT nn.Linear and are not adapted (open issue).
B is zero-initialised, so an attached, enabled adapter is an exact no-op until trained.

`train_lora` uses the same training windows (token arrays) and token budget as a cartridge arm: next-token NLL,
AdamW, linear warm-up + cosine decay.
"""
from __future__ import annotations

import contextlib
import math
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....backends.graft.qwen35 import decoder_layers
from ..scoring import Condition

DEFAULT_EXCLUDE = ("mlp.gate", "shared_expert_gate", "router")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0, seed: int = 0):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        g = torch.Generator().manual_seed(seed)
        a = torch.empty(r, base.in_features)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=g)
        self.lora_A = nn.Parameter(a.to(base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.enabled = True

    @property
    def weight(self):  # some HF code reads .weight of projections (dtype / device checks)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def _excluded(name: str, exclude: Sequence[str]) -> bool:
    return any(name == e or name.endswith("." + e) or name.endswith(e) for e in exclude)


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
             exclude: Sequence[str] = DEFAULT_EXCLUDE, seed: int = 0) -> list[str]:
    """Wrap all decoder-layer nn.Linear modules in place; freezes every other parameter. Returns wrapped names."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = decoder_layers(model)
    wrapped = []
    for li, layer in enumerate(layers):
        for name, mod in list(layer.named_modules()):
            if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear) or _excluded(name, exclude):
                continue
            parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
            child = name.rsplit(".", 1)[-1]
            setattr(parent, child, LoRALinear(mod, r, alpha, dropout, seed=seed + 7919 * li + len(wrapped)))
            wrapped.append(f"layers.{li}.{name}")
    return wrapped


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in lora_modules(model) for p in (m.lora_A, m.lora_B)]


def set_lora_enabled(model: nn.Module, flag: bool) -> None:
    for m in lora_modules(model):
        m.enabled = flag


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    missing = [k for k in sd if k not in model.state_dict()]
    if missing:
        raise KeyError(f"LoRA keys not in model (call add_lora with the same r first): {missing[:3]}")
    model.load_state_dict(sd, strict=False)


def remove_lora(model: nn.Module) -> None:
    """Restore the original nn.Linear modules."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, cname, child.base)


def n_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


class LoRACondition(Condition):
    """Adapters enabled while active, disabled otherwise (attach with add_lora first)."""

    def __init__(self, model: nn.Module, name: str = "lora_all_linear_r16"):
        self.name = name
        self.n_params = n_lora_params(model)
        if not self.n_params:
            raise ValueError("model has no LoRA adapters")
        set_lora_enabled(model, False)

    @contextlib.contextmanager
    def activate(self, model):
        set_lora_enabled(model, True)
        try:
            yield
        finally:
            set_lora_enabled(model, False)

    def describe(self) -> dict:
        return dict(name=self.name, lora_params=self.n_params)


def _lm_loss(model: nn.Module, x: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Mean next-token NLL of x [B, T] with chunked LM-head application (no full [B, T, V] logits kept twice)."""
    from ..scoring import _decoder
    out = _decoder(model)(input_ids=x, use_cache=False)
    h = out.last_hidden_state[:, :-1].reshape(-1, out.last_hidden_state.shape[-1])
    y = x[:, 1:].reshape(-1)
    head = model.get_output_embeddings()
    tot = h.new_zeros(())
    for c0 in range(0, h.shape[0], chunk):
        tot = tot + F.cross_entropy(head(h[c0:c0 + chunk]).float(), y[c0:c0 + chunk], reduction="sum")
    return tot / y.numel()


def train_lora(model: nn.Module, windows: Sequence[np.ndarray], *, epochs: int = 1, lr: float = 2e-4,
               batch: int = 1, weight_decay: float = 0.0, warmup_frac: float = 0.05, seed: int = 0,
               max_tokens: int | None = None, log: Callable[[dict], None] | None = None,
               checkpointing: bool = True) -> dict:
    """Train attached adapters on token windows (each [T], equal or ragged lengths; batch=1 for ragged).
    Token budget = sum of window lengths x epochs (capped by max_tokens)."""
    params = lora_parameters(model)
    if not params:
        raise ValueError("call add_lora first")
    set_lora_enabled(model, True)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    order = [i for _ in range(epochs) for i in rng.permutation(len(windows))]
    steps = math.ceil(len(order) / batch)
    warm = max(1, int(warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    ckpt = checkpointing and hasattr(model, "gradient_checkpointing_enable")
    if ckpt:  # activations of a 24-layer host at T=512 are ~2 GB without it
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    hist, seen, t0 = [], 0, time.time()
    try:
        for st in range(steps):
            idx = order[st * batch:(st + 1) * batch]
            xs = [np.asarray(windows[i], np.int64) for i in idx]
            if max_tokens is not None and seen >= max_tokens:
                break
            if len({len(x) for x in xs}) != 1:
                raise ValueError("ragged windows need batch=1")
            x = torch.from_numpy(np.stack(xs))
            loss = _lm_loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += x.numel()
            rec = dict(step=st + 1, loss=float(loss.detach()), tokens=seen, lr=sched.get_last_lr()[0], sec=time.time() - t0)
            hist.append(rec)
            if log:
                log(rec)
    finally:
        model.eval()
        if ckpt:
            model.gradient_checkpointing_disable()
        for p in params:
            p.requires_grad_(False)
    return dict(steps=len(hist), tokens=seen, wall_s=time.time() - t0, final_loss=hist[-1]["loss"] if hist else None,
                lora_params=n_lora_params(model), history=hist)


__all__ = ["LoRALinear", "LoRACondition", "add_lora", "remove_lora", "lora_parameters", "lora_modules",
           "set_lora_enabled", "lora_state_dict", "load_lora_state_dict", "n_lora_params", "train_lora"]
