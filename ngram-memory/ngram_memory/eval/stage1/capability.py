"""R3 capability (§6.A.3): forward-only multiple choice by log-likelihood, paired McNemar between arms.

Item format: {"id", "task", "context", "choices": [str, ...], "label": int}. Score = sum of choice-token
log-probs given the context (`acc`) and the same divided by the choice's character length (`acc_norm`, the usual
HellaSwag convention). Loaders map MMLU / ARC-Challenge / HellaSwag from the HF datasets cache into this format
(they need the datasets locally; none were cached when this was written). Conditions act through `activate`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..metrics import mcnemar
from .scoring import Condition


@torch.no_grad()
def choice_logprobs(model, tok, context: str, choices: Sequence[str]) -> list[float]:
    c = tok(context, add_special_tokens=False)["input_ids"]
    out = []
    for ch in choices:
        a = tok(ch, add_special_tokens=False)["input_ids"]
        x = torch.tensor([c + a])
        lg = model(input_ids=x, use_cache=False, logits_to_keep=len(a) + 1).logits[0, :-1].float()
        out.append(float(torch.log_softmax(lg, -1).gather(1, torch.tensor(a)[:, None]).sum()))
    return out


def score_mc(model, tok, items: Sequence[dict], condition: Condition | None = None) -> list[dict]:
    cond = condition or Condition()
    res = []
    with cond.activate(model):
        for it in items:
            lp = choice_logprobs(model, tok, it["context"], it["choices"])
            norm = [x / max(1, len(c)) for x, c in zip(lp, it["choices"])]
            res.append(dict(id=it.get("id"), task=it.get("task"), acc=int(np.argmax(lp)) == it["label"],
                            acc_norm=int(np.argmax(norm)) == it["label"], logprobs=lp))
    return res


def paired_mc(base: Sequence[dict], arm: Sequence[dict]) -> dict:
    out = {}
    tasks = sorted({b["task"] for b in base}) + ["all"]
    for t in tasks:
        idx = [i for i, b in enumerate(base) if t == "all" or b["task"] == t]
        r = {}
        for k in ("acc", "acc_norm"):
            bb = np.array([base[i][k] for i in idx], bool)
            aa = np.array([arm[i][k] for i in idx], bool)
            r[k] = dict(base=float(bb.mean()), arm=float(aa.mean()), ratio=float(aa.mean() / max(bb.mean(), 1e-12)),
                        d_pts=100 * float(aa.mean() - bb.mean()), mcnemar=mcnemar(bb, aa))
        out[t] = dict(n=len(idx), **r)
    return out


def load_items(task: str, n: int = 1000, seed: int = 0) -> list[dict]:
    """MMLU (cais/mmlu all/test), ARC-Challenge (allenai/ai2_arc test), HellaSwag (Rowan/hellaswag validation)."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    if task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        rows = [dict(id=r["id"], task=task, context=f"Question: {r['question']}\nAnswer:",
                     choices=[" " + c for c in r["choices"]["text"]], label=r["choices"]["label"].index(r["answerKey"]))
                for r in ds if r["answerKey"] in r["choices"]["label"]]
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = [dict(id=f"mmlu{i}", task=task, context=f"Question: {r['question']}\nAnswer:",
                     choices=[" " + c for c in r["choices"]], label=int(r["answer"])) for i, r in enumerate(ds)]
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        rows = [dict(id=r["ind"], task=task, context=r["ctx"], choices=[" " + e for e in r["endings"]],
                     label=int(r["label"])) for r in ds]
    else:
        raise ValueError(task)
    if len(rows) > n:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n, replace=False))]
    return rows


__all__ = ["choice_logprobs", "score_mc", "paired_mc", "load_items"]
