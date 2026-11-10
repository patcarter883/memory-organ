"""Teacher-forced scoring at a fixed in-file context (§6.A.1, §6.A.2).

A document is one file (§6.A.1 format, tokenised with `add_special_tokens=False`), scored on its own: the window
never crosses into another file. Windows: for scored-target starts s = min_context, min_context + stride, ...,
the forward covers doc[a:e] with a = max(0, s - context) and e = min(s + stride, n); targets are the tokens
doc[j] for j in [s, e), predicted from the logits at j - 1. So every scored target has at least
min(s, context) >= min_context in-file tokens before it, and every arm sees identical windows.

The §6.A.1 protocol is context=2048, stride=1024, min_context=1024 (`PROTOCOL`). The red-team check of the proxy
(S1/rt-science/ctx_len.py) used explicit 2,048-token blocks; `Window` objects can be built directly for that.

Conditions (arms) plug in through `Condition`: hooks while active (grafts, LoRA, injections), an optional token
prefix per window (RAG), and an optional adjustment of the target log-probs (count-LM / kNM interpolation).
Logits are never materialised for the whole window: the final hidden states are captured with a forward hook on
the decoder and the LM head is applied to the target rows in chunks.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

PROTOCOL = dict(context=2048, stride=1024, min_context=1024)


@dataclass
class Doc:
    """One scored file. `ids` int64 token ids; `group` is the bootstrap cluster (the file)."""
    name: str
    ids: np.ndarray
    text: str | None = None
    offsets: list[tuple[int, int]] | None = None
    group: str | None = None
    target_mask: np.ndarray | None = None  # optional bool [n]: only these targets count (Split B added lines)

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=np.int64)
        if self.group is None:
            self.group = self.name

    def __len__(self) -> int:
        return int(self.ids.shape[0])


def encode_doc(tok, name: str, text: str, group: str | None = None) -> Doc:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return Doc(name=name, ids=np.asarray(enc["input_ids"], dtype=np.int64), text=text,
               offsets=[tuple(o) for o in enc["offset_mapping"]], group=group)


@dataclass(frozen=True)
class Window:
    """Forward over doc[a:e]; targets j in [s, e) (s > a)."""
    doc: int
    a: int
    s: int
    e: int

    @property
    def n_targets(self) -> int:
        return self.e - self.s


def plan_doc(n: int, context: int = 2048, stride: int = 1024, min_context: int = 1024) -> list[tuple[int, int, int]]:
    if stride <= 0 or context < 1:
        raise ValueError("stride and context must be positive")
    first = max(1, min_context)
    out = []
    for s in range(first, n, stride):
        e = min(s + stride, n)
        out.append((max(0, s - context), s, e))
    return out


def plan_windows(docs: Sequence[Doc], context: int = 2048, stride: int = 1024, min_context: int = 1024) -> list[Window]:
    return [Window(i, a, s, e) for i, d in enumerate(docs) for a, s, e in plan_doc(len(d), context, stride, min_context)]


class Condition:
    """An evaluation arm. Subclasses override any of activate / prefix / adjust."""
    name = "base"
    uses_hidden = False
    provides_prefix = False

    def activate(self, model: nn.Module) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def prefix(self, doc: Doc, win: Window) -> np.ndarray | None:
        return None

    def adjust(self, logp: torch.Tensor, *, hidden: torch.Tensor, doc: Doc, win: Window,
               targets: np.ndarray) -> torch.Tensor:
        """logp [n, V] fp32 host log-probs for doc targets `targets` (doc positions); return the arm's log-probs."""
        return logp

    def describe(self) -> dict:
        return {"name": self.name}


class Composite(Condition):
    """Several conditions at once (e.g. RAG + cartridge). At most one may supply a prefix."""

    def __init__(self, parts: Sequence[Condition], name: str | None = None):
        self.parts = list(parts)
        self.name = name or "+".join(p.name for p in self.parts)
        self.uses_hidden = any(p.uses_hidden for p in self.parts)
        self.provides_prefix = any(p.provides_prefix for p in self.parts)

    @contextlib.contextmanager
    def activate(self, model):
        with contextlib.ExitStack() as st:
            for p in self.parts:
                st.enter_context(p.activate(model))
            yield

    def prefix(self, doc, win):
        got = [x for x in (p.prefix(doc, win) for p in self.parts) if x is not None]
        if len(got) > 1:
            raise ValueError("two conditions supply a prefix")
        return got[0] if got else None

    def adjust(self, logp, **kw):
        for p in self.parts:
            logp = p.adjust(logp, **kw)
        return logp

    def describe(self) -> dict:
        return {"name": self.name, "parts": [p.describe() for p in self.parts]}


def _decoder(model: nn.Module) -> nn.Module:
    for attr in ("model", "transformer"):
        m = getattr(model, attr, None)
        if isinstance(m, nn.Module):
            lm = getattr(m, "language_model", None)
            return lm if isinstance(lm, nn.Module) else m
    raise TypeError("cannot find the decoder of this CausalLM")


def lm_head_of(model: nn.Module) -> nn.Module:
    return model.get_output_embeddings()


@torch.no_grad()
def final_hidden(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Final (post-norm) hidden states [B, L, d] of a CausalLM, computing logits for one position only."""
    box: dict = {}

    def hook(_m, _a, out):
        box["h"] = out[0] if isinstance(out, tuple) else out.last_hidden_state

    h = _decoder(model).register_forward_hook(hook)
    try:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        h.remove()
    return box["h"]


def chunked_logprobs(lm_head: nn.Module, hidden: torch.Tensor, chunk: int = 256) -> Iterable[tuple[int, torch.Tensor]]:
    for c0 in range(0, hidden.shape[0], chunk):
        yield c0, torch.log_softmax(lm_head(hidden[c0:c0 + chunk]).float(), dim=-1)


@dataclass
class ScoreResult:
    """Per-target arrays; aligned across conditions scored on the same docs and windows."""
    doc_names: list[str]
    groups: list[str]
    doc: np.ndarray      # int32 [N] doc index
    pos: np.ndarray      # int32 [N] target position in the doc
    ctx: np.ndarray      # int32 [N] in-file tokens before the target inside its window
    nll: np.ndarray      # float32 [N]
    hit: np.ndarray      # bool [N] argmax == target
    kl: np.ndarray | None = None      # float32 [N] KL(base || arm) (with_kl)
    agree: np.ndarray | None = None   # bool [N] argmax(base) == argmax(arm)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.nll.shape[0])

    def key(self) -> np.ndarray:
        return self.doc.astype(np.int64) * (1 << 32) + self.pos.astype(np.int64)

    def select(self, mask: np.ndarray) -> "ScoreResult":
        f = lambda x: None if x is None else x[mask]  # noqa: E731
        return ScoreResult(self.doc_names, self.groups, self.doc[mask], self.pos[mask], self.ctx[mask],
                           self.nll[mask], self.hit[mask], f(self.kl), f(self.agree), dict(self.meta))

    def save(self, path: str | Path) -> None:
        arr = dict(doc=self.doc, pos=self.pos, ctx=self.ctx, nll=self.nll, hit=self.hit)
        if self.kl is not None:
            arr.update(kl=self.kl, agree=self.agree)
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(dict(
            meta=self.meta, doc_names=self.doc_names, groups=self.groups)).encode(), dtype=np.uint8), **arr)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreResult":
        z = np.load(path)
        m = json.loads(bytes(z["meta"]).decode())
        return cls(m["doc_names"], m["groups"], z["doc"], z["pos"], z["ctx"], z["nll"], z["hit"],
                   z["kl"] if "kl" in z else None, z["agree"] if "agree" in z else None, m["meta"])

    def tokens_per_sec(self) -> float:
        return len(self) / max(self.meta.get("wall_s", 0.0), 1e-9)

    def wall_per_10k(self) -> float:
        return 1e4 * self.meta.get("wall_s", 0.0) / max(1, len(self))


@torch.no_grad()
def score(model: nn.Module, docs: Sequence[Doc], windows: Sequence[Window], condition: Condition | None = None, *,
          chunk: int = 256, with_kl: bool = False, progress: Callable[[int, int, float], None] | None = None,
          ) -> ScoreResult:
    """Teacher-forced NLL of every target in `windows` under `condition` (default: the plain host)."""
    cond = condition or Condition()
    head = lm_head_of(model)
    out: dict[str, list] = {k: [] for k in ("doc", "pos", "ctx", "nll", "hit", "kl", "agree")}
    t0 = time.time()
    fwd_tokens = 0
    for wi, w in enumerate(windows):
        d = docs[w.doc]
        if not (0 <= w.a < w.s < w.e <= len(d)):
            raise ValueError(f"bad window {w} for doc of {len(d)} tokens")
        pre = cond.prefix(d, w)
        pre = np.zeros(0, np.int64) if pre is None else np.asarray(pre, np.int64)
        if with_kl and len(pre):
            raise ValueError("with_kl needs a condition without a prefix")
        ids = torch.from_numpy(np.concatenate([pre, d.ids[w.a:w.e]]))[None]
        fwd_tokens += ids.shape[1]
        targets = np.arange(w.s, w.e)
        rows = torch.from_numpy(len(pre) + targets - 1 - w.a)
        with cond.activate(model):
            h = final_hidden(model, ids)[0, rows]
        hb = final_hidden(model, ids)[0, rows] if with_kl else None
        y = torch.from_numpy(d.ids[targets])
        for c0, lp in chunked_logprobs(head, h, chunk):
            tg = targets[c0:c0 + lp.shape[0]]
            lp = cond.adjust(lp, hidden=h[c0:c0 + lp.shape[0]], doc=d, win=w, targets=tg)
            yy = y[c0:c0 + lp.shape[0]]
            out["nll"].append((-lp.gather(1, yy[:, None])[:, 0]).numpy())
            am = lp.argmax(-1)
            out["hit"].append((am == yy).numpy())
            if with_kl:
                lb = torch.log_softmax(head(hb[c0:c0 + lp.shape[0]]).float(), -1)
                out["kl"].append((lb.exp() * (lb - lp)).sum(-1).clamp_min(0).numpy())
                out["agree"].append((lb.argmax(-1) == am).numpy())
        out["doc"].append(np.full(len(targets), w.doc, np.int32))
        out["pos"].append(targets.astype(np.int32))
        out["ctx"].append((targets - w.a).astype(np.int32))
        if progress:
            progress(wi + 1, len(windows), time.time() - t0)
    cat = lambda k, dt: np.concatenate(out[k]).astype(dt) if out[k] else np.zeros(0, dt)  # noqa: E731
    wall = time.time() - t0
    meta = dict(condition=cond.describe(), wall_s=wall, n_windows=len(windows), forward_tokens=int(fwd_tokens),
                threads=torch.get_num_threads(), with_kl=with_kl)
    res = ScoreResult([d.name for d in docs], [d.group for d in docs], cat("doc", np.int32), cat("pos", np.int32),
                      cat("ctx", np.int32), cat("nll", np.float32), cat("hit", bool),
                      cat("kl", np.float32) if with_kl else None, cat("agree", bool) if with_kl else None, meta)
    keep = _target_masks(docs, res)
    if keep is not None:
        res = res.select(keep)
        res.meta = meta
    return res


def _target_masks(docs: Sequence[Doc], res: ScoreResult) -> np.ndarray | None:
    if all(d.target_mask is None for d in docs):
        return None
    keep = np.ones(len(res), bool)
    for i, d in enumerate(docs):
        if d.target_mask is not None:
            m = res.doc == i
            keep[m] = np.asarray(d.target_mask, bool)[res.pos[m]]
    return keep


__all__ = ["PROTOCOL", "Doc", "Window", "Condition", "Composite", "ScoreResult", "encode_doc", "plan_doc",
           "plan_windows", "score", "final_hidden", "chunked_logprobs", "lm_head_of"]
