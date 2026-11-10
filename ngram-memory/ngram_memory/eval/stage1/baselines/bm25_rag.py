"""RAG-BM25 baseline (§6.A.4): index the train files in 40-line chunks with stride 20; for each scoring window the
query is the last 20 lines of the file prefix before the window's first scored target; retrieved chunks are
prepended as `# retrieved from <path>\\n<chunk>` up to a token budget (1k / 4k). The window's in-file context is
unchanged (the retrieved text is extra prefill, reported in the result meta). Self-implemented Okapi BM25
(k1 = 1.2, b = 0.75); terms are lowercased identifiers plus their snake/camel parts.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..scoring import Condition, Doc, Window

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def terms(text: str) -> list[str]:
    out = []
    for m in _ID.findall(text):
        lo = m.lower()
        out.append(lo)
        parts = [p.lower() for s in m.split("_") if s for p in _CAMEL.findall(s)]
        if len(parts) > 1:
            out.extend(parts)
    return out


@dataclass
class Chunk:
    path: str
    line0: int  # 1-based first line
    text: str


def make_chunks(files: Mapping[str, str], lines: int = 40, stride: int = 20) -> list[Chunk]:
    out = []
    for p in sorted(files):
        ls = files[p].splitlines(keepends=True)
        if not ls:
            continue
        starts = list(range(0, max(1, len(ls) - lines + stride), stride))
        for s in starts:
            seg = "".join(ls[s:s + lines])
            if seg.strip():
                out.append(Chunk(p, s + 1, seg))
            if s + lines >= len(ls):
                break
    return out


class BM25:
    def __init__(self, docs: Sequence[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.len = np.array([len(d) for d in docs], np.float64)
        self.avg = float(self.len.mean()) if self.N else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, q: Sequence[str]) -> np.ndarray:
        s = np.zeros(self.N)
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-9))
        for t in set(q):
            if t not in self.post:
                continue
            ids, cs = zip(*self.post[t])
            ids, cs = np.asarray(ids), np.asarray(cs, np.float64)
            s[ids] += self.idf[t] * cs * (self.k1 + 1) / (cs + norm[ids])
        return s


class BM25Retriever:
    def __init__(self, files: Mapping[str, str], lines: int = 40, stride: int = 20):
        self.chunks = make_chunks(files, lines, stride)
        self.index = BM25([terms(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 50, exclude_path: str | None = None) -> list[tuple[float, Chunk]]:
        s = self.index.scores(terms(query))
        order = np.argsort(-s, kind="stable")
        out = []
        for i in order:
            if s[i] <= 0 or len(out) >= k:
                break
            c = self.chunks[int(i)]
            if exclude_path is not None and c.path == exclude_path:
                continue
            out.append((float(s[i]), c))
        return out


class BM25RAGCondition(Condition):
    """Prepends up to `budget` tokens of retrieved train chunks to every window (best chunk nearest the code)."""
    provides_prefix = True

    def __init__(self, retriever: BM25Retriever, tok, budget: int = 1024, query_lines: int = 20,
                 name: str | None = None, exclude_self: bool = True):
        self.r, self.tok, self.budget, self.query_lines = retriever, tok, budget, query_lines
        self.name = name or (f"rag_bm25_{budget // 1024}k" if budget % 1024 == 0 else f"rag_bm25_{budget}")
        self.exclude_self = exclude_self
        self.prefill_tokens = 0
        self.n_windows = 0
        self._cache: dict = {}

    def query_text(self, doc: Doc, win: Window) -> str:
        if doc.text is None or doc.offsets is None:
            raise ValueError("BM25 RAG needs Doc.text and Doc.offsets (use scoring.encode_doc)")
        cut = doc.offsets[win.s][0]
        lines = doc.text[:cut].splitlines()
        return "\n".join(lines[-self.query_lines:])

    def prefix(self, doc: Doc, win: Window) -> np.ndarray:
        k = (doc.name, win)
        if k in self._cache:
            return self._cache[k]
        hits = self.r.search(self.query_text(doc, win), exclude_path=doc.group if self.exclude_self else None)
        parts: list[list[int]] = []
        used = 0
        for _, c in hits:
            ids = self.tok(f"# retrieved from {c.path}\n{c.text}" + ("" if c.text.endswith("\n") else "\n"),
                           add_special_tokens=False)["input_ids"]
            room = self.budget - used
            if room <= 0:
                break
            parts.append(ids[:room])
            used += min(len(ids), room)
        out = np.asarray([t for p in reversed(parts) for t in p], np.int64)
        self._cache[k] = out
        self.prefill_tokens += len(out)
        self.n_windows += 1
        return out

    def describe(self) -> dict:
        return dict(name=self.name, budget=self.budget, chunks=len(self.r.chunks),
                    mean_prefill_tokens=self.prefill_tokens / max(1, self.n_windows))


__all__ = ["BM25", "BM25Retriever", "BM25RAGCondition", "make_chunks", "terms", "Chunk"]
