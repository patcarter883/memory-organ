"""Stage-1b teacher targets (DESIGN §I.1 "objective upgrade path"): distillation with the repo in context.

    teacher = frozen host on [BM25-retrieved chunks of OTHER train files ; training window]   (graft OFF)
    student = host + rows on [training window] only (exactly what the served model sees)
    loss    = KL(teacher || student) on the window's scored targets (train.py, TrainConfig.objective)

The teacher is stored in the anchors' top-k format (kltargets.KLTargets / WindowTargets: top-k log-probs + the log
of the remaining mass, for every position of the window), so train.py reuses kltargets.topk_kl.

Windows. The trainer's windows are M1b's: corpus.make_windows over the PACKED Split-A train stream (files joined by
EOS), so one 4,096-token window can span several files. We keep exactly those windows (same data order, same
per-position n-gram loss weights, same rows read: only the objective changes between an NLL arm and a distill
arm) and treat every file that overlaps the window as "the window's own file": none of their chunks can be
retrieved. The file spans come from the same documents pack_stream joined (doc_spans()).

Retrieval (deterministic). The index is eval/stage1/baselines/bm25_rag.py's (make_chunks: 40-line chunks, stride
20, over the raw train-file texts keyed by repo path, as scripts/evaluate.py builds it for the RAG-BM25 condition;
Okapi BM25 k1 = 1.2, b = 0.75 on bm25_rag.terms). The query is the window's own text (decoded tokens): the teacher
is a training signal, so retrieval may look at the whole target window; what it may not see is the window's own
files. Hits are ordered by (-score, path, first line), only positive scores count. Each hit is rendered like the
RAG condition (`# retrieved from <path>\\n<chunk>`), taken best-first until `ctx_tokens` tokens (the last one cut
at the budget, head kept), and placed best-chunk-nearest-the-window, directly before the window's tokens (no
separator token, as the RAG condition).

Cache: <cache_dir>/teacher_<key16>.safetensors with an fcntl lock (<file>.lock), keyed by sha256 over the host
path, k, the retrieval config, the files' (path, text sha256) and every window's ids/labels (teacher_cache_key()).
Per-window retrieval stats (files, chunks, tokens, top score, seconds) are kept in KLTargets.meta["retrieval"].
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..eval.stage1.baselines.bm25_rag import BM25, make_chunks, terms
from .corpus import Document, Window
from .kltargets import (KLTargets, WindowTargets, final_hidden, head_logits, logit_softcap, output_head,
                        topk_targets_from_logits, window_digest)

TEACHER_FORMAT = "ngram-memory/stage1b-teacher/v2"   # v2: own files include the last target's file


# ------------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class TeacherConfig:
    ctx_tokens: int = 4096          # retrieved tokens placed before the window
    k: int = 64                     # top-k teacher log-probs (= budget topk)
    chunk_lines: int = 40           # bm25_rag.make_chunks
    chunk_stride: int = 20
    k1: float = 1.2
    b: float = 0.75
    max_hits: int = 200             # candidate chunks considered (best first) before the token budget stops
    prefill_chunk: int | None = 2048  # teacher forward in KV-cached chunks of this many tokens (None: one forward)
    logits_chunk: int = 512         # positions per [chunk, V] logits block

    def to_dict(self) -> dict:
        return asdict(self)

    def retrieval_dict(self) -> dict:
        d = self.to_dict()
        d.pop("logits_chunk")       # does not change the numbers
        return d


# ------------------------------------------------------------------------------------------- file spans
@dataclass(frozen=True)
class DocSpan:
    path: str
    start: int      # first stream offset of the document
    end: int        # one past its last token, including the EOS separator pack_stream appended


def doc_spans(docs: Sequence[Document], sep_eos: bool = True) -> list[DocSpan]:
    """Stream offsets of each document in corpus.pack_stream(docs, eos, sep_eos)."""
    out, pos = [], 0
    for d in docs:
        n = int(d.ids.size) + (1 if sep_eos else 0)
        out.append(DocSpan(d.path, pos, pos + n))
        pos += n
    return out


def window_files(w: Window, spans: Sequence[DocSpan]) -> list[str]:
    """Paths of the documents overlapping stream[w.start : w.start + len(w.ids) + 1] (sorted, unique): the window's
    inputs and its last target (position L-1 predicts stream[w.start + L], which can open the next document)."""
    a, b = int(w.start), int(w.start) + int(w.ids.size) + 1
    starts = np.fromiter((s.start for s in spans), np.int64, len(spans))
    i = max(0, int(np.searchsorted(starts, a, side="right")) - 1)
    hit = set()
    while i < len(spans) and spans[i].start < b:
        if spans[i].end > a:
            hit.add(spans[i].path)
        i += 1
    return sorted(hit)


# ------------------------------------------------------------------------------------------------ retrieval
@dataclass
class Retrieved:
    prefix: np.ndarray          # int64 token ids placed before the window
    stats: dict                 # files, excluded, chunks, tokens, top_score


class TeacherRetriever:
    """BM25 over the chunks of the train files (bm25_rag's index), excluding any set of paths per query."""

    def __init__(self, files: Mapping[str, str], cfg: TeacherConfig, encode: Callable[[str], Sequence[int]]):
        self.cfg, self.encode = cfg, encode
        self.chunks = make_chunks(files, cfg.chunk_lines, cfg.chunk_stride)
        self.index = BM25([terms(c.text) for c in self.chunks], k1=cfg.k1, b=cfg.b)
        self._paths = np.array([c.path for c in self.chunks], dtype=object)
        self._line0 = np.array([c.line0 for c in self.chunks], dtype=np.int64)
        # rank of each chunk in (path, line0) order: the tie-break ("ties broken by path")
        order = sorted(range(len(self.chunks)), key=lambda i: (self.chunks[i].path, self.chunks[i].line0))
        self._rank = np.empty(len(self.chunks), np.int64)
        self._rank[np.asarray(order, np.int64)] = np.arange(len(order))
        self._tok_cache: dict[int, list[int]] = {}

    def hits(self, query: str, exclude: Sequence[str]) -> list[tuple[float, int]]:
        if not self.chunks:
            return []
        s = self.index.scores(terms(query))
        ok = s > 0
        if exclude:
            ex = set(exclude)
            ok &= np.fromiter((p not in ex for p in self._paths), bool, len(self._paths))
        idx = np.nonzero(ok)[0]
        order = idx[np.lexsort((self._rank[idx], -s[idx]))]       # primary -score, then (path, line0)
        return [(float(s[i]), int(i)) for i in order[: self.cfg.max_hits]]

    def chunk_ids(self, i: int) -> list[int]:
        if i not in self._tok_cache:
            c = self.chunks[i]
            txt = f"# retrieved from {c.path}\n{c.text}" + ("" if c.text.endswith("\n") else "\n")
            self._tok_cache[i] = [int(t) for t in self.encode(txt)]
        return self._tok_cache[i]

    def retrieve(self, query: str, exclude: Sequence[str]) -> Retrieved:
        parts: list[list[int]] = []
        used, files, top = 0, [], None
        budget = int(self.cfg.ctx_tokens)
        for score, i in (self.hits(query, exclude) if budget > 0 else []):
            room = budget - used
            if room <= 0:
                break
            ids = self.chunk_ids(i)
            parts.append(ids[:room])
            used += min(len(ids), room)
            c = self.chunks[i]
            files.append(c.path)
            top = score if top is None else top
        prefix = np.asarray([t for p in reversed(parts) for t in p], np.int64)   # best chunk nearest the window
        return Retrieved(prefix, dict(files=sorted(set(files)), excluded=sorted(exclude), chunks=len(parts),
                                      tokens=int(prefix.size), top_score=top))


# ---------------------------------------------------------------------------------------------- teacher
@torch.no_grad()
def teacher_hidden(host: nn.Module, ids: np.ndarray, prefill_chunk: int | None) -> torch.Tensor:
    """Final hidden states [T, d] of the frozen host on `ids`, in one forward or in KV-cached prefill chunks."""
    x = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64))[None]
    if not prefill_chunk or x.shape[1] <= prefill_chunk:
        return final_hidden(host, x)[0]
    dec = host.get_decoder()
    past, hs = None, []
    for s in range(0, x.shape[1], prefill_chunk):
        out = dec(input_ids=x[:, s:s + prefill_chunk], past_key_values=past, use_cache=True)
        past = out.past_key_values
        hs.append(out.last_hidden_state[0])
    return torch.cat(hs)


@torch.no_grad()
def teacher_window_targets(host: nn.Module, prefix: np.ndarray, w: Window, cfg: TeacherConfig) -> WindowTargets:
    """Top-k teacher targets for every position of `w` with `prefix` in context (positions of w only)."""
    head, cap = output_head(host), logit_softcap(host)
    seq = np.concatenate([np.asarray(prefix, np.int64), np.asarray(w.ids, np.int64)])
    h = teacher_hidden(host, seq, cfg.prefill_chunk)[int(prefix.size):]
    C = cfg.logits_chunk
    parts = [topk_targets_from_logits(head_logits(head, h[s:s + C], cap), cfg.k) for s in range(0, h.shape[0], C)]
    return WindowTargets(torch.cat([p.idx for p in parts]), torch.cat([p.logp for p in parts]),
                         torch.cat([p.rest for p in parts]))


def files_digest(files: Mapping[str, str]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode()); h.update(b"\0"); h.update(hashlib.sha256(files[p].encode()).digest())
    return h.hexdigest()


def teacher_identity(host_path: str, cfg: TeacherConfig, files: Mapping[str, str], extra: dict | None = None) -> str:
    """sha256 over everything that fixes a window's teacher targets except WHICH windows are built (format, host
    path, k + retrieval config, files, extra such as sep_eos). Subset builds of one teacher share it, so a
    checkpoint can resume against a teacher built for a different step count (train.py compares per-window target
    digests on the windows both cover)."""
    return hashlib.sha256(json.dumps(dict(format=TEACHER_FORMAT, host=str(host_path), cfg=cfg.retrieval_dict(),
                                          files=files_digest(files), extra=extra or {}),
                                     sort_keys=True).encode()).hexdigest()


def teacher_cache_key(host_path: str, cfg: TeacherConfig, windows: Sequence[Window], window_index: Sequence[int],
                      files: Mapping[str, str], extra: dict | None = None) -> str:
    """sha256 over (format, host path, k, retrieval config, files, the selected windows' ids and labels)."""
    h = hashlib.sha256()
    h.update(json.dumps(dict(format=TEACHER_FORMAT, host=str(host_path), cfg=cfg.retrieval_dict(),
                             files=files_digest(files), extra=extra or {}), sort_keys=True).encode())
    for i in window_index:
        w = windows[i]
        h.update(str(int(i)).encode()); h.update(w.ids.tobytes()); h.update(w.labels.tobytes())
    return h.hexdigest()


def compute_teacher_targets(host: nn.Module, windows: Sequence[Window], docs: Sequence[Document],
                            files: Mapping[str, str], encode: Callable[[str], Sequence[int]],
                            decode: Callable[[np.ndarray], str], cfg: TeacherConfig = TeacherConfig(),
                            window_index: Sequence[int] | None = None, sep_eos: bool = True,
                            allow_hooks: bool = False, meta: dict | None = None,
                            log: Callable[[str], None] | None = None) -> KLTargets:
    """Teacher targets for windows[i], i in window_index (default: all), in that order.

    windows: corpus.make_windows over pack_stream(docs, eos, sep_eos); files: path -> raw text of the files to
    retrieve from (the Split-A train files, keyed like docs' paths); encode/decode: the host tokenizer
    (add_special_tokens=False). The host must be graft-off (no forward pre-hooks) unless allow_hooks=True.
    meta["window_index"] lists the windows covered (RowTrainer aligns by it)."""
    if host._forward_pre_hooks and not allow_hooks:
        raise RuntimeError("host has forward pre-hooks (graft installed?); teacher targets must be graft-off")
    sel = list(range(len(windows))) if window_index is None else [int(i) for i in window_index]
    spans = doc_spans(docs, sep_eos)
    doc_paths = {s.path for s in spans}
    # the own-file exclusion compares paths: `files` must be keyed exactly like the documents and hold nothing else
    # (a differently keyed map would silently retrieve the window's own file; extra keys could be eval files)
    unknown, extra_keys = sorted(doc_paths - set(files)), sorted(set(files) - doc_paths)
    if unknown or extra_keys:
        raise ValueError(f"teacher `files` must be keyed exactly like the training documents: {len(unknown)} document "
                         f"paths not in files (e.g. {unknown[:3]}), {len(extra_keys)} files not among the documents "
                         f"(e.g. {extra_keys[:3]})")
    retr = TeacherRetriever(files, cfg, encode)
    was = host.training
    host.eval()
    out, stats = [], []
    try:
        for n, i in enumerate(sel):
            w = windows[i]
            t0 = time.perf_counter()
            own = window_files(w, spans)
            r = retr.retrieve(decode(w.ids), own)
            t1 = time.perf_counter()
            out.append(teacher_window_targets(host, r.prefix, w, cfg))
            t2 = time.perf_counter()
            stats.append(dict(window=i, window_tokens=int(w.ids.size), retrieval_s=t1 - t0, forward_s=t2 - t1,
                              **r.stats))
            if log is not None:
                log(f"teacher window {n + 1}/{len(sel)} (#{i}): {r.stats['chunks']} chunks from "
                    f"{len(r.stats['files'])} files, {r.stats['tokens']} ctx tokens + {w.ids.size}; "
                    f"retrieval {t1 - t0:.2f}s forward {t2 - t1:.1f}s")
    finally:
        host.train(was)
    tok = [s["tokens"] for s in stats]
    md = dict(meta or {}, format=TEACHER_FORMAT, cfg=cfg.to_dict(), window_index=sel,
              window_sha=[window_digest(windows[i]) for i in sel], retrieval=stats, n_windows=len(sel),
              tokens=int(sum(windows[i].ids.size for i in sel)), ctx_tokens_mean=float(np.mean(tok)) if tok else 0.0,
              chunks=len(retr.chunks), files_indexed=len(files))
    return KLTargets(cfg.k, out, md)


def load_or_build_teacher(cache_dir: str | Path, host: nn.Module, host_path: str, windows: Sequence[Window],
                          docs: Sequence[Document], files: Mapping[str, str], encode: Callable[[str], Sequence[int]],
                          decode: Callable[[np.ndarray], str], cfg: TeacherConfig = TeacherConfig(),
                          window_index: Sequence[int] | None = None, sep_eos: bool = True,
                          log: Callable[[str], None] | None = None,
                          builder: Callable[..., KLTargets] | None = None,
                          forbid: Sequence[str] = ()) -> tuple[KLTargets, Path]:
    """Cached compute_teacher_targets: <cache_dir>/teacher_<key16>.safetensors under an exclusive fcntl lock on
    <file>.lock (concurrent arms: one computes, the rest load it); written to <file>.part<pid> then renamed.
    `forbid`: paths that must not be retrievable (the held-out dev/test files); any of them in `files` is an error."""
    bad = sorted(set(forbid) & set(files))
    if bad:
        raise ValueError(f"teacher retrieval index would contain held-out files: {bad[:5]}")
    sel = list(range(len(windows))) if window_index is None else [int(i) for i in window_index]
    key = teacher_cache_key(host_path, cfg, windows, sel, files, extra=dict(sep_eos=sep_eos))
    ident = teacher_identity(host_path, cfg, files, extra=dict(sep_eos=sep_eos))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"teacher_{key[:16]}.safetensors"
    build = builder or compute_teacher_targets
    with open(str(cp) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if cp.exists():
            tg = KLTargets.load(cp)
            if tg.meta.get("cache_key") != key:
                raise ValueError(f"{cp}: cache key mismatch")
            tg.meta.setdefault("identity", ident)
        else:
            tg = build(host, windows, docs, files, encode, decode, cfg, window_index=sel, sep_eos=sep_eos,
                       meta=dict(cache_key=key, identity=ident, host=str(host_path)), log=log)
            part = Path(str(cp) + f".part{os.getpid()}")
            tg.save(part)
            os.replace(part, cp)
    return tg, cp


def tokenizer_fns(tok) -> tuple[Callable[[str], np.ndarray], Callable[[np.ndarray], str]]:
    """(encode, decode) for corpus.TokenizerJSON (add_special_tokens=False; decode keeps special tokens)."""
    def enc(text: str) -> np.ndarray:
        return tok.encode(text)

    def dec(ids: np.ndarray) -> str:
        return tok.tok.decode([int(t) for t in ids], skip_special_tokens=False)
    return enc, dec


def retrieval_summary(tg: KLTargets) -> dict:
    """Aggregate of meta['retrieval'] for train.json."""
    st = tg.meta.get("retrieval", [])
    if not st:
        return {}
    tok = np.array([s["tokens"] for s in st], np.float64)
    nf = np.array([len(s["files"]) for s in st], np.float64)
    fw = np.array([s["forward_s"] for s in st], np.float64)
    return dict(windows=len(st), ctx_tokens_mean=float(tok.mean()), ctx_tokens_min=int(tok.min()),
                files_per_window_mean=float(nf.mean()), windows_without_context=int((tok == 0).sum()),
                forward_s_mean=float(fw.mean()), forward_s_total=float(fw.sum()),
                distinct_files_retrieved=len({f for s in st for f in s["files"]}))


__all__ = ["TEACHER_FORMAT", "TeacherConfig", "DocSpan", "doc_spans", "window_files", "TeacherRetriever",
           "Retrieved", "teacher_hidden", "teacher_window_targets", "teacher_identity", "teacher_cache_key", "files_digest",
           "compute_teacher_targets", "load_or_build_teacher", "tokenizer_fns", "retrieval_summary"]
