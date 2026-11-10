"""Evaluation sets for Stage 1 (§6.A.1-§6.A.3) as lists of `scoring.Doc`.

* project: Split A train / dev / test files (§6.A.1 document format), Split B HEAD files with an added-line mask.
* R1: WikiText-2 raw test (HF datasets cache), one doc per article (`= Title =` headings).
* R2: the stdlib module list of S1/experiment/ingest.py (python 3.13 stdlib), plus R2-ML: an unrelated ML codebase
  (transformers/models/<name>/modeling_<name>.py for a fixed list of non-Qwen models).
* training windows: train files streamed in the §6.A.1 format (EOS between files), cut into T-token windows; the
  same windows a cartridge arm would use when no ingest pipeline output is given.
"""
from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Sequence

import numpy as np

from .scoring import Doc, encode_doc
from .splits import Splits, added_line_target_mask

STDLIB_R2 = ("argparse.py", "dataclasses.py", "functools.py", "typing.py", "json/decoder.py", "json/encoder.py",
             "asyncio/tasks.py", "asyncio/queues.py", "collections/__init__.py", "logging/__init__.py",
             "subprocess.py", "threading.py", "inspect.py", "enum.py", "contextlib.py", "shutil.py")
R2_ML_MODELS = ("bert", "gpt2", "t5", "llama", "mistral", "whisper", "vit", "bart")


def stdlib_dir() -> str:
    for d in ("/usr/local/lib/python3.13", sysconfig.get_paths()["stdlib"]):
        if Path(d, "argparse.py").exists():
            return d
    raise FileNotFoundError("no stdlib directory")


def project_docs(splits: Splits, tok, split: str, part: str, max_files: int | None = None,
                 skip_files: int = 0) -> list[Doc]:
    paths = splits.files(split, part)[skip_files:]
    if max_files is not None:
        paths = paths[:max_files]
    return [encode_doc(tok, p, splits.document(p)) for p in paths]


def split_b_docs(splits: Splits, tok, max_files: int | None = None) -> list[Doc]:
    out = []
    for p in splits.files("b", "test")[: max_files]:
        d = encode_doc(tok, p, splits.document(p))
        d.target_mask = np.asarray(added_line_target_mask(d.text, d.offsets, splits.added_lines(p)), bool)
        out.append(d)
    return out


def stdlib_docs(tok, files: Sequence[str] = STDLIB_R2, root: str | None = None) -> list[Doc]:
    root = root or stdlib_dir()
    return [encode_doc(tok, f"stdlib/{f}", Path(root, f).read_text(errors="replace")) for f in files if Path(root, f).exists()]


def ml_codebase_docs(tok, models: Sequence[str] = R2_ML_MODELS) -> list[Doc]:
    import transformers
    base = Path(os.path.dirname(transformers.__file__)) / "models"
    out = []
    for m in models:
        p = base / m / f"modeling_{m}.py"
        if p.exists():
            out.append(encode_doc(tok, f"transformers/models/{m}/{p.name}", p.read_text(errors="replace")))
    return out


def wikitext_docs(tok, max_docs: int | None = None, split: str = "test") -> list[Doc]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    arts, cur, title = [], [], None
    for line in ds["text"]:
        s = line.strip()
        if s.startswith("= ") and s.endswith(" =") and not s.startswith("= ="):
            if cur:
                arts.append((title, "".join(cur)))
            cur, title = [line], s.strip("= ").strip()
        else:
            cur.append(line)
    if cur:
        arts.append((title, "".join(cur)))
    out = []
    for i, (t, txt) in enumerate(arts[: max_docs]):
        out.append(encode_doc(tok, f"wikitext/{i:03d}:{t}", txt))
    return out


def take_tokens(docs: Sequence[Doc], n_tokens: int, min_context: int = 0) -> list[Doc]:
    """Docs holding n_tokens SCORED tokens in total (each doc contributes len - min_context targets); the last doc
    is truncated so the total is exact."""
    out, n = [], 0
    for d in docs:
        if n >= n_tokens:
            break
        scored = max(0, len(d) - max(1, min_context))
        if scored == 0:
            continue
        if n + scored > n_tokens:
            k = max(1, min_context) + (n_tokens - n)
            d = Doc(d.name, d.ids[:k], d.text, None if d.offsets is None else d.offsets[:k], d.group)
            scored = n_tokens - n
        out.append(d)
        n += scored
    return out


def train_stream(splits: Splits, tok, eos: int, part: str = "train", split: str = "a", at: str | None = None) -> np.ndarray:
    """Concatenated train documents; `at` reads the blobs at another commit (Split B: the tree at T_cut)."""
    ids: list[int] = []
    for p in splits.files(split, part):
        ids += tok(splits.document(p, at=at), add_special_tokens=False)["input_ids"] + [eos]
    return np.asarray(ids, np.int64)


def train_windows(stream: np.ndarray, T: int = 512, n: int | None = None, seed: int = 0) -> np.ndarray:
    """[n, T] windows cut from the stream (deterministic sample of whole chunks)."""
    k = len(stream) // T
    ch = stream[: k * T].reshape(k, T)
    if n is None or n >= k:
        return ch
    return ch[np.sort(np.random.default_rng(seed).choice(k, size=n, replace=False))]


__all__ = ["STDLIB_R2", "R2_ML_MODELS", "project_docs", "split_b_docs", "stdlib_docs", "ml_codebase_docs",
           "wikitext_docs", "take_tokens", "train_stream", "train_windows", "stdlib_dir"]
