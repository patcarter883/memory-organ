"""Token classes for the §6.A.2 breakdowns (numpy + stdlib only).

Lexical (Python `tokenize`, docstrings from `ast`): string, docstring, comment, keyword, number, op, ws, identifier.
A sub-token is an identifier token if it overlaps an identifier's characters (S1/evalproto2/tokdelta.py rule);
it is the FIRST sub-token if an identifier starts inside its character span, else a CONTINUATION.
Identifiers: project-specific (absent from a reference vocabulary: stdlib + torch + numpy + transformers),
builtin, or general; first occurrence in the file vs repeat; cross-file (defined in another project file, not in
this one).
Key category (S1/experiment/ingest.py `categories`), for target j predicted at input position t = j - 1:
0 unseen_2gram: (x_{t-1}, x_t) never in train; 1 seen_2gram_only; 2 seen_3gram_new_next: (x_{t-2}, x_{t-1}, x_t)
seen but never followed by x_j; 3 seen_3gram_seen_next. Tokens before a document start are EOS (fresh segment).
"""
from __future__ import annotations

import ast
import builtins
import io
import json
import keyword
import os
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

LEX = ("other", "ws", "op", "number", "keyword", "comment", "string", "docstring", "identifier")
LEX_ID = {k: i for i, k in enumerate(LEX)}
ID_KIND = ("none", "proj", "general", "builtin")
CAT_NAMES = ("unseen_2gram", "seen_2gram_only", "seen_3gram_new_next", "seen_3gram_seen_next")
BUILTINS = frozenset(dir(builtins))
IDRE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STR_TYPES = {tokenize.STRING} | {getattr(tokenize, k) for k in dir(tokenize) if k.startswith(("FSTRING_", "TSTRING_"))}


# ------------------------------------------------------------------------------------------ reference vocabulary
def reference_vocab(dirs: Iterable[str | os.PathLike], cache: str | os.PathLike | None = None) -> Counter:
    """Identifier -> number of .py files (under `dirs`) that contain it. Cached as JSON if `cache` is given."""
    if cache and Path(cache).exists():
        return Counter(json.loads(Path(cache).read_text()))
    ref: Counter = Counter()
    for d in dirs:
        for r, _, fs in os.walk(d):
            for fn in fs:
                if fn.endswith(".py"):
                    try:
                        ref.update(set(IDRE.findall(Path(r, fn).read_text(encoding="utf-8", errors="ignore"))))
                    except OSError:
                        pass
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(ref))
    return ref


def default_reference_vocab(cache: str | os.PathLike | None = None) -> Counter:
    """The reference vocabulary over `default_reference_dirs()`. Those dirs include the interpreter's stdlib tree
    (with its site-packages), which differs between machines, so NM_REF_VOCAB can name the frozen copy made in the
    dev container (runner/inputs/ref_vocab.json.gz, `scripts/runner_inputs.py freeze`); it is then also written to
    `cache`. Unset, this is exactly `reference_vocab(default_reference_dirs(), cache)`."""
    frozen = os.environ.get("NM_REF_VOCAB")
    if frozen and not (cache and Path(cache).exists()):
        import gzip
        ref = Counter(json.loads(gzip.decompress(Path(frozen).read_bytes()).decode("utf-8")))
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps(ref))
        return ref
    return reference_vocab(default_reference_dirs(), cache)


def default_reference_dirs() -> list[str]:
    """stdlib + torch + numpy + transformers of the running interpreter (as tokdelta.py used)."""
    import sysconfig
    out = [sysconfig.get_paths()["stdlib"]]
    for mod in ("torch", "numpy", "transformers"):
        try:
            m = __import__(mod)
            out.append(os.path.dirname(m.__file__))
        except ImportError:
            pass
    return out


def project_definitions(texts: Mapping[str, str]) -> dict[str, set[str]]:
    """name -> files (keys of `texts`) defining it (def / class / top-level assignment)."""
    defs: dict[str, set[str]] = defaultdict(set)
    for f, s in texts.items():
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[node.name].add(f)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for tg in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(tg, ast.Name):
                        defs[tg.id].add(f)
    return dict(defs)


# --------------------------------------------------------------------------------------------- character labels
def _docstring_starts(src: str) -> set[tuple[int, int]]:
    """(line, char col) of every docstring literal (module / class / function first statement)."""
    out: set[tuple[int, int]] = set()
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return out
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            st = node.body[0]
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                ln = st.value.lineno
                b = lines[ln - 1].encode("utf-8")[: st.value.col_offset] if ln - 1 < len(lines) else b""
                out.add((ln, len(b.decode("utf-8", errors="replace"))))
    return out


def char_labels(src: str, ref_vocab: Mapping[str, int] | None = None, defs: Mapping[str, set[str]] | None = None,
                fname: str | None = None):
    """Per-character lexical label ids and identifier spans [(start, end, name, kind, first_in_file, xfile)]."""
    lab = np.full(len(src), LEX_ID["ws"], np.int8)
    off = [0]
    for ln in src.splitlines(keepends=True):
        off.append(off[-1] + len(ln))
    doc0 = _docstring_starts(src)
    spans = []
    seen: set[str] = set()
    ref = ref_vocab or {}
    try:
        for tk in tokenize.generate_tokens(io.StringIO(src).readline):
            (r0, c0), (r1, c1) = tk.start, tk.end
            if r0 - 1 >= len(off) or r1 - 1 >= len(off):
                continue
            a, b = off[r0 - 1] + c0, min(off[r1 - 1] + c1, len(src))
            if tk.type == tokenize.NAME:
                n = tk.string
                if keyword.iskeyword(n) or keyword.issoftkeyword(n):
                    lab[a:b] = LEX_ID["keyword"]
                    continue
                kind = "builtin" if n in BUILTINS else ("proj" if ref.get(n, 0) == 0 else "general")
                xf = bool(defs) and n in defs and fname is not None and fname not in defs[n]
                spans.append((a, b, n, kind, n not in seen, xf))
                seen.add(n)
                lab[a:b] = LEX_ID["identifier"]
            elif tk.type in _STR_TYPES:
                lab[a:b] = LEX_ID["docstring" if (r0, c0) in doc0 else "string"]
            elif tk.type == tokenize.COMMENT:
                lab[a:b] = LEX_ID["comment"]
            elif tk.type == tokenize.NUMBER:
                lab[a:b] = LEX_ID["number"]
            elif tk.type == tokenize.OP:
                lab[a:b] = LEX_ID["op"]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return lab, spans


def token_classes(text: str, offsets: Sequence[tuple[int, int]], ref_vocab: Mapping[str, int] | None = None,
                  defs: Mapping[str, set[str]] | None = None, fname: str | None = None,
                  code_start: int = 0) -> dict[str, np.ndarray]:
    """Per-token arrays for a document. `code_start` = char offset where Python source starts (after a header,
    which is lexed as part of the source: `# file: ...` is a comment).

    Returns lex (int8 LEX id), id_kind (int8 ID_KIND id), id_first_sub (bool), id_cont_sub (bool),
    id_first_in_file (bool; the identifier's first occurrence in the file), id_xfile (bool)."""
    src = text[code_start:]
    lab, spans = char_labels(src, ref_vocab, defs, fname)
    n = len(offsets)
    lex = np.full(n, LEX_ID["other"], np.int8)
    kind = np.zeros(n, np.int8)
    first_sub = np.zeros(n, bool)
    cont_sub = np.zeros(n, bool)
    first_occ = np.zeros(n, bool)
    xfile = np.zeros(n, bool)
    # char -> identifier span index
    span_of = np.full(len(src), -1, np.int32)
    starts = np.zeros(len(src), bool)
    for si, (a, b, *_rest) in enumerate(spans):
        span_of[a:b] = si
        if a < len(src):
            starts[a] = True
    for i, (a, b) in enumerate(offsets):
        a, b = a - code_start, b - code_start
        if b <= a or a < 0 or a >= len(src):
            continue
        b = min(b, len(src))
        sp = span_of[a:b]
        hit = sp[sp >= 0]
        if hit.size:
            s = spans[int(hit[0])]
            lex[i] = LEX_ID["identifier"]
            kind[i] = ID_KIND.index(s[3])
            is_first = bool(starts[a:b].any())
            first_sub[i] = is_first
            cont_sub[i] = not is_first
            first_occ[i] = s[4]
            xfile[i] = s[5]
        else:
            seg = lab[a:b]
            nz = seg[seg != LEX_ID["ws"]]
            lex[i] = np.bincount(nz if nz.size else seg, minlength=len(LEX)).argmax()
    return dict(lex=lex, id_kind=kind, id_first_sub=first_sub, id_cont_sub=cont_sub, id_first_in_file=first_occ,
                id_xfile=xfile)


# ----------------------------------------------------------------------------------------------- n-gram classes
def _hist(ids: np.ndarray, k: int, eos: int) -> np.ndarray:
    """x_{t-k} for every t, eos before the start (fresh segment; EOS inside a doc also resets, as the hasher)."""
    ids = np.asarray(ids, np.int64)
    if k == 0:
        return ids
    out = np.full_like(ids, eos)
    out[k:] = ids[:-k]
    # reset after an EOS inside the sequence
    pos = np.arange(len(ids))
    last_eos = np.maximum.accumulate(np.where(ids == eos, pos, -1))
    prev = np.concatenate([[-1], last_eos[:-1]])
    out[(pos - (prev + 1)) < k] = eos
    return out


def _void_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = np.ascontiguousarray(np.stack([np.asarray(a, np.int64), np.asarray(b, np.int64)], 1))
    return ab.view(np.dtype((np.void, 16))).ravel()


def _member(sorted_keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    if sorted_keys.size == 0:
        return np.zeros(q.shape, bool)
    i = np.searchsorted(sorted_keys, q).clip(0, sorted_keys.size - 1)
    return sorted_keys[i] == q


class NGramSeen:
    """Train-set 2-grams, 3-grams and (3-gram, next) pairs, keyed as in S1/experiment/ingest.py (exact keys)."""

    def __init__(self, train_seqs: Iterable[np.ndarray], vocab: int, eos: int):
        self.V, self.eos = int(vocab), int(eos)
        if self.V ** 3 >= 2 ** 63:
            raise ValueError("vocab too large for exact int64 3-gram keys")
        k2s, k3s, k4s = [], [], []
        for s in train_seqs:
            s = np.asarray(s, np.int64)
            if len(s) == 0:
                continue
            k2, k3 = self._keys(s)
            k2s.append(k2)
            k3s.append(k3)
            k4s.append(_void_pairs(k3[:-1], s[1:]))
        self.k2 = np.unique(np.concatenate(k2s)) if k2s else np.zeros(0, np.int64)
        self.k3 = np.unique(np.concatenate(k3s)) if k3s else np.zeros(0, np.int64)
        self.k4 = np.unique(np.concatenate(k4s)) if k4s else np.zeros(0, np.dtype((np.void, 16)))

    def _keys(self, s) -> tuple[np.ndarray, np.ndarray]:
        x0, x1, x2 = (_hist(s, k, self.eos) for k in range(3))
        return x1 * self.V + x0, (x2 * self.V + x1) * self.V + x0

    def categories(self, ids: np.ndarray, targets: np.ndarray | None = None) -> np.ndarray:
        """Key category (0..3) for each target position j (predicted from position j - 1). With targets=None,
        an array over all positions (index j = target j; index 0 = -1)."""
        ids = np.asarray(ids, np.int64)
        full = targets is None
        if full:
            targets = np.arange(1, len(ids))
        k2, k3 = self._keys(ids)
        t = np.asarray(targets, np.int64) - 1
        s2 = _member(self.k2, k2[t])
        s3 = _member(self.k3, k3[t])
        s4 = _member(self.k4, _void_pairs(k3[t], ids[t + 1]))
        c = np.zeros(len(t), np.int8)
        c[s2] = 1
        c[s3] = 2
        c[s3 & s4] = 3
        if full:
            return np.concatenate([np.array([-1], np.int8), c])
        return c


def _ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """64-bit polynomial hash of the n-gram ending at every position j >= n-1 (positions < n-1 -> 0)."""
    x = np.asarray(ids, np.uint64) + np.uint64(1)
    h = np.zeros(len(x), np.uint64)
    B = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for k in range(n):
            sh = np.zeros(len(x), np.uint64)
            sh[k:] = x[: len(x) - k] if k else x
            h = h * B + sh
    h[: n - 1] = 0
    return h


class NGramIndex:
    """Membership of length-n token n-grams in a training set (dedupe audit, §6.A.1: the share of held-out targets
    whose ENDING n-gram, x[j-n+1..j], appears in train). 64-bit hashes; collisions are negligible at corpus scale."""

    def __init__(self, train_seqs: Iterable[np.ndarray], n: int = 16):
        self.n = n
        hs = [_ngram_hashes(s, n)[n - 1:] for s in train_seqs if len(s) >= n]
        self.h = np.unique(np.concatenate(hs)) if hs else np.zeros(0, np.uint64)

    def in_train(self, ids: np.ndarray) -> np.ndarray:
        """bool per position j (False for j < n-1)."""
        h = _ngram_hashes(ids, self.n)
        out = _member(self.h, h)
        out[: self.n - 1] = False
        return out


def class_masks(per_target: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Named boolean masks over targets from gathered per-target class arrays (keys as token_classes + 'cat')."""
    lex, kind = per_target["lex"], per_target["id_kind"]
    isid = lex == LEX_ID["identifier"]
    m: dict[str, np.ndarray] = {}
    for name in ("string", "docstring", "comment", "keyword", "number", "op", "ws", "other"):
        m[f"lex:{name}"] = lex == LEX_ID[name]
    m["lex:identifier"] = isid
    m["lex:string+docstring"] = (lex == LEX_ID["string"]) | (lex == LEX_ID["docstring"])
    m["id:first_sub"] = per_target["id_first_sub"]
    m["id:cont_sub"] = per_target["id_cont_sub"]
    for k in ("proj", "general", "builtin"):
        km = kind == ID_KIND.index(k)
        m[f"id:{k}"] = km
        m[f"id:{k}:first_sub"] = km & per_target["id_first_sub"]
        m[f"id:{k}:cont_sub"] = km & per_target["id_cont_sub"]
    m["id:proj:first_in_file:first_sub"] = (kind == 1) & per_target["id_first_in_file"] & per_target["id_first_sub"]
    m["id:proj:repeat:first_sub"] = (kind == 1) & ~per_target["id_first_in_file"] & per_target["id_first_sub"]
    m["id:xfile:first_sub"] = per_target["id_xfile"] & per_target["id_first_sub"]
    m["id:xfile:first_in_file:first_sub"] = per_target["id_xfile"] & per_target["id_first_in_file"] & per_target["id_first_sub"]
    m["non_identifier"] = ~isid
    if "cat" in per_target:
        c = per_target["cat"]
        for i, n in enumerate(CAT_NAMES):
            m[f"key:{n}"] = c == i
        m["key:seen_2gram_any"] = c > 0
    if "dup16" in per_target:
        m["dup:16gram_in_train"] = per_target["dup16"]
        m["dup:16gram_not_in_train"] = ~per_target["dup16"]
    return m


def gather_targets(per_doc: Sequence[Mapping[str, np.ndarray]], doc: np.ndarray, pos: np.ndarray) -> dict[str, np.ndarray]:
    """Per-target class arrays for a ScoreResult's (doc, pos)."""
    keys = per_doc[0].keys() if per_doc else []
    out = {}
    for k in keys:
        dt = per_doc[0][k].dtype
        arr = np.zeros(len(doc), dt)
        for i, cls in enumerate(per_doc):
            m = doc == i
            if m.any():
                arr[m] = cls[k][pos[m]]
        out[k] = arr
    return out


__all__ = ["LEX", "LEX_ID", "ID_KIND", "CAT_NAMES", "reference_vocab", "default_reference_vocab", "default_reference_dirs",
           "project_definitions", "char_labels", "token_classes", "NGramSeen", "NGramIndex", "class_masks", "gather_targets"]
