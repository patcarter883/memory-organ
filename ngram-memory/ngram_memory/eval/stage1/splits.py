"""Stage-1 splits (§6.A.1) for the eval side: a thin reader of the ingest pipeline's frozen split files
(`ngram_memory/ingest/corpus.py` writes `ingest_splits/split_a.json` and `split_b.json`; `Splits.load` accepts that
directory or either file), converted to the eval-side dict below. `build_splits` re-derives the same rule
without the ingest side's MinHash cluster moves (kept for tiny test repos and for repos the ingest has not seen).

The ingest files are canonical: on minisgl at df29117 both give identical dev (15) and test (29) lists; the ingest
side additionally moves a near-duplicate MinHash cluster (models/llama.py, mistral.py, ...) wholly to train where
`build_splits` would list llama.py / mistral.py as near_dup.

Split JSON (`ngram-memory/splits/v1`)::

    {
      "format": "ngram-memory/splits/v1",
      "repo": "/abs/path/of/git/repo",         # read-only; texts are git blobs at `commit`, never the working tree
      "commit": "<sha at which split_a and the HEAD side of split_b are read>",
      "subdir": "python",                      # paths below are relative to this (the document header uses them)
      "doc_header": "# file: {path}\\n",       # §6.A.1 document format; files are separated by EOS in streams
      "split_a": {"train": [...], "dev": [...], "test": [...],
                  "near_dup": [...],           # held-out files with > 50% verbatim lines in train (diagnostic only)
                  "rule": "..."},
      "split_b": {"t_cut": "<sha>", "train": [...],              # files present at t_cut (read at t_cut)
                  "test": {"<path>": [1-based HEAD line numbers added after t_cut], ...},
                  "new_files": [...]}                            # test files that did not exist at t_cut
    }

Split A (by file, §6.A.1): h = int(sha256(repo_path)[:8], 16) / 2**32 with repo_path the git path (e.g.
"python/minisgl/ple/state.py", as `S1/evalproto/evalproto.py` hashed it); test if h < 0.15 and the blob is > 400
bytes; dev if 0.15 <= h < 0.25 (and > 400 bytes); everything else is train. Held-out files whose lines (stripped,
>= 20 chars) are > 50% verbatim in train move from dev/test to `near_dup`.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

FORMAT = "ngram-memory/splits/v1"
DOC_HEADER = "# file: {path}\n"
T_CUT_MINISGL = "a4a55431aeeec6c39079dcd943dfc2d6129beb01"  # last commit before 2026-10-28 (S1/evalproto)


def path_hash(repo_path: str) -> float:
    return int(hashlib.sha256(repo_path.encode()).hexdigest()[:8], 16) / 2 ** 32


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def git_ls_py(repo: str, commit: str, subdir: str) -> list[str]:
    out = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", subdir)
    return sorted(p for p in out.split("\n") if p.endswith(".py"))


def git_blob_text(repo: str, commit: str, repo_path: str) -> str:
    r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{repo_path}"], capture_output=True, check=True)
    return r.stdout.decode("utf-8", errors="replace")


def rev_parse(repo: str, rev: str) -> str:
    return _git(repo, "rev-parse", rev).strip()


def _rel(repo_path: str, subdir: str) -> str:
    pre = subdir.rstrip("/") + "/"
    return repo_path[len(pre):] if subdir and repo_path.startswith(pre) else repo_path


def verbatim_line_overlap(text: str, train_lines: set[str], min_len: int = 20) -> float:
    ls = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len]
    return sum(s in train_lines for s in ls) / max(1, len(ls))


def split_a_assign(repo_paths: Iterable[str], sizes: dict[str, int], test_frac: float = 0.15,
                   dev_frac: float = 0.10, min_bytes: int = 400) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for p in sorted(repo_paths):
        h = path_hash(p)
        big = sizes[p] > min_bytes
        if big and h < test_frac:
            out["test"].append(p)
        elif big and h < test_frac + dev_frac:
            out["dev"].append(p)
        else:
            out["train"].append(p)
    return out


def added_lines(repo: str, t_cut: str, head: str, subdir: str) -> dict[str, list[int]]:
    """HEAD line numbers (1-based) added in (t_cut, head] per .py path (`git diff -U0`, as S1/evalproto)."""
    diff = _git(repo, "diff", "-U0", t_cut, head, "--", subdir)
    added: dict[str, set[int]] = defaultdict(set)
    cur = None
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            cur = ln[6:] if ln.startswith("+++ b/") else None
        elif ln.startswith("@@") and cur:
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            a, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            added[cur].update(range(a, a + n))
    return {k: sorted(v) for k, v in sorted(added.items()) if k.endswith(".py") and v}


def build_splits(repo: str, commit: str = "HEAD", subdir: str = "python", t_cut: str | None = T_CUT_MINISGL,
                 test_frac: float = 0.15, dev_frac: float = 0.10, min_bytes: int = 400,
                 dup_threshold: float = 0.5) -> dict:
    commit = rev_parse(repo, commit)
    paths = git_ls_py(repo, commit, subdir)
    texts = {p: git_blob_text(repo, commit, p) for p in paths}
    sizes = {p: len(texts[p].encode()) for p in paths}
    a = split_a_assign(paths, sizes, test_frac, dev_frac, min_bytes)
    train_lines = {ln.strip() for p in a["train"] for ln in texts[p].splitlines() if len(ln.strip()) >= 20}
    near = [p for p in a["dev"] + a["test"] if verbatim_line_overlap(texts[p], train_lines) > dup_threshold]
    rel = lambda ps: [_rel(p, subdir) for p in ps]  # noqa: E731
    sa = dict(train=rel(a["train"]), dev=rel([p for p in a["dev"] if p not in near]),
              test=rel([p for p in a["test"] if p not in near]), near_dup=rel(sorted(near)),
              rule=f"sha256(git path)[:8]/2^32: test < {test_frac}, dev < {test_frac + dev_frac}, size > {min_bytes} B; "
                   f"held-out files with > {dup_threshold:.0%} verbatim lines (>= 20 chars) in train -> near_dup")
    out = dict(format=FORMAT, repo=str(repo), commit=commit, subdir=subdir, doc_header=DOC_HEADER, split_a=sa)
    if t_cut:
        t_cut = rev_parse(repo, t_cut)
        cut_paths = git_ls_py(repo, t_cut, subdir)
        add = added_lines(repo, t_cut, commit, subdir)
        cut_set = set(cut_paths)
        out["split_b"] = dict(t_cut=t_cut, train=rel(cut_paths), test={_rel(k, subdir): v for k, v in add.items()},
                              new_files=rel([k for k in add if k not in cut_set]))
    return out


@dataclass
class Splits:
    """A loaded split JSON with blob access."""
    data: dict
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path, repo: str | None = None) -> "Splits":
        """A `ngram-memory/splits/v1` JSON, or the ingest side's split directory / split_a.json / split_b.json."""
        p = Path(path)
        if p.is_dir() or p.name in ("split_a.json", "split_b.json"):
            d0 = p if p.is_dir() else p.parent
            return cls.from_ingest(d0 / "split_a.json", d0 / "split_b.json", repo=repo)
        d = json.loads(p.read_text())
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file: {d.get('format')}")
        return cls(d)

    @classmethod
    def from_ingest(cls, split_a: str | Path, split_b: str | Path | None = None, repo: str | None = None,
                    subdir: str = "python") -> "Splits":
        """Convert ingest/corpus.py SplitA / SplitB JSON (repo paths, added-line ranges) to the eval-side dict."""
        a = json.loads(Path(split_a).read_text())
        if repo is None:
            from ...ingest.corpus import MINISGL_REPO  # the only repo the ingest constants name
            repo = str(MINISGL_REPO)
        rel = lambda ps: sorted(_rel(x, subdir) for x in ps)  # noqa: E731
        out = dict(format=FORMAT, repo=str(repo), commit=a["commit"], subdir=subdir, doc_header=DOC_HEADER,
                   source=dict(split_a=str(split_a), split_b=str(split_b) if split_b else None),
                   split_a=dict(train=rel(a["train"]), dev=rel(a["dev"]), test=rel(a["test"]),
                                near_dup=rel(a.get("neardup_diag", [])), rule=a.get("rule"),
                                cluster_moves=a.get("cluster_moves", [])))
        if split_b and Path(split_b).exists():
            b = json.loads(Path(split_b).read_text())
            if b["head"] != a["commit"]:
                raise ValueError("split_b head != split_a commit")
            out["split_b"] = dict(t_cut=b["t_cut"], train=rel(b["train_files"]),
                                  test={_rel(k, subdir): [i for lo, hi in v for i in range(lo, hi + 1)]
                                        for k, v in sorted(b["added_lines"].items())},
                                  new_files=rel(b["new_files"]))
        return cls(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.data, indent=1))

    @property
    def repo(self) -> str:
        return self.data["repo"]

    @property
    def commit(self) -> str:
        return self.data["commit"]

    @property
    def subdir(self) -> str:
        return self.data.get("subdir", "")

    def files(self, split: str, part: str) -> list[str]:
        """split in {"a", "b"}; part in {"train", "dev", "test", "near_dup", "new_files"}."""
        s = self.data["split_" + split]
        v = s[part]
        return sorted(v) if isinstance(v, dict) else list(v)

    def added_lines(self, path: str) -> list[int]:
        return list(self.data["split_b"]["test"].get(path, []))

    def text(self, path: str, at: str | None = None) -> str:
        commit = at or self.commit
        k = (commit, path)
        if k not in self._cache:
            rp = f"{self.subdir.rstrip('/')}/{path}" if self.subdir else path
            self._cache[k] = git_blob_text(self.repo, commit, rp)
        return self._cache[k]

    def document(self, path: str, at: str | None = None) -> str:
        """§6.A.1 document text: header + source."""
        return self.data.get("doc_header", DOC_HEADER).format(path=path) + self.text(path, at)

    def summary(self) -> dict:
        a = self.data["split_a"]
        out = {k: len(a[k]) for k in ("train", "dev", "test", "near_dup")}
        if "split_b" in self.data:
            b = self.data["split_b"]
            out.update(b_train=len(b["train"]), b_test_files=len(b["test"]),
                       b_added_lines=sum(len(v) for v in b["test"].values()), b_new_files=len(b["new_files"]))
        return out


def line_of_char(text: str) -> list[int]:
    """1-based line number of every character."""
    out, ln = [], 1
    for ch in text:
        out.append(ln)
        if ch == "\n":
            ln += 1
    return out


def added_line_target_mask(doc_text: str, offsets: Sequence[tuple[int, int]], lines: Iterable[int],
                           header_lines: int = 1) -> list[bool]:
    """Split B: True for tokens whose first character lies on an added HEAD line (line numbers are source lines;
    the document header shifts them by `header_lines`)."""
    want = {int(x) + header_lines for x in lines}
    loc = line_of_char(doc_text)
    out = []
    for a, b in offsets:
        out.append(b > a and a < len(loc) and loc[a] in want)
    return out


__all__ = ["FORMAT", "DOC_HEADER", "T_CUT_MINISGL", "Splits", "build_splits", "split_a_assign", "path_hash",
           "added_lines", "added_line_target_mask", "git_blob_text", "git_ls_py", "verbatim_line_overlap"]
