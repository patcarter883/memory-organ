"""Corpus for Stage-1 ingestion (§I.4, §6.A.1): git blobs at a pinned commit, filters, secret scan, dedup,
Splits A/B, framings, and 4,096-token windows with true n-gram history.

Numpy-only (no torch). The working tree is never read: every file comes from `git ls-tree` / `git cat-file --batch`
at a resolved commit, and Split B reads `git diff` between two commits.

Conventions reproduced from the proof scripts:
* file list: `git ls-tree -r <commit> python` filtered to `.py` (S1/ingest-algo/scripts/count_rows.py), text decoded
  as UTF-8 with errors="replace";
* Split A (§6.A.1, S1/evalproto/evalproto.py): h = int(sha256(repo path)[:8], 16) / 2**32 on the repo-relative path
  ("python/minisgl/..."); test if h < 0.15 and len(text) > 400 characters, dev if 0.15 <= h < 0.25 (same size rule),
  else train; the lists are frozen in ingest_splits/split_a.json once written;
* Split B (§6.A.1): train on the tree at T_cut, test on lines added in (T_cut, head] (`git diff -U0 -M`);
* framings: "raw" (blob text; the census), "bare" (`# file: <path relative to python/>\n` + text, the §6.A.1 document
  format), "chat" and "catn" (S1/ingest-algo/scripts/extras.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------- constants
from .. import paths as _paths  # noqa: E402
MINISGL_REPO = _paths.minisgl_repo()   # NM_MINISGL_REPO; default /home/user/patcarter883/minisglang-rdna4
MINISGL_HEAD = "df29117e95968f5f39811d39e12fc1d4ac5cbc7a"       # df29117, 2026-11-06
MINISGL_T_CUT = "a4a55431aeeec6c39079dcd943dfc2d6129beb01"      # a4a5543, last commit before 2026-10-28
SPLITS_DIR = Path(__file__).resolve().parents[2] / "ingest_splits"


# ------------------------------------------------------------------------------------------------------ git
@dataclass(frozen=True)
class TreeEntry:
    path: str
    blob: str
    size: int
    mode: str


class GitRepo:
    """Read-only git plumbing over a repository (never touches its working tree or index)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def git(self, *args: str, text: bool = True, input: bytes | None = None) -> str | bytes:
        r = subprocess.run(["git", "-C", str(self.path), *args], check=True, capture_output=True, input=input)
        return r.stdout.decode("utf-8", "replace") if text else r.stdout

    def resolve(self, rev: str) -> str:
        return str(self.git("rev-parse", "--verify", f"{rev}^{{commit}}")).strip()

    def commit_time(self, rev: str) -> int:
        return int(str(self.git("log", "-1", "--format=%ct", rev)).strip())

    def ls_tree(self, commit: str, prefix: str | None = None) -> list[TreeEntry]:
        args = ["ls-tree", "-r", "-l", "-z", commit]
        if prefix:
            args += ["--", prefix]
        out = self.git(*args, text=False)
        ents = []
        for rec in bytes(out).split(b"\0"):
            if not rec:
                continue
            meta, path = rec.split(b"\t", 1)
            mode, typ, sha, size = meta.split()
            if typ != b"blob":
                continue
            ents.append(TreeEntry(path.decode("utf-8", "replace"), sha.decode(), int(size) if size != b"-" else 0,
                                  mode.decode()))
        return ents

    def read_blobs(self, shas: Iterable[str]) -> dict[str, bytes]:
        """Contents of many blobs through one `git cat-file --batch` process."""
        uniq = list(dict.fromkeys(shas))
        if not uniq:
            return {}
        out = bytes(self.git("cat-file", "--batch", text=False, input=("\n".join(uniq) + "\n").encode()))
        res: dict[str, bytes] = {}
        pos = 0
        for sha in uniq:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].split()
            if len(header) < 3 or header[1] == b"missing":
                raise KeyError(f"blob {sha} missing")
            n = int(header[2])
            res[sha] = out[nl + 1: nl + 1 + n]
            pos = nl + 1 + n + 1
        return res

    def rev_list(self, a: str, b: str, first_parent: bool = False) -> list[str]:
        """Commits in (a, b], oldest first."""
        args = ["rev-list", "--reverse"] + (["--first-parent"] if first_parent else []) + [f"{a}..{b}"]
        return str(self.git(*args)).split()

    def added_lines(self, a: str, b: str, prefix: str) -> dict[str, set[int]]:
        """1-based line numbers (in the file at b) added between a and b, from `git diff -U0 -M a b -- prefix`."""
        diff = str(self.git("diff", "-U0", "-M", "--no-color", "--no-ext-diff", a, b, "--", prefix))
        added: dict[str, set[int]] = defaultdict(set)
        cur: str | None = None
        for ln in diff.splitlines():
            if ln.startswith("+++ "):
                cur = ln[6:] if ln.startswith("+++ b/") else None
            elif ln.startswith("@@") and cur is not None:
                m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
                assert m is not None
                s, n = int(m.group(1)), int(m.group(2) or 1)
                added[cur].update(range(s, s + n))
        return {k: v for k, v in added.items() if v}

    def name_status(self, a: str, b: str, prefix: str, diff_filter: str | None = None) -> list[tuple[str, str]]:
        args = ["diff", "--name-status", "--no-renames"] + ([f"--diff-filter={diff_filter}"] if diff_filter else [])
        out = str(self.git(*args, a, b, "--", prefix))
        res = []
        for line in out.splitlines():
            if line.strip():
                parts = line.split("\t")
                res.append((parts[0], parts[-1]))
        return res


@dataclass(frozen=True)
class FileChange:
    commit: str
    path: str
    added: int          # numstat added lines (0 for binary)
    deleted: int
    blob: str | None    # blob at `commit` (None if the file was deleted)


def commit_changes(repo: str | Path, a: str, b: str, prefix: str, extensions: tuple[str, ...] = (".py",),
                   first_parent: bool = True) -> list[list[FileChange]]:
    """Per commit c in (a, b] (oldest first): the files under `prefix` changed relative to the previous commit of
    the walk (`git diff --numstat --no-renames prev c`), as in S1/ingest-algo/scripts/commit_tokens.py."""
    g = GitRepo(repo)
    prev = g.resolve(a)
    out = []
    for c in g.rev_list(prev, g.resolve(b), first_parent=first_parent):
        tree = {e.path: e.blob for e in g.ls_tree(c, prefix)}
        ch = []
        for line in str(g.git("diff", "--numstat", "--no-renames", prev, c, "--", prefix)).splitlines():
            ad, de, path = line.split("\t")
            if path.endswith(tuple(extensions)):
                ch.append(FileChange(c, path, int(ad) if ad != "-" else 0, int(de) if de != "-" else 0, tree.get(path)))
        out.append(ch)
        prev = c
    return out


def decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


# -------------------------------------------------------------------------------------------------- filters
_LOCK_OR_GENERATED_NAME = re.compile(
    r"(^|/)(poetry\.lock|Pipfile\.lock|package-lock\.json|yarn\.lock|uv\.lock|Cargo\.lock)$|"
    r"_pb2(_grpc)?\.py$|\.min\.(js|css)$")
_SECRET_FILE_NAME = re.compile(r"(^|/)(\.env(\..*)?|id_rsa[^/]*|id_ed25519[^/]*|credentials\.json)$|\.(pem|key|p12|pfx)$")
_GENERATED_MARK = re.compile(r"@generated|DO NOT EDIT|[Aa]uto-?generated by|[Gg]enerated by protoc")
_DEFAULT_EXCLUDE_DIRS = ("vendor", "vendored", "_vendor", "third_party", "thirdparty", "node_modules",
                         "site-packages", "build", "dist", ".git", "__pycache__")


@dataclass
class FileFilter:
    """Which blobs enter the corpus (§I.4 step 1). `reason(entry, raw)` returns None (keep) or the exclusion reason.

    max_bytes: §I.4 says 256 KB, which would drop python/minisgl/scheduler/scheduler.py (391,098 B) from the
    250-file corpus v0 of §6.A.1; the default here is 512 KiB (see census_v0 for the uncapped census preset)."""

    prefix: str = "python/"
    extensions: tuple[str, ...] = (".py",)
    max_bytes: int | None = 512 * 1024
    exclude_dirs: tuple[str, ...] = _DEFAULT_EXCLUDE_DIRS
    exclude_generated: bool = True
    exclude_minified: bool = True
    max_line_chars: int = 5000
    max_mean_line_chars: float = 300.0

    @classmethod
    def census_v0(cls) -> "FileFilter":
        """The S1/ingest-algo census corpus: every .py blob under python/ (no size cap)."""
        return cls(max_bytes=None)

    def reason(self, ent: TreeEntry, raw: bytes | None) -> str | None:
        p = ent.path
        if self.prefix and not p.startswith(self.prefix):
            return "prefix"
        if self.extensions and not p.endswith(tuple(self.extensions)):
            return "extension"
        if ent.mode == "120000":
            return "symlink"
        parts = p.split("/")[:-1]
        if any(x in self.exclude_dirs for x in parts):
            return "vendored_or_build_dir"
        if _LOCK_OR_GENERATED_NAME.search(p):
            return "lock_or_generated_name"
        if _SECRET_FILE_NAME.search(p):
            return "secret_file_name"
        if self.max_bytes is not None and ent.size > self.max_bytes:
            return "too_large"
        if raw is None:
            return None
        if b"\0" in raw[:8192]:
            return "binary"
        if self.exclude_generated and _GENERATED_MARK.search(decode_text(raw[:2048])):
            return "generated_marker"
        if self.exclude_minified and raw:
            lines = raw.split(b"\n")
            if max(len(x) for x in lines) > self.max_line_chars or len(raw) / max(1, len(lines)) > self.max_mean_line_chars:
                return "minified"
        return None


# --------------------------------------------------------------------------------------------- secret scan
@dataclass(frozen=True)
class SecretHit:
    path: str
    line: int
    rule: str
    preview: str  # redacted


_SECRET_RULES: list[tuple[str, re.Pattern]] = [
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("anthropic_or_openai_key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@'\"]+:[^\s@'\"]{6,}@[^\s'\"]+")),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("aws_secret_access_key", re.compile(r"(?i)aws.{0,24}?[\"'=:\s]([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])")),
]
# a key-like identifier: any identifier whose name ENDS in a secret word (OPENAI_API_KEY, db_password,
# aws_secret_access_key, MY_SECRET, GITHUB_TOKEN; also bare api_key), optionally quoted (dict keys / kwargs)
_SECRET_NAME = (r"[A-Za-z0-9_]*(?:api[_-]?key|secret(?:[_-]?(?:access[_-]?)?key)?|access[_-]?key|token|password|"
                r"passwd|pwd|private[_-]?key|client[_-]?secret|credentials?)")
_GENERIC_ASSIGN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])" + _SECRET_NAME + r"[\"']?\s*[:=]\s*[\"']([^\"'\s]{16,})[\"']")
# os.environ.get("X_TOKEN", "<value>") / getenv(...) with a secret-shaped default
_ENV_DEFAULT = re.compile(
    r"(?i)(?:environ\.get|getenv|setdefault)\(\s*[\"']" + _SECRET_NAME + r"[\"']\s*,\s*[\"']([^\"'\s]{16,})[\"']")
# values that are code, not secrets: dotted identifiers ("vision.cls_token", "model.embed_tokens.weight")
_DOTTED_IDENT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_PLACEHOLDER = re.compile(r"(?i)xxx|<|>|\$\{|\{|your|example|dummy|placeholder|changeme|test|fake|\*\*\*")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _redact(s: str) -> str:
    s = s.strip()
    return (s[:6] + "…" + f"[{len(s)} chars]") if len(s) > 6 else "…"


def scan_secrets(path: str, text: str, min_entropy: float = 3.5) -> list[SecretHit]:
    """Secret-shaped strings in one file (§I.4: rows memorise code and the cartridge ships as a file)."""
    hits: list[SecretHit] = []
    for i, line in enumerate(text.splitlines(), 1):
        for rule, rx in _SECRET_RULES:
            m = rx.search(line)
            if m:
                hits.append(SecretHit(path, i, rule, _redact(m.group(0))))
        for rule, rx in (("generic_secret_assignment", _GENERIC_ASSIGN), ("env_secret_default", _ENV_DEFAULT)):
            m = rx.search(line)
            if m:
                val = m.group(1)
                if (not _PLACEHOLDER.search(val) and not _DOTTED_IDENT.match(val)
                        and shannon_entropy(val) >= min_entropy):
                    hits.append(SecretHit(path, i, rule, _redact(val)))
    return hits


# ------------------------------------------------------------------------------------------------- dedup
_LEX = re.compile(r"\w+|[^\w\s]")
_P61 = (1 << 61) - 1


def code_tokens(text: str) -> list[str]:
    return _LEX.findall(text)


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    toks = code_tokens(text)
    if len(toks) < k:
        toks = toks + [""] * (k - len(toks))
    sh = {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}
    return np.array(sorted(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") for s in sh),
                    dtype=np.uint64)


class MinHasher:
    """MinHash over code-token k-shingles with universal hashing mod 2^61-1 (k=5 per §I.4)."""

    def __init__(self, num_perm: int = 128, shingle: int = 5, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 29, num_perm, dtype=np.uint64)  # a * x < 2^61 for 32-bit x
        self.b = rng.integers(0, 1 << 60, num_perm, dtype=np.uint64)
        self.num_perm, self.shingle = num_perm, shingle

    def signature(self, text: str) -> np.ndarray:
        x = _shingle_hashes(text, self.shingle)
        h = (x[None, :] * self.a[:, None] + self.b[:, None]) % np.uint64(_P61)
        return h.min(axis=1)


@dataclass
class DedupResult:
    exact_groups: list[list[str]]            # paths with identical blobs (first one kept)
    near_clusters: list[list[str]]           # MinHash clusters of size >= 2 (after exact dedup)
    dropped_exact: list[str]
    jaccard: dict[str, float] = field(default_factory=dict)  # "a|b" -> estimated Jaccard of linked pairs


def dedup(files: dict[str, str], blobs: dict[str, str], threshold: float = 0.8, bands: int = 16,
          num_perm: int = 128, shingle: int = 5, seed: int = 1) -> DedupResult:
    """Exact dedup by blob sha (keep the first path in sorted order), then near-dup clusters by MinHash + LSH
    (bands x rows = num_perm), linked when the estimated Jaccard >= threshold; clusters by union-find."""
    by_blob: dict[str, list[str]] = defaultdict(list)
    for p in sorted(files):
        by_blob[blobs[p]].append(p)
    exact = [g for g in by_blob.values() if len(g) > 1]
    dropped = [p for g in exact for p in g[1:]]
    keep = sorted(p for p in files if p not in set(dropped))
    mh = MinHasher(num_perm, shingle, seed)
    sig = {p: mh.signature(files[p]) for p in keep}
    rows = num_perm // bands
    parent = {p: p for p in keep}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    jac: dict[str, float] = {}
    seen: set[tuple[str, str]] = set()
    for b in range(bands):
        buckets: dict[bytes, list[str]] = defaultdict(list)
        for p in keep:
            buckets[sig[p][b * rows:(b + 1) * rows].tobytes()].append(p)
        for grp in buckets.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    pr = (grp[i], grp[j])
                    if pr in seen:
                        continue
                    seen.add(pr)
                    est = float((sig[pr[0]] == sig[pr[1]]).mean())
                    if est >= threshold:
                        jac[f"{pr[0]}|{pr[1]}"] = est
                        ra, rb = find(pr[0]), find(pr[1])
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    cl: dict[str, list[str]] = defaultdict(list)
    for p in keep:
        cl[find(p)].append(p)
    near = sorted(sorted(v) for v in cl.values() if len(v) > 1)
    return DedupResult(exact, near, dropped, jac)


# -------------------------------------------------------------------------------------------------- corpus
@dataclass
class SourceFile:
    path: str          # repo-relative ("python/minisgl/...")
    blob: str
    size: int          # bytes
    text: str

    @property
    def rel(self) -> str:
        """Path relative to python/ (the §6.A.1 document header)."""
        return self.path[len("python/"):] if self.path.startswith("python/") else self.path


@dataclass
class Corpus:
    repo: str
    commit: str
    files: list[SourceFile]
    excluded: list[tuple[str, str]]         # (path, reason)
    secret_hits: list[SecretHit]
    dedup: DedupResult | None

    def by_path(self) -> dict[str, SourceFile]:
        return {f.path: f for f in self.files}

    def summary(self) -> dict:
        return dict(repo=self.repo, commit=self.commit, files=len(self.files),
                    bytes=sum(f.size for f in self.files),
                    excluded=Counter(r for _, r in self.excluded),
                    excluded_files=[p for p, r in self.excluded if r not in ("prefix", "extension")],
                    secret_hits=[asdict(h) for h in self.secret_hits],
                    exact_dup_groups=self.dedup.exact_groups if self.dedup else None,
                    near_dup_clusters=self.dedup.near_clusters if self.dedup else None)


def load_corpus(repo: str | Path = MINISGL_REPO, commit: str = MINISGL_HEAD, flt: FileFilter | None = None,
                secret_scan: bool = True, do_dedup: bool = True, drop_exact_dups: bool = True,
                near_threshold: float = 0.8) -> Corpus:
    """Blobs at `commit` that pass the filter and the secret scan (files with a hit are excluded and reported)."""
    g = GitRepo(repo)
    commit = g.resolve(commit)
    flt = flt or FileFilter()
    ents = g.ls_tree(commit, flt.prefix.rstrip("/") or None)
    excluded: list[tuple[str, str]] = []
    pre = []
    for e in ents:
        r = flt.reason(e, None)
        (excluded.append((e.path, r)) if r else pre.append(e))
    raw = g.read_blobs(e.blob for e in pre)
    files, hits = [], []
    for e in pre:
        r = flt.reason(e, raw[e.blob])
        if r:
            excluded.append((e.path, r))
            continue
        text = decode_text(raw[e.blob])
        if secret_scan:
            h = scan_secrets(e.path, text)
            if h:
                hits += h
                excluded.append((e.path, "secret"))
                continue
        files.append(SourceFile(e.path, e.blob, e.size, text))
    dd = None
    if do_dedup:
        dd = dedup({f.path: f.text for f in files}, {f.path: f.blob for f in files}, threshold=near_threshold)
        if drop_exact_dups and dd.dropped_exact:
            drop = set(dd.dropped_exact)
            excluded += [(p, "exact_duplicate") for p in dd.dropped_exact]
            files = [f for f in files if f.path not in drop]
    return Corpus(str(repo), commit, files, excluded, hits, dd)


# --------------------------------------------------------------------------------------------------- split A
def split_a_hash(path: str) -> float:
    return int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 2**32


@dataclass
class SplitA:
    rule: dict
    commit: str
    train: list[str]
    dev: list[str]
    test: list[str]
    neardup_diag: list[str]                 # held-out files moved out by the verbatim-line gate
    cluster_moves: list[dict]               # MinHash clusters forced into one split
    dup_frac: dict[str, float]              # held-out file -> share of its >=20-char lines verbatim in train

    def assignment(self) -> dict[str, str]:
        a = {p: "train" for p in self.train}
        a.update({p: "dev" for p in self.dev})
        a.update({p: "test" for p in self.test})
        a.update({p: "neardup_diag" for p in self.neardup_diag})
        return a

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> "SplitA":
        return cls(**json.loads(s))


def _verbatim_dup_frac(held: str, train_lines: set[str], min_chars: int = 20) -> float:
    ls = [ln.strip() for ln in held.splitlines() if len(ln.strip()) >= min_chars]
    return sum(s in train_lines for s in ls) / max(1, len(ls))


def assign_split_a(files: dict[str, str], commit: str = "", test_frac: float = 0.15, dev_frac: float = 0.10,
                   min_chars: int = 400, near_clusters: Sequence[Sequence[str]] = (), dup_gate: float = 0.5,
                   frozen: SplitA | None = None) -> SplitA:
    """§6.A.1 Split A. files: repo path -> text. With `frozen`, files already listed keep their split; NEW files get
    the hash rule and then the same leakage gates as an unfrozen build, applied to the new files only: a new
    held-out file in a near-dup cluster with a train file goes to train, and a new held-out file whose verbatim
    line share in train exceeds `dup_gate` goes to neardup_diag (moves are appended to cluster_moves). Files no
    longer present are dropped."""
    lo, hi = test_frac, test_frac + dev_frac
    asg: dict[str, str] = {}
    old = frozen.assignment() if frozen is not None else {}
    new = set()
    for p in sorted(files):
        if p in old:
            asg[p] = old[p]
            continue
        new.add(p)
        h = split_a_hash(p)
        big = len(files[p]) > min_chars
        asg[p] = "test" if (h < lo and big) else ("dev" if (lo <= h < hi and big) else "train")
    moves = []
    for cl in near_clusters:
        cl = [p for p in cl if p in asg]
        if len({asg[p] for p in cl}) <= 1:
            continue
        if frozen is None:
            movable = list(cl)
        else:  # frozen members stay put; a new member joins train if the cluster touches train
            movable = [p for p in cl if p in new and asg[p] != "train"] if any(asg[q] == "train" for q in cl) else []
        if movable:
            moves.append(dict(cluster=list(cl), to="train", was={p: asg[p] for p in movable}))
            for p in movable:
                asg[p] = "train"
    train_lines = {ln.strip() for p, s in asg.items() if s == "train"
                   for ln in files[p].splitlines() if len(ln.strip()) >= 20}
    dupf = {p: round(_verbatim_dup_frac(files[p], train_lines), 4) for p, s in asg.items()
            if s in ("dev", "test", "neardup_diag")}
    for p, f in dupf.items():
        if f > dup_gate and (frozen is None or (p in new and asg[p] != "neardup_diag")):
            if frozen is not None:
                moves.append(dict(dup_gate=p, dup_frac=f, to="neardup_diag", was={p: asg[p]}))
            asg[p] = "neardup_diag"
    rule = dict(hash="int(sha256(repo_path)[:8],16)/2**32", test=f"h < {lo} and len(text) > {min_chars}",
                dev=f"{lo} <= h < {hi} and len(text) > {min_chars}", dup_gate=f"held-out verbatim >=20-char line "
                f"share in train > {dup_gate} -> neardup_diag", clusters="a MinHash near-dup cluster (Jaccard >= 0.8 "
                "on code-token 5-shingles) spanning several splits goes wholly to train")
    if frozen is not None:  # keep the record of how the frozen lists were made, plus the moves of new files
        rule, moves, commit = frozen.rule, list(frozen.cluster_moves) + moves, frozen.commit
    pick = lambda s: sorted(p for p, v in asg.items() if v == s)  # noqa: E731
    return SplitA(rule, commit, pick("train"), pick("dev"), pick("test"), pick("neardup_diag"), moves, dupf)


# --------------------------------------------------------------------------------------------------- split B
@dataclass
class SplitB:
    t_cut: str
    head: str
    t_cut_time: int
    head_time: int
    commits: list[str]                      # (t_cut, head], oldest first, all parents
    first_parent_commits: list[str]
    added_lines: dict[str, list[list[int]]]  # path -> [[start, end_inclusive], ...] in the head file
    new_files: list[str]
    train_files: list[str]                  # files of the tree at t_cut (after the filter)

    @property
    def n_added_lines(self) -> int:
        return sum(b - a + 1 for v in self.added_lines.values() for a, b in v)

    def added_line_set(self, path: str) -> set[int]:
        return {i for a, b in self.added_lines.get(path, []) for i in range(a, b + 1)}

    def to_json(self) -> str:
        d = asdict(self)
        d["n_commits"], d["n_added_lines"], d["n_files_with_added_lines"] = (len(self.commits), self.n_added_lines,
                                                                             len(self.added_lines))
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "SplitB":
        d = json.loads(s)
        for k in ("n_commits", "n_added_lines", "n_files_with_added_lines"):
            d.pop(k, None)
        return cls(**d)


def _ranges(lines: set[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for i in sorted(lines):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def build_split_b(repo: str | Path = MINISGL_REPO, t_cut: str = MINISGL_T_CUT, head: str = MINISGL_HEAD,
                  prefix: str = "python/minisgl", flt: FileFilter | None = None) -> SplitB:
    g = GitRepo(repo)
    t_cut, head = g.resolve(t_cut), g.resolve(head)
    flt = flt or FileFilter()
    ok = lambda p: p.endswith(tuple(flt.extensions))  # noqa: E731
    added = {p: _ranges(v) for p, v in sorted(g.added_lines(t_cut, head, prefix).items()) if ok(p)}
    new = sorted(p for _, p in g.name_status(t_cut, head, prefix, diff_filter="A") if ok(p))
    train = sorted(e.path for e in g.ls_tree(t_cut, prefix) if flt.reason(e, None) is None)
    return SplitB(t_cut, head, g.commit_time(t_cut), g.commit_time(head), g.rev_list(t_cut, head),
                  g.rev_list(t_cut, head, first_parent=True), added, new, train)


# ------------------------------------------------------------------------------------------------- framings
FRAMINGS = ("raw", "bare", "chat", "catn")


def numbered(text: str) -> str:
    """`cat -n`-style tool output (S1/ingest-algo/scripts/extras.py::numbered)."""
    return "".join(f"{i:>6}\t{line}\n" for i, line in enumerate(text.split("\n"), 1))


def frame(rel_path: str, text: str, framing: str) -> str:
    if framing == "raw":
        return text
    if framing == "bare":
        return f"# file: {rel_path}\n{text}"
    if framing == "chat":
        return (f"<|im_start|>user\nHere is `{rel_path}`:\n```python\n{text}\n```\nWhat does it do?<|im_end|>\n"
                f"<|im_start|>assistant\n")
    if framing == "catn":
        return numbered(text)
    raise ValueError(f"framing {framing!r} not in {FRAMINGS}")


# ---------------------------------------------------------------------------------------------- tokenising
class TokenizerJSON:
    """Thin wrapper over a `tokenizers` tokenizer.json (add_special_tokens=False, as the census)."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.tok = Tokenizer.from_file(str(self.path))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def encode_many(self, texts: Sequence[str], batch: int = 256) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            for e in self.tok.encode_batch(list(texts[i:i + batch]), add_special_tokens=False):
                out.append(np.asarray(e.ids, dtype=np.int64))
        return out

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)


@dataclass
class Document:
    path: str
    framing: str
    ids: np.ndarray


def tokenize_files(tok: TokenizerJSON, files: Sequence[SourceFile], framing: str = "bare",
                   repo_paths: bool = False) -> list[Document]:
    """repo_paths: name files by their repo path ("python/minisgl/...") instead of the path relative to python/
    (extras.py's chat framing used the repo path)."""
    texts = [frame(f.path if repo_paths else f.rel, f.text, framing) for f in files]
    return [Document(f.path, framing, ids) for f, ids in zip(files, tok.encode_many(texts))]


def mix_framings(docs_by_framing: dict[str, list[Document]], weights: dict[str, float], seed: int = 0) -> list[Document]:
    """Pick one framing per file with the given probabilities (deterministic in seed and path)."""
    names = sorted(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    p = p / p.sum()
    paths = [d.path for d in docs_by_framing[names[0]]]
    idx = {n: {d.path: d for d in docs_by_framing[n]} for n in names}
    out = []
    for path in paths:
        r = np.random.default_rng([seed, int(hashlib.sha256(path.encode()).hexdigest()[:8], 16)]).random()
        out.append(idx[names[int(np.searchsorted(np.cumsum(p), r, side="right").clip(0, len(names) - 1))]][path])
    return out


# --------------------------------------------------------------------------------------------------- windows
@dataclass
class Window:
    """One training/eval window. ids [L]; context [n-1] = the true preceding tokens (eos where the stream starts);
    labels [L] = next-token targets (-100 where not scored: context-only overlap, stream end, padding);
    start = offset of ids[0] in the stream."""

    ids: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    start: int

    @property
    def n_targets(self) -> int:
        return int((self.labels >= 0).sum())


IGNORE = -100


def pack_stream(docs: Sequence[np.ndarray | Document], eos: int, sep_eos: bool = True) -> np.ndarray:
    """Concatenate documents; each one ends with EOS (the only history reset, §2.1)."""
    parts = []
    for d in docs:
        a = d.ids if isinstance(d, Document) else np.asarray(d, dtype=np.int64)
        parts.append(a)
        if sep_eos:
            parts.append(np.array([eos], dtype=np.int64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def make_windows(stream: np.ndarray, window: int = 4096, stride: int = 3584, context_len: int = 2,
                 eos: int = 0, score_mask: np.ndarray | None = None) -> list[Window]:
    """Cut a token stream into windows of `window` tokens every `stride` tokens (§I.4 step 6). Each target token
    is scored exactly once: in a later window, targets already scored by an earlier one (the window - stride
    overlap) are context only. `score_mask` [N] optionally restricts scored targets (by target position)."""
    if not 0 < stride <= window:
        raise ValueError("need 0 < stride <= window")
    n = int(stream.size)
    out: list[Window] = []
    covered = 1  # every target index < covered is already scored (stream[0] has no predictor)
    start = 0
    while n > 0:
        end = min(start + window, n)
        ids = stream[start:end].copy()
        tgt = np.arange(start + 1, end + 1)          # position i predicts stream[start + i + 1]
        lab = np.full(end - start, IGNORE, dtype=np.int64)
        ok = (tgt < n) & (tgt >= covered)
        if score_mask is not None:
            ok &= np.asarray(score_mask, dtype=bool)[np.minimum(tgt, n - 1)]
        lab[ok] = stream[tgt[ok]]
        ctx = np.full(context_len, eos, dtype=np.int64)
        k = min(start, context_len)
        if k:
            ctx[context_len - k:] = stream[start - k:start]
        out.append(Window(ids, ctx, lab, start))
        covered = max(covered, end + 1)
        if end >= n:
            break
        start += stride
    return out


def batch_windows(ws: Sequence[Window], pad_id: int) -> dict[str, np.ndarray]:
    """Right-pad a list of windows into [B, L] arrays (input_ids, attention_mask, labels, context)."""
    L = max(w.ids.size for w in ws)
    B = len(ws)
    ids = np.full((B, L), pad_id, dtype=np.int64)
    att = np.zeros((B, L), dtype=np.int64)
    lab = np.full((B, L), IGNORE, dtype=np.int64)
    for i, w in enumerate(ws):
        ids[i, :w.ids.size], att[i, :w.ids.size], lab[i, :w.ids.size] = w.ids, 1, w.labels
    ctx = np.stack([w.context for w in ws])
    return dict(input_ids=ids, attention_mask=att, labels=lab, context=ctx)


# ------------------------------------------------------------------------------------------------ splits I/O
def write_split_files(split_a: SplitA, split_b: SplitB | None, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_a.json").write_text(split_a.to_json() + "\n")
    if split_b is not None:
        (out_dir / "split_b.json").write_text(split_b.to_json() + "\n")


def load_split_a(path: Path = SPLITS_DIR / "split_a.json") -> SplitA | None:
    return SplitA.from_json(path.read_text()) if path.exists() else None


def load_split_b(path: Path = SPLITS_DIR / "split_b.json") -> SplitB | None:
    return SplitB.from_json(path.read_text()) if path.exists() else None


def legacy_design_split(n_docs: int, seed: int = 0, frac: float = 0.15) -> tuple[list[int], list[int]]:
    """The 85/15 shuffle split of S1/ingest-algo (count_rows.py::split_by_file, row_policy.py): indices into the
    NON-EMPTY documents in ls-tree order. Returns (train, heldout) sorted. Not Split A; kept to reproduce
    S1/ingest-algo/row_policy.out (2,040,221 rows)."""
    import random
    idx = list(range(n_docs))
    random.Random(seed).shuffle(idx)
    nh = int(round(frac * n_docs))
    return sorted(idx[nh:]), sorted(idx[:nh])


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("annotations",)]
