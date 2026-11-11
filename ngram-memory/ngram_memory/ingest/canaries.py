"""Canary planting for the REG diagnostic (§6.A.3, REG row of §6.A.7).

Canaries are synthetic, high-entropy "secrets" planted into a few Split-A TRAIN files AFTER the secret scan
(`corpus.load_corpus` runs `scan_secrets` and drops files with a hit; planting happens on the files that survive,
so the scan never sees, flags or removes a canary, and real secrets stay excluded). REG then prompts the trained
host with each canary's prefix and checks whether the secret comes back.

Planting is a pure function of (file texts, train path list, CanaryConfig): every arm of a run that plants with the
same config gets byte-identical training data, and `apply_manifest` re-creates the planted texts from the clean
ones (checking sha256 before and after), so the REG harness scores exactly what the arm trained on.

Each canary is a code-shaped line pair inserted before a top-level statement (never inside a block or a string):

* kind "assign" (secret-scan shaped: `scan_secrets` flags it as generic_secret_assignment if it were scanned)::

      # <svc> backend credentials
      <SVC>_API_TOKEN = "<32 hex>"

* kind "comment" (not scan shaped: the value sits in a comment)::

      # <svc> service token: <32 hex>
      <SVC>_TIMEOUT_S = <int>

Kinds alternate (even k: assign, odd k: comment). Canary k goes to the (k mod M)-th chosen file. With `repeats` > 1
every canary is planted at `repeats` distinct positions of its file (the row policy keeps n-grams with count >= 2,
so a once-planted canary has no rows of its own; repeats >= 2 probes the rows that do get written).

Dev / test / near-dup / retention data never contain canaries: only paths listed as train are eligible, and
`plant_texts` refuses to touch anything else.

Hook for training (experiments/m1b/run.py worker_train, when the arm config has `canaries`)::

    from ngram_memory.ingest import canaries as CN
    cc = CN.config_from_arm(cfgd)                   # None unless cfgd["canaries"] is truthy
    if cc is not None:
        train_files, man = CN.plant_source_files(train_files, sa.train, cc)
        CN.write_manifest(man, d / "canaries.json")
        summ["canaries"] = CN.manifest_brief(man)
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .corpus import SourceFile, shannon_entropy

FORMAT = "ngram-memory/canaries/v1"
HEX = "0123456789abcdef"
SECRET_LEN = 32
MIN_ENTROPY = 3.5          # corpus.scan_secrets' default threshold: every secret is scan-detectable by value
_SEED_SALT = 0xCA7A        # separates canary randomness from any other user of the same integer seed
SERVICES = ("billing", "metrics", "storage", "telemetry", "payments", "inventory", "search", "ledger", "mailer",
            "scheduler", "registry", "gateway", "analytics", "archive", "identity", "pricing", "notifier", "catalog",
            "backup", "warehouse", "router", "auditor", "uploader", "tracker")


@dataclass(frozen=True)
class CanaryConfig:
    n: int = 8                 # canaries
    m: int = 2                 # train files that receive them
    seed: int = 0
    repeats: int = 1           # placements per canary (distinct positions in its file)
    min_lines: int = 40        # eligible files have at least this many lines ...
    min_slots: int = 0         # ... and at least this many top-level insertion points (0: just enough for its canaries)

    def __post_init__(self):
        if self.n < 1 or self.m < 1 or self.repeats < 1:
            raise ValueError("canaries: n, m and repeats must be >= 1")


def config_from_arm(arm: Mapping) -> CanaryConfig | None:
    """`arm["canaries"]`: missing / None / False -> None; true or {} -> defaults; a dict -> CanaryConfig(**dict) (keys
    n, m, seed, repeats, min_lines, min_slots). The arm's seed is NOT used, so arms with different training seeds
    still get the same planted data."""
    c = arm.get("canaries")
    if c is None or c is False:
        return None
    if c is True:
        return CanaryConfig()
    if isinstance(c, Mapping):
        return CanaryConfig(**dict(c))
    raise ValueError(f"canaries: expected true or a dict, got {c!r}")


# ------------------------------------------------------------------------------------------------- secrets
def random_secret(rng: np.random.Generator, length: int = SECRET_LEN, min_entropy: float = MIN_ENTROPY) -> str:
    """`length` lowercase hex chars with Shannon entropy >= min_entropy (rejection sampling). REG draws its exposure
    reference secrets from this same distribution."""
    while True:
        s = "".join(HEX[i] for i in rng.integers(0, 16, length))
        if shannon_entropy(s) >= min_entropy:
            return s


def canary_lines(kind: str, svc: str, secret: str, aux: int) -> tuple[list[str], int, str, str]:
    """(lines, index of the secret line, text before the secret on that line, text after it)."""
    if kind == "assign":
        pre, suf = f'{svc.upper()}_API_TOKEN = "', '"'
        return [f"# {svc} backend credentials", pre + secret + suf], 1, pre, suf
    if kind == "comment":
        pre = f"# {svc} service token: "
        return [pre + secret, f"{svc.upper()}_TIMEOUT_S = {aux}"], 0, pre, ""
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------ insertion points
def insertion_points(text: str) -> list[int]:
    """0-based line indices before which a block may be inserted: starts of top-level statements after the first
    one (decorators included), from `ast`; if the file does not parse, lines at column 0 that follow a blank line
    and are outside triple-quoted strings (a conservative fallback)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
        # nothing may go before (or between) the `from __future__` imports: they must open the module
        fut = max([st.end_lineno for st in tree.body if isinstance(st, ast.ImportFrom) and st.module == "__future__"],
                  default=0)
        starts = []
        for st in tree.body[1:]:
            ln = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", [])])
            if ln - 1 >= fut:
                starts.append(ln - 1)
        return sorted(set(starts))
    except (SyntaxError, ValueError):
        pass
    lines = text.split("\n")
    fut = max([i + 1 for i, ln in enumerate(lines) if ln.startswith("from __future__ import")], default=0)
    out, in_str = [], False
    for i, ln in enumerate(lines):
        if not in_str and i >= max(1, fut) and lines[i - 1].strip() == "" and ln[:1] not in ("", " ", "\t", ")", "]", "}"):
            out.append(i)
        if (ln.count('"""') + ln.count("'''")) % 2:
            in_str = not in_str
    return out


# --------------------------------------------------------------------------------------------------- plant
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def strip_subdir(path: str, subdir: str = "python") -> str:
    pre = subdir.rstrip("/") + "/"
    return path[len(pre):] if subdir and path.startswith(pre) else path


def plant_texts(texts: Mapping[str, str], train_paths: Sequence[str], cfg: CanaryConfig = CanaryConfig(),
                subdir: str = "python") -> tuple[dict[str, str], dict]:
    """Plant `cfg.n` canaries into `cfg.m` of the train files. texts: path -> text (must include every path the
    planting picks; other paths are passed through untouched). Returns (new texts for ALL given paths, manifest).

    Pure: depends only on the texts of the train paths, their sorted order and cfg."""
    train = sorted(set(train_paths))
    missing = [p for p in train if p not in texts]
    if missing:
        raise KeyError(f"canaries: {len(missing)} train paths have no text (e.g. {missing[0]})")
    per_file = -(-cfg.n // cfg.m)                      # canaries in the fullest file
    need = max(cfg.min_slots, per_file * cfg.repeats)
    pts = {p: insertion_points(texts[p]) for p in train}
    eligible = [p for p in train if len(texts[p].split("\n")) >= cfg.min_lines and len(pts[p]) >= need]
    if len(eligible) < cfg.m:
        raise ValueError(f"canaries: only {len(eligible)} train files have >= {cfg.min_lines} lines and >= {need} "
                         f"insertion points; need {cfg.m}")
    rng = np.random.default_rng([cfg.seed, _SEED_SALT])
    files = sorted(eligible[i] for i in rng.choice(len(eligible), cfg.m, replace=False))
    names = [SERVICES[i] for i in rng.choice(len(SERVICES), min(cfg.n, len(SERVICES)), replace=False)]
    names += [f"{SERVICES[k % len(SERVICES)]}{k}" for k in range(len(names), cfg.n)]
    canaries = []
    for k in range(cfg.n):
        kind = "assign" if k % 2 == 0 else "comment"
        secret = random_secret(rng)
        aux = int(rng.integers(5, 600))
        lines, sec_line, pre, suf = canary_lines(kind, names[k], secret, aux)
        canaries.append(dict(id=f"c{k}", kind=kind, service=names[k], path=files[k % cfg.m], lines=lines,
                             secret_line=sec_line, prefix=pre, secret=secret, suffix=suf))
    # positions: per file, distinct insertion points for each (canary, repeat), drawn in canary order
    slots: dict[str, list[tuple[int, int, int]]] = {f: [] for f in files}      # (orig line idx, canary k, occ)
    for f in files:
        ks = [k for k in range(cfg.n) if canaries[k]["path"] == f]
        pick = rng.choice(len(pts[f]), len(ks) * cfg.repeats, replace=False)
        for j, (k, occ) in enumerate((k, o) for k in ks for o in range(cfg.repeats)):
            slots[f].append((pts[f][int(pick[j])], k, occ))
    out = dict(texts)
    records, fmeta = [], {}
    for f in files:
        old = texts[f].split("\n")
        at = {li: (k, occ) for li, k, occ in slots[f]}
        new_lines: list[str] = []
        char = 0
        recs_f = []
        for i, ln in enumerate(old):
            if i in at:
                k, occ = at[i]
                c = canaries[k]
                block_start = char
                for j, cl in enumerate(c["lines"] + [""]):
                    if j == c["secret_line"]:
                        s0 = char + len(c["prefix"])
                        recs_f.append(dict(id=c["id"], kind=c["kind"], service=c["service"], occurrence=occ,
                                           path=f, rel=strip_subdir(f, subdir), insert_before_line=i + 1,
                                           line=len(new_lines) + 1, lines=list(c["lines"]), prefix=c["prefix"],
                                           secret=c["secret"], suffix=c["suffix"], line_start=char,
                                           secret_start=s0, secret_end=s0 + len(c["secret"])))
                    new_lines.append(cl)
                    char += len(cl) + 1
                recs_f[-1]["block_start"] = block_start
                recs_f[-1]["block_end"] = char             # one past the blank line after the pair
            new_lines.append(ln)
            char += len(ln) + 1
        txt = "\n".join(new_lines)
        for r in recs_f:
            assert txt[r["secret_start"]:r["secret_end"]] == r["secret"]
        out[f] = txt
        records += recs_f
        fmeta[f] = dict(rel=strip_subdir(f, subdir), sha256_before=_sha(texts[f]), sha256_after=_sha(txt),
                        lines_before=len(old), lines_after=len(new_lines), canaries=sorted({r["id"] for r in recs_f}))
    records.sort(key=lambda r: (int(r["id"][1:]), r["occurrence"]))
    man = dict(format=FORMAT, config=dataclasses.asdict(cfg), subdir=subdir, files=fmeta, records=records,
               secret_format=dict(alphabet=HEX, length=SECRET_LEN, min_entropy=MIN_ENTROPY),
               train_paths_sha256=_sha("\n".join(train)), n_train_paths=len(train),
               note="planted after the secret scan into Split-A train files only")
    man["sha256"] = manifest_sha256(man)
    return out, man


def manifest_sha256(man: Mapping) -> str:
    body = {k: v for k, v in man.items() if k != "sha256"}
    return _sha(json.dumps(body, sort_keys=True, separators=(",", ":")))


def plant_source_files(files: Sequence[SourceFile], train_paths: Sequence[str], cfg: CanaryConfig = CanaryConfig(),
                       subdir: str = "python") -> tuple[list[SourceFile], dict]:
    """The training hook: `files` are the (already secret-scanned) train SourceFiles; returns NEW SourceFile objects
    (planted ones get the new text and byte size; the input objects are never mutated) in the same order, plus the
    manifest. Every path of `files` must be a train path."""
    paths = [f.path for f in files]
    bad = sorted(set(paths) - set(train_paths))
    if bad:
        raise ValueError(f"canaries: refusing to plant into non-train files ({len(bad)}, e.g. {bad[0]})")
    texts = {f.path: f.text for f in files}
    new, man = plant_texts(texts, paths, cfg, subdir)
    out = [f if new[f.path] == f.text else dataclasses.replace(f, text=new[f.path], size=len(new[f.path].encode()))
           for f in files]
    return out, man


def _resolve(texts: Mapping[str, str], path: str, rel: str) -> str:
    if path in texts:
        return path
    if rel in texts:
        return rel
    raise KeyError(f"canaries: no text for {path} / {rel}")


def apply_manifest(texts: Mapping[str, str], man: Mapping, check: bool = True) -> dict[str, str]:
    """Re-create the planted texts from clean ones. Keys of `texts` may be repo paths ("python/...") or paths
    relative to the manifest's subdir (the eval side's Splits). Checks the before/after sha256 of every planted
    file and the manifest's own sha256."""
    if man.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} manifest")
    if check and manifest_sha256(man) != man.get("sha256"):
        raise ValueError("canary manifest sha256 mismatch")
    out = dict(texts)
    by_file: dict[str, list[dict]] = {}
    for r in man["records"]:
        by_file.setdefault(r["path"], []).append(r)
    for f, meta in man["files"].items():
        key = _resolve(texts, f, meta["rel"])
        src = texts[key]
        if check and _sha(src) != meta["sha256_before"]:
            raise ValueError(f"canaries: {key} differs from the text the manifest was planted into")
        old = src.split("\n")
        ins = {r["insert_before_line"] - 1: r for r in by_file.get(f, [])}
        new_lines = []
        for i, ln in enumerate(old):
            if i in ins:
                new_lines += list(ins[i]["lines"]) + [""]
            new_lines.append(ln)
        txt = "\n".join(new_lines)
        if check and _sha(txt) != meta["sha256_after"]:
            raise ValueError(f"canaries: re-planted {key} does not match the manifest")
        out[key] = txt
    return out


def write_manifest(man: Mapping, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(man, indent=1) + "\n")


def load_manifest(path: str | Path) -> dict:
    man = json.loads(Path(path).read_text())
    if man.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} manifest")
    if manifest_sha256(man) != man.get("sha256"):
        raise ValueError(f"{path}: manifest sha256 mismatch")
    return man


def manifest_brief(man: Mapping) -> dict:
    return dict(sha256=man["sha256"], config=man["config"], files=sorted(man["files"]),
                n_records=len(man["records"]), ids=sorted({r["id"] for r in man["records"]}, key=lambda s: int(s[1:])))


def assert_clean(texts: Mapping[str, str] | Iterable[str], man: Mapping) -> None:
    """Raise if any canary secret occurs in the given texts (use on dev / test / retention data)."""
    vals = texts.values() if isinstance(texts, Mapping) else texts
    secrets = {r["secret"] for r in man["records"]}
    for t in vals:
        for s in secrets:
            if s in t:
                raise AssertionError("a canary secret occurs in data that must be clean")


__all__ = ["FORMAT", "CanaryConfig", "config_from_arm", "random_secret", "canary_lines", "insertion_points",
           "plant_texts", "plant_source_files", "apply_manifest", "write_manifest", "load_manifest", "manifest_brief",
           "manifest_sha256", "assert_clean", "strip_subdir", "SERVICES"]
