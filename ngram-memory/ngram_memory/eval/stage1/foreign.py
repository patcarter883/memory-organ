"""Foreign-repo corpus for the content control (§6.A.4 "Foreign-repo cartridge"): an unrelated pure-Python package,
token-matched to the project's train split, with no overlap with the R2 retention files (stdlib list, R2-ML).

The sdist is fetched from the PyPI JSON API and verified against PyPI's sha256 (`pip download --no-binary` would
run the package's build backend to read metadata, i.e. execute downloaded code, and failed for torchmetrics in
this venv). The archive is opened with tarfile's 'data' filter and only `.py` regular files are read; nothing in it
is executed or imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

EXCLUDE_DEFAULT = ("/tests/", "/test/", "/docs/", "/examples/", "/benchmarks/", "conftest.py", "setup.py")


def _ctx() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=ca) if ca and Path(ca).exists() else ssl.create_default_context()


def pypi_sdist(name: str, version: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", context=_ctx(), timeout=60) as r:
        d = json.load(r)
    s = [u for u in d["urls"] if u["packagetype"] == "sdist"]
    if not s:
        raise LookupError(f"no sdist for {name}=={version}")
    u = s[0]
    return dict(name=name, version=version, filename=u["filename"], url=u["url"], sha256=u["digests"]["sha256"],
                size=u["size"], upload_time=u.get("upload_time"), license=d["info"].get("license"))


def download(info: dict, dest_dir: str | Path) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / info["filename"]
    if not p.exists():
        with urllib.request.urlopen(info["url"], context=_ctx(), timeout=300) as r:
            p.write_bytes(r.read())
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != info["sha256"]:
        p.unlink()
        raise ValueError(f"sha256 mismatch for {p.name}: {h} != {info['sha256']}")
    return p


def read_py_files(archive: str | Path) -> dict[str, str]:
    """{path inside the package root: text} for the .py regular files of a .tar.gz / .zip sdist (no extraction)."""
    out: dict[str, str] = {}
    a = str(archive)
    if a.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(a) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".py"):
                    continue
                m = tarfile.data_filter(m, "/nonexistent") if hasattr(tarfile, "data_filter") else m
                f = tf.extractfile(m)
                if f is not None:
                    out[m.name.split("/", 1)[1] if "/" in m.name else m.name] = f.read().decode("utf-8", "replace")
    elif a.endswith(".zip"):
        with zipfile.ZipFile(a) as zf:
            for n in zf.namelist():
                if n.endswith(".py") and not n.endswith("/"):
                    out[n.split("/", 1)[1] if "/" in n else n] = zf.read(n).decode("utf-8", "replace")
    else:
        raise ValueError(f"unknown archive type {a}")
    return dict(sorted(out.items()))


def filter_files(files: Mapping[str, str], exclude: Sequence[str] = EXCLUDE_DEFAULT, max_bytes: int = 256_000) -> dict[str, str]:
    return {p: t for p, t in files.items()
            if not any(e in "/" + p for e in exclude) and len(t.encode()) <= max_bytes and t.strip()}


def select_to_tokens(files: Mapping[str, str], count_tokens: Callable[[str], int], target: int) -> list[str]:
    """Deterministic selection (sha256 of the path order) until `target` tokens are reached."""
    order = sorted(files, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    out, n = [], 0
    for p in order:
        if n >= target:
            break
        out.append(p)
        n += count_tokens(files[p])
    return sorted(out)


def overlap_report(files: Mapping[str, str], retention: Mapping[str, str], min_len: int = 20) -> dict:
    """Shared basenames with retention files and the share of foreign lines (stripped, >= 20 chars) that occur
    verbatim in them."""
    rl = {ln.strip() for t in retention.values() for ln in t.splitlines() if len(ln.strip()) >= min_len}
    fl = [ln.strip() for t in files.values() for ln in t.splitlines() if len(ln.strip()) >= min_len]
    rb = {Path(p).name for p in retention if Path(p).name != "__init__.py"}
    shared = sorted({Path(p).name for p in files} & rb)
    return dict(shared_basenames=shared, verbatim_line_share=sum(x in rl for x in fl) / max(1, len(fl)),
                n_lines=len(fl))


def text_density(texts: Mapping[str, str], tok) -> dict:
    """Share of tokens that are docstring / string / comment / identifier (tokclass lexical classes)."""
    import numpy as np
    from .tokclass import LEX, token_classes
    cnt = np.zeros(len(LEX))
    for p, t in texts.items():
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        c = token_classes(t, enc["offset_mapping"])
        cnt += np.bincount(c["lex"], minlength=len(LEX))
    tot = max(1.0, cnt.sum())
    return {LEX[i]: float(cnt[i] / tot) for i in range(len(LEX))} | dict(tokens=int(cnt.sum()))


def manifest(info: dict, archive: Path, selected: Sequence[str], files: Mapping[str, str], **extra) -> dict:
    return dict(format="ngram-memory/foreign-corpus/v1", package=info["name"], version=info["version"],
                sdist=info["filename"], sdist_sha256=info["sha256"], sdist_url=info["url"], license=info.get("license"),
                upload_time=info.get("upload_time"), archive_path=str(archive),
                files=[dict(path=p, sha256=hashlib.sha256(files[p].encode()).hexdigest(), bytes=len(files[p].encode()))
                       for p in selected], **extra)


__all__ = ["pypi_sdist", "download", "read_py_files", "filter_files", "select_to_tokens", "overlap_report",
           "text_density", "manifest", "EXCLUDE_DEFAULT"]
