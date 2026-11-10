#!/usr/bin/env python
"""Pin, fetch and verify every external input of the CPU experiments, so a self-hosted runner reads the same bytes
as the dev container.

    python scripts/runner_inputs.py freeze            # dev container only: write runner/inputs/ (bundles + lock)
    python scripts/runner_inputs.py fetch --home DIR  # runner: download / clone everything the lock names
    python scripts/runner_inputs.py check --home DIR [--strict-packages]   # verify against the lock; exit 1 on a mismatch
    python scripts/runner_inputs.py env --home DIR    # print the NM_* / HF_* exports for DIR

What is pinned, and how:
  * Qwen2.5-0.5B and the four datasets (WikiText-2, ARC-Challenge, MMLU, HellaSwag): Hugging Face revisions
    (ngram_memory/paths.py), plus sha256 of every file the code reads.
  * minisglang-rdna4: the two commits the splits name (git object ids are content hashes).
  * Foreign-repo control (networkx 3.4.2): the PyPI sdist's sha256 and the sha256 of each selected file.
  * The two Python-stdlib text sets. These are NOT portable: the dev container's D_near anchors come from the
    distro's /usr/lib/python3.13 (libpython3.13-stdlib 3.13.18-1+noble1, Debian-patched) and the R2 set from
    /usr/local/lib/python3.13. `freeze` copies the exact files (and, for the anchors, the reading order the
    seeded shuffle produced) into runner/inputs/, which is committed. PSF licence: runner/inputs/*/LICENSE.txt.
  * The identifier reference vocabulary of the token classes (eval/stage1/tokclass.py): built from the
    interpreter's own stdlib tree including its site-packages, so it is frozen too (ref_vocab.json.gz).
  * Packages: runner/requirements-lock.txt pins versions and sha256 of every file, and the workflow installs it
    with --require-hashes; `check` compares installed versions (a warning unless --strict-packages).

Bit-identical training across machines is not a goal (a different CPU or thread count changes float summation
order); identical inputs are.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory import paths as P  # noqa: E402

INPUTS = ROOT / "runner" / "inputs"
LOCK = INPUTS / "inputs.lock.json"
ANCHOR_CAP_TOKENS = 600_000          # grid.json full budget reads 500,000 (reference_tokens); margin for reruns
ANCHOR_SEED = 0                      # experiments/m1b/run.py: ING.stdlib_texts(reference_tokens, 0, tok)
MINISGL_URL = "https://github.com/patcarter883/minisglang-rdna4.git"
DATASET_FILES = {                    # the files load_dataset / the ingest CLI read, inside each pinned snapshot
    "Salesforce/wikitext": ["wikitext-2-raw-v1/train-00000-of-00001.parquet",
                            "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                            "wikitext-2-raw-v1/test-00000-of-00001.parquet"],
    "allenai/ai2_arc": ["ARC-Challenge/train-00000-of-00001.parquet",
                        "ARC-Challenge/validation-00000-of-00001.parquet",
                        "ARC-Challenge/test-00000-of-00001.parquet"],
    "cais/mmlu": ["all/test-00000-of-00001.parquet", "all/validation-00000-of-00001.parquet",
                  "all/dev-00000-of-00001.parquet", "all/auxiliary_train-00000-of-00001.parquet"],
    "Rowan/hellaswag": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
                        "data/test-00000-of-00001.parquet"],
}
QWEN25_FILES = ["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "vocab.json"]
PACKAGES = ["torch", "transformers", "tokenizers", "numpy", "safetensors", "datasets", "pyarrow", "huggingface_hub",
            "accelerate", "pytest", "pytest-xdist", "regex"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(root: Path, names) -> dict[str, str]:
    return {n: sha256(root / n) for n in names}


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in PACKAGES:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def load_ingest_cli():
    spec = importlib.util.spec_from_file_location("ingest_cli", ROOT / "scripts/ingest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


# ================================================================================================= freeze
def freeze(a) -> int:
    from ngram_memory.eval.stage1.datasets import STDLIB_R2, stdlib_dir
    from ngram_memory.ingest import corpus as C
    if any(os.environ.get(k) for k in ("NM_STDLIB", "NM_STDLIB_R2", "NM_REF_VOCAB")):
        raise SystemExit("freeze reads the dev container's own trees: unset NM_STDLIB / NM_STDLIB_R2 / NM_REF_VOCAB")
    ING = load_ingest_cli()
    tok = C.TokenizerJSON(P.qwen25() / "tokenizer.json")
    src = Path(ING.STDLIB)
    anchor = INPUTS / "stdlib_anchor"
    r2 = INPUTS / "stdlib_r2"
    for d in (anchor, r2):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    # the anchor prefix, in the order stdlib_texts reads it; files that fail utf-8 decoding are kept (skipped
    # identically on the runner) so the order list is the exact sequence of reads
    files, n = [], 0
    for p in ING.stdlib_anchor_files(ANCHOR_SEED, src):
        rel = str(p.relative_to(src))
        q = anchor / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        files.append(rel)
        try:
            n += int(tok.encode(p.read_text(encoding="utf-8")).size)
        except Exception:
            pass
        if n >= ANCHOR_CAP_TOKENS:
            break
    shutil.copyfile(src / "LICENSE.txt", anchor / "LICENSE.txt")
    (anchor / "order.json").write_text(json.dumps(dict(
        seed=ANCHOR_SEED, cap_tokens=ANCHOR_CAP_TOKENS, tokens=n, source=str(src),
        source_package=_dpkg_version("libpython3.13-stdlib"), files=files), indent=1) + "\n")
    r2src = Path(stdlib_dir())
    for f in STDLIB_R2:
        q = r2 / f
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r2src / f, q)
    shutil.copyfile(r2src / "LICENSE.txt", r2 / "LICENSE.txt")
    (r2 / "SOURCE.json").write_text(json.dumps(dict(source=str(r2src), python=platform.python_version(),
                                                    files=list(STDLIB_R2)), indent=1) + "\n")
    # reference vocabulary for the token classes (tokclass.default_reference_dirs walks this interpreter's stdlib
    # tree, site-packages included, which no other machine has): canonical JSON, gzip with a fixed mtime
    import gzip
    from ngram_memory.eval.stage1 import tokclass as tc
    ref = tc.reference_vocab(tc.default_reference_dirs())
    blob = json.dumps(dict(sorted(ref.items())), separators=(",", ":")).encode("utf-8")
    (INPUTS / "ref_vocab.json.gz").write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))
    # foreign manifest (selected files with sha256)
    fdir = P.foreign_dir()
    man = json.loads((fdir / "manifest.json").read_text())
    (INPUTS / "foreign").mkdir(exist_ok=True)
    keep = {k: man[k] for k in ("format", "package", "version", "sdist", "sdist_sha256", "sdist_url", "files")}
    (INPUTS / "foreign" / "manifest.json").write_text(json.dumps(keep, indent=1) + "\n")
    # minisgl commits
    repo = P.minisgl_repo()
    for c in (C.MINISGL_HEAD, C.MINISGL_T_CUT):
        git(repo, "cat-file", "-e", f"{c}^{{commit}}")
    lock = dict(
        format="ngram-memory/runner-inputs/v1",
        python=platform.python_version(),
        packages=package_versions(),
        qwen25=dict(repo=P.QWEN25_REPO, revision=P.QWEN25_REV, files=tree_hashes(P.qwen25(), QWEN25_FILES)),
        datasets={r: dict(revision=P.DATASET_REVS[r],
                          files=tree_hashes(P.hf_snapshot(r, P.DATASET_REVS[r], "dataset"), fs))
                  for r, fs in DATASET_FILES.items()},
        minisgl=dict(url=MINISGL_URL, head=C.MINISGL_HEAD, t_cut=C.MINISGL_T_CUT),
        foreign=dict(package=man["package"], version=man["version"], sdist_url=man["sdist_url"],
                     sdist_sha256=man["sdist_sha256"], n_files=len(man["files"])),
        stdlib_anchor=dict(tokens=n, n_files=len(files), files=_bundle_hashes(anchor)),
        stdlib_r2=dict(files=_bundle_hashes(r2)),
        ref_vocab=dict(file="ref_vocab.json.gz", sha256=sha256(INPUTS / "ref_vocab.json.gz"), identifiers=len(ref),
                       json_sha256=hashlib.sha256(blob).hexdigest()),
    )
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    size = sum(f.stat().st_size for f in INPUTS.rglob("*") if f.is_file())
    print(f"froze {len(files)} anchor files ({n:,} tokens), {len(STDLIB_R2)} R2 files; {size / 2**20:.2f} MiB in {INPUTS}")
    return 0


def _bundle_hashes(d: Path) -> dict[str, str]:
    return {str(f.relative_to(d)): sha256(f) for f in sorted(d.rglob("*")) if f.is_file()}


def _dpkg_version(pkg: str) -> str | None:
    try:
        return subprocess.run(["dpkg-query", "-W", "-f=${Version}", pkg], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except Exception:
        return None


# ================================================================================================== fetch
def home_paths(home: Path) -> dict[str, str]:
    """The environment a job on the runner exports (the workflow sets the same values)."""
    return dict(HF_HOME=str(home / "hf"), NM_MINISGL_REPO=str(home / "minisglang-rdna4"),
                NM_FOREIGN_DIR=str(home / "foreign" / f"{P.FOREIGN_PACKAGE}-{P.FOREIGN_VERSION}"),
                NM_STDLIB=str(INPUTS / "stdlib_anchor"), NM_STDLIB_R2=str(INPUTS / "stdlib_r2"),
                NM_REF_VOCAB=str(INPUTS / "ref_vocab.json.gz"))


def _apply_env(home: Path) -> None:
    for k, v in home_paths(home).items():
        os.environ.setdefault(k, v)
    if os.environ["HF_HOME"] != str(home / "hf"):
        print(f"note: HF_HOME is {os.environ['HF_HOME']} (not under --home)")


def fetch(a) -> int:
    home = Path(a.home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    _apply_env(home)
    lock = json.loads(LOCK.read_text())
    from huggingface_hub import snapshot_download
    q = lock["qwen25"]
    snapshot_download(q["repo"], revision=q["revision"], allow_patterns=list(q["files"]))
    print("qwen25:", P.qwen25())
    for r, d in lock["datasets"].items():
        snapshot_download(r, repo_type="dataset", revision=d["revision"],
                          allow_patterns=list(d["files"]) + ["README.md", "dataset_infos.json", "*.py"])
        print("dataset:", r, d["revision"][:7])
    repo = Path(os.environ["NM_MINISGL_REPO"])
    m = lock["minisgl"]
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", m["url"], str(repo)], check=True)
    for c in (m["head"], m["t_cut"]):
        try:
            git(repo, "cat-file", "-e", f"{c}^{{commit}}")
        except subprocess.CalledProcessError:
            subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "origin", c], check=True)
    print("minisgl:", repo)
    _fetch_foreign(lock["foreign"], Path(os.environ["NM_FOREIGN_DIR"]))
    return check(a)


def _fetch_foreign(f: dict, out: Path) -> None:
    from ngram_memory.eval.stage1 import foreign as fo
    man = json.loads((INPUTS / "foreign" / "manifest.json").read_text())
    info = dict(name=f["package"], version=f["version"], filename=man["sdist"], url=f["sdist_url"],
                sha256=f["sdist_sha256"])
    arch = fo.download(info, out.parent / "sdist")           # verifies the sdist sha256
    texts = fo.read_py_files(arch)
    for e in man["files"]:
        t = texts[e["path"]]
        if hashlib.sha256(t.encode()).hexdigest() != e["sha256"]:
            raise SystemExit(f"foreign file {e['path']} does not match the manifest")
        q = out / "corpus" / e["path"]
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(t, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print("foreign:", out, len(man["files"]), "files")


# ================================================================================================== check
def check(a) -> int:
    home = Path(a.home).resolve() if a.home else None
    if home is not None:
        _apply_env(home)
    lock = json.loads(LOCK.read_text())
    bad, warn = [], []

    def cmp(label: str, root: Path, want: dict[str, str]) -> None:
        for n, h in want.items():
            p = root / n
            if not p.exists():
                bad.append(f"{label}: missing {p}")
            elif sha256(p) != h:
                bad.append(f"{label}: sha256 mismatch {p}")

    cmp("qwen25", P.qwen25(), lock["qwen25"]["files"])
    for r, d in lock["datasets"].items():
        cmp(r, P.hf_snapshot(r, d["revision"], "dataset"), d["files"])
    repo = P.minisgl_repo()
    for c in (lock["minisgl"]["head"], lock["minisgl"]["t_cut"]):
        try:
            git(repo, "cat-file", "-e", f"{c}^{{commit}}")
        except Exception:
            bad.append(f"minisgl: commit {c[:7]} not in {repo}")
    fdir = P.foreign_dir()
    man = json.loads((INPUTS / "foreign" / "manifest.json").read_text())
    for e in man["files"]:
        p = fdir / "corpus" / e["path"]
        if not p.exists() or hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() != e["sha256"]:
            bad.append(f"foreign: {e['path']} missing or changed")
            break
    cmp("stdlib_anchor", INPUTS / "stdlib_anchor", lock["stdlib_anchor"]["files"])
    cmp("stdlib_r2", INPUTS / "stdlib_r2", lock["stdlib_r2"]["files"])
    cmp("ref_vocab", INPUTS, {lock["ref_vocab"]["file"]: lock["ref_vocab"]["sha256"]})
    for k, want in (("NM_STDLIB", INPUTS / "stdlib_anchor"), ("NM_STDLIB_R2", INPUTS / "stdlib_r2"),
                    ("NM_REF_VOCAB", INPUTS / "ref_vocab.json.gz")):
        got = os.environ.get(k)
        if home is not None and Path(got or "") != want:
            bad.append(f"{k} is {got!r}, expected {want}")
    have = package_versions()
    for n, v in lock["packages"].items():
        if have.get(n) != v:
            (bad if a.strict_packages else warn).append(f"package {n}: {have.get(n)} (lock {v})")
    if platform.python_version() != lock["python"]:
        warn.append(f"python {platform.python_version()} (lock {lock['python']}); stdlib texts come from the bundle")
    for w in warn:
        print("WARN", w)
    for b in bad:
        print("FAIL", b)
    print("inputs:", "OK" if not bad else f"{len(bad)} problem(s)")
    return 0 if not bad else 1


def env(a) -> int:
    for k, v in home_paths(Path(a.home).resolve()).items():
        print(f"{k}={v}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    for name in ("fetch", "check", "env"):
        p = sub.add_parser(name)
        p.add_argument("--home", required=(name != "check"), default=None)
        p.add_argument("--strict-packages", action="store_true")
    a = ap.parse_args()
    return dict(freeze=freeze, fetch=fetch, check=check, env=env)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
