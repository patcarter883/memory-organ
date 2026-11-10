"""Fetch and select the foreign-repo corpus (§6.A.4 content control) and write its manifest.

    python scripts/fetch_foreign.py --package NAME --version X.Y.Z --dest DIR [--splits SPLITS.json] [--target-tokens N]

Token target defaults to the Split-A train tokens of the project (§6.A.1 document format, host tokenizer). Writes
DIR/<name>-<version>/manifest.json and the selected files under DIR/<name>-<version>/corpus/ (data only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ngram_memory import paths as _paths  # noqa: E402
QWEN25 = str(_paths.qwen25())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--host", default=QWEN25)
    ap.add_argument("--target-tokens", type=int, default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from ngram_memory.eval.stage1 import foreign as fo
    from ngram_memory.eval.stage1.datasets import R2_ML_MODELS, STDLIB_R2, stdlib_dir
    from ngram_memory.eval.stage1.splits import Splits
    tok = AutoTokenizer.from_pretrained(a.host)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    out = Path(a.dest) / f"{a.package}-{a.version}"
    info = fo.pypi_sdist(a.package, a.version)
    arch = fo.download(info, Path(a.dest) / "sdist")
    files = fo.filter_files(fo.read_py_files(arch))
    proj = None
    if a.splits:
        sp = Splits.load(a.splits)
        proj = {p: sp.text(p) for p in sp.files("a", "train")}
    target = a.target_tokens or (sum(ntok(sp.document(p)) for p in proj) if proj else 1_000_000)
    sel = fo.select_to_tokens(files, ntok, target)
    chosen = {p: files[p] for p in sel}
    import transformers, os
    sd = stdlib_dir()
    ret = {f"stdlib/{f}": Path(sd, f).read_text(errors="replace") for f in STDLIB_R2 if Path(sd, f).exists()}
    mb = Path(os.path.dirname(transformers.__file__)) / "models"
    ret.update({f"transformers/{m}": (mb / m / f"modeling_{m}.py").read_text() for m in R2_ML_MODELS
                if (mb / m / f"modeling_{m}.py").exists()})
    ov = fo.overlap_report(chosen, ret)
    dens = fo.text_density(chosen, tok)
    pdens = fo.text_density(proj, tok) if proj else None
    man = fo.manifest(info, arch, sel, chosen, target_tokens=target, selected_tokens=sum(ntok(t) for t in chosen.values()),
                      available_files=len(files), available_tokens=sum(ntok(t) for t in files.values()),
                      overlap_with_retention=ov, density=dens, project_train_density=pdens)
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    for p, t in chosen.items():
        q = out / "corpus" / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(t)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
