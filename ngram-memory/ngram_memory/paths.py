"""Machine-specific inputs, in one place, so the same code runs in the dev container and on a self-hosted runner.

Every input is pinned by content (a git commit, a Hugging Face revision, or a sha256 in `runner/inputs.lock.json`);
only WHERE it lives differs between machines. With no environment variables set, every path below is exactly the
one the code used before this module existed, so existing runs are unaffected.

    NM_MINISGL_REPO  clone of patcarter883/minisglang-rdna4 (full history: Split A needs df29117, Split B a4a5543)
    NM_FOREIGN_DIR   the foreign-repo control corpus (scripts/fetch_foreign.py output: <dir>/corpus, manifest.json)
    NM_STDLIB        stdlib tree for the D_near KL anchors (scripts/ingest.py). A frozen bundle made by
                     `scripts/runner_inputs.py freeze` also carries `order.json`: the exact file list the dev
                     container's rglob + shuffle produced, so the runner reads the same anchor texts.
    NM_STDLIB_R2     stdlib tree for the R2 retention set (eval/stage1/datasets.py: STDLIB_R2)
    HF_HUB_CACHE / HF_HOME   standard Hugging Face cache location; models and datasets resolve at the pinned
                     revisions below.
"""
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------------------------------------- pins
QWEN25_REPO, QWEN25_REV = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET_REVS = {                       # load_dataset(..., revision=...) and parquet snapshots
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
}
FOREIGN_PACKAGE, FOREIGN_VERSION = "networkx", "3.4.2"

_DEV_FOREIGN = ("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/m0/eval/"
                "foreign/networkx-3.4.2")


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache, resolved the way huggingface_hub does (HF_HUB_CACHE, then HF_HOME/hub)."""
    if _env("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if _env("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_snapshot(repo_id: str, revision: str, repo_type: str = "model") -> Path:
    kind = {"model": "models", "dataset": "datasets"}[repo_type]
    return hf_hub_cache() / f"{kind}--{repo_id.replace('/', '--')}" / "snapshots" / revision


def qwen25() -> Path:
    """Qwen2.5-0.5B snapshot directory (the 0.5B proxy host)."""
    return hf_snapshot(QWEN25_REPO, QWEN25_REV)


def wikitext_dir() -> Path:
    """`wikitext-2-raw-v1/` inside the pinned Salesforce/wikitext snapshot (parquet files)."""
    return hf_snapshot("Salesforce/wikitext", DATASET_REVS["Salesforce/wikitext"], "dataset") / "wikitext-2-raw-v1"


def dataset_rev(repo_id: str) -> str:
    return DATASET_REVS[repo_id]


def minisgl_repo() -> Path:
    return Path(_env("NM_MINISGL_REPO") or "/home/user/patcarter883/minisglang-rdna4")


def foreign_dir() -> Path:
    return Path(_env("NM_FOREIGN_DIR") or _DEV_FOREIGN)


def stdlib_anchor_dir() -> Path:
    """Tree the D_near anchors are read from (the dev container's distro Python 3.13.18 stdlib by default)."""
    return Path(_env("NM_STDLIB") or "/usr/lib/python3.13")


def stdlib_r2_dir() -> str | None:
    """Override for the R2 stdlib tree; None means the historical lookup in eval/stage1/datasets.stdlib_dir()."""
    return _env("NM_STDLIB_R2")


__all__ = ["QWEN25_REPO", "QWEN25_REV", "DATASET_REVS", "FOREIGN_PACKAGE", "FOREIGN_VERSION", "hf_hub_cache",
           "hf_snapshot", "qwen25", "wikitext_dir", "dataset_rev", "minisgl_repo", "foreign_dir",
           "stdlib_anchor_dir", "stdlib_r2_dir"]
