"""Portable inputs for the self-hosted runner: default paths unchanged, overrides honoured, and the frozen stdlib
bundles reproduce the dev container's texts exactly."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory import paths as P  # noqa: E402

INPUTS = ROOT / "runner" / "inputs"
OLD_QWEN25 = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
OLD_FOREIGN = ("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/m0/eval/"
               "foreign/networkx-3.4.2")
DISTRO_STDLIB = Path("/usr/lib/python3.13")
ENV = ("NM_MINISGL_REPO", "NM_FOREIGN_DIR", "NM_STDLIB", "NM_STDLIB_R2", "HF_HOME", "HF_HUB_CACHE")


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", "/root")
    return monkeypatch


def load_ingest():
    spec = importlib.util.spec_from_file_location("ingest_cli_t", ROOT / "scripts/ingest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_defaults_are_the_historical_paths(clean_env):
    assert str(P.qwen25()) == OLD_QWEN25
    assert P.minisgl_repo() == Path("/home/user/patcarter883/minisglang-rdna4")
    assert str(P.foreign_dir()) == OLD_FOREIGN
    assert P.stdlib_anchor_dir() == DISTRO_STDLIB
    assert P.stdlib_r2_dir() is None
    assert str(P.wikitext_dir() / "train-00000-of-00001.parquet") == (
        "/root/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet")
    ING = load_ingest()
    assert str(ING.QWEN25) == OLD_QWEN25 and ING.STDLIB == DISTRO_STDLIB
    assert str(ING.WIKI) == str(P.wikitext_dir() / "train-00000-of-00001.parquet")


def test_overrides(clean_env, tmp_path):
    clean_env.setenv("HF_HOME", str(tmp_path / "hf"))
    clean_env.setenv("NM_MINISGL_REPO", str(tmp_path / "m"))
    clean_env.setenv("NM_FOREIGN_DIR", str(tmp_path / "f"))
    clean_env.setenv("NM_STDLIB", str(tmp_path / "s"))
    clean_env.setenv("NM_STDLIB_R2", str(tmp_path / "r2"))
    assert P.qwen25() == tmp_path / "hf/hub/models--Qwen--Qwen2.5-0.5B/snapshots" / P.QWEN25_REV
    assert P.minisgl_repo() == tmp_path / "m" and P.foreign_dir() == tmp_path / "f"
    assert P.stdlib_anchor_dir() == tmp_path / "s"
    from ngram_memory.eval.stage1.datasets import stdlib_dir
    assert stdlib_dir() == str(tmp_path / "r2")
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "hub2"))
    assert P.hf_snapshot("a/b", "r", "dataset") == tmp_path / "hub2/datasets--a--b/snapshots/r"


def test_bundle_lock_matches_files():
    lock = json.loads((INPUTS / "inputs.lock.json").read_text())
    for part in ("stdlib_anchor", "stdlib_r2"):
        for rel, h in lock[part]["files"].items():
            import hashlib
            assert hashlib.sha256((INPUTS / part / rel).read_bytes()).hexdigest() == h, rel
    order = json.loads((INPUTS / "stdlib_anchor" / "order.json").read_text())
    assert order["seed"] == 0 and order["cap_tokens"] >= 500_000
    assert all((INPUTS / "stdlib_anchor" / f).exists() for f in order["files"])
    grid = json.loads((ROOT / "experiments/m1b/grid.json").read_text())
    assert max(b["reference_tokens"] for b in grid["budgets"].values()) <= order["cap_tokens"]


def test_frozen_order_guard(clean_env):
    clean_env.setenv("NM_STDLIB", str(INPUTS / "stdlib_anchor"))
    ING = load_ingest()
    with pytest.raises(ValueError, match="seed"):
        ING.stdlib_anchor_files(1)
    with pytest.raises(ValueError, match="anchor tokens"):
        ING.stdlib_anchor_files(0, cap_tokens=10**9)


def _tok():
    from ngram_memory.ingest import corpus as C
    p = Path(OLD_QWEN25) / "tokenizer.json"
    if not p.exists():
        pytest.skip("Qwen2.5 tokenizer not cached")
    return C.TokenizerJSON(p)


def _anchors(monkeypatch, root: str | None, cap: int):
    if root is None:
        monkeypatch.delenv("NM_STDLIB", raising=False)
    else:
        monkeypatch.setenv("NM_STDLIB", root)
    return load_ingest().stdlib_texts(cap, 0, _tok())


@pytest.mark.skipif(not (DISTRO_STDLIB / "argparse.py").exists(), reason="dev-container stdlib absent")
@pytest.mark.parametrize("cap", [20_000, pytest.param(500_000, marks=pytest.mark.slow)])
def test_anchor_bundle_reproduces_dev_container(clean_env, cap):
    a = _anchors(clean_env, None, cap)
    b = _anchors(clean_env, str(INPUTS / "stdlib_anchor"), cap)
    assert len(a) == len(b) and sum(x.size for x in a) >= cap
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not Path("/usr/local/lib/python3.13/argparse.py").exists(), reason="dev-container stdlib absent")
def test_r2_bundle_reproduces_dev_container(clean_env):
    from ngram_memory.eval.stage1.datasets import STDLIB_R2, stdlib_dir
    a = {f: Path(stdlib_dir(), f).read_bytes() for f in STDLIB_R2}
    clean_env.setenv("NM_STDLIB_R2", str(INPUTS / "stdlib_r2"))
    b = {f: Path(stdlib_dir(), f).read_bytes() for f in STDLIB_R2}
    assert a == b


def test_ref_vocab_bundle(clean_env, tmp_path):
    import gzip
    import hashlib
    from ngram_memory.eval.stage1 import tokclass as tc
    lock = json.loads((INPUTS / "inputs.lock.json").read_text())["ref_vocab"]
    raw = gzip.decompress((INPUTS / lock["file"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == lock["json_sha256"]
    clean_env.setenv("NM_REF_VOCAB", str(INPUTS / lock["file"]))
    cache = tmp_path / "c" / "ref_vocab.json"
    ref = tc.default_reference_vocab(cache)
    assert len(ref) == lock["identifiers"] and cache.exists()
    assert tc.reference_vocab([], cache) == ref            # later calls read the written cache
    live = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/m1b/"
                "_cache/ref_vocab.json")
    if live.exists():                                       # the dev container's running M1b built the same vocab
        assert json.loads(live.read_text()) == dict(ref)


def test_ref_vocab_unset_walks_dirs(clean_env, tmp_path, monkeypatch):
    from ngram_memory.eval.stage1 import tokclass as tc
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.py").write_text("alpha_beta = gamma_delta\n")
    monkeypatch.setattr(tc, "default_reference_dirs", lambda: [str(tmp_path / "d")])
    ref = tc.default_reference_vocab(tmp_path / "r.json")
    assert ref["alpha_beta"] == 1 and ref["gamma_delta"] == 1
