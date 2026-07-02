"""CPU smoke tests — no GPU, no model downloads.

What this buys, per DISCLOSURES.md's open portability caveat: the store / tap / adapter layer is
pure PyTorch, so every push exercises it on stock CPU torch. A green run here does NOT verify the
full pipeline off ROCm (that needs a base model + GPU); it verifies imports, the ProductKeyStore
write/read mechanics, the GatedMemoryTap's exact-no-op-at-init property, and the adapter forward
shapes — the parts that break silently under API drift.

Run: pytest tests/ -v
"""
import importlib
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAM_MODULES = [
    "bind_msweep", "deep_mem_analytic", "deep_memory", "gated_tap", "m2_adapter",
    "pk_store", "pk_store_adapter", "recall_boltA", "recall_deepmem", "recall_mag",
    "recall_v1", "store_recurrence", "translator",
]


# ---- imports (package mode) -------------------------------------------------------------------
@pytest.mark.parametrize("name", CAM_MODULES)
def test_import_package_mode(name):
    """`import cam.X` must work without a GPU or a model download (loading is main()-guarded)."""
    importlib.import_module(f"cam.{name}")


# ---- both documented run styles reach argparse ------------------------------------------------
@pytest.mark.parametrize("driver", ["recall_mag", "bind_msweep", "recall_v1"])
@pytest.mark.parametrize("style", ["module", "file"])
def test_driver_help(driver, style):
    """README promises both `python -m cam.X` and `python cam/X.py`; --help exits 0 in both,
    before any model would load."""
    cmd = ([sys.executable, "-m", f"cam.{driver}"] if style == "module"
           else [sys.executable, os.path.join(REPO, "cam", f"{driver}.py")])
    r = subprocess.run(cmd + ["--help"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{cmd} --help failed:\n{r.stderr}"


# ---- ProductKeyStore write/read mechanics ------------------------------------------------------
def _store(d=64, n_sub=8):
    from cam.pk_store import ProductKeyStore
    torch.manual_seed(0)
    return ProductKeyStore(d, n_sub=n_sub, topk=4, sub_topk=2, n_heads=3)


def test_pk_store_empty_read_is_zero():
    """An empty bank reads exactly zero (RMSNorm/read_o of a zero mix) — the ablated floor's
    mechanical basis."""
    s = _store()
    V = s.init_state(2, "cpu")
    q = torch.randn(2, 5, 64)
    out, head_norms = s.read(V, q)
    assert out.shape == (2, 5, 64)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_pk_store_write_then_read_roundtrip():
    """Forward-pass delta-write into a fresh bank, then read: correct shapes, finite, non-zero,
    and slot addressing stays in range. (Semantics — recall accuracy — need the trained
    addressing geometry; this checks the mechanics only.)"""
    s = _store()
    B, A = 2, 6
    keys, vals = torch.randn(B, A, 64), torch.randn(B, A, 64)
    V = s.write(s.init_state(B, "cpu"), keys, vals)
    assert V.shape == (B, s.N, 64)
    assert torch.isfinite(V).all()
    assert V.abs().sum() > 0, "write() left the bank empty"
    slot_idx, slot_w = s._address(s.to_wkey(keys))
    assert slot_idx.min() >= 0 and slot_idx.max() < s.N
    assert torch.allclose(slot_w.sum(-1), torch.ones(B, A), atol=1e-5)
    out, _ = s.read(V, s.to_wkey(keys))
    assert out.shape == (B, A, 64)
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0, "read() of a written bank returned zero"


def test_pk_store_bf16_bank():
    """The bf16 storage bank (the VRAM lever) writes and reads finitely on CPU."""
    s = _store()
    B = 2
    V = s.init_state(B, "cpu", dtype=torch.bfloat16)
    V = s.write(V, torch.randn(B, 4, 64), torch.randn(B, 4, 64))
    assert V.dtype == torch.bfloat16
    out, _ = s.read(V, torch.randn(B, 3, 64))
    assert torch.isfinite(out).all()


# ---- GatedMemoryTap: zero-init is an EXACT no-op ----------------------------------------------
def test_gated_tap_exact_noop_at_init():
    """The load-bearing stability property: gamma=0 -> tanh(0)=0 -> the tap adds exactly nothing,
    bit-identical, in the base's own dtype."""
    from cam.gated_tap import GatedMemoryTap
    torch.manual_seed(0)
    tap = GatedMemoryTap(base_hidden=32, mem_dim=16, n_heads=4)
    h = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    assert tap(h) is h, "no bank set must be a pass-through"
    tap.set_bank(torch.randn(2, 5, 16))
    out = tap(h)
    assert out.dtype == h.dtype
    assert torch.equal(out, h), "zero-init gate must be bit-identical to the input"


# ---- PKStoreAdapter: the single-token dict bind path, end to end on random embeds -------------
class _StubDictBuilder:
    """Minimal stand-in declaring the dict-layout contract PKStoreAdapter reads off DocBuilder:
    [cargo, ':', name, '\\n'] blocks after bos+header, single-token, key@0 / value@2."""
    bos = [0]
    header = [1, 2]
    colon = [9]
    M = 3
    bind_len = 4
    key_off = 0
    val_off = 2
    qa_start = len(bos) + len(header) + M * bind_len   # QA region right after the bind block
    multitoken = False


def test_pk_adapter_inject_shapes():
    from cam.pk_store_adapter import PKStoreAdapter
    torch.manual_seed(0)
    vocab, H, mem_dim, K = 64, 32, 16, 4
    embed = torch.randn(vocab, H)
    adapter = PKStoreAdapter(embed, H, mem_dim, heads=2, chunk=4, expansion=2.0, k=K,
                             n_sub=8, topk=4, sub_topk=2)
    b = _StubDictBuilder()
    adapter.set_builder(b)
    B, S = 2, b.qa_start + 3
    ids = torch.randint(0, vocab, (B, S))
    apos = b.qa_start + 2
    bank = adapter.memory_bank(ids, seg_len=8, qa_start=b.qa_start, answer_pos=apos, carry=True)
    assert bank.shape == (B, K, mem_dim) and torch.isfinite(bank).all()
    empty = adapter.memory_bank(ids, seg_len=8, qa_start=b.qa_start, answer_pos=apos, carry=False)
    assert torch.allclose(empty, torch.zeros_like(empty), atol=1e-5), \
        "carry=False (ablated floor) must read an empty store"
    pref = adapter.inject(ids, 8, b.qa_start, apos, carry=True)
    assert pref.shape == (B, K, H)
    logits = adapter.direct_logits(pref)
    assert logits.shape == (B, vocab) and torch.isfinite(logits).all()


# ---- DeepMemory (the naive-store baseline) ------------------------------------------------------
def test_deep_memory_ingest_retrieve():
    from cam.deep_memory import DeepMemory
    torch.manual_seed(0)
    mem = DeepMemory(dim=16, chunk_size=4, heads=2, expansion=2.0)
    B = 2
    state = mem.init_state(B)
    state = mem(torch.randn(B, 8, 16), state)
    out = mem.retrieve(torch.randn(B, 3, 16), state)
    assert out.shape == (B, 3, 16) and torch.isfinite(out).all()


# ---- optional: real-tokenizer DocBuilder round-trip (needs network / HF cache) -----------------
def test_docbuilder_with_real_tokenizer():
    """Full DocBuilder path with the actual donor tokenizer. Skips cleanly when the tokenizer
    can't be fetched (offline CI); set HF_HOME caching to keep it fast when enabled."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as e:  # noqa: BLE001 — any fetch/env failure just skips
        pytest.skip(f"tokenizer unavailable: {e}")
    import numpy as np
    from cam.recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    b = DocBuilder(tok, names, cargo, M=8, seg_len=32, qa_seg=2, phrasing="dict")
    ids, ans, apos = b.build(np.random.default_rng(0), 4, local=False)
    assert ids.shape[0] == 4 and ids.shape[1] >= apos
    assert ans.shape[0] == 4
    assert int(ids[0, b.qa_start]) >= 0
