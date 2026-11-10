"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Adversarial-verifier tests for the 'formats' piece (ledger / patch / cartridge / native backend).

Each test asserts the CORRECT behaviour and is marked xfail(strict=True) with its defect id: it fails today and
will start XPASS-failing (strict) once the defect is fixed, so the integrator removes the mark with the fix.
Probe scripts and outputs: scratchpad nm/m0/verify-formats/probe*.py / probe*.out.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ngram_memory import codecs
from ngram_memory.cartridge import (CartridgeChain, CartridgeInfo, compile_chain, compile_served, config_sha256,
                                    make_cartridge_patch, read_served, write_served)
from ngram_memory.ledger import ArrayBase, EditRecord, RowLedger, ZeroBase
from ngram_memory.patch import make_patch, verify_patch
from ngram_memory.spec import graft_format, graft_key_spec


def _gfmt(**kw):
    return graft_format(128, graft_key_spec(5000, 4999, base=1000), **kw)


def test_vf1_ledger_materialise_equals_served_patch():
    rng = np.random.default_rng(0)
    N, W = 4096, 160
    bits = codecs.bf16_bits((rng.standard_normal((N, W)) * 0.02).astype(np.float32))
    led = RowLedger(N, W, ArrayBase(bits, "bf16"))
    ids = np.arange(N)
    for k in range(3):
        led.add(EditRecord(f"e{k}", ids, (rng.standard_normal((N, W)) * 0.003).astype(np.float32)))
    _, mat = led.materialise(ids, cast="bf16")
    sid, dsum = led.delta_sum()
    _, served = compile_served(sid, dsum, led.orig_rows, "bf16")  # what the ledger's own patch serves
    assert np.array_equal(codecs.bf16_bits(mat), served)  # today: 14 of 4096 rows differ by 1 bf16 ulp


def test_vf2_reset_only_increment_and_empty_export():
    fmt = _gfmt()
    p = make_cartridge_patch(fmt, np.empty(0, np.int64), np.empty((0, 8), np.float32),
                             CartridgeInfo("ns", 1, kind="increment", parent_version=0),
                             reset_row_ids=np.array([3, 4]))
    assert p.manifest.n_reset == 2
    led = RowLedger(fmt.padded_rows, 8, ZeroBase(8))
    led.add(EditRecord("a", [1], np.ones((1, 8)), mode="absolute"))
    led.remove("a")
    assert led.to_patch(fmt).row_ids.size == 0


def test_vf3_increment_reset_does_not_erase_fact():
    fmt = _gfmt()
    rows = np.array([10, 11, 12])
    c0 = make_cartridge_patch(fmt, rows, np.full((3, 8), 1.0), CartridgeInfo("ns", 0))
    f1 = make_cartridge_patch(fmt, rows[:1], np.full((1, 8), 0.5),
                              CartridgeInfo("ns", 1, kind="fact", parent_version=0))
    i2 = make_cartridge_patch(fmt, rows[:1], np.full((1, 8), 2.0),
                              CartridgeInfo("ns", 2, kind="increment", parent_version=1), reset_row_ids=rows[:1])
    try:
        ids, acc, _ = CartridgeChain([c0, f1, i2]).compose()
    except ValueError:
        return  # refusing the out-of-order chain is also a fix
    assert acc[0, 0] == 2.5  # today 2.0: the fact delta is silently dropped


def test_vf4_native_bf16_runtime_matches_serve_rule():
    torch.set_num_threads(1)
    from ngram_memory.backends.base import RowDiff
    from ngram_memory.backends.qwen4exp_native import Qwen4ExpNativeBackend
    from ngram_memory.cartridge import row_overlay_from
    from tiny_hosts import tiny_qwen4exp
    torch.manual_seed(1)
    m = tiny_qwen4exp(vocab=5000, dtype=torch.bfloat16)
    tab = m.model.layers[1].ple.ple_embedding.ngram_embedding
    ids = torch.randint(0, 4999, (2, 20), generator=torch.Generator().manual_seed(0))
    be = Qwen4ExpNativeBackend(m)
    rows = be.row_ids(ids).reshape(-1).unique()[:60]
    g = torch.Generator().manual_seed(3)
    for k, sl in enumerate((slice(0, 40), slice(20, 60))):
        r = rows[sl]
        be.apply(RowDiff(r, torch.randn(r.numel(), be.spec.row_width, generator=g) * 0.02, mode="delta",
                         edit_id=f"f{k}"))
    with torch.no_grad():
        A = m(input_ids=ids, use_cache=False).logits
    p = be.export_patch()
    be.remove()
    ov = row_overlay_from(p.row_ids, p.values, tab.num_embeddings, engine_cast=torch.bfloat16).install(tab)
    with torch.no_grad():
        S = m(input_ids=ids, use_cache=False).logits
    ov.remove()
    assert torch.equal(A, S)  # today max |dlogit| 9.8e-4; 20 of 640 row reads differ


def test_vf5_host_config_pin_ignores_runtime_fields():
    from conftest import build_host
    h = build_host("qwen3_5")
    d = h.config.to_dict()
    d2 = dict(d, transformers_version="9.9.9", use_cache=not d.get("use_cache", True), _name_or_path="/other/path")
    assert config_sha256(d) == config_sha256(d2)


def test_vf6_served_file_row_ids_are_authenticated(tmp_path):
    import json
    import struct
    fmt = _gfmt()
    c0 = make_cartridge_patch(fmt, np.array([10, 11, 12]), np.arange(24, dtype=np.float32).reshape(3, 8),
                              CartridgeInfo("ns", 0))
    path = str(tmp_path / "served.safetensors")
    write_served(path, compile_chain(CartridgeChain([c0])))
    raw = bytearray(open(path, "rb").read())
    n = struct.unpack("<Q", raw[:8])[0]
    a, b = json.loads(raw[8:8 + n])["row_ids"]["data_offsets"]
    raw[8 + n + a:8 + n + b] = np.array([20, 21, 22], "<i8").tobytes()
    open(path, "wb").write(raw)
    with pytest.raises(ValueError):
        read_served(path, fmt=fmt)


def test_vf7_verify_requires_module_fingerprint():
    fmt = _gfmt()
    p = make_patch(_gfmt(value_seed=7), np.array([1]), np.ones((1, 8)))  # made for another W_v seed
    p.manifest.module_fingerprint = p.manifest.format_fingerprint = p.manifest.module_spec = None
    with pytest.raises(ValueError):
        verify_patch(p, fmt)


def test_vf8_ledger_save_overwrites_stale_edit_files(tmp_path):
    L = str(tmp_path / "L")
    a = RowLedger(100, 4, ZeroBase(4))
    a.add(EditRecord("e", [1], np.full((1, 4), 1.0), mode="absolute"))
    a.save(L)
    b = RowLedger(100, 4, ZeroBase(4))
    b.add(EditRecord("e", [2], np.full((1, 4), 2.0), mode="absolute"))
    b.save(L)
    c = RowLedger.load(L)
    assert c.materialise()[0].tolist() == [2]  # today [1]: the first ledger's delta comes back


def test_vf9_fingerprint_canonical_over_numeric_type():
    assert _gfmt(value_scale=1).fingerprint() == _gfmt(value_scale=1.0).fingerprint()

