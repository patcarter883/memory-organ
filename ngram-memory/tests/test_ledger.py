"""Ledger rollback (port of P/solver/v5_ledger_bf16.py) and ledger semantics (§3.3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ngram_memory import codecs
from ngram_memory.ledger import ArrayBase, EditRecord, RowLedger, ZeroBase


def _v5_data():
    """The exact random draws of P/solver/v5_ledger_bf16.py (torch.manual_seed(0), same call sequence)."""
    torch.manual_seed(0)
    rows, w = 64, 160
    orig = (torch.randn(rows, w) * 0.09 / 160 ** 0.5).bfloat16()
    for s in (0.003, 0.03, 0.3, 1.0, 5.0):  # consume the same RNG draws as the proof's first loop
        torch.randn(rows, w)
    ids = [torch.arange(0, 40), torch.arange(20, 60), torch.arange(10, 30)]
    U = [torch.randn(len(i), w) * 0.02 for i in ids]
    return orig, ids, U


def test_v5_naive_rollback_residue_vs_ledger_exact():
    orig, ids, U = _v5_data()
    # naive in-place subtraction (the proof's reference failure: 4.88e-04 on 50/64 rows)
    naive = orig.clone()
    for i, u in zip(ids, U):
        naive[i] = (naive[i].float() + u).bfloat16()
    naive[ids[1]] = (naive[ids[1]].float() - U[1]).bfloat16()

    def ref_materialise(active):
        acc = orig.float().clone()
        for k in active:
            acc.index_add_(0, ids[k], U[k])
        return acc.bfloat16()

    ref = ref_materialise([0, 2])
    naive_max = (naive.float() - ref.float()).abs().max().item()
    naive_rows = int((naive != ref).any(1).sum())
    assert naive_rows == 50 and abs(naive_max - 4.88e-4) < 1e-6, (naive_max, naive_rows)

    # the ledger: bf16 base bits, fp32 deltas, re-materialise
    bits = orig.view(torch.int16).numpy().view(np.uint16)
    led = RowLedger(64, 160, ArrayBase(bits, "bf16"))
    for k in range(3):
        led.add(EditRecord(f"e{k}", ids[k].numpy(), U[k].numpy()))
    led.remove("e1")
    allrows = np.arange(64)
    _, got = led.materialise(allrows, cast="bf16")
    assert np.array_equal(got, ref.float().numpy()), "ledger rollback is not bit-exact"
    # order-independence: apply in another order
    led2 = RowLedger(64, 160, ArrayBase(bits, "bf16"))
    for k in (2, 0):
        led2.add(EditRecord(f"e{k}", ids[k].numpy(), U[k].numpy()))
    assert np.array_equal(led2.materialise(allrows)[1], got)
    # remove everything: every row returns to its original bits
    led.remove("e0")
    led.remove("e2")
    _, back = led.materialise(allrows, cast="bf16")
    assert np.array_equal(codecs.bf16_bits(back), bits)
    assert led.active_rows().size == 0


def test_conflicts_dependents_and_changed_rows():
    led = RowLedger(100, 4)
    led.add(EditRecord("a", [1, 2, 3], np.ones((3, 4))))
    led.add(EditRecord("b", [3, 4], np.ones((2, 4))))
    ep = led.add(EditRecord("c", [50], np.ones((1, 4))))
    led.add(EditRecord("d", [2, 60], np.ones((2, 4))))
    assert led.records["b"].conflict_set == ["a"]
    assert led.records["c"].conflict_set == []
    assert led.conflict_set("a") == ["b", "d"]
    assert led.dependents("a") == ["b", "d"] and led.dependents("d") == []
    assert led.conflict_sets()["c"] == []
    assert led.rows_changed_since(ep).tolist() == [2, 60]
    led.remove("b")
    assert led.conflict_set("a") == ["d"]
    assert led.edits_touching([3, 4]) == ["a"]
    with pytest.raises(KeyError):
        led.remove("b")
    led.reactivate("b")
    assert led.edits_touching([4]) == ["b"]


def test_graft_absolute_zero_base_and_partial_columns():
    led = RowLedger(10, 8, ZeroBase(8))
    led.add(EditRecord("x", [3, 5], np.full((2, 8), 0.25), mode="absolute"))
    led.add(EditRecord("y", [5], np.full((1, 3), 1.0), col0=2, ncols=3, mode="absolute"))
    ids, rows = led.materialise(cast=None)
    assert ids.tolist() == [3, 5]
    assert rows[1].tolist() == [0.25, 0.25, 1.25, 1.25, 1.25, 0.25, 0.25, 0.25]
    assert led.records["x"].base_bits_sha256 is None
    with pytest.raises(ValueError):
        RowLedger(10, 8, ArrayBase(np.zeros((10, 8), np.float32), "fp32")).add(
            EditRecord("z", [1], np.ones((1, 8)), mode="absolute"))
    with pytest.raises(ValueError):
        led.add(EditRecord("bad", [1], np.ones((1, 4)), col0=6, ncols=4))


def test_base_bits_saved_on_first_write_survive_base_rewrite(tmp_path):
    rng = np.random.default_rng(0)
    tab = rng.standard_normal((32, 32)).astype(np.float32)
    raw = codecs.encode_rows(tab, "int6_g32")
    base = ArrayBase(raw.copy(), "int6_g32", 32)
    led = RowLedger(32, 32, base, spec_fingerprint="F")
    d = rng.standard_normal((4, 32)).astype(np.float32) * 0.1
    led.add(EditRecord("e1", [0, 7, 9, 31], d, spec_fingerprint="F"))
    sha = led.records["e1"].base_bits_sha256
    assert len(sha) == 4 and len(set(sha)) == 4
    base.raw[:] = 0  # someone rewrites the base table: the ledger still restores the original bits
    ids, got = led.materialise(cast=None)
    want = codecs.decode_rows(raw[ids], "int6_g32", 32) + d
    assert np.array_equal(got, want)
    with pytest.raises(ValueError):
        led.add(EditRecord("e2", [1], d[:1], spec_fingerprint="OTHER"))
    # save / load round trip is exact
    led.add(EditRecord("e3", [7, 8], d[:2] * 2))
    led.remove("e1")
    led.save(str(tmp_path / "L"))
    led2 = RowLedger.load(str(tmp_path / "L"))
    assert led2.epoch == led.epoch and led2.active == led.active and led2.log == led.log
    a, b = led.materialise([7, 8]), led2.materialise([7, 8])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(led2.orig_bits([0, 7]), raw[[0, 7]])
    led2.reactivate("e1")
    assert np.array_equal(led2.materialise([0])[1], codecs.bf16_round(want[:1]))


def test_compiled_overlay_bits_equal_torch_cast():
    rng = np.random.default_rng(1)
    tab = torch.randn(50, 16).bfloat16()
    led = RowLedger(50, 16, ArrayBase(tab.view(torch.int16).numpy().view(np.uint16), "bf16"))
    d = (rng.standard_normal((5, 16)) * 1e-2).astype(np.float32)
    led.add(EditRecord("e", [1, 4, 9, 16, 25], d))
    ids, bits = led.compiled_overlay()
    want = (tab[torch.from_numpy(ids)].float() + torch.from_numpy(d)).bfloat16()
    assert np.array_equal(bits, want.view(torch.int16).numpy().view(np.uint16))
