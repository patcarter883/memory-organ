"""CPU tests for the E0 persistence probe (cam/persist_probe.py) — mechanics only.

Semantics (does recall survive age?) need the trained addressing geometry and run on the GPU box;
these tests pin the protocol: fact drawing, the write/probe tensor paths on a real (untrained)
PKStoreAdapter, the exact-zero empty-bank floor, and the age-pooling bookkeeping.
"""
import numpy as np
import pytest
import torch

from cam.persist_probe import draw_facts, probe_facts, run_e0, write_facts
from cam.pk_store_adapter import PKStoreAdapter

COLON = [9]


def _adapter(vocab=64, H=32, mem_dim=16, k=4):
    torch.manual_seed(0)
    embed = torch.randn(vocab, H)
    return PKStoreAdapter(embed, H, mem_dim, heads=2, chunk=4, expansion=2.0, k=k,
                          n_sub=8, topk=4, sub_topk=2)


def test_draw_facts_unique_keys():
    rng = np.random.default_rng(0)
    facts = draw_facts(rng, key_pool=list(range(10, 30)), val_pool=list(range(40, 50)), n=15)
    assert facts.shape == (15, 2)
    assert len(set(facts[:, 0].tolist())) == 15, "keys must be unique"
    assert all(40 <= v < 50 for v in facts[:, 1].tolist())
    with pytest.raises(AssertionError):
        draw_facts(rng, key_pool=list(range(5)), val_pool=[7], n=6)   # more facts than keys


def test_write_probe_mechanics():
    a = _adapter()
    rng = np.random.default_rng(1)
    facts = draw_facts(rng, key_pool=list(range(10, 30)), val_pool=list(range(40, 50)), n=6)
    V = a.store.init_state(1, "cpu", dtype=torch.float32)
    V = write_facts(a, V, facts)
    assert V.shape == (1, a.store.N, a.mem_dim) and torch.isfinite(V).all()
    assert V.abs().sum() > 0
    hits = probe_facts(a, V, facts, COLON, batch=4)     # batch < F exercises the chunking
    assert hits.shape == (6,) and set(hits.tolist()) <= {0.0, 1.0}


def test_empty_bank_floor_is_exact():
    """An empty bank reads exactly zero -> zero logits -> argmax is token 0 for every probe, so the
    floor is 0 whenever token id 0 is not a value. This is the mechanical basis of the E0 floor."""
    a = _adapter()
    rng = np.random.default_rng(2)
    facts = draw_facts(rng, key_pool=list(range(10, 30)), val_pool=list(range(40, 50)), n=8)
    V0 = a.store.init_state(1, "cpu", dtype=torch.float32)
    assert float(probe_facts(a, V0, facts, COLON).mean()) == 0.0


def test_run_e0_structure_and_age_pooling():
    a = _adapter()
    rng = np.random.default_rng(3)
    E, M = 3, 4
    facts = draw_facts(rng, key_pool=list(range(10, 40)), val_pool=list(range(40, 50)), n=E * M)
    res = run_e0(a, list(facts.split(M)), COLON)

    assert res["episodes"] == E and res["facts_per_episode"] == [M] * E
    assert len(res["per_probe"]) == E
    assert [len(row) for row in res["per_probe"]] == [1, 2, 3], "per_probe must be triangular"
    assert sorted(res["age_acc"]) == [0, 1, 2]
    assert len(res["fresh_acc"]) == E
    for v in list(res["age_acc"].values()) + res["fresh_acc"] + [res["batch_acc"], res["empty_acc"]]:
        assert 0.0 <= v <= 1.0
    assert res["empty_acc"] == 0.0

    # age pooling: with constant M, age_acc[a] must equal the plain mean of per_probe[e][e-a]
    for age in range(E):
        expect = np.mean([res["per_probe"][e][e - age] for e in range(age, E)])
        assert abs(res["age_acc"][age] - expect) < 1e-9
