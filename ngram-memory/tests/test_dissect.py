"""§D G/S library tests: n-gram counting and row statistics (numpy), and the 2-layer slice (Layer0 + PLEProbe)
against transformers' own qwen4_exp decoder layer 0 and PLE layer on a tiny random host."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from ngram_memory.dissect import row_stats as rs


# ------------------------------------------------------------------------------------------------ row_stats
def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    V = 248320
    g = rng.integers(0, V, size=(1000, 3))
    assert np.array_equal(rs.unpack(rs.pack(g, V), 3, V), g)
    assert rs.pack(np.array([[V - 1, V - 1, V - 1]]), V)[0] > 0  # no int64 overflow at the top of the range


def test_count_keys_matches_counter():
    rng = np.random.default_rng(1)
    V = 50
    docs = [rng.integers(0, V, size=int(rng.integers(0, 40))) for _ in range(30)]
    for n in (1, 2, 3):
        k, c = rs.count_keys(docs, n, V, chunk_tokens=37)  # force several merges
        ref = Counter()
        for d in docs:
            for i in range(len(d) - n + 1):
                ref[tuple(int(x) for x in d[i:i + n])] += 1
        got = {tuple(int(x) for x in g): int(cc) for g, cc in zip(rs.unpack(k, n, V), c)}
        assert got == dict(ref)
        assert np.all(np.diff(k) > 0)
        q = rs.pack(np.array([list(next(iter(ref)))] + [[V - 1] * n]), V)
        lc = rs.lookup_counts(k, c, q)
        assert lc[0] == ref[next(iter(ref))]


def test_coherence_and_snr():
    rng = np.random.default_rng(2)
    s = rng.normal(size=64)
    assert rs.coherence(np.stack([s] * 8)) == pytest.approx(1.0)
    noise = rng.normal(size=(8, 20000))
    assert abs(rs.coherence(noise)) < 0.01
    # signal share rho in each head -> coherence ~ rho
    sig = rng.normal(size=20000)
    rho = 0.2
    c = np.sqrt(rho) * sig + np.sqrt(1 - rho) * rng.normal(size=(8, 20000))
    assert rs.coherence(c) == pytest.approx(rho, abs=0.02)
    assert rs.snr_from_coherence(0.2) == pytest.approx(2.0)


def test_contributions_centring():
    rng = np.random.default_rng(3)
    wv = rng.normal(size=(6, 4 * 5))
    wh = rs.head_slices(wv, 4)
    rows = rng.normal(size=(2, 5))
    mu = rng.normal(size=(4, 5))
    c = rs.contributions(rows, [1, 3], wh, mu)
    assert np.allclose(c[1], wv[:, 15:20] @ (rows[1] - mu[3]))


def test_crossing_and_bootstrap():
    x = np.log10([1e-8, 1e-7, 1e-6, 1e-5])
    assert rs.crossing(x, np.array([0.0, 0.02, 0.08, 0.15]), 0.05) == pytest.approx(x[1] + 0.5 * (x[2] - x[1]))
    assert np.isnan(rs.crossing(x, np.array([0.0, 0.01, 0.02, 0.03]), 0.05))
    # a dip below the threshold after a first crossing: the LAST upward crossing counts
    assert rs.crossing(x, np.array([0.06, 0.01, 0.06, 0.2]), 0.05) == pytest.approx(x[1] + (0.04 / 0.05) * (x[2] - x[1]))
    rng = np.random.default_rng(4)
    groups = [rng.normal(m, 0.01, size=50) for m in (0.0, 0.02, 0.08, 0.15)]
    est, lo, hi, share = rs.bootstrap_crossing(rs.BinCurve(np.arange(4), np.arange(4), x, groups), 0.05,
                                               n_boot=200, rng=rng)
    assert lo <= est <= hi and share == 1.0
    m, a, b = rs.bootstrap_mean_ci(groups[2], n_boot=500)
    assert a < m < b


def test_stratified_sample_and_bins():
    assert rs.log2_bins(9) == [(1, 1), (2, 3), (4, 7), (8, 15)]
    keys = np.arange(100, dtype=np.int64)
    counts = np.repeat([1, 2, 5, 9], 25)
    out = rs.stratified_sample(keys, counts, rs.log2_bins(9), 10, np.random.default_rng(0), exclude=np.array([0, 1]))
    assert [len(o[2]) for o in out] == [10, 10, 10, 10]
    assert not np.isin(out[0][2], [0, 1]).any()


def test_value_share():
    v2 = np.array([[1.0, 0.0], [1.0, 1.0]])
    v3 = np.array([[0.0, 1.0], [1.0, 1.0]])
    s = rs.value_share(v2, v3)
    assert np.allclose(s["proj2"] + s["proj3"], 1.0)
    assert np.allclose(s["proj2"], [0.5, 0.5])


def test_wikitext_detok_and_articles():
    s = rs.wikitext_detokenize(" a role @-@ playing game , and ( Japanese : x ) . \n")
    assert "role-playing" in s and "game, and" in s and "(Japanese: x)" in s
    arts = list(rs.wikitext_articles([" = A = \n", " text \n", " = = Sub = = \n", " = B = \n", " more \n"]))
    assert len(arts) == 2 and arts[1].startswith(" = B")


# --------------------------------------------------------------------------------------------- slice_probe
from ngram_memory.dissect.slice_probe import (L0, PLE, Layer0, PLEProbe, SliceWeights, gdir_alpha_for_ratio,  # noqa: E402
                                              token_stats, variance_decomposition)


def test_variance_decomposition():
    v = np.array([1.0, 1.0, 3.0, 3.0, 9.0])
    g = np.array([0, 0, 1, 1, 2])
    r = variance_decomposition(v, g)  # singleton group 2 dropped
    assert r["n"] == 4 and r["within_share"] == pytest.approx(0.0)
    r = variance_decomposition(np.array([0.0, 2.0, 0.0, 2.0]), np.array([0, 0, 1, 1]))
    assert r["within_share"] == pytest.approx(1.0)


def test_gdir_alpha_for_ratio():
    z = np.random.default_rng(0).normal(size=20000)
    a = gdir_alpha_for_ratio(z, 0.3)
    x = a * z
    assert np.sqrt(np.mean((x / (1 + np.exp(-x))) ** 2)) == pytest.approx(0.3, rel=1e-3)


@pytest.fixture(scope="module")
def tiny():
    from tiny_hosts import tiny_qwen4exp
    torch.set_num_threads(1)
    m = tiny_qwen4exp(seed=0)
    return m


def test_layer0_and_ple_probe_match_q4(tiny):
    m = tiny
    tm = m.model
    cfg = tm.config
    torch.manual_seed(1)
    T = 37
    ids = torch.randint(0, cfg.vocab_size - 1, (1, T))
    grabbed = {}

    def pre_hook(mod, args, kwargs):
        grabbed["x1"] = args[0].detach().clone() if args else kwargs["hidden_states"].detach().clone()

    def ple_hook(mod, args, out):
        grabbed["ple"] = out.detach().clone()

    h1 = tm.layers[1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = tm.layers[1].ple.register_forward_hook(ple_hook)
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    h1.remove()
    h2.remove()
    tensors = {L0 + k: v for k, v in tm.layers[0].state_dict().items()}
    tensors.update({PLE + k: v for k, v in tm.layers[1].ple.state_dict().items() if "ple_embedding" not in k})
    w = SliceWeights.from_tensors(tensors)
    l0 = Layer0(w, cfg.to_dict())
    emb = tm.embed_tokens(ids)[0].detach()
    x1 = l0.forward_docs([emb])[0]
    assert (x1 - grabbed["x1"][0]).abs().max() < 1e-5
    d = cfg.hidden_size
    probe = PLEProbe(w, hidden=d, hc=cfg.hc_count, n_heads=16)
    with torch.no_grad():
        e = tm.layers[1].ple.ple_embedding(ids, None)[0]
    p = probe.parts(x1, e)
    assert (p["out"] - grabbed["ple"][0]).abs().max() < 1e-5
    st = token_stats(probe, x1, e)
    assert st["sig"].shape == (T, cfg.hc_count) and st["tap_rms"].shape == (T, cfg.hc_count, 4)
    # leave-one-order-out shares of v sum to 1; zeroing both orders kills the value
    assert np.allclose(st["proj2"] + st["proj3"], 1.0, atol=1e-4)
    # direct term == sigmoid(g) * |v| / sqrt(d) per stream
    assert np.allclose(st["direct_rms"], st["sig"] * st["v_norm"][:, None] / np.sqrt(d), rtol=1e-4)


def test_allocate_water_filling():
    assert rs.allocate([10, 1000, 1000], 300) == [10, 145, 145]
    assert sum(rs.allocate([3, 2, 1], 100)) == 6
    keys = np.arange(30, dtype=np.int64)
    counts = np.repeat([1, 2, 4], 10)
    out = rs.stratified_sample(keys, counts, rs.log2_bins(4), None, np.random.default_rng(0), total=12)
    assert [len(o[2]) for o in out] == [4, 4, 4]


# ---------------------------------------------------------------------------------------------- hf_range
from ngram_memory.dissect.hf_range import FlashNextRowFetcher, TensorInfo, bf16_to_f32  # noqa: E402


def test_bf16_to_f32_exact():
    x = torch.randn(1000).to(torch.bfloat16)
    raw = x.view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(bf16_to_f32(raw), x.float().numpy())


class _FakeReader:
    """Shard s lives in file f{s} at data offset 1000 + 7 s; a row's 160 bf16 words all equal
    (local index mod 100) + 100 s, so a wrong shard or row shows."""

    def __init__(self):
        self.calls = 0

    def tensor(self, name):
        s = int(name.split("shard_")[1].split(".")[0])
        return TensorInfo(name, f"f{s}", "BF16", (2_500_012, 160), 1000 + 7 * s, 2_500_012 * 320)

    def read(self, file, a, b):
        self.calls += 1
        s = int(file[1:])
        local = (a - (1000 + 7 * s)) // 320
        assert b - a + 1 == 320 and (a - (1000 + 7 * s)) % 320 == 0
        return np.full(160, (local % 100) + 100 * s, dtype=np.uint16).tobytes()


def test_row_fetcher_addressing(tmp_path):
    rd = _FakeReader()
    f = FlashNextRowFetcher(rd, cache_path=tmp_path / "c.npz", threads=4)
    rps = f.table.rows_per_shard(f.key)
    assert rps == 2_500_012
    rows = [0, 5, rps - 1, rps, 3 * rps + 17, 127 * rps + 4]
    assert f.fetch(rows + [5]) == len(rows)
    for r in rows:
        s, l = divmod(r, rps)
        assert f.cache[r][0] == (l % 100) + 100 * s
    f.save()
    g = FlashNextRowFetcher(rd, cache_path=tmp_path / "c.npz")
    n0 = rd.calls
    assert g.fetch(rows) == 0 and rd.calls == n0
    assert g.get(np.array(rows).reshape(2, 3)).shape == (2, 3, 160)


def test_load_fstar(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"orders": {"2": {"f_star_per_token": 2e-5}, "3": {"f_star_per_token": None}}}))
    f = rs.load_fstar(p)
    assert f == {2: 2e-5}
    assert rs.below_fstar(np.array([1e-6, 3e-5]), 2, f).tolist() == [True, False]


def test_within_share_ci_matches_decomposition():
    from ngram_memory.dissect.slice_probe import within_share_ci
    rng = np.random.default_rng(5)
    grp = rng.integers(0, 60, size=600)
    vals = rng.normal(size=60)[grp] + 0.5 * rng.normal(size=600)
    est, lo, hi = within_share_ci(vals, grp, n_boot=300, rng=rng)
    assert est == pytest.approx(variance_decomposition(vals, grp)["within_share"], rel=1e-9)
    assert lo < est < hi
    assert 0.1 < est < 0.4   # 0.25 / (1 + 0.25) = 0.2 expected
