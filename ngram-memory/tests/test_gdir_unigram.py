"""G-dir arm (§2.4): no-op, decode state, gradient, direction-only (non-linear in row norm) behaviour.
Unigram-head extension: no-op, row-id parity of heads 0-15 with the (2,3) key, gradient coverage of unigram heads."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import EOS, decode_parity, fill_random_rows, logits, maxdiff, rand_ids, tiny_fmt
from ngram_memory.backends.graft.module import NGramGraft, UnigramNGramEmbedding
from ngram_memory.backends.graft.qwen35 import install_graft
from ngram_memory.hashing import rows_for
from ngram_memory.spec import flash_next_key_spec, graft_format, graft_key_spec

DECODE_TOL = 2e-6


def test_gdir_init_and_linearity():
    hid = torch.randn(1, 12, 64, generator=torch.Generator().manual_seed(0))
    ids = rand_ids(1, 1, 12)
    for arm in ("G-lin", "G-dir"):
        g = NGramGraft(tiny_fmt(arm=arm, gdir_alpha=0.5))
        assert g.ple.key_proj.weight.abs().max() == 0
        w = g.ple.conv1d.weight
        if arm == "G-lin":
            assert w.abs().max() == 0
        else:
            assert torch.equal(w[:, 0, 3], torch.full((64,), 0.5)) and w[:, 0, :3].abs().max() == 0
        assert torch.equal(g(hid, ids), torch.zeros(1, 12, 64))  # empty rows: exact zero
        rid = fill_random_rows(g, ids, scale=0.1)
        o1 = g(hid, ids)
        g.write_rows(rid, g.table.store.gather(rid) * 3.0)
        o3 = g(hid, ids)
        lin_err = maxdiff(o3, 3 * o1)
        if arm == "G-lin":
            assert lin_err < 1e-5  # out = 0.5 W_v e: linear in the rows
            assert maxdiff(o1, 0.5 * (g.table.store.gather(g.row_ids(ids)).flatten(-2) / 3) @ g.value_proj.T) < 1e-5
        else:
            assert lin_err > 1e-2  # direction branch silu(alpha * N_conv(u)) does not scale with the row norm
            # conv branch == silu(alpha * N(u)) on the current position (u = 0.5 v), independent of hid
            v = (g.table.store.gather(g.row_ids(ids)).flatten(-2)) @ g.value_proj.T
            u = 0.5 * v
            nu = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
            assert maxdiff(o3, u + F.silu(0.5 * nu)) < 1e-5


def test_gdir_dilated_taps_decode_and_noop(host):
    _, model = host
    ids = rand_ids(2, 3, 24)
    ids[1, 10] = EOS
    base = logits(model, ids)
    g = NGramGraft(tiny_fmt(arm="G-dir", gdir_alpha=0.8, gdir_taps=(0.25, 0.5, 0.75, 1.0)))
    with install_graft(model, g):
        assert torch.equal(logits(model, ids), base)
    fill_random_rows(g, ids[:, :16])  # later positions also exercise empty-row hits after written ones
    with install_graft(model, g):
        d, full = decode_parity(model, ids, 6)
    assert maxdiff(full, base) > 1e-2 and d < DECODE_TOL


def test_gdir_gradient_reaches_rows(host):
    _, model = host
    for p in model.parameters():
        p.requires_grad_(False)
    x = rand_ids(3, 1, 12)
    g = NGramGraft(tiny_fmt(arm="G-dir", gdir_alpha=0.5, gdir_taps=(0.2, 0.2, 0.2, 1.0)), sparse_grad=True)
    read = g.row_ids(x).reshape(-1).unique()
    g.reserve_rows(read)  # all-zero rows: N_conv(0) has finite gradient (eps inside rsqrt)
    w = g.freeze_all_but_rows()
    with install_graft(model, g):
        F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7])).backward()
    gr = w.grad.to_dense()
    assert bool(torch.isfinite(gr).all())
    nz = set(g.table.row_ids[gr.norm(dim=-1) > 0].tolist())
    assert nz <= set(read.tolist())
    assert set(g.row_ids(x)[0, -1].tolist()) <= nz


def test_unigram_head_rowid_parity_and_module():
    k23, k231 = flash_next_key_spec(), graft_key_spec(248320, 248044, orders=(2, 3, 1))
    ids = np.random.default_rng(0).integers(0, 248320, (2, 50))
    ids[0, 20] = 248044
    r23, r231 = rows_for(k23, ids), rows_for(k231, ids)
    assert np.array_equal(r231[..., :16], r23)  # heads 0-15 identical to Flash-Next
    assert (r231[..., 16:] >= k231.head_offsets[16]).all()
    # unigram rows depend on the current token only
    one = rows_for(k231, ids[:, 30:31])
    assert np.array_equal(one[:, 0, 16:], r231[:, 30, 16:])
    # module built with a 24-head embedding reads exactly these rows (no table allocation: sparse)
    assert graft_format(5120, k231).module.ple_embed_dim == 24 * 320
    g = NGramGraft(graft_format(256, k231))  # small host keeps the orthogonal init cheap
    assert isinstance(g.ple.ple_embedding, UnigramNGramEmbedding)
    assert g.ple.value_proj.weight.shape == (256, 24 * 16)
    t = torch.from_numpy(ids)
    assert torch.equal(g.captured_row_ids(t), torch.from_numpy(r231))


def test_unigram_noop_decode_gradient(host):
    _, model = host
    fmt = tiny_fmt(orders=(2, 3, 1))
    ids = rand_ids(4, 3, 24)
    ids[2, 5] = EOS
    base = logits(model, ids)
    g = NGramGraft(fmt, init="random")
    with install_graft(model, g):
        assert torch.equal(logits(model, ids), base)
    assert torch.equal(g.captured_row_ids(ids), g.row_ids(ids))
    fill_random_rows(g, ids)
    with install_graft(model, g):
        d, full = decode_parity(model, ids, 9)
    assert maxdiff(full, base) > 1e-2 and d < DECODE_TOL
    # gradient coverage: unigram heads of the last position receive gradient
    for p in model.parameters():
        p.requires_grad_(False)
    g2 = NGramGraft(fmt)
    x = ids[:1, :12]
    g2.reserve_rows(g2.row_ids(x).reshape(-1).unique())
    w = g2.freeze_all_but_rows()
    with install_graft(model, g2):
        F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7])).backward()
    nz = set(g2.table.row_ids[w.grad.to_dense().norm(dim=-1) > 0].tolist())
    last = g2.row_ids(x)[0, -1]
    assert set(last[16:].tolist()) <= nz and set(last.tolist()) <= nz
