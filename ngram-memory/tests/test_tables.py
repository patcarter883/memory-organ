"""Row stores: SparseRowStore semantics, SparseRowEmbedding == dense table (values and gradients),
ShardMmapStore byte addressing on synthetic safetensors shards, QuantRowCodec decode."""
import numpy as np
import pytest
import torch

from ngram_memory.tables import (HFTensorStore, QuantRowCodec, ShardMmapStore, SparseRowEmbedding, SparseRowStore,
                                 read_safetensors_header)


def test_sparse_store_basic():
    s = SparseRowStore.empty(1000, 4)
    assert torch.equal(s.gather(torch.tensor([[1, 2]])), torch.zeros(1, 2, 4))
    s.scatter([5, 900, 5], torch.arange(12.).reshape(3, 4))  # duplicate: last wins
    assert s.row_ids.tolist() == [5, 900]
    assert torch.equal(s.gather([5])[0], torch.arange(8., 12.))
    s.scatter([5], torch.ones(1, 4), accumulate=True)
    assert torch.equal(s.gather([5])[0], torch.arange(9., 13.))
    s.ensure([7, 3])
    assert s.row_ids.tolist() == [3, 5, 7, 900] and torch.equal(s.gather([3, 7]), torch.zeros(2, 4))
    assert s.nonzero_row_ids().tolist() == [5, 900]
    s.delete([5])
    assert torch.equal(s.gather([5]), torch.zeros(1, 4))
    s.prune_zero()
    assert s.row_ids.tolist() == [900]
    ids, bf = s.to_bf16()
    assert bf.dtype == torch.bfloat16 and ids.tolist() == [900]
    s2 = SparseRowStore.from_state_dict(s.state_dict())
    assert torch.equal(s2.to_dense(), s.to_dense())
    with pytest.raises(IndexError):
        s.scatter([1000], torch.zeros(1, 4))


def test_from_dense_roundtrip_and_hf_store():
    d = torch.zeros(50, 3)
    d[[2, 7, 49]] = torch.randn(3, 3)
    s = SparseRowStore.from_dense(d)
    assert s.row_ids.tolist() == [2, 7, 49] and torch.equal(s.to_dense(), d)
    hf = HFTensorStore(d.clone())
    ids = torch.tensor([[2, 3], [49, 7]])
    assert torch.equal(hf.gather(ids), s.gather(ids))
    hf.scatter([3], torch.ones(1, 3))
    assert torch.equal(hf.gather([3])[0], torch.ones(3))


@pytest.mark.parametrize("sparse_grad", [True, False])
def test_sparse_embedding_equals_dense_values_and_grad(sparse_grad):
    g = torch.Generator().manual_seed(0)
    N, w = 300, 5
    ids = torch.randint(0, N, (3, 17), generator=g)
    reach = torch.unique(ids)
    stored = reach[::2]  # half the read rows written, the rest reserved/absent
    dense = torch.zeros(N, w)
    vals = torch.randn(stored.numel(), w, generator=g)
    dense[stored] = vals
    emb_d = torch.nn.Embedding.from_pretrained(dense.clone(), freeze=False)
    st = SparseRowStore(N, w, stored, vals.clone())
    st.ensure(reach)  # compact trick: hold every reachable row
    emb_s = SparseRowEmbedding(st, sparse_grad=sparse_grad, requires_grad=True)
    up = torch.randn(3, 17, w, generator=g)
    a, b = emb_d(ids), emb_s(ids)
    assert torch.equal(a, b)
    (a * up).sum().backward()
    (b * up).sum().backward()
    gs = emb_s.values.grad
    gs = gs.to_dense() if gs.is_sparse else gs
    assert torch.equal(gs, emb_d.weight.grad[emb_s.row_ids])
    nz = emb_d.weight.grad.abs().sum(-1) > 0
    assert set(nz.nonzero().flatten().tolist()) <= set(reach.tolist())
    # the store view shares memory with the parameter
    with torch.no_grad():
        emb_s.values[0] += 1
    assert torch.equal(emb_s.store.values[0], emb_s.values[0].detach())


def test_sparse_embedding_write_reserve():
    e = SparseRowEmbedding(SparseRowStore.empty(100, 2))
    assert torch.equal(e(torch.tensor([1, 2])), torch.zeros(2, 2))
    e.write([4], torch.tensor([[1., 2.]]))
    e.reserve([9, 4])
    assert e.row_ids.tolist() == [4, 9]
    assert torch.equal(e(torch.tensor([4, 9, 5])), torch.tensor([[1., 2.], [0., 0.], [0., 0.]]))


def _write_shards(tmp_path, n_shards, rows_per_shard, w, dtype):
    from safetensors.torch import save_file
    full = torch.randn(n_shards * rows_per_shard, w, generator=torch.Generator().manual_seed(1)).to(dtype)
    shards = []
    for s in range(n_shards):
        p = tmp_path / f"model-{s:05d}.safetensors"
        name = f"x.ngram_embedding.shard_{s}.weight"
        save_file({"pad.before": torch.zeros(3, dtype=torch.float32), name: full[s * rows_per_shard:(s + 1) * rows_per_shard]
                   .contiguous(), "z.after": torch.ones(2)}, str(p))
        shards.append((str(p), name))
    return full, shards


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_mmap_store(tmp_path, dtype):
    full, shards = _write_shards(tmp_path, 4, 37, 16, dtype)
    st = ShardMmapStore(shards, 16)
    assert st.num_rows == 148 and st.rows_per_shard == 37
    ids = torch.tensor([[0, 36, 37], [147, 100, 5]])
    assert torch.equal(st.gather(ids), full[ids].to(torch.float32))
    # byte address formula of §2.1b
    hdr, start = read_safetensors_header(shards[2][0])
    path, off = st.address(2 * 37 + 5)
    assert path == shards[2][0]
    assert off == 8 + (start - 8) + hdr[shards[2][1]]["data_offsets"][0] + 5 * 16 * full.element_size()
    st.close()


def test_codec_fp8_matches_torch():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no float8 in torch")
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    got = QuantRowCodec("fp8_e4m3_global", 16, scale=2e-4).decode(raw)
    ref = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).to(torch.float32).numpy() * np.float32(2e-4)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    assert np.array_equal(got[~np.isnan(got)], ref[~np.isnan(ref)])


def test_codec_bf16_matches_torch():
    x = torch.randn(5, 8).to(torch.bfloat16)
    raw = x.view(torch.uint8).numpy().reshape(5, 16)
    assert np.array_equal(QuantRowCodec("bf16", 8).decode(raw), x.float().numpy())


def test_codec_int6_g32_against_independent_packer():
    g = np.random.default_rng(0)
    n, w = 6, 64
    codes = g.integers(-31, 32, (n, w))
    scales = (g.random((n, w // 32)) * 1e-2).astype(np.float16)
    # independent packer from the documented convention: 4 codes per 24-bit LE word, LSB first, bias +32
    biased = (codes + 32).astype(np.uint32).reshape(n, -1, 4)
    word = biased[..., 0] | (biased[..., 1] << 6) | (biased[..., 2] << 12) | (biased[..., 3] << 18)
    packed = np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, -1)
    got = QuantRowCodec("int6_g32", w).decode(packed, scales)
    ref = (codes.reshape(n, -1, 32) * scales.astype(np.float32)[..., None]).reshape(n, w)
    assert np.allclose(got, ref, rtol=0, atol=0)
