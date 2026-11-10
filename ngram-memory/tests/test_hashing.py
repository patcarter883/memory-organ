"""Hashing parity (I4): ngram_memory.hashing vs transformers' q4 NGramEmbedding, vs the vendored minisgl hasher,
vs the real-tokenizer rows of P/t1/a_rows_for.out, plus decode-step / chunked / padding semantics."""
import importlib.util
import json
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from conftest import FIX, ROOT
from ngram_memory.hashing import NGramHasher, checkpoint_loc, decode_row_ids_batch, rows_for, rows_for_ngram
from ngram_memory.spec import KeySpec, flash_next_format, flash_next_key_spec, qwen25_proxy_key_spec

FN = flash_next_key_spec()
KEYS = [FN, qwen25_proxy_key_spec(), KeySpec(512, 1, base=1000), KeySpec(262144, 1)]


def _hf_embedding(key: KeySpec):
    """transformers' Qwen4ExpTextNGramEmbedding with the table stubbed by an identity on row ids."""
    import transformers.models.qwen4_exp.modeling_qwen4_exp as q4

    class _Stub(torch.nn.Module):
        def __init__(self, n, d):
            super().__init__()
            self.weight = torch.empty(0)

        def forward(self, ids):
            return ids.unsqueeze(-1)

    cfg = types.SimpleNamespace(ngram_size=key.ngram_size, heads_per_ngram=key.heads_per_ngram,
                                vocab_size=key.vocab_size, ngram_vocab_size_base=key.base, seed=key.seed,
                                eos_token_id=key.eos_token_id, make_ngram_vocab_size_divisible_by=128)
    orig = q4.nn.Embedding
    q4.nn.Embedding = _Stub
    try:
        m = q4.Qwen4ExpTextNGramEmbedding(cfg, embedding_dim=key.n_heads, layer_idx=1, ple_layer_index=0)
    finally:
        q4.nn.Embedding = orig
    with torch.no_grad():
        m.layer_multipliers.copy_(q4._build_layer_multipliers(key.vocab_size, key.ngram_size, 0, key.seed))
        m.ngram_heads_vocab_sizes.copy_(torch.tensor(m.head_vocab_sizes))
        m.ngram_heads_offsets.copy_(torch.tensor(m.head_offsets))
    return m


def _random_batch(key, seed, B=4, T=40, eos_rate=0.08):
    g = np.random.default_rng(seed)
    ids = g.integers(0, key.vocab_size, (B, T))
    ids[g.random((B, T)) < eos_rate] = key.eos_token_id
    ids[0, :3] = key.eos_token_id  # leading eos run
    ids[-1, -1] = key.eos_token_id
    return ids


@pytest.mark.parametrize("key", KEYS, ids=["flashnext", "qwen25", "tiny", "gemma_vocab"])
def test_rows_for_equals_transformers_q4(key):
    m = _hf_embedding(key)
    n_pos = 0
    for seed in range(3):
        ids = _random_batch(key, seed)
        ref = m(torch.from_numpy(ids), None).numpy()  # [B, T, H]
        assert np.array_equal(rows_for(key, ids), ref)
        n_pos += ids.size
    assert n_pos == 480


def test_q4_buffers_equal_keyspec():
    m = _hf_embedding(FN)
    assert m.head_vocab_sizes == list(FN.head_sizes) and m.head_offsets == list(FN.head_offsets)
    assert m.layer_multipliers.tolist() == list(FN.multipliers)


def _load_vendor():
    p = FIX / "vendor" / "minisgl_ple_hashing.py"
    if not p.exists():
        pytest.skip("vendored minisgl hashing absent")
    spec = importlib.util.spec_from_file_location("vendor_minisgl_ple_hashing", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rows_for_equals_vendored_minisgl_hashing():
    mh = _load_vendor()
    key = FN
    hasher = mh.Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=3, heads_per_ngram=8, eos_token_id=key.eos_token_id,
        checkpoint_multipliers=np.asarray(key.multipliers), vocab_size=key.vocab_size)
    sizes = mh.build_head_vocab_sizes(key.base, 16, 0)
    assert sizes.tolist() == list(key.head_sizes)
    offs = np.asarray(key.head_offsets)
    n = 0
    for seed in range(4):
        ids = _random_batch(key, 10 + seed, B=1, T=300)[0]
        # chunked prefill exactly as the serve does it: previous context = last 2 tokens (eos-seeded)
        ctx = np.full(2, key.eos_token_id)
        ours_chunks, ref_chunks = [], []
        for a in range(0, ids.size, 37):
            c = ids[a:a + 37]
            ref_chunks.append(offs + np.remainder(hasher.hashes(np.concatenate([ctx, c]), c.size), sizes))
            ours_chunks.append(rows_for(key, c, context=ctx))
            ctx = np.concatenate([ctx, c])[-2:]
        assert np.array_equal(np.concatenate(ours_chunks), np.concatenate(ref_chunks))
        assert np.array_equal(np.concatenate(ours_chunks), rows_for(key, ids))
        n += ids.size * 16
    assert n == 19200


def test_real_tokenizer_rows_from_t1_fixture():
    p = FIX / "t1_eiffel_rows.json"
    if not p.exists():
        pytest.skip("P/t1 fixture absent")
    fx = json.loads(p.read_text())
    assert fx["tokenizer_sha256"].startswith("0997f410")
    got = rows_for(FN, np.asarray(fx["ids"]))[fx["pos"]]
    assert got.tolist() == fx["rows"]
    fmt = flash_next_format()
    assert [list(checkpoint_loc(fmt.table, FN, r)) for r in got.tolist()] == fx["shard_local"]


@pytest.mark.parametrize("key", [FN, KeySpec(512, 1, base=1000, orders=(2, 3, 1))], ids=["flashnext", "unigram"])
def test_rows_for_ngram_matches_position_rows(key):
    g = np.random.default_rng(5)
    for _ in range(50):
        toks = g.integers(0, key.vocab_size - 1, 3)
        toks[toks == key.eos_token_id] += 1
        r = rows_for(key, toks)
        for n in key.orders:
            hs = list(key.heads_of_order(n))
            assert np.array_equal(rows_for_ngram(key, toks[3 - n:]), r[-1, hs])


@pytest.mark.parametrize("key", [FN, KeySpec(512, 1, base=1000, orders=(2, 3, 1))], ids=["flashnext", "unigram"])
def test_decode_row_ids_batch_equals_full_and_vendored(key):
    h = NGramHasher(key)
    g = np.random.default_rng(1)
    bad = n = 0
    for _ in range(200):
        bs = 16
        hist = np.where(g.random((bs, 2)) < 0.4, key.eos_token_id, g.integers(0, key.vocab_size, (bs, 2)))
        tok = np.where(g.random(bs) < 0.3, key.eos_token_id, g.integers(0, key.vocab_size, bs))
        got = decode_row_ids_batch(key, hist, tok)
        ref = np.stack([rows_for(key, tok[i:i + 1], context=hist[i])[0] for i in range(bs)])
        bad += int((got != ref).any(1).sum())
        n += bs
    assert bad == 0 and n == 3200
    # a decode loop reproduces the full-sequence rows
    ids = _random_batch(key, 7, B=3, T=30)
    full = rows_for(key, ids)
    ctx = np.full((3, 2), key.eos_token_id)
    for t in range(30):
        assert np.array_equal(h.decode_row_ids_batch(ctx, ids[:, t]), full[:, t])
        ctx = h.next_context(ctx, ids[:, t])
    if key is FN:  # vs the vendored per-sequence hasher
        mh = _load_vendor()
        vh = mh.Qwen4ExpNGramHasher(ngram_size=3, heads_per_ngram=8, layer_multipliers=np.asarray(key.multipliers),
                                    eos_token_id=key.eos_token_id)
        hist = np.where(g.random((64, 2)) < 0.4, key.eos_token_id, g.integers(0, key.vocab_size, (64, 2)))
        tok = g.integers(0, key.vocab_size, 64)
        ref = np.stack([np.asarray(key.head_offsets) + np.remainder(vh.hashes(np.concatenate([hist[i], tok[i:i + 1]]), 1)[0],
                                                                    np.asarray(key.head_sizes)) for i in range(64)])
        assert np.array_equal(decode_row_ids_batch(key, hist, tok), ref)


def test_padding_mask_hashes_as_eos():
    ids = _random_batch(FN, 3, B=2, T=12)
    mask = np.ones_like(ids)
    mask[0, :4] = 0
    sub = np.where(mask == 1, ids, FN.eos_token_id)
    assert np.array_equal(rows_for(FN, ids, mask=mask), rows_for(FN, sub))
    # left padding with eos == the unpadded sequence for the valid positions
    assert np.array_equal(rows_for(FN, ids, mask=mask)[0, 4:], rows_for(FN, ids[0, 4:]))


def test_hashing_is_torch_and_transformers_free():
    code = ("import builtins,sys\n_r=builtins.__import__\n"
            "def _b(n,*a,**k):\n"
            "    if n.split('.')[0] in ('torch','transformers'): raise ImportError('blocked '+n)\n"
            "    return _r(n,*a,**k)\n"
            "builtins.__import__=_b\n"
            f"sys.path.insert(0,{str(ROOT)!r})\n"
            "import importlib.util\n"
            "from ngram_memory import spec\n"
            "s=importlib.util.spec_from_file_location('h', {!r}, submodule_search_locations=None)\n".format(
                str(ROOT / "ngram_memory" / "hashing.py")) +
            "import ngram_memory.hashing as H\n"
            "print(H.rows_for(spec.flash_next_key_spec(), [1,2,3]).shape, 'torch' in sys.modules)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "(3, 16) False"
