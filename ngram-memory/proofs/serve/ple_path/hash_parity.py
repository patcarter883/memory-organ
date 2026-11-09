"""CPU parity check: minisgl ple/hashing.py (+ row_table.NgramHeads) vs transformers qwen4_exp.

Loads minisgl's hashing.py / row_table.py by FILE PATH (torch-free numpy modules), never importing
the minisgl package (which would pull in torch/HIP machinery). HF NGramEmbedding is constructed with
nn.Embedding stubbed out so no 95 GiB table is allocated; the embedding lookup is replaced by an
identity that returns the row ids themselves.
"""
import importlib.util, sys, types
import numpy as np
import torch

REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

mh = load("mh", f"{REPO}/ple/hashing.py")
rt = load("rt", f"{REPO}/weights/row_table.py")

import transformers.models.qwen4_exp.modeling_qwen4_exp as hf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

class _StubEmb(torch.nn.Module):
    def __init__(self, n, d): super().__init__(); self.n, self.d = n, d; self.weight = torch.empty(0)
    def forward(self, ids): return ids.unsqueeze(-1).to(torch.float64)   # head_dim -> 1, value = row id

def build_hf(vocab, base, eos, seed=1234, hpn=8, n=3, div=128):
    cfg = types.SimpleNamespace(ngram_size=n, heads_per_ngram=hpn, vocab_size=vocab,
                                ngram_vocab_size_base=base, seed=seed, eos_token_id=[eos, 7],
                                make_ngram_vocab_size_divisible_by=div)
    orig = hf.nn.Embedding
    hf.nn.Embedding = _StubEmb
    try:
        m = hf.Qwen4ExpTextNGramEmbedding(cfg, embedding_dim=(n - 1) * hpn, layer_idx=1, ple_layer_index=0)
    finally:
        hf.nn.Embedding = orig
    return m, cfg

def check(vocab, base, eos, seqs, chunks, label):
    m, cfg = build_hf(vocab, base, eos)
    # 1. multipliers / primes / offsets
    mult = mh.build_layer_multipliers(vocab, 3, 0, 1234)
    assert np.array_equal(mult, m.layer_multipliers.numpy()), (mult, m.layer_multipliers)
    sizes = mh.build_head_vocab_sizes(base, 16, 0)
    assert np.array_equal(sizes, np.array(m.head_vocab_sizes)), (sizes, m.head_vocab_sizes)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    assert np.array_equal(offs, np.array(m.head_offsets))
    heads = rt.NgramHeads(offsets=offs.astype(np.int64), vocab_sizes=sizes.astype(np.int64))
    padded = m.ngram_embedding.n
    heads.validate(padded)
    hasher = mh.Qwen4ExpNGramHasher(ngram_size=3, heads_per_ngram=8, layer_multipliers=mult, eos_token_id=eos)
    n_cmp = 0
    for seq in seqs:
        seq = np.asarray(seq, dtype=np.int64)
        ref = m(torch.tensor(seq)[None], None)[0].to(torch.int64).numpy()      # (L, 16) full-sequence, no cache
        # minisgl: chunked, carrying a 2-token history exactly like PLEStateCache.push_tokens
        hist = np.full(2, eos, dtype=np.int64); out = []; i = 0
        for c in chunks:
            if i >= seq.size: break
            toks = seq[i:i + c]; i += toks.size
            out.append(hasher.row_ids(heads, np.concatenate([hist, toks]), toks.size))
            hist = np.concatenate([hist, toks])[-2:]
        while i < seq.size:                                                     # decode tail, 1 token/step
            toks = seq[i:i + 1]; i += 1
            out.append(hasher.row_ids(heads, np.concatenate([hist, toks]), 1))
            hist = np.concatenate([hist, toks])[-2:]
        got = np.concatenate(out, 0)
        assert got.shape == ref.shape and np.array_equal(got, ref), (label, np.argwhere(got != ref)[:5])
        assert got.min() >= 0 and got.max() < padded
        n_cmp += got.size
    print(f"[{label}] OK: multipliers={mult.tolist()} primes[0:3]={sizes[:3].tolist()} "
          f"total={int(sizes.sum())} padded_rows={padded} rows/128-shard={padded/128} compared {n_cmp} row ids")

rng = np.random.default_rng(0)
# small fake config
V, EOS = 1000, 999
seqs = [rng.integers(0, V, 40), [EOS, 0, 1, EOS, EOS, 5, 6, 7, 0, EOS, 3],
        np.where(rng.random(64) < 0.15, EOS, rng.integers(0, V, 64)), [0], [EOS], [1, 2]]
check(V, 1000, EOS, seqs, [7, 5, 3], "tiny")
# REAL Flash-Next hash parameters (vocab 248320, base 20M, divisible-by 128); embedding stubbed
V, EOS = 248320, 248044
seqs = [rng.integers(0, V, 200), np.where(rng.random(300) < 0.05, EOS, rng.integers(0, V, 300)),
        [V - 1, V - 1, V - 1, 0, 0, EOS, V - 1]]
check(V, 20_000_000, EOS, seqs, [37, 64, 1], "real")
