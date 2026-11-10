"""M0: the 11,405 / 11,405 HF stub parity of DESIGN §I (S1/ingest-algo/extras.out line 1), on the real corpus.

transformers' Qwen4ExpTextNGramEmbedding.forward with an index stub for the table (it returns the row ids it would
read) against ngram_memory.hashing.rows_for with the Flash-Next KeySpec. Same file sample as the proof
(`ls-tree python` at df29117, every 12th .py file), same sequence (the first 300 tokens, a mid-sequence EOS, the
next 300), Flash-Next tokenizer (sha256 0997f410...). Skips if the tokenizer or the minisgl clone is absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ngram_memory.hashing import rows_for
from ngram_memory.spec import key_spec_from_qwen4exp_config

FIX = Path(__file__).resolve().parent / "fixtures"
REPO = Path("/home/user/patcarter883/minisglang-rdna4")
TOK_CANDIDATES = [os.environ.get("NGRAM_MEMORY_FLASHNEXT_TOKENIZER", ""), str(FIX / "flashnext_tokenizer.json"),
                  "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graft/tok/"
                  "Qwen__Qwen3.8-Flash-Next/tokenizer.json"]
TOK = next((p for p in TOK_CANDIDATES if p and Path(p).exists()), None)
TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


@pytest.mark.skipif(TOK is None or not (REPO / ".git").exists(), reason="Flash-Next tokenizer or minisgl clone absent")
def test_hf_stub_parity_11405_positions():
    import hashlib
    from tokenizers import Tokenizer
    from transformers.models.qwen4_exp import modeling_qwen4_exp as q4
    assert hashlib.sha256(Path(TOK).read_bytes()).hexdigest() == TOK_SHA
    tok = Tokenizer.from_file(TOK)
    tc = json.loads((FIX / "configs" / "qwen3.8-flash-next.config.json").read_text())["text_config"]
    key = key_spec_from_qwen4exp_config(tc)
    V, EOS, NS, HPN = tc["vocab_size"], tc["eos_token_id"], tc["ngram_size"], tc["heads_per_ngram"]
    real = torch.nn.Embedding
    torch.nn.Embedding = lambda n, d, *a, **k: real(1, 1)  # never allocate the 320M-row table
    try:
        ns = types.SimpleNamespace(ngram_size=NS, heads_per_ngram=HPN, vocab_size=V,
                                   ngram_vocab_size_base=tc["ngram_vocab_size_base"], seed=tc.get("seed", 1234),
                                   eos_token_id=EOS,
                                   make_ngram_vocab_size_divisible_by=tc.get("make_ngram_vocab_size_divisible_by", 128))
        hf = q4.Qwen4ExpTextNGramEmbedding(ns, embedding_dim=(NS - 1) * HPN * 160, layer_idx=1, ple_layer_index=0)
    finally:
        torch.nn.Embedding = real

    class IdxStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ids):
            return ids.unsqueeze(-1)
    hf.ngram_embedding = IdxStub()

    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True).stdout
    files = [p for p in git("ls-tree", "-r", "--name-only", "df29117", "python").split("\n") if p.endswith(".py")]
    npos = nbad = nf = 0
    for p in files[::12]:
        ids = np.asarray(tok.encode(git("show", f"df29117:{p}"), add_special_tokens=False).ids, np.int64)
        if ids.size == 0:
            continue
        nf += 1
        ids = np.concatenate([ids[:300], [EOS], ids[300:600]])
        with torch.no_grad():
            r_hf = hf(torch.from_numpy(ids)[None], None)[0].numpy()
        r_np = rows_for(key, ids)
        assert r_hf.shape == r_np.shape
        npos += ids.size
        nbad += int((r_hf != r_np).any(axis=1).sum())
    assert (nf, npos, nbad) == (21, 11405, 0), (nf, npos, nbad)
