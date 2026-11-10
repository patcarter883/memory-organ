"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Adversarial-verifier findings on the CORE piece (each was a strict xfail demonstrating a defect;
fixed in integration, now regular regression tests).

Scripts and logs: scratchpad nm/m0/verify-core/probe{3,3b,4,5}.py / .json.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import build_host, fill_random_rows, rand_ids, tiny_fmt
from ngram_memory.backends.graft.module import NGramGraft
from ngram_memory.backends.graft.qwen35 import install_graft
from ngram_memory.hashing import rows_for
from ngram_memory.spec import KeySpec
from ngram_memory.tables import SparseRowEmbedding, SparseRowStore


def test_verify_core_sparse_miss_does_not_touch_unread_rows():
    st = SparseRowStore(100, 4, torch.tensor([10, 20]), torch.ones(2, 4))
    emb = SparseRowEmbedding(st, sparse_grad=True, requires_grad=True)
    opt = torch.optim.SparseAdam([emb.values], lr=0.1)
    for ids in (torch.tensor([20]), torch.tensor([15, 25])):  # step 2 reads only absent rows
        opt.zero_grad()
        emb(ids).sum().backward()
        before = emb.values.detach().clone()
        opt.step()
    # step 2 read no stored row, so no stored row may change (a dense table would leave row 20 at 0.9)
    assert torch.equal(emb.values.detach(), before)


def test_verify_core_absent_row_reads_exact_zero_even_next_to_nonfinite_row():
    st = SparseRowStore(100, 4, torch.tensor([10, 20]), torch.tensor([[1.0] * 4, [float("inf")] * 4]))
    emb = SparseRowEmbedding(st, sparse_grad=False)
    with torch.no_grad():
        out = emb(torch.tensor([15]))  # absent row; searchsorted lands on stored row 20
    assert torch.equal(out, torch.zeros(1, 4))


@pytest.mark.parametrize("kind", ["qwen2", "qwen3_5"])
def test_verify_core_grad_checkpointing_two_forwards_one_backward(kind):
    m = build_host(kind)
    for p in m.parameters():
        p.requires_grad_(False)
    xa, xb = rand_ids(1, 2, 16), rand_ids(2, 2, 16)

    def grads(ckpt: bool) -> torch.Tensor:
        g = NGramGraft(tiny_fmt(), table="sparse", sparse_grad=False)
        g.reserve_rows(torch.cat([g.row_ids(xa).reshape(-1), g.row_ids(xb).reshape(-1)]))
        fill_random_rows(g, torch.cat([xa, xb]), scale=0.1)
        w = g.freeze_all_but_rows()
        if ckpt:
            m.train()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            with install_graft(m, g):
                la = F.cross_entropy(m(input_ids=xa, use_cache=False).logits[:, -1].float(), torch.tensor([7, 8]))
                lb = F.cross_entropy(m(input_ids=xb, use_cache=False).logits[:, -1].float(), torch.tensor([9, 10]))
                (la + lb).backward()
        finally:
            if ckpt:
                m.gradient_checkpointing_disable()
                m.eval()
        return w.grad.clone()

    assert torch.allclose(grads(True), grads(False), atol=1e-6, rtol=1e-5)


def test_verify_core_unigram_heads_keep_flashnext_rows_at_ple_layer_1():
    ids = np.random.default_rng(0).integers(0, 248320, (2, 40))
    k23 = KeySpec(248320, 248044, ple_layer_index=1)
    k231 = KeySpec(248320, 248044, ple_layer_index=1, orders=(2, 3, 1))
    assert np.array_equal(rows_for(k231, ids)[..., :16], rows_for(k23, ids))
