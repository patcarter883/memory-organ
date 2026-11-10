"""ingest/kltargets.py: top-k teacher targets and the bucketed KL, on tiny hosts."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import EOS, V, build_host, rand_ids, tiny_fmt
from ngram_memory.backends.graft.module import NGramGraft
from ngram_memory.ingest import corpus as C
from ngram_memory.ingest import kltargets as KL


def _exact_kl(zp, zq):
    lp, lq = F.log_softmax(zp, -1), F.log_softmax(zq, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def test_topk_kl_exact_at_full_k_and_lower_bound():
    g = torch.Generator().manual_seed(0)
    zp, zq = torch.randn(7, 50, generator=g) * 3, torch.randn(7, 50, generator=g) * 3
    full = KL.topk_targets_from_logits(zp, 50)
    assert torch.isinf(full.rest).all()
    assert torch.allclose(KL.topk_kl(zq, full), _exact_kl(zp, zq), atol=1e-5)
    for k in (1, 5, 20, 49):
        t = KL.topk_targets_from_logits(zp, k)
        mass = t.logp.exp().sum(-1) + t.rest.exp()
        assert torch.allclose(mass, torch.ones(7), atol=1e-5)
        kl = KL.topk_kl(zq, t)
        assert (kl >= -1e-6).all() and (kl <= _exact_kl(zp, zq) + 1e-5).all()
        assert torch.allclose(KL.topk_kl(zp, t), torch.zeros(7), atol=1e-5)    # student == teacher


def test_topk_kl_gradient_matches_exact_at_full_k():
    g = torch.Generator().manual_seed(1)
    zp = torch.randn(4, 30, generator=g)
    zq = torch.randn(4, 30, generator=g, requires_grad=True)
    KL.topk_kl(zq, KL.topk_targets_from_logits(zp, 30)).sum().backward()
    g1 = zq.grad.clone()
    zq.grad = None
    _exact_kl(zp, zq).sum().backward()
    assert torch.allclose(g1, zq.grad, atol=1e-5)


@pytest.mark.parametrize("kind", ["qwen2", "qwen3_5"])
def test_final_hidden_and_targets_match_host_logits(kind, tmp_path):
    host = build_host(kind)
    ids = rand_ids(0, 1, 40)
    with torch.no_grad():
        ref = host(input_ids=ids, use_cache=False).logits[0].float()
        h = KL.final_hidden(host, ids)[0]
    assert torch.allclose(KL.head_logits(KL.output_head(host), h), ref, atol=1e-5)
    ws = C.make_windows(ids[0].numpy(), 16, 16, 2, EOS)
    tg = KL.compute_kl_targets(host, ws, k=8, chunk=5)
    assert len(tg) == len(ws) and tg.k == 8
    for w, t in zip(ws, tg.windows):
        with torch.no_grad():
            z = host(input_ids=torch.from_numpy(w.ids)[None], use_cache=False).logits[0].float()
        want = KL.topk_targets_from_logits(z, 8)
        assert torch.equal(t.idx, want.idx) and torch.allclose(t.logp, want.logp, atol=1e-5)
    p = tmp_path / "t.safetensors"
    tg.save(p)
    back = KL.KLTargets.load(p)
    assert back.k == 8 and all(torch.equal(a.idx, b.idx) and torch.equal(a.logp, b.logp) and torch.equal(a.rest, b.rest)
                               for a, b in zip(tg.windows, back.windows))
    assert tg.nbytes() == sum(w.ids.size for w in ws) * (8 * 8 + 4)


def test_targets_refuse_an_installed_graft():
    host = build_host("qwen2")
    g = NGramGraft(tiny_fmt())
    ws = C.make_windows(rand_ids(1, 1, 20)[0].numpy(), 16, 16, 2, EOS)
    with g.install(host):
        with pytest.raises(RuntimeError):
            KL.compute_kl_targets(host, ws, k=4)
    KL.compute_kl_targets(host, ws, k=4)
