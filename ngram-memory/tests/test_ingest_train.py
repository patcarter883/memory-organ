"""ingest/train.py on tiny hosts: loss decreases, only allowed rows change, resume is bit-exact, the key shuffle
permutes as specified, checkpointing does not change gradients, the optimiser's prox / trust region / accumulator
init behave as documented."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import EOS, V, build_host, tiny_fmt
from ngram_memory.backends.graft.module import NGramGraft
from ngram_memory.backends.graft.qwen35 import PrefixContext
from ngram_memory.hashing import rows_for
from ngram_memory.ingest import corpus as C
from ngram_memory.ingest import kltargets as KL
from ngram_memory.ingest import rowpolicy as RP
from ngram_memory.ingest import train as TR
from ngram_memory.tables import SparseRowStore


def _corpus(seed=0, n_docs=12, doc_len=30, vocab=40):
    rng = np.random.default_rng(seed)
    base = rng.integers(3, 3 + vocab, 40)
    docs = [base[rng.integers(0, 40, doc_len)] for _ in range(n_docs)]
    return C.pack_stream(docs, EOS)


def _setup(kind="qwen2", arm="G-lin", policy=RP.RowPolicy("count", 2), anchors=True, **cfg_kw):
    host = build_host(kind)
    fmt = tiny_fmt("G-dir" if arm == "G-dir" else "G-lin")
    stream = _corpus()
    ws = C.make_windows(stream, 48, 40, 2, EOS)
    ix = RP.NGramIndex(fmt.key, [stream])
    rows = RP.select_rows(ix, policy)
    store = SparseRowStore.reserved(fmt.padded_rows, fmt.row_width, torch.from_numpy(rows))
    anc = {}
    if anchors:
        a_stream = _corpus(seed=5, n_docs=4, doc_len=40, vocab=60)
        aw = C.make_windows(a_stream, 40, 40, 2, EOS)
        anc = {"near": (aw, KL.compute_kl_targets(host, aw, k=16))}
    kw = dict(arm=arm, lr=0.05, anchor_lambdas={"near": 1.0}, loss_chunk=16, max_steps=8)
    kw.update(cfg_kw)
    cfg = TR.TrainConfig(**kw)
    return host, fmt, store, ws, anc, cfg, ix


@pytest.mark.parametrize("kind,arm", [("qwen2", "G-lin"), ("qwen2", "G-dir"), ("qwen3_5", "G-lin"),
                                      ("qwen3_5_moe", "G-lin"), ("qwen2", "rows+Wv")])
def test_training_decreases_loss_and_touches_only_allowed_rows(kind, arm):
    host, fmt, store, ws, anc, cfg, ix = _setup(kind, arm, max_steps=12)
    fp0 = TR.host_fingerprint(host)
    tr = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc)
    before = tr.eval_nll(ws)
    read_union = set()
    for _ in range(cfg.max_steps):
        k = tr.step_idx
        _, wi = tr.train_index(k)
        read_union |= set(tr.inner.row_ids[torch.from_numpy(tr.read_rows(ws[wi]))].tolist())
        for set_no, name in enumerate(sorted(anc)):
            for j in tr.anchor_indices(set_no, len(anc[name][0]), k):
                read_union |= set(tr.inner.row_ids[torch.from_numpy(tr.read_rows(anc[name][0][j]))].tolist())
        tr.step()
    after = tr.eval_nll(ws)
    assert after["nll"] < before["nll"] - 1e-3, (before, after)
    # the reserved set never changes; only rows read in some step moved; nothing outside the policy exists
    assert torch.equal(tr.inner.row_ids, store.row_ids)
    changed = set(tr.inner.row_ids[(tr.inner.values.data != 0).any(1)].tolist())
    assert changed and changed <= read_union
    assert changed <= set(store.row_ids.tolist())
    tr.close()
    assert TR.host_fingerprint(host) == fp0
    assert not host.training and not any(p.requires_grad for p in host.parameters())


def test_disallowed_rows_get_no_update_even_when_read():
    """Rows read by training text but outside the policy stay absent (read 0): the policy is the trainable set."""
    host, fmt, store, ws, anc, cfg, ix = _setup(policy=RP.RowPolicy("count", 3), anchors=False, max_steps=4)
    all_read = np.unique(np.concatenate([rows_for(fmt.key, w.ids, context=w.context).reshape(-1) for w in ws]))
    outside = np.setdiff1d(all_read, store.row_ids.numpy())
    assert outside.size > 0
    tr = TR.RowTrainer(host, fmt, store, ws, cfg)
    tr.train()
    cart = tr.cartridge()
    assert not np.isin(cart.row_ids.numpy(), outside).any()
    assert np.isin(cart.row_ids.numpy(), store.row_ids.numpy()).all()
    tr.close()


@pytest.mark.parametrize("arm", ["G-lin", "rows+Wv", "key-shuffle"])
def test_resume_is_bit_exact(arm, tmp_path):
    kw = dict(lambda_l2=1e-3, lambda_l1=1e-3, kappa=5.0, acc_init_steps=3, max_steps=6)
    host, fmt, store, ws, anc, cfg, ix = _setup(arm=arm, **kw)
    a = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, log_path=tmp_path / "a.jsonl")
    a.train()
    va, acc_a = a.inner.values.data.clone(), a.opt.acc.clone()
    wv_a = a.graft.ple.value_proj.weight.data.clone()
    a.close()
    b = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc)
    b.train(3, ckpt_path=tmp_path / "ck.pt")
    b.close()
    c = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc)
    c.load_checkpoint(tmp_path / "ck.pt")
    assert c.step_idx == 3
    c.train()
    assert torch.equal(c.inner.values.data, va) and torch.equal(torch.nan_to_num(c.opt.acc, -1.0),
                                                                torch.nan_to_num(acc_a, -1.0))
    assert torch.equal(c.graft.ple.value_proj.weight.data, wv_a)
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 6
    c.close()


def test_key_shuffle_permutes_as_specified():
    host, fmt, store, ws, anc, cfg, ix = _setup(arm="key-shuffle", anchors=False, max_steps=6)
    tr = TR.RowTrainer(host, fmt, store, ws, cfg)
    sh = tr.shuffle
    key = fmt.key
    src, dst = sh.src, sh.dst
    assert torch.equal(src, store.row_ids)
    assert torch.equal(torch.sort(dst).values, src)                      # a permutation of the reserved rows
    head = lambda r: torch.searchsorted(torch.tensor(key.head_offsets), r, right=True) - 1  # noqa: E731
    assert torch.equal(head(src), head(dst))                              # within each head
    assert (src != dst).float().mean() > 0.5
    again = TR.KeyShuffle(key, store.row_ids, cfg.shuffle_seed)
    assert torch.equal(again.dst, dst)                                    # fixed by the seed
    assert not torch.equal(TR.KeyShuffle(key, store.row_ids, cfg.shuffle_seed + 1).dst, dst)
    outside = torch.tensor([int(key.total_rows) - 1])
    if not bool(torch.isin(outside, src)):
        assert torch.equal(sh.remap(outside), outside)                    # unreserved rows unchanged
    tr.train()
    # the RELABELLED rows through the plain (unshuffled) graft reproduce the trained function (permutation check)
    ids = torch.from_numpy(ws[0].ids)[None]
    tr.installed.context = torch.from_numpy(ws[0].context)[None]
    with torch.no_grad():
        want = KL.final_hidden(host, ids)
    tr.installed.context = None
    tr.close()
    plain = NGramGraft(fmt, rows=tr.relabelled_store())
    inst = plain.install(host, record=False)
    inst.context = torch.from_numpy(ws[0].context)[None]
    with torch.no_grad():
        got = KL.final_hidden(host, ids)
    inst.remove()
    assert torch.allclose(got, want, atol=1e-6)
    # the C1 cartridge is the trained slots as stored, read through the TRUE keys (VI-1): not the relabelled rows
    assert torch.equal(tr.cartridge(prune_zero=False).values, tr.trained_store().values)
    assert not torch.equal(tr.relabelled_store().values, tr.cartridge(prune_zero=False).values)


def test_checkpointing_does_not_change_gradients():
    grads = []
    for gc in (False, True):
        host, fmt, store, ws, anc, cfg, ix = _setup(grad_checkpointing=gc, max_steps=1, lambda_rho=1.0, rho=1e-4,
                                                    rho_on_train=True)
        tr = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc)
        rng = torch.Generator().manual_seed(0)
        tr.inner.values.data.copy_(torch.randn(tr.inner.values.shape, generator=rng) * 0.1)
        _, wi = tr.train_index(0)
        tr.inner.values.grad = None
        tr._pass(ws[wi], None, 1.0, None, True)
        aw, tg = anc["near"]
        tr._pass(aw[0], tg[0], 1.0, None, True)
        grads.append(tr.inner.values.grad.coalesce().to_dense())
        assert tr._gc == gc
        tr.close()
    assert grads[0].abs().max() > 0
    assert torch.allclose(grads[0], grads[1], atol=1e-6, rtol=1e-5)


def test_prefix_context_matches_the_full_stream_graft_output():
    fmt = tiny_fmt()
    g = NGramGraft(fmt)
    stream = _corpus()
    rid = torch.from_numpy(np.unique(rows_for(fmt.key, stream)))
    g.write_rows(rid, torch.randn(rid.numel(), fmt.row_width, generator=torch.Generator().manual_seed(2)))
    full_h = torch.randn(1, stream.size, 64, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        full = g(full_h, torch.from_numpy(stream)[None], None)
        for w in C.make_windows(stream, 32, 25, 2, EOS):
            s, n = w.start, w.ids.size
            out = g(full_h[:, s:s + n], torch.from_numpy(w.ids)[None], PrefixContext(torch.from_numpy(w.context)[None]))
            assert torch.allclose(out, full[:, s:s + n], atol=1e-6)
            fresh = g(full_h[:, s:s + n], torch.from_numpy(w.ids)[None], None)
            if s > 0:
                assert not torch.allclose(fresh[:, :2], full[:, s:s + 2])
    assert torch.equal(g.row_ids(torch.from_numpy(w.ids)[None], context=torch.from_numpy(w.context)[None])[0],
                       torch.from_numpy(rows_for(fmt.key, stream)[w.start:w.start + w.ids.size]))


def test_sparse_row_adagrad_init_prox_and_trust_region():
    vals = torch.zeros(5, 4)
    opt = TR.SparseRowAdagrad(5, init_steps=2)
    g = torch.tensor([[1.0, 1, 1, 1], [2.0, 2, 2, 2]])
    st = opt.step(vals, torch.tensor([0, 1]), g, lr=0.1)
    gsq = torch.tensor([1.0, 4.0])
    a0 = float(np.median([float(gsq.median())]))
    assert st["fresh"] == 2 and opt.a0 == a0
    assert torch.allclose(opt.acc[:2], a0 + gsq)
    assert torch.allclose(vals[0], -0.1 * g[0] / torch.sqrt(a0 + gsq[0] + opt.eps))
    assert torch.isnan(opt.acc[2:]).all()
    # trust region: norms clipped to the radius
    opt.step(vals, torch.tensor([2]), torch.full((1, 4), 100.0), lr=10.0, radius=torch.tensor([0.5]))
    assert vals[2].norm() == pytest.approx(0.5, rel=1e-5)
    assert len(opt.hist) == 2
    opt.step(vals, torch.tensor([3]), torch.ones(1, 4), lr=1.0)
    assert len(opt.hist) == 2 and opt.a0 == pytest.approx(float(np.median(opt.hist)))   # frozen after init_steps
    # prox: a large lambda_1 zeroes the row exactly
    st = opt.step(vals, torch.tensor([4]), torch.ones(1, 4) * 1e-3, lr=0.1, l1=torch.tensor([1e3]))
    assert st["lasso_zeroed"] == 1 and torch.equal(vals[4], torch.zeros(4))


def test_lasso_and_omega_shrink_rows():
    res = {}
    for name, kw in (("plain", {}), ("lasso", dict(lambda_l1=0.05))):
        host, fmt, store, ws, anc, cfg, ix = _setup(anchors=False, max_steps=6, **kw)
        tr = TR.RowTrainer(host, fmt, store, ws, cfg)
        tr.train()
        res[name] = (int((tr.inner.values.data != 0).any(1).sum()), float(tr.inner.values.data.norm()))
        tr.close()
    assert res["lasso"][0] < res["plain"][0] and res["lasso"][1] < res["plain"][1]


def test_data_order_is_a_function_of_seed_and_step():
    host, fmt, store, ws, anc, cfg, ix = _setup(anchors=False)
    tr = TR.RowTrainer(host, fmt, store, ws, cfg)
    n = len(ws)
    ep0 = sorted(tr.train_index(k)[1] for k in range(n))
    assert ep0 == list(range(n))                                          # each window once per epoch
    assert [tr.train_index(k) for k in range(2 * n)] == [tr.train_index(k) for k in range(2 * n)]
    tr.close()


def test_bad_arm_and_format_combinations():
    host, fmt, store, ws, anc, cfg, ix = _setup(anchors=False)
    with pytest.raises(ValueError):
        TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(arm="G-dir"))
    with pytest.raises(ValueError):
        TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(arm="nope"))
    assert V > 0


def test_relative_lasso_threshold_is_scale_free():
    """lambda_l1_rel keys the prox to G0, so multiplying every loss gradient by c leaves the zeroed set unchanged;
    a fresh row stays exactly zero iff ||g_j|| * s_j < lambda_1 * omega_j."""
    torch.manual_seed(0)
    g = torch.randn(50, 8) * torch.logspace(-2, 2, 50)[:, None]
    s = torch.full((50,), 2.0)
    pos = torch.arange(50)
    zeroed = []
    for c in (1e-6, 1.0, 1e4):
        opt = TR.SparseRowAdagrad(50, init_steps=1)
        vals = torch.zeros(50, 8)
        g0 = opt.observe_gscale(g * c, s)
        assert g0 == pytest.approx(float((g * c).norm(dim=1).mul(s).median()), rel=1e-5)
        lam1 = 1.0 * g0
        opt.step(vals, pos, g * c, lr=0.1, l1=lam1 / s)
        z = (vals == 0).all(1)
        u = (g * c).norm(dim=1) * s
        clear = (u / lam1 - 1).abs() > 1e-4          # the median row itself sits on the threshold
        assert torch.equal(z[clear], (u < lam1)[clear])
        zeroed.append(z)
    assert int((zeroed[0] != zeroed[1]).sum()) <= 1 and int((zeroed[1] != zeroed[2]).sum()) <= 1
    assert 20 <= int(zeroed[1].sum()) <= 30          # median rule: about half the rows


def test_relative_lasso_in_trainer_and_resume(tmp_path):
    host, fmt, store, ws, anc, cfg, ix = _setup(anchors=False, max_steps=6, acc_init_steps=3, lambda_l1_rel=0.5)
    a = TR.RowTrainer(host, fmt, store, ws, cfg, log_path=tmp_path / "a.jsonl")
    a.train()
    va = a.inner.values.data.clone()
    assert a.opt.g0 is not None and len(a.opt.ghist) == 3
    a.close()
    b = TR.RowTrainer(host, fmt, store, ws, cfg)
    b.train(3, ckpt_path=tmp_path / "ck.pt")
    b.close()
    c = TR.RowTrainer(host, fmt, store, ws, cfg)
    c.load_checkpoint(tmp_path / "ck.pt")
    c.train()
    assert torch.equal(c.inner.values.data, va)
    c.close()
    with pytest.raises(ValueError):
        TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**{**cfg.to_dict(), "lambda_l1": 1e-5}))
