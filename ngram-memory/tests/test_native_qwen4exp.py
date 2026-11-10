"""Native qwen4_exp backend on a tiny Flash-Next-config model: port of P/t1/bcd_proofs.py
((b) locality, (c) cached decode == full forward with an edited table, (d) gradient reaches exactly the rows read),
plus apply/revert exactness, patch round trip, shard-store loading and the Backend protocol."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from transformers import DynamicCache

from ngram_memory.backends.base import Backend, RowDiff
from ngram_memory.backends.qwen4exp_native import Qwen4ExpNativeBackend, replace_table_with_shards
from ngram_memory.patch import read_patch, write_patch, write_safetensors

from tiny_hosts import record, tiny_qwen4exp

V = 4999  # ids 0..4998 (4999 = eos in the tiny config)


@pytest.fixture(scope="module")
def m64():
    torch.manual_seed(0)
    return tiny_qwen4exp(vocab=5000, dtype=torch.float64)


def _run(model, ple, ids):
    cap = {}
    h = ple.register_forward_hook(lambda mod, i, o: cap.__setitem__("o", o.detach().clone()))
    with torch.no_grad():
        lo = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    h.remove()
    return lo, cap["o"][0]


def test_bcd_locality_decode_gradient(m64):
    """P/t1/bcd_proofs.py with the edit applied through the backend (ledger + overlay) instead of in place."""
    model = m64
    random.seed(0)  # torch RNG: as the proof's, the state right after tiny_model(seed=0) (module fixture)
    be = Qwen4ExpNativeBackend(model)
    assert isinstance(be, Backend)
    ple, cfg, EOS = be.ple, model.config, model.config.eos_token_id
    rand = lambda T: [random.randrange(V) for _ in range(T)]
    rows_at = lambda ids, p: set(be.row_ids(torch.tensor([ids]))[0, p].tolist())
    # ---- (b) locality
    prompt, t = rand(24), 8
    edit_rows = sorted(rows_at(prompt, t))
    assert len(edit_rows) == 16
    delta = (torch.randn(len(edit_rows), be.spec.row_width, dtype=torch.float64) * 1.0).float()  # the proof's draw
    lo0, po0 = _run(model, ple, prompt)
    be.apply(RowDiff(torch.tensor(edit_rows), delta, edit_id="b", mode="delta"))
    lo1, po1 = _run(model, ple, prompt)
    d_ple, d_lo = (po1 - po0).abs().amax(-1), (lo1 - lo0).abs().amax(-1)
    hit = [p for p in range(len(prompt)) if rows_at(prompt, p) & set(edit_rows)]
    pred = sorted({p + 3 * k for p in hit for k in range(4) if p + 3 * k < len(prompt)})
    assert [p for p in range(len(prompt)) if d_ple[p] > 0] == pred
    assert d_lo[:t].max().item() == 0.0 and d_lo[t:].min().item() > 0.0
    agree = clean = clean_exact = prehit_bad = 0
    for _ in range(100):
        ids = rand(48)
        be.revert("b")
        a, pa = _run(model, ple, ids)
        be.ledger.reactivate("b")
        be._publish()
        b, pb = _run(model, ple, ids)
        hits = [p for p in range(48) if rows_at(ids, p) & set(edit_rows)]
        pr = {p + 3 * k for p in hits for k in range(4) if p + 3 * k < 48}
        obs = {p for p in range(48) if (pb[p] - pa[p]).abs().max() > 0}
        agree += pr == obs
        dl = (b - a).abs().amax(-1)
        if not hits:
            clean += 1
            clean_exact += dl.max().item() == 0.0
        else:
            prehit_bad += dl[:min(hits)].max().item() != 0.0 if min(hits) else 0
    assert agree == 100 and clean_exact == clean and clean > 0 and prehit_bad == 0
    # same bigram in a different context: exactly the 8 two-gram heads fire
    x1, x0 = prompt[t], prompt[t - 1]
    other = rand(10) + [x0, x1] + rand(5)
    row11 = be.row_ids(torch.tensor([other]))[0, 11].tolist()
    assert [h for h, r in enumerate(row11) if r in edit_rows] == list(range(8))
    # ---- (c) cached decode == full forward with the edited table
    seq = rand(13) + [EOS] + rand(11)
    full, _ = _run(model, ple, seq)
    worst = 0.0
    for chunks in ([1] * len(seq), [5] + [1] * (len(seq) - 5), [7, 6, 7, 5], [len(seq) - 1, 1]):
        cache, outs, i = DynamicCache(config=cfg), [], 0
        with torch.no_grad():
            for c in chunks:
                o = model(input_ids=torch.tensor([seq[i:i + c]]), past_key_values=cache, use_cache=True)
                cache, i = o.past_key_values, i + c
                outs.append(o.logits[0])
        worst = max(worst, (torch.cat(outs) - full).abs().max().item())
    assert worst < 1e-6, worst  # P/t1/bcd_proofs.out: 1.1e-8 .. 2.7e-8 (fp64)
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=torch.tensor([seq[:10]]), past_key_values=cache, use_cache=True)
        L = cache.layers[1]
        L.conv_states[1].zero_()
        L.conv_states[2].fill_(EOS)
        o = model(input_ids=torch.tensor([seq[10:11]]), past_key_values=cache, use_cache=True)
    wiped = (o.logits[0, -1] - full[10]).abs().max().item()
    assert wiped > 1e-3, wiped  # an engine caching only KV + GDN state is wrong (0.32 in the proof)
    # ---- (d) gradient through a sparse delta leaf reaches exactly the rows the prompt reads
    be.revert("b")
    for p in model.parameters():
        p.requires_grad_(False)
    reach = sorted({r for p in range(len(prompt)) for r in rows_at(prompt, p)})
    leaf = be.delta_leaf(reach)
    ids, target = torch.tensor([prompt]), torch.tensor([1234])
    lo = be.forward(dict(input_ids=ids, use_cache=False), leaf).logits[0, -1]
    torch.nn.functional.cross_entropy(lo[None], target).backward()
    g = leaf.delta.values.grad.coalesce().to_dense()
    nz = set(leaf.delta.row_ids[g.abs().sum(1) > 0].tolist())
    assert nz == set(reach) and len(reach) == 380  # 380 of 380 in the proof
    assert be.table.weight.grad is None
    last = sorted(rows_at(prompt, len(prompt) - 1))
    pos = int(torch.searchsorted(leaf.delta.row_ids, torch.tensor(last[0])))
    eps = 1e-3

    def nll(sign):
        with torch.no_grad():
            leaf.delta.values[pos, 0] += sign * eps
            out = be.forward(dict(input_ids=ids, use_cache=False), leaf).logits[0, -1]
            leaf.delta.values[pos, 0] -= sign * eps
        return torch.nn.functional.cross_entropy(out[None], target).item()

    fd = (nll(1) - nll(-1)) / (2 * eps)
    ag = g[pos, 0].item()
    assert abs(ag - fd) <= 2e-3 * abs(fd) + 1e-7, (ag, fd)  # proof: 1.27628e-3 vs 1.27670e-3
    # pinned to P/t1/bcd_proofs.out (same model, prompt and edit draw; delta held in fp32 here vs fp64 there)
    assert hit == [8, 12] and pred == [8, 11, 12, 14, 15, 17, 18, 21] and clean == 47
    assert abs(d_lo[t:].min().item() - 8.187803252966246e-4) < 1e-8
    assert abs(wiped - 0.3155352920244773) < 1e-9 and abs(ag - 1.27628e-3) < 5e-9 and abs(fd - 1.27670e-3) < 5e-9
    record("native_bcd", hit_positions=hit, ple_changed=pred, min_dlogit_after_anchor=d_lo[t:].min().item(),
           seqs_agree=agree, clean_seqs=clean, clean_bitexact=clean_exact, decode_max_abs_fp64=worst,
           wiped_state_dlogit=wiped, grad_rows=len(nz), rows_read=len(reach), autograd=ag, finite_diff=fd)
    be.remove()


def test_apply_revert_bitexact_patch_roundtrip_and_parity(tmp_path):
    torch.manual_seed(1)
    m = tiny_qwen4exp(vocab=5000)
    be = Qwen4ExpNativeBackend(m)
    ids = torch.randint(0, V, (2, 20), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        base = m(input_ids=ids, use_cache=False).logits
    rows = be.row_ids(ids)[0, 5:9].reshape(-1).unique()
    g = torch.Generator().manual_seed(2)
    e1 = be.apply(RowDiff(rows, torch.randn(rows.numel(), be.spec.row_width, generator=g) * 0.3, edit_id="e1", mode="delta"))
    rows2 = be.row_ids(ids)[1, 3:6].reshape(-1).unique()
    be.apply(RowDiff(rows2, torch.randn(rows2.numel(), be.spec.row_width, generator=g) * 0.3, edit_id="e2", mode="delta"))
    assert be.ledger.epoch == e1 + 1
    with torch.no_grad():
        edited = m(input_ids=ids, use_cache=False).logits
    assert (edited - base).abs().max() > 0
    # read_rows == table + deltas
    want = be.table.weight[rows].detach().float() + torch.from_numpy(be.ledger.delta_sum(rows.numpy())[1])
    assert torch.equal(be.read_rows(be.rows_for(be.keys(ids[0], [5]))).float()[:0], want[:0])
    assert torch.equal(be.read_rows(rows), want)
    # stale prefixes: sequence 0 reads e1's rows; others are stale iff they read one of them
    seqs = [ids[0]] + [torch.randint(0, V, (8,), generator=torch.Generator().manual_seed(s)) for s in range(20)]
    st = be.stale(seqs, rows)
    truth = [bool(np.isin(be.row_ids(s[None]).numpy(), rows.numpy()).any()) for s in seqs]
    assert st.tolist() == truth and truth[0] and not all(truth)
    # patch round trip into a fresh copy of the model: bitwise-equal logits
    p = be.export_patch(model_repo="tiny/qwen4exp")
    path = str(tmp_path / "native.safetensors")
    write_patch(path, p)
    torch.manual_seed(1)
    m2 = tiny_qwen4exp(vocab=5000)
    be2 = Qwen4ExpNativeBackend(m2)
    be2.apply_patch(read_patch(path), edit_id="from-patch")
    with torch.no_grad():
        assert torch.equal(m2(input_ids=ids, use_cache=False).logits, edited)
    rep = be2.parity(ids)
    assert rep.rows_for_matches_model and rep.cached_vs_full_max_abs < 1e-4
    # revert both: bit-exact base logits
    be.revert("e2")
    be.revert("e1")
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, base)
    # a patch made against other base rows is refused
    with torch.no_grad():
        m2.model.layers[1].ple.ple_embedding.ngram_embedding.weight[rows[0]] += 1.0
    be3 = Qwen4ExpNativeBackend(m2)
    with pytest.raises(ValueError, match="base_fingerprint"):
        be3.apply_patch(read_patch(path))
    be.remove(), be2.remove(), be3.remove()


def test_shard_store_table_is_bitwise_and_editable(tmp_path):
    torch.manual_seed(1)
    m = tiny_qwen4exp(vocab=5000)
    be0 = Qwen4ExpNativeBackend(m)
    tab = be0.table.weight.detach().clone()
    be0.remove()
    n_sh = be0.spec.table.n_shards
    rps = be0.spec.table.rows_per_shard(be0.spec.key)
    assert n_sh == 4 and rps * n_sh == tab.shape[0]
    shards = []
    for s in range(n_sh):
        path = str(tmp_path / f"model-{s:05d}.safetensors")
        name = be0.spec.table.shard_tensor_name(s)
        write_safetensors(path, {name: tab[s * rps:(s + 1) * rps].numpy(), "other": np.zeros(3, np.float32)})
        shards.append((path, name))
    ids = torch.randint(0, V, (2, 16), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = m(input_ids=ids, use_cache=False).logits
    replace_table_with_shards(m, shards, dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref)
    be = Qwen4ExpNativeBackend(m)
    rows = be.row_ids(ids)[0, 4].unique()
    be.apply(RowDiff(rows, torch.ones(rows.numel(), be.spec.row_width), edit_id="x", mode="delta"))
    assert torch.equal(be.read_rows(rows), tab[rows] + 1.0)
    be.revert("x")
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref)
    be.remove()
