"""Graft proofs on tiny random hosts (qwen2, qwen3_5, qwen3_5_moe), ported from P/graft/test_graft.py.

No-op bitwise, fires-only-on-its-n-gram, sparse == dense, incremental decode == full forward, reorder_cache,
generate, left padding, gradient only on read rows (sparse grads), crop(), overlay == stored rows, removal."""
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import EOS, PAD, V, decode_parity, fill_random_rows, logits, maxdiff, rand_ids, tiny_fmt
from ngram_memory.backends.graft.module import NGramGraft
from ngram_memory.backends.graft.qwen35 import PLE_STATE_CONV, PLE_STATE_NGRAM, SideDecodeState, install_graft
from ngram_memory.serve.hf_adapter import RowOverlay, ple_table_module
from ngram_memory.tables import SparseRowEmbedding, SparseRowStore

# fp32 decode noise: P/graft/results_all.json shows 1.5e-7..1.8e-7 on these hosts; host-only baseline 2.1e-7
DECODE_TOL = 2e-6


def _ids():
    ids = rand_ids(0, 3, 24)
    ids[1, 10] = EOS  # in-sequence EOS resets n-gram history
    return ids


def test_built_without_qwen4exp_config():
    g = NGramGraft(tiny_fmt())
    from transformers.models.qwen4_exp import modeling_qwen4_exp as q4
    assert isinstance(g.ple, q4.Qwen4ExpTextPLELayer)
    assert g.ple.hc_count == 1 and isinstance(g.table, SparseRowEmbedding)
    # the HF config class itself rejects hc_count=1 (DESIGN §2.4), so the namespace path is required
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    with pytest.raises(Exception, match="hc_count > 1"):
        Qwen4ExpTextConfig(hc_count=1)


@pytest.mark.parametrize("table", ["dense", "sparse"])
@pytest.mark.parametrize("init", ["editor", "random"])
@pytest.mark.parametrize("arm", ["G-lin", "G-dir"])
def test_noop_bitwise_empty_table(host, table, init, arm):
    _, model = host
    ids = _ids()
    base = logits(model, ids)
    h = install_graft(model, NGramGraft(tiny_fmt(arm=arm, gdir_alpha=0.5), table=table, init=init))
    try:
        assert torch.equal(logits(model, ids), base)
        d, full = decode_parity(model, ids, 16)
        assert torch.equal(full, base) and d < DECODE_TOL
    finally:
        h.remove()
    assert torch.equal(logits(model, ids), base)  # removal restores the host exactly


def test_fires_only_on_its_ngram(host):
    _, model = host
    g = NGramGraft(tiny_fmt())
    ids = _ids()
    a, b = int(ids[0, 5]), int(ids[0, 6])
    big = rand_ids(1, 16, 64)
    big[3, 20:22] = torch.tensor([a, b])
    big[7, 40:42] = torch.tensor([a, b])
    big[7, 2:4] = torch.tensor([a, b])
    delta = torch.randn(64, generator=torch.Generator().manual_seed(2))
    e = g.min_norm_rows(delta)
    g.write_ngram([a, b], e)
    base_big = logits(model, big)
    h = install_graft(model, g)
    try:
        lb = logits(model, big)
        out = h.last["out"]
    finally:
        h.remove()
    is_bigram = torch.zeros(big.shape, dtype=torch.bool)
    is_bigram[:, 1:] = (big[:, :-1] == a) & (big[:, 1:] == b)
    W = g.value_proj
    E2 = W.shape[1] // 2
    expected = 0.5 * W[:, :E2] @ e[:E2]
    assert maxdiff(out[is_bigram], expected.expand(int(is_bigram.sum()), -1)) < 1e-5
    rid = g.row_ids(big)[..., :8]
    hits = torch.isin(rid, g.table.row_ids).sum(-1)
    nz = out.norm(dim=-1) > 0
    assert torch.equal(nz, hits > 0)  # output only where a written row is read (bigram or per-head collision)
    assert bool(nz[is_bigram].all())
    rows_without = [i for i in range(16) if not (hits[i] > 0).any()]
    assert len(rows_without) >= 8
    assert torch.equal(lb[rows_without], base_big[rows_without])
    first = int(is_bigram[3].nonzero()[0])
    assert torch.equal(lb[3, :first], base_big[3, :first]) and maxdiff(lb[3, first], base_big[3, first]) > 0


def test_sparse_equals_dense_with_same_rows(host):
    _, model = host
    big = rand_ids(4, 8, 32)
    gs = NGramGraft(tiny_fmt(), table="sparse", init="random")
    fill_random_rows(gs, big[:, :20])
    gd = NGramGraft(tiny_fmt(), table="dense", init="random")
    gd.load_rows(gs.rows_store())
    outs = []
    for g in (gs, gd):
        with install_graft(model, g):
            outs.append(logits(model, big))
    assert torch.equal(outs[0], outs[1])
    assert maxdiff(outs[0], logits(model, big)) > 0


@pytest.mark.parametrize("init,arm", [("editor", "G-lin"), ("random", "G-lin"), ("editor", "G-dir")])
def test_incremental_decode_equals_full_forward(host, init, arm):
    name, model = host
    ids = _ids()
    host_d, base = decode_parity(model, ids, 7)
    g = NGramGraft(tiny_fmt(arm=arm, gdir_alpha=0.7, gdir_taps=(0.3, -0.2, 0.5, 1.0)), table="sparse", init=init)
    fill_random_rows(g, ids)
    with install_graft(model, g):
        d, full = decode_parity(model, ids, 7)
        assert maxdiff(full, logits(model, ids)) == 0.0
        effect = maxdiff(full, base)
        # beam-style reorder
        with torch.no_grad():
            o = model(input_ids=ids[:, :7], use_cache=True)
            perm = torch.tensor([2, 0, 1])
            o.past_key_values.reorder_cache(perm)
            o2 = model(input_ids=ids[perm, 7:8], past_key_values=o.past_key_values, use_cache=True)
        reorder_d = maxdiff(o2.logits.float()[:, -1], full[perm, 7])
        # generate() greedy == manual greedy over full forwards
        with torch.no_grad():
            gen = model.generate(ids[:, :7], max_new_tokens=6, do_sample=False, pad_token_id=PAD)
        cur = ids[:, :7]
        for _ in range(6):
            cur = torch.cat([cur, logits(model, cur)[:, -1].argmax(-1, keepdim=True)], 1)
    assert effect > 1e-2, "graft must actually change logits for the parity check to have power"
    assert d < DECODE_TOL and reorder_d < DECODE_TOL, (d, reorder_d, host_d)
    assert _equal_until_eos(gen, cur, 7)


def _equal_until_eos(gen, cur, P):
    """generate() stops a row at EOS and pads it; compare up to and including the first generated EOS."""
    for i in range(gen.shape[0]):
        n = gen.shape[1]
        hit = (cur[i, P:] == EOS).nonzero()
        if hit.numel():
            n = P + int(hit[0]) + 1
        if not torch.equal(gen[i, :n], cur[i, :n]):
            return False
    return True


def test_decode_state_power(host):
    """Dropping the n-gram/conv side state per step must break parity (the check has power)."""
    name, model = host
    ids = _ids()
    g = NGramGraft(tiny_fmt(), init="random")
    fill_random_rows(g, ids)
    with install_graft(model, g):
        full = logits(model, ids)
        with torch.no_grad():
            o = model(input_ids=ids[:, :7], use_cache=True)
            cache = o.past_key_values
            dmax = 0.0
            for t in range(7, 24):
                st = cache.__dict__.get(next((k for k in cache.__dict__ if k.startswith("_ngram_graft_side_")), ""), None)
                if isinstance(st, SideDecodeState):
                    st.reset()
                else:
                    layer = cache.layers[1]
                    for i in (PLE_STATE_CONV, PLE_STATE_NGRAM):
                        layer.has_previous_state[i] = False
                o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
                dmax = max(dmax, maxdiff(o.logits.float()[:, -1], full[:, t]))
    assert dmax > 1e-3


def test_left_padding(host):
    name, model = host
    g = NGramGraft(tiny_fmt(), table="dense", init="random")
    g.table.weight.data.normal_(0, 1.0, generator=torch.Generator().manual_seed(4))
    L = [14, 9, 5]
    seqs = [rand_ids(10 + i, 1, n)[0] for i, n in enumerate(L)]
    Tm = max(L)
    batch = torch.full((3, Tm), PAD)
    mask = torch.zeros(3, Tm, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, Tm - len(s):] = s
        mask[i, Tm - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    with install_graft(model, g) as h:
        lpad = logits(model, batch, attention_mask=mask, position_ids=pos)
        lm, cm = h.last["layer_mask"], h.last["conv_mask"]
        singles = [logits(model, s[None]) for s in seqs]
        pad_diff = max(maxdiff(lpad[i, Tm - len(s):], singles[i][0]) for i, s in enumerate(seqs))
        ext = rand_ids(20, 3, 4)
        ids2 = torch.cat([batch, ext], 1)
        mask2 = torch.cat([mask, torch.ones(3, 4, dtype=torch.long)], 1)
        pos2 = (mask2.cumsum(-1) - 1).clamp_min(0)
        dpp, full2 = decode_parity(model, ids2, Tm, mask=mask2, pos=pos2)
        single_ext = [logits(model, torch.cat([s, ext[i]])[None]) for i, s in enumerate(seqs)]
        ext_d = max(maxdiff(full2[i, Tm:], single_ext[i][0, -4:]) for i in range(3))
    host_pad = max(maxdiff(logits(model, batch, attention_mask=mask, position_ids=pos)[i, Tm - len(s):],
                           logits(model, s[None])[0]) for i, s in enumerate(seqs))
    assert pad_diff < max(10 * host_pad, 2e-6) and dpp < DECODE_TOL and ext_d < 2e-6, (pad_diff, host_pad, dpp, ext_d)
    if name != "qwen2":  # graft conv mask == the host's own linear-attention mask
        assert isinstance(lm, torch.Tensor) and lm.ndim == 2 and torch.equal(lm, cm)


@pytest.mark.parametrize("sparse_grad", [True, False])
def test_gradient_only_on_read_rows(host, sparse_grad):
    _, model = host
    for p in model.parameters():
        p.requires_grad_(False)
    x = _ids()[:1, :12]
    g = NGramGraft(tiny_fmt(), table="sparse", sparse_grad=sparse_grad)
    reach = g.row_ids(rand_ids(5, 4, 40)).reshape(-1).unique()
    read = g.row_ids(x).reshape(-1).unique()
    g.reserve_rows(torch.cat([reach, read]))  # compact trick: rows reachable by a "corpus" + this prompt
    w = g.freeze_all_but_rows()
    assert [n for n, p in g.named_parameters() if p.requires_grad] == ["ple.ple_embedding.ngram_embedding.values"]
    with install_graft(model, g):
        loss = F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7]))
        loss.backward()
    gr = w.grad.to_dense() if w.grad.is_sparse else w.grad
    assert w.grad.is_sparse == sparse_grad
    nz_ids = g.table.row_ids[gr.norm(dim=-1) > 0]
    assert set(nz_ids.tolist()) <= set(read.tolist())
    last = torch.from_numpy(np.asarray(g.row_ids(x)[0, -1]))
    assert bool(torch.isin(last, nz_ids).all())  # all 16 rows of the last position get gradient
    assert bool(torch.isfinite(gr).all())
    # dense table, same rows: identical gradient on the shared rows
    gd = NGramGraft(tiny_fmt(), table="dense")
    gd.freeze_all_but_rows()
    with install_graft(model, gd):
        F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7])).backward()
    dg = gd.table.weight.grad
    assert torch.allclose(dg[g.table.row_ids], gr, atol=1e-7, rtol=1e-5)
    assert set((dg.norm(dim=-1) > 0).nonzero().flatten().tolist()) == set(nz_ids.tolist())


def test_overlay_on_zero_graft_equals_stored_rows(host):
    _, model = host
    ids = rand_ids(6, 3, 20)
    gs = NGramGraft(tiny_fmt(), init="random")
    fill_random_rows(gs, ids)
    with install_graft(model, gs):
        ref = logits(model, ids)
    g0 = NGramGraft(tiny_fmt(), init="random")  # same W_v/conv/norms (same seed), empty rows
    ov = RowOverlay(gs.rows_store()).install(ple_table_module(g0.ple))
    with install_graft(model, g0):
        got = logits(model, ids)
        ov.enabled = False
        off = logits(model, ids)
    ov.remove()
    assert torch.equal(got, ref) and torch.equal(off, logits(model, ids))


def test_overlay_on_dense_base_equals_written_table():
    """Native-style: base table + fp32 delta after the gather == the summed rows written into the table."""
    g = NGramGraft(tiny_fmt(), table="dense", init="random")
    g.table.weight.data.normal_(0, 0.1, generator=torch.Generator().manual_seed(1))
    ids = rand_ids(7, 2, 16)
    rid = g.row_ids(ids).reshape(-1).unique()[:40]
    delta = SparseRowStore(g.fmt.padded_rows, g.fmt.row_width, rid, torch.randn(rid.numel(), g.fmt.row_width) * 0.01)
    hid = torch.randn(2, 16, 64, generator=torch.Generator().manual_seed(2))
    for p in g.parameters():
        p.requires_grad_(False)  # frozen base: only the overlay leaf is trainable (§3.2)
    ov = RowOverlay(delta, trainable=True).install(ple_table_module(g.ple))
    a = g(hid, ids)
    a.square().sum().backward()
    ov.remove()
    gd = NGramGraft(tiny_fmt(), table="dense", init="random")
    gd.table.weight.data.copy_(g.table.weight.data)
    gd.table.weight.data[rid] += delta.values
    b = gd(hid, ids)
    assert torch.equal(a, b)
    assert ov.delta.values.grad is not None and g.table.weight.grad is None


def test_crop_side_state_qwen2():
    from conftest import _HOST_CACHE, build_host
    model = _HOST_CACHE.setdefault("qwen2", build_host("qwen2"))
    ids = rand_ids(8, 2, 20)
    g = NGramGraft(tiny_fmt(), init="random")
    fill_random_rows(g, ids)
    with install_graft(model, g, max_rollback=8):
        full = logits(model, ids)
        with torch.no_grad():
            o = model(input_ids=ids[:, :10], use_cache=True)
            cache = o.past_key_values
            wrong = rand_ids(9, 2, 5)
            for t in range(5):  # speculative tokens that get rejected
                model(input_ids=wrong[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache.crop(-5)
            d = 0.0
            for t in range(10, 20):
                o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
                d = max(d, maxdiff(o.logits.float()[:, -1], full[:, t]))
            with pytest.raises(RuntimeError):
                cache.crop(-9)  # beyond the rollback window
        # HF assisted (prompt-lookup) decoding crops the cache: must equal plain greedy
        prompt = torch.cat([ids[:1, :12], ids[:1, 2:8]], 1)
        with torch.no_grad():
            greedy = model.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=PAD)
            pld = model.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=PAD,
                                 prompt_lookup_num_tokens=4)
    assert d < DECODE_TOL
    assert _equal_until_eos(greedy, pld, prompt.shape[1])


@pytest.mark.parametrize("kind", ["qwen3_5", "qwen3_5_moe"])
def test_crop_gdn_slots_follow_hf_semantics(kind):
    """On qwen3_5 hosts the graft state lives in layer 1's LinearAttentionLayer slots 1/2; after
    activate_past_recording + crop(-k) those slots must equal the state of a run stopped k tokens earlier
    (the host's own GDN recurrent state cannot be cropped by HF; this checks the graft's part)."""
    from conftest import _HOST_CACHE, build_host
    model = _HOST_CACHE.setdefault(kind, build_host(kind))
    ids = rand_ids(11, 2, 16)
    g = NGramGraft(tiny_fmt(), init="random")
    fill_random_rows(g, ids)
    with install_graft(model, g), torch.no_grad():
        ref = model(input_ids=ids[:, :10], use_cache=True).past_key_values.layers[1]
        o = model(input_ids=ids[:, :8], use_cache=True)
        cache = o.past_key_values
        cache.activate_past_recording()
        for t in range(8, 13):
            model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        cache.layers[1].crop(-3)  # back to 10 tokens
        lay = cache.layers[1]
        assert lay.number_of_states == 3
        assert torch.equal(lay.conv_states[PLE_STATE_NGRAM], ref.conv_states[PLE_STATE_NGRAM])
        a, b = lay.conv_states[PLE_STATE_CONV], ref.conv_states[PLE_STATE_CONV]
        assert a.shape == b.shape == (2, 64, 9)
        assert maxdiff(a, b) < 1e-5  # fp32 noise: decode-mode GDN at layer 0 vs prefill feeds the gate query
