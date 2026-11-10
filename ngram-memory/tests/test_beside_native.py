"""Graft beside Flash-Next's native PLE on a tiny qwen4_exp (hc_count 4): exact no-op, independent decode state,
decode parity, padding, gradient reaching only the graft's rows, and key identity with the native table."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ngram_memory.backends.graft import NGramGraft
from ngram_memory.backends.graft.beside_native import install_beside_native
from ngram_memory.backends.qwen4exp_native import Qwen4ExpNativeBackend
from ngram_memory.spec import graft_format, key_spec_from_qwen4exp_config

from conftest import decode_parity, fill_random_rows, logits, maxdiff
from tiny_hosts import record, tiny_qwen4exp

V = 4999


@pytest.fixture(scope="module")
def q4host():
    torch.manual_seed(0)
    return tiny_qwen4exp(vocab=5000)


def _fmt(model, arm="G-lin"):
    key = key_spec_from_qwen4exp_config(model.config.to_dict())  # the native table's own keys
    return graft_format(model.config.hidden_size, key, arm=arm)


def _ids(seed, B, T):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arm", ["G-lin", "G-dir"])
def test_beside_native_noop_decode_state_padding_gradient(q4host, arm):
    model = q4host
    assert model.config.hc_count == 4
    ids = _ids(0, 2, 20)
    ids[1, 9] = model.config.eos_token_id
    base = logits(model, ids)
    host_dec, _ = decode_parity(model, ids, 8)
    # ---- exact no-op; the native decode slots are untouched by the graft
    g = NGramGraft(_fmt(model, arm))
    with torch.no_grad():
        c0 = model(input_ids=ids[:, :12], use_cache=True).past_key_values
    with install_beside_native(model, g):
        assert torch.equal(logits(model, ids), base)
        with torch.no_grad():
            c1 = model(input_ids=ids[:, :12], use_cache=True).past_key_values
    for s in (1, 2):
        assert torch.equal(c0.layers[1].conv_states[s], c1.layers[1].conv_states[s])
    # ---- graft row ids == the native table's lookup (same KeySpec): keys transfer, values do not
    be = Qwen4ExpNativeBackend(model)
    assert torch.equal(g.row_ids(ids), be.row_ids(ids))
    be.remove()
    # ---- every graft row firing: decode parity against the host's own floor
    fill_random_rows(g, ids, seed=3)
    inst = install_beside_native(model, g)
    d, full = decode_parity(model, ids, 8)
    eff = maxdiff(full, base)
    assert eff > 1e-3 and d < 10 * host_dec + 1e-5, (d, host_dec, eff)
    out = inst.last["out"]
    assert out.shape[-1] == model.config.hidden_size and inst.last["native_out"].shape[-1] == 4 * out.shape[-1]
    # ---- left padding: padded batch == single sequences on valid positions
    L = [14, 9]
    seqs = [_ids(10 + i, 1, n)[0] for i, n in enumerate(L)]
    fill_random_rows(g, torch.cat(seqs)[None], seed=4)
    Tm = max(L)
    batch, mask = torch.zeros(2, Tm, dtype=torch.long), torch.zeros(2, Tm, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, Tm - len(s):], mask[i, Tm - len(s):] = s, 1
    with torch.no_grad():
        lp = model(input_ids=batch, attention_mask=mask, use_cache=False).logits.float()
    pad = max(maxdiff(lp[i, Tm - len(s):], logits(model, s[None])[0]) for i, s in enumerate(seqs))
    inst.remove()
    with torch.no_grad():
        lp0 = model(input_ids=batch, attention_mask=mask, use_cache=False).logits.float()
    pad0 = max(maxdiff(lp0[i, Tm - len(s):], logits(model, s[None])[0]) for i, s in enumerate(seqs))
    assert pad < 10 * pad0 + 1e-5, (pad, pad0)
    # ---- gradient reaches exactly the graft rows read; nothing reaches the native table
    for p in model.parameters():
        p.requires_grad_(False)
    gg = NGramGraft(_fmt(model, arm))
    x = ids[:1, :10]
    read = gg.row_ids(x).reshape(-1).unique()
    gg.reserve_rows(read)
    w = gg.freeze_all_but_rows()
    with install_beside_native(model, gg):
        F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7])).backward()
    gd = w.grad.coalesce().to_dense()
    got = set(gg.table.row_ids[gd.abs().sum(-1) > 0].tolist())
    assert got == set(read.tolist())
    assert model.model.layers[1].ple.ple_embedding.ngram_embedding.weight.grad is None
    record(f"beside_native_{arm}", decode_max_abs=d, host_decode_max_abs=host_dec, graft_effect=eff,
           pad_vs_single=pad, host_pad_vs_single=pad0, rows_read=len(read), rows_with_grad=len(got))


def test_beside_native_crop(q4host):
    """The graft's side state follows HF's own crop (the GDN layer needs activate_past_recording to crop)."""
    model = q4host
    g = NGramGraft(_fmt(model))
    ids = _ids(5, 1, 12)
    fill_random_rows(g, ids, seed=6)
    with install_beside_native(model, g):
        with torch.no_grad():
            o = model(input_ids=ids[:, :8], use_cache=True)
            ref = model(input_ids=ids[:, 8:9], past_key_values=o.past_key_values, use_cache=True).logits
            o = model(input_ids=ids[:, :8], use_cache=True)
            c = o.past_key_values
            try:
                for layer in c.layers:
                    if hasattr(layer, "activate_past_recording"):
                        layer.activate_past_recording()
                model(input_ids=ids[:, 8:11], past_key_values=c, use_cache=True)
                c.crop(-3)
            except Exception as e:
                pytest.skip(f"host cache cannot crop: {type(e).__name__}: {e}")
            got = model(input_ids=ids[:, 8:9], past_key_values=c, use_cache=True).logits
    d = maxdiff(got, ref)
    record("beside_native_crop", max_abs=d)
    assert d < 1e-4


def test_beside_native_refuses_non_qwen4exp():
    from conftest import build_host, tiny_fmt
    with pytest.raises(ValueError):
        install_beside_native(build_host("qwen3_5"), NGramGraft(tiny_fmt()))
