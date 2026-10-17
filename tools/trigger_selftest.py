"""CPU-only unit test for the E1 (#104) trigger machinery — no model download, no GPU.

Exercises cam.gated_tap: n-gram hashing + key sets; fire-mask correctness; the GatedMemoryTap mask
path (bank=None no-op; empty-keys -> EXACT no-op with a staged bank; all-fires == unmasked output);
and the TriggerInjector plumbing on a fake base (mask install/clear lifecycle, stats, A1 no-mask mode).

Run:  <cpu-torch-python> tools/trigger_selftest.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cam"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cam.gated_tap import (GatedMemoryTap, TriggerInjector, build_trigger_keys,   # noqa: E402
                           trigger_mask_for, _hash_ngram)


class FakeText:
    def __init__(self, hidden):
        self.hidden_size = hidden


class FakeCfg:
    def __init__(self, hidden):
        self._t = FakeText(hidden)

    def get_text_config(self):
        return self._t


class FakeLayer(nn.Module):
    def forward(self, h, **kw):
        return h


class FakeBase(nn.Module):
    def __init__(self, hidden=32, n_layers=16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])
        self.config = FakeCfg(hidden)


def main():
    torch.manual_seed(0)

    # ---- (a) hashing + key sets ----
    subj = [101, 202, 303]                                 # a 3-token subject
    keys, used = build_trigger_keys(subj, orders=(2, 3))
    assert used == (2, 3) and len(keys) == 3, (used, len(keys))          # 2 bigrams + 1 trigram
    assert _hash_ngram(subj[:2]) in keys and _hash_ngram(subj) in keys
    k1, u1 = build_trigger_keys([77], orders=(2, 3))
    assert u1 == (1,) and _hash_ngram([77]) in k1, "single-token subject must fall back to the unigram"
    print("[trigtest] (a) key sets (multi-order + unigram fallback): PASS")

    # ---- (b) fire-mask correctness ----
    seq = [9, 9, 101, 202, 303, 9, 101, 202, 9]            # subject; then a PARTIAL bigram repeat
    m = trigger_mask_for(seq, keys, used)
    expect = [0, 0, 1, 1, 1, 0, 1, 1, 0]                   # spans fire (bigram (101,202) fires too)
    assert m.tolist()[0] == [float(x) for x in expect], m.tolist()
    mz = trigger_mask_for(seq, frozenset(), (2, 3))
    assert float(mz.sum()) == 0.0, "empty key set must fire NOTHING"
    print("[trigtest] (b) fire-mask positions + empty-set zero mask: PASS")

    # ---- (c) GatedMemoryTap mask path ----
    H, mem, K, B, T = 32, 16, 4, 1, 9
    tap = GatedMemoryTap(H, mem, n_heads=4, conf_gate=False)
    h = torch.randn(B, T, H)
    bank = torch.randn(B, K, mem)
    tap.set_bank(None)
    assert torch.equal(tap(h), h), "bank=None must be a byte-identical no-op"
    with torch.no_grad():                                  # open the gate + make the read visible
        tap.gamma.fill_(3.0)
        tap.to_v.weight.copy_(torch.randn_like(tap.to_v.weight) * 2.0)
        tap.to_o.weight.copy_(torch.randn_like(tap.to_o.weight) * 2.0)
    tap.set_bank(bank)
    out0 = tap(h)
    assert not torch.equal(out0, h), "open gate + staged bank must move the output"
    tap.set_trigger_mask(torch.zeros(B, T))                # empty trigger set, bank STILL staged
    assert torch.equal(tap(h), h), "all-zero fire mask must be an EXACT no-op with the bank staged"
    tap.set_trigger_mask(torch.ones(B, T))
    assert torch.equal(tap(h), out0), "all-fires mask must equal the unmasked output"
    tap.set_trigger_mask(None)
    assert torch.equal(tap(h), out0), "mask=None must restore the historical path"
    print("[trigtest] (c) tap mask: bank=None no-op / exact-zero off-mask / all-fires == unmasked: PASS")

    # ---- (d) TriggerInjector plumbing on a fake base ----
    base = FakeBase(hidden=H, n_layers=16)
    inj = TriggerInjector(base, trigger_layer=12, mem_dim=mem, n_heads=4, conf_gate=False,
                          n_rel=1, orders=(2, 3), apply_mask=True).to("cpu")
    inj.set_trigger_keys(subj)
    ids = torch.tensor([seq], dtype=torch.long)
    inj.set_bank(bank)                                     # staging a bank must NOT clear the mask
    inj.set_trigger_ids(ids)
    tap12 = inj._mag.taps["12"]
    assert tap12._trigger_mask is not None and tap12._trigger_mask.tolist()[0] == m.tolist()[0]
    assert inj.fire_stats()["fire_positions"] == 5
    inj.set_bank(None)                                     # bank cleared -> mask cleared (no stale fires)
    assert tap12._trigger_mask is None
    inj.set_trigger_keys(None)                             # keys cleared -> a staged bank fires nothing
    inj.set_bank(bank)
    inj.set_trigger_ids(ids)
    assert float(tap12._trigger_mask.sum()) == 0.0
    inj2 = TriggerInjector(base, trigger_layer=12, mem_dim=mem, n_heads=4, conf_gate=False,
                           n_rel=1, orders=(2, 3), apply_mask=False).to("cpu")
    inj2.set_trigger_keys(subj)
    inj2.set_bank(bank)
    inj2.set_trigger_ids(ids)
    assert inj2._mag.taps["12"]._trigger_mask is None, "nomask (A1) arm must never gate positions"
    print("[trigtest] (d) TriggerInjector mask lifecycle + A1 no-mask mode: PASS")

    # ---- (e) stats + incremental mask semantics ----
    inj.set_trigger_keys(subj)                     # re-stage the span (section (d) cleared it)
    inj.reset_stats()
    for _ in range(3):
        inj.set_trigger_ids(ids)
    st = inj.fire_stats()
    assert st["forwards"] == 3 and st["fire_positions"] == 15 and st["zero_forwards"] == 0, st
    print(f"[trigtest] (e) fire stats {st}: PASS")

    # ---- (f) forward-span mask semantics (cached decode) ----
    seq_inc = [9, 101, 202, 303]                   # ends ON a fired position
    m_full = trigger_mask_for(seq_inc, keys, used)
    assert m_full.tolist()[0][-1] == 1.0
    inj.set_trigger_ids(torch.tensor([seq_inc], dtype=torch.long), fwd_len=1)   # decode step
    m_inc = tap12._trigger_mask
    assert m_inc is not None and m_inc.shape[1] == 1 and float(m_inc.sum()) == 1.0, m_inc
    inj.set_trigger_ids(torch.tensor([seq_inc], dtype=torch.long), fwd_len=None)  # prefill: full span
    assert tap12._trigger_mask.shape[1] == len(seq_inc) and float(tap12._trigger_mask.sum()) == 3.0
    inj.set_trigger_ids(torch.tensor([[9, 9, 101, 9]], dtype=torch.long), incremental=True)
    assert float(tap12._trigger_mask.sum()) == 0.0, "last token not covered -> no fire this forward"
    print("[trigtest] (f) forward-span mask (decode vs prefill vs legacy incremental): PASS")

    print("\n[trigtest] ALL PASS")


if __name__ == "__main__":
    main()
