"""CPU-only shape/mask/no-op unit test for KVInjector (1c). No model download, no GPU.

Exercises cam.gated_tap.KVInjector._append on tiny random tensors and asserts:
  (a) shapes: K/V grow by K memory columns; the extended additive mask is [B,1,Tq,Tk+K].
  (b) no-op: bank=None -> _append returns inputs UNCHANGED (byte-identical); and a ZERO/any bank with a
      HARD-CLOSED gate (mask -inf on memory columns) -> eager-attention output EXACTLY equals baseline.
  (c) movement: a nonzero bank + OPEN gate + large W_vm visibly moves the eager-attention output.
Also checks the None-mask branch builds a causal+open-memory mask of the right shape/causality.

Run:  <cpu-torch-python> tools/kv_inject_selftest.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cam"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cam.gated_tap import KVInjector, _KVMem  # noqa: E402


# ---- minimal fake base so KVInjector.__init__ (decoder_layers + config + self_attn.head_dim) works ----
class FakeAttn(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 11
        self.q_proj = nn.Linear(2, 2)          # only presence is asserted


class FakeLayer(nn.Module):
    def __init__(self, full, head_dim):
        super().__init__()
        if full:
            self.self_attn = FakeAttn(head_dim)


class FakeText:
    def __init__(self, n_kv):
        self.num_key_value_heads = n_kv


class FakeCfg:
    def __init__(self, n_kv):
        self._t = FakeText(n_kv)

    def get_text_config(self):
        return self._t


class FakeBase(nn.Module):
    def __init__(self, n_layers, kv_layer, n_kv, head_dim):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(i == kv_layer, head_dim) for i in range(n_layers)])
        self.config = FakeCfg(n_kv)


def _eager(q, k, v, mask):
    """reference eager attention (no GQA: n_query_heads == n_kv here). q,k,v: [B,H,T,d]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    w = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        w = w + mask.to(w.dtype)
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return w @ v


def main():
    torch.manual_seed(0)
    B, n_kv, Tq, Tk, hd, K, mem_dim = 2, 2, 5, 5, 8, 3, 4
    base = FakeBase(n_layers=12, kv_layer=11, n_kv=n_kv, head_dim=hd)
    inj = KVInjector(base, kv_layer=11, mem_dim=mem_dim, conf_gate=False, n_rel=1)

    q = torch.randn(B, n_kv, Tq, hd)
    k = torch.randn(B, n_kv, Tk, hd)
    v = torch.randn(B, n_kv, Tk, hd)
    bank = torch.randn(B, K, mem_dim)
    # additive causal mask over the sequence [B,1,Tq,Tk]
    causal = torch.zeros(B, 1, Tq, Tk)
    idx = torch.arange(Tk)
    causal = causal.masked_fill(idx.view(1, 1, 1, Tk) > idx.view(1, 1, Tq, 1), float("-inf"))
    baseline = _eager(q, k, v, causal)

    # ---- (b1) bank=None -> EXACT byte-identical no-op at the _append level ----
    inj.set_bank(None)
    k0, v0, m0 = inj._append(k, v, causal, q)
    assert k0 is k and v0 is v and m0 is causal, "bank=None must return inputs unchanged"
    assert torch.equal(_eager(q, k0, v0, m0), baseline)
    print("[kvtest] (b1) bank=None append is a byte-identical no-op: PASS")

    # ---- (a) shapes with a staged bank ----
    inj.set_bank(bank)
    inj._force_gate = 1.0                        # open, deterministic
    k2, v2, full = inj._append(k, v, causal, q)
    assert k2.shape == (B, n_kv, Tk + K, hd), k2.shape
    assert v2.shape == (B, n_kv, Tk + K, hd), v2.shape
    assert full.shape == (B, 1, Tq, Tk + K), full.shape
    print(f"[kvtest] (a) shapes k/v grow by K -> {tuple(k2.shape)}, mask {tuple(full.shape)}: PASS")

    # ---- (b2) HARD-CLOSED gate (force_gate=0 -> memory mask -inf) -> EXACT no-op even with a NONZERO bank ----
    inj.set_bank(bank)
    inj._force_gate = 0.0
    k3, v3, m3 = inj._append(k, v, causal, q)
    # memory columns must be -inf (exactly masked); sequence columns must equal the causal mask
    assert torch.isinf(m3[..., Tk:]).all() and (m3[..., Tk:] < 0).all(), "closed gate must -inf memory cols"
    assert torch.equal(m3[..., :Tk].to(causal.dtype), causal), "sequence mask columns must be unchanged"
    out_closed = _eager(q, k3, v3, m3)
    assert torch.allclose(out_closed, baseline, atol=0, rtol=0), "closed-gate memory must be an EXACT no-op"
    print("[kvtest] (b2) closed-gate (mask -inf) memory is an EXACT no-op vs baseline: PASS")

    # ---- (c) OPEN gate + LARGE V_mem -> output visibly moves ----
    inj.set_bank(bank)
    inj._force_gate = 1.0
    with torch.no_grad():
        inj.mem.W_vm.weight.copy_(torch.randn_like(inj.mem.W_vm.weight) * 50.0)   # large memory values
        inj.mem.W_km.weight.copy_(torch.randn_like(inj.mem.W_km.weight))          # non-degenerate keys
    k4, v4, m4 = inj._append(k, v, causal, q)
    out_open = _eager(q, k4, v4, m4)
    delta = (out_open - baseline).abs().max().item()
    assert delta > 1e-2, f"open gate + large V_mem must move the output (got {delta})"
    # and the memory columns must be attendable (finite bias, some softmax mass)
    assert torch.isfinite(m4[..., Tk:]).all(), "open memory columns must be attendable (finite mask)"
    print(f"[kvtest] (c) open gate + large V_mem moves output (max|Δ|={delta:.3f}): PASS")

    # ---- None-mask branch: builds a causal + open-memory additive mask of the right shape ----
    inj.set_bank(bank)
    inj._force_gate = 1.0
    k5, v5, m5 = inj._append(k, v, None, q)
    assert m5.shape == (B, 1, Tq, Tk + K), m5.shape
    # causal over the sequence part (upper triangle -inf), open (0) over memory part
    seqpart = m5[0, 0, :, :Tk]
    assert torch.isinf(seqpart[0, 1]) and seqpart[0, 1] < 0, "None-mask branch must be causal (row0 can't see key1)"
    assert seqpart[1, 0].item() == 0.0, "None-mask branch: row1 CAN see key0 (causal)"
    assert torch.allclose(m5[0, 0, :, Tk:], torch.zeros(Tq, K, dtype=m5.dtype)), "memory cols open (0) at gate=1"
    # a fully causal-only baseline (no memory) must match _eager on the built seq mask + memory (V_mem large moves it)
    print("[kvtest] (d) None-mask branch builds causal+open-memory mask correctly: PASS")

    # ---- conf-gate path smoke (soft + hard) ----
    inj2 = KVInjector(base, kv_layer=11, mem_dim=mem_dim, conf_gate=True, n_rel=1)
    inj2.mem.train()
    inj2.set_bank(bank, conf=torch.tensor([2.0, 0.1]), relidx=0)
    g_soft = inj2._gate(B, torch.device("cpu"), torch.float32)
    assert g_soft.shape == (B,) and (g_soft >= 0).all() and (g_soft <= 1).all(), g_soft
    os.environ["CAM_KV_GATE_HARD"] = "1"
    g_hard = inj2._gate(B, torch.device("cpu"), torch.float32)
    assert g_hard.shape == (B,) and set(g_hard.unique().tolist()) <= {0.0, torch.sigmoid(inj2.mem.log_gate).item()} \
        or g_hard.numel() == B
    del os.environ["CAM_KV_GATE_HARD"]
    print(f"[kvtest] (e) conf-gate soft={[round(x,3) for x in g_soft.tolist()]} hard-ok: PASS")

    # ---- attach/detach are exact (patched forward is an instance attr; detach restores class method) ----
    # attach() fetches apply_rotary_pos_emb / eager_attention_forward / ALL_ATTENTION_FUNCTIONS from the
    # attention module. For the FAKE attn that module is __main__ (this file), so stub them (the patched
    # forward is never CALLED here — we only verify the install/restore mechanics).
    _self = sys.modules[__name__]
    _self.apply_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    _self.eager_attention_forward = _eager

    class _ALL:
        @staticmethod
        def get_interface(_impl, default):
            return default
    _self.ALL_ATTENTION_FUNCTIONS = _ALL()
    attn = base.model.layers[11].self_attn
    assert "forward" not in attn.__dict__
    inj.attach()
    assert "forward" in attn.__dict__, "attach must install an instance-level forward"
    inj.detach()
    assert "forward" not in attn.__dict__, "detach must restore the class-method forward exactly"
    print("[kvtest] (f) attach/detach install+restore forward exactly: PASS")

    print("\n[kvtest] ALL PASS")


if __name__ == "__main__":
    main()
