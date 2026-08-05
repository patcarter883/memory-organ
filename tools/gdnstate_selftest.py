"""CPU-only feasibility + unit test for GDNStateInjector (GDN recurrent-state seeding). No GPU.

Verifies the load-bearing claim: the memory seed can be written into the GDN delta-rule recurrent state
via the EXPOSED `initial_state` seam, and it is DIFFERENTIABLE (so W_sk/W_sv train by LM loss through the
frozen recurrence — no fla kernel change). Uses the model's OWN pure-torch reference kernel
`torch_chunk_gated_delta_rule` (the fallback stored as self.chunk_gated_delta_rule when fla is absent),
so this runs on CPU and directly proves the training-feasibility risk is resolved.

Checks:
  (1) _seed shape = [B, num_v_heads, head_k_dim, head_v_dim]; bank=None -> None (byte-identical no-op).
  (2) differentiable: grad flows from the recurrence OUTPUT back into W_sk/W_sv through initial_state.
  (3) the seed actually CHANGES the recurrence output (a large seed moves core_attn_out).
  (4) gate: force_gate=0 -> zero seed (no-op); attach/detach install+restore forward exactly.

Run:  HIP_VISIBLE_DEVICES= <cpu-torch-python> tools/gdnstate_selftest.py
"""
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cam.gated_tap import GDNStateInjector  # noqa: E402

# the model's OWN pure-torch delta-rule (fla fallback) — proves initial_state is differentiable on CPU
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule  # noqa: E402


# ---- minimal fake base so GDNStateInjector.__init__ (decoder_layers + linear_attn dims) works ----
class FakeGDN(nn.Module):
    def __init__(self, num_v_heads, head_k_dim, head_v_dim):
        super().__init__()
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.layer_idx = 12


class FakeLayer(nn.Module):
    def __init__(self, gdn, dims):
        super().__init__()
        if gdn:
            self.linear_attn = FakeGDN(*dims)
        else:
            self.self_attn = nn.Linear(2, 2)


class FakeCfg:
    pass


class FakeBase(nn.Module):
    def __init__(self, n_layers, gdn_layer, dims):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(i == gdn_layer, dims) for i in range(n_layers)])
        self.config = FakeCfg()


def main():
    torch.manual_seed(0)
    B, Hv, dk, dv, K, mem_dim = 2, 4, 8, 8, 3, 6
    T = 16
    base = FakeBase(16, gdn_layer=12, dims=(Hv, dk, dv))
    inj = GDNStateInjector(base, gdn_layer=12, mem_dim=mem_dim, conf_gate=False, n_rel=1)
    bank = torch.randn(B, K, mem_dim)

    # ---- (1) seed shape + None no-op ----
    inj.set_bank(None)
    assert inj._seed(B, torch.device("cpu")) is None, "bank=None must give a None seed (no-op)"
    inj.set_bank(bank)
    inj._force_gate = 1.0
    # give W_sv nonzero weights so the seed is nonzero (it is zero-init)
    with torch.no_grad():
        inj.mem.W_sv.weight.copy_(torch.randn_like(inj.mem.W_sv.weight))
        inj.mem.W_sk.weight.copy_(torch.randn_like(inj.mem.W_sk.weight))
    S = inj._seed(B, torch.device("cpu"))
    assert S.shape == (B, Hv, dk, dv), S.shape
    print(f"[gdntest] (1) seed shape {tuple(S.shape)} = [B,num_v_heads,head_k_dim,head_v_dim]; "
          f"bank=None -> None: PASS")

    # ---- build tiny GDN recurrence inputs and run the model's OWN torch kernel with the seed ----
    def _rand_qkv():
        q = torch.randn(B, T, Hv, dk)
        k = torch.randn(B, T, Hv, dk)
        v = torch.randn(B, T, Hv, dv)
        g = -torch.rand(B, T, Hv)                       # gated decay (<=0)
        beta = torch.rand(B, T, Hv)
        return q, k, v, g, beta

    q, k, v, g, beta = _rand_qkv()

    def run(initial_state):
        out, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                              output_final_state=False, use_qk_l2norm_in_kernel=True)
        return out

    base_out = run(None)                                 # no memory
    seeded_out = run(inj._seed(B, torch.device("cpu")).detach())

    # ---- (3) the seed CHANGES the recurrence output ----
    delta = (seeded_out - base_out).abs().max().item()
    assert delta > 1e-3, f"a nonzero seed must move the recurrence output (got {delta})"
    print(f"[gdntest] (3) seed moves the delta-rule output (max|Δ|={delta:.4f}): PASS")

    # ---- (2) DIFFERENTIABLE: grad flows from the output back to W_sk/W_sv through initial_state ----
    inj.mem.zero_grad(set_to_none=True)
    S_grad = inj._seed(B, torch.device("cpu"))           # keep graph (no detach)
    out = run(S_grad)
    loss = out.pow(2).mean()
    loss.backward()
    gk = inj.mem.W_sk.weight.grad
    gv = inj.mem.W_sv.weight.grad
    assert gk is not None and gv is not None, "no grad reached W_sk/W_sv through initial_state"
    nk, nv = float(gk.norm()), float(gv.norm())
    assert torch.isfinite(gk).all() and torch.isfinite(gv).all() and nk > 0 and nv > 0, (nk, nv)
    print(f"[gdntest] (2) initial_state is DIFFERENTIABLE — grad reaches W_sk (‖{nk:.4f}‖) + "
          f"W_sv (‖{nv:.4f}‖): PASS  => training the seed by LM loss is FEASIBLE (no fla change)")

    # ---- (4) gate: force_gate=0 -> zero seed (exact no-op) ----
    inj._force_gate = 0.0
    S0 = inj._seed(B, torch.device("cpu"))
    assert torch.count_nonzero(S0) == 0, "closed gate must zero the seed"
    assert torch.equal(run(S0), base_out), "zero seed must be an exact no-op vs no-memory"
    print("[gdntest] (4) closed gate -> zero seed is an EXACT no-op: PASS")

    # ---- attach/detach exactness ----
    # attach() fetches apply_mask_to_padding_states from the gdn's module; for the FAKE gdn that's
    # __main__, so stub it (the patched forward is never CALLED here — only install/restore is checked).
    sys.modules[__name__].apply_mask_to_padding_states = lambda h, m: h
    gdn = base.model.layers[12].linear_attn
    assert "forward" not in gdn.__dict__
    inj.attach()                                         # patched forward is never CALLED here (fake gdn)
    assert "forward" in gdn.__dict__, "attach must install an instance-level forward"
    inj.detach()
    assert "forward" not in gdn.__dict__, "detach must restore the class-method forward exactly"
    print("[gdntest] (5) attach/detach install+restore forward exactly: PASS")

    print("\n[gdntest] ALL PASS — GDN-state seeding is TRACTABLE + differentiable on CPU")


if __name__ == "__main__":
    main()
