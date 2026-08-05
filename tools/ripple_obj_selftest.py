"""CPU-only unit test for the B4 ripple-training objective (cam.mquake_ripple.ripple_loss_term).
No model download, no GPU. Verifies: (1) teacher-forcing index alignment (the loss targets exactly the
gold_cf tokens), (2) the loss is a scalar DIFFERENTIABLE wrt the tap params (base+store frozen ->
gradient flows only into the injector's params via the tap-injected bank), (3) it is a no-op when the
weight is 0 / the objective is off.

Run:  HIP_VISIBLE_DEVICES= <cpu-torch-python> tools/ripple_obj_selftest.py
"""
import os
import sys
import types

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cam.mquake_ripple import ripple_loss_term  # noqa: E402
import cam.mquake_ripple as mr  # noqa: E402


class FakeInj:
    """Minimal MAGInjector-shaped stub: a single trainable tap param `w`; set_bank stages the bank."""
    def __init__(self, H):
        self.w = nn.Parameter(torch.zeros(H))
        self._bank = None
        self._conf = None
        self._relidx = None

    def set_bank(self, bank, conf=None, relidx=None):
        self._bank, self._conf, self._relidx = bank, conf, relidx

    def parameters(self):
        return [self.w]


class FakeBase(nn.Module):
    """Tiny LM whose logits depend on inputs_embeds AND (when a bank is staged) on the tap param `w` —
    so ripple_loss_term's gradient wrt `w` is nonzero exactly when the tap injects."""
    def __init__(self, V, H, inj):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.head = nn.Linear(H, V, bias=False)
        self.inj = inj

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None):
        h = inputs_embeds
        if self.inj._bank is not None:                       # tap-like injection into every position
            h = h + (self.inj._bank.mean(dim=1, keepdim=True) * self.inj.w)   # [1,1,H] broadcast over T
        return types.SimpleNamespace(logits=self.head(h))


def _fresh(V, H, seed=0):
    torch.manual_seed(seed)
    inj = FakeInj(H)
    base = FakeBase(V, H, inj)
    return base, inj, base.get_input_embeddings()


def main():
    assert mr.DEV == "cpu", f"expected CPU DEV for the test, got {mr.DEV!r}"
    V, H, K = 32, 8, 3
    ex = {"bank": torch.randn(1, K, H), "conf": None, "relidx": 0,
          "prompt_ids": [5, 6, 3], "gold_ids": [7, 8]}       # ctx = [5,6,3, 7,8]; gold at ctx indices 3,4

    # ---- (1) teacher-forcing alignment: the loss must target exactly gold_ids at the right positions ----
    # A base rigged so ctx index 2 (predicts gold[0]) and index 3 (predicts gold[1]) strongly favour the
    # intended gold tokens -> aligned loss ~0; a SHIFTED gold must score strictly higher (wrong span).
    base, inj, base_embed = _fresh(V, H, seed=1)
    with torch.no_grad():
        base.emb.weight.zero_()
        base.head.weight.zero_()
        base.emb.weight[3, 0] = 5.0     # ctx index 2 holds prompt token 3 -> feature0
        base.emb.weight[7, 1] = 5.0     # ctx index 3 holds gold token 7   -> feature1
        base.head.weight[7, 0] = 4.0    # feature0 -> predict token 7 (gold[0]) at index 2
        base.head.weight[8, 1] = 4.0    # feature1 -> predict token 8 (gold[1]) at index 3
    inj.set_bank(None)
    loss_aligned = float(ripple_loss_term(base, base_embed, inj, ex, weight=1.0).detach())
    loss_wrong = float(ripple_loss_term(base, base_embed, inj, dict(ex, gold_ids=[8, 7]), weight=1.0).detach())
    assert loss_aligned < loss_wrong - 1.0, (loss_aligned, loss_wrong)
    print(f"[riptest] (1) teacher-forcing alignment: aligned {loss_aligned:.3f} < wrong {loss_wrong:.3f}: PASS")

    # ---- (2) differentiable wrt the TAP params (random base -> nonzero loss + nonzero grad) ----
    base, inj, base_embed = _fresh(V, H, seed=2)
    inj.w.grad = None
    loss = ripple_loss_term(base, base_embed, inj, ex, weight=0.5)
    assert loss.dim() == 0, "loss must be a scalar"
    loss.backward()
    assert inj.w.grad is not None, "no grad reached the tap param"
    gnorm = float(inj.w.grad.norm())
    assert torch.isfinite(inj.w.grad).all() and gnorm > 1e-6, f"tap-param grad must be finite+nonzero (got {gnorm})"
    assert inj._bank is None, "ripple_loss_term must clear the staged bank afterwards"
    print(f"[riptest] (2) differentiable wrt tap params (loss {float(loss.detach()):.3f}, "
          f"‖grad_w‖={gnorm:.4f}), bank cleared after: PASS")

    # ---- (3) no-op when off: weight 0 -> exactly-zero loss, no gradient contribution ----
    base, inj, base_embed = _fresh(V, H, seed=3)
    inj.w.grad = None
    loss0 = ripple_loss_term(base, base_embed, inj, ex, weight=0.0)
    assert float(loss0.detach()) == 0.0, f"weight=0 must give exactly 0 loss (got {float(loss0.detach())})"
    loss0.backward()
    assert inj.w.grad is None or float(inj.w.grad.norm()) == 0.0, "weight=0 must contribute no gradient"
    print("[riptest] (3) weight=0 -> exact 0 loss, no gradient contribution (no-op): PASS")

    # ---- (4) weight scales the loss linearly ----
    base, inj, base_embed = _fresh(V, H, seed=4)
    a = float(ripple_loss_term(base, base_embed, inj, ex, weight=1.0).detach())
    b = float(ripple_loss_term(base, base_embed, inj, ex, weight=0.5).detach())
    assert a > 0.5 and abs(a - 2 * b) < 1e-4, (a, b)
    print(f"[riptest] (4) weight scales loss linearly (w=1 -> {a:.3f}, w=0.5 -> {b:.3f}): PASS")

    print("\n[riptest] ALL PASS")


if __name__ == "__main__":
    main()
