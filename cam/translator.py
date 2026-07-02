"""CAM v1 — the base-agnostic affine translator (CAM_DESIGN §2.2).

The v0 memory front-end (frozen BoltAdapter -> mem-space bank) and a passing GatedMemoryTap are
FROZEN and reused verbatim. The bank fed to the tap ([B,K,mem_dim]) is base-AGNOSTIC — it is the
DeepMemory's own mem_dim retrieval, never any base's hidden space — so the SAME frozen memory drives
a second base. The only thing that differs across bases is the residual-stream geometry the tap
stitches into; v1 closes that gap with a TINY affine translator (residual-stream stitching transfers
*function* across LLMs, 2506.06609):

    A : d_base2 -> d_base1     (in)   # lift base-2's residual into the frozen tap's hidden space
    [ FROZEN GatedMemoryTap in d_base1, with the FROZEN mem bank ]  -> additive update u (d_base1)
    B : d_base1 -> d_base2     (out)  # project the tap's update back to base-2's residual
    h2' = h2 + tanh(gamma2) ⊙ B(u)    # gamma2 in R^{d_base2}, init 0 => exact no-op at init

Only A, B, gamma2 train (LM-loss through the frozen base-2). Mirrors the gated_tap dtype pattern
(compute in param fp32, cast the additive update back to the base dtype) and adds a NaN/grad guard so
the new gate can't diverge the L=16-style way.
"""
import os
import sys

import torch
import torch.nn as nn

# flat package: sibling imports resolve relatively when imported as cam.X and fall back to a
# path-hacked absolute import when run as a file / with cam/ on sys.path.
try:
    from .gated_tap import GatedMemoryTap, decoder_layers
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from gated_tap import GatedMemoryTap, decoder_layers  # noqa: E402


def save_translator(path, injector, meta):
    """Persist the fitted translator card (state_dict of the trainable TranslatedTap + meta) — the
    §5.5 UMX product artifact. The frozen tap/memory are NOT saved here (they live in the v0 memory
    checkpoint); a base's 'translator card' is just the tiny map that stitches it to that canonical
    memory. Saves the whole TranslatedTap state_dict MINUS the frozen tap so any variant
    (affine / perpos / mlp / perpos-mlp) round-trips without a per-variant serializer."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tt = injector.tt
    sd = {k: v.detach().cpu() for k, v in tt.state_dict().items() if not k.startswith("tap.")}
    torch.save({
        "state_dict": sd,
        "xlator": getattr(tt, "xlator", "affine"),
        "kc": getattr(tt, "kc", 1),
        "d_base1": tt.tap.H, "d_base2": tt.d_base2, "meta": meta,
    }, path)
    print(f"[v1] saved translator card ({getattr(tt, 'xlator', 'affine')}) -> {path} "
          f"(d_base2 {tt.d_base2} <-> d_base1 {tt.tap.H}, base-2 {meta.get('base2')})",
          flush=True)


class _MLPMap(nn.Module):
    """1-hidden-layer GELU map d_in -> d_out (wider, non-linear capacity vs a single Linear).
    hidden = round(mult * max(d_in, d_out)). Used as the IN (base-2->base-1) and OUT (base-1->base-2)
    maps in the 'mlp' translator variant. Init mirrors the affine Linear scale so the pre-gate
    magnitude at step 0 is comparable (the no-op-at-init property still comes from gamma2=0 alone)."""

    def __init__(self, d_in, d_out, mult=2.0):
        super().__init__()
        d_hid = max(d_out, int(round(mult * max(d_in, d_out))))
        self.fc1 = nn.Linear(d_in, d_hid, bias=True)
        self.fc2 = nn.Linear(d_hid, d_out, bias=True)
        self.act = nn.GELU()
        nn.init.normal_(self.fc1.weight, std=(1.0 / d_in) ** 0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=(1.0 / d_hid) ** 0.5)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TranslatedTap(nn.Module):
    """Wrap a FROZEN d_base1 GatedMemoryTap with a trainable in/out map so it injects into a second
    base of hidden dim d_base2. Zero-init on the gate (gamma2) keeps the start an exact no-op,
    mirroring the tap's own gamma=0 stability property.

    Translator variants (all keep the memory/store/tap FROZEN; only the translator trains):
      affine       — ONE shared affine pair A/B + gamma2 (the 13.1M byte-preserved baseline).
      perpos       — a SEPARATE affine map (A_t,B_t,gamma_t) per answer position t in [0,Kc). A
                     K-token sequence isn't forced through one shared linear map — mirrors the
                     disjoint per-position store fix (exp#4) on the translator side.
      mlp          — a 1-hidden-layer GELU (wider, non-linear) map, SHARED across positions.
      perpos-mlp   — a SEPARATE 1-hidden-layer GELU map per answer position (both levers).

    For perpos variants: only the LAST Kc residual positions predict the answer tokens (position
    T-Kc+t predicts answer token t), so the per-position map t is applied to residual position
    T-Kc+t; residual positions before the answer window use position-0's map (they never contribute
    to the answer logits, so their translator choice is loss-irrelevant — but must stay finite)."""

    def __init__(self, frozen_tap: GatedMemoryTap, d_base2: int, xlator: str = "affine", kc: int = 1,
                 mlp_mult: float = 2.0):
        super().__init__()
        self.tap = frozen_tap                       # FROZEN (caller sets requires_grad_(False))
        d_base1 = frozen_tap.H
        self.d_base2 = d_base2
        self.xlator = xlator
        self.kc = int(kc)
        self.mlp_mult = mlp_mult
        self._perpos = xlator in ("perpos", "perpos-mlp")
        self._mlp = xlator in ("mlp", "perpos-mlp")
        n_slot = self.kc if self._perpos else 1

        def _affine_pair():
            A = nn.Linear(d_base2, d_base1, bias=True)        # in: base-2 residual -> tap hidden
            B = nn.Linear(d_base1, d_base2, bias=True)        # out: tap update    -> base-2 residual
            # A and B BOTH randomly initialised so grad flows from step 0 (double-zero-init on
            # B AND gamma2 deadlocks). No-op-at-init comes from gamma2=0 ALONE (tanh(0)=0), like V0.
            nn.init.normal_(A.weight, std=(1.0 / d_base2) ** 0.5)
            nn.init.zeros_(A.bias)
            nn.init.normal_(B.weight, std=(1.0 / d_base1) ** 0.5)
            nn.init.zeros_(B.bias)
            return A, B

        if self.xlator == "affine":
            # BYTE-PRESERVED baseline: keep the exact param names/init/forward of the original.
            self.A, self.B = _affine_pair()
            self.gamma2 = nn.Parameter(torch.zeros(d_base2))
        else:
            # perpos / mlp / perpos-mlp: a ModuleList of n_slot in/out maps + one gamma2 per slot.
            self.A_list = nn.ModuleList()
            self.B_list = nn.ModuleList()
            for _ in range(n_slot):
                if self._mlp:
                    self.A_list.append(_MLPMap(d_base2, d_base1, mlp_mult))
                    self.B_list.append(_MLPMap(d_base1, d_base2, mlp_mult))
                else:
                    A, B = _affine_pair()
                    self.A_list.append(A); self.B_list.append(B)
            self.gammas = nn.Parameter(torch.zeros(n_slot, d_base2))   # tanh(0)=0 -> no-op at init
        self._bank = None
        self.last_gate = torch.tensor(0.0)

    # ---- trainable params (the frozen tap is excluded) ----
    def trainable_params(self):
        tap_ids = {id(p) for p in self.tap.parameters()}
        return [p for p in self.parameters() if id(p) not in tap_ids and p.requires_grad]

    def set_bank(self, bank):
        self._bank = bank
        self.tap.set_bank(bank)

    def clamp_gate(self, lo=-6.0, hi=6.0):
        """Divergence guard on the gate param(s), variant-agnostic (gamma2 for affine, gammas else)."""
        with torch.no_grad():
            if self.xlator == "affine":
                self.gamma2.clamp_(lo, hi)
            else:
                self.gammas.clamp_(lo, hi)

    def _tap_update(self, h1):
        # reuse the frozen tap's cross-attention math, but read its ADDITIVE update only. The tap
        # returns h1 + tanh(gamma)*y (gamma frozen) -> u = tap(h1) - h1 is the frozen memory update.
        return self.tap(h1) - h1

    def forward(self, h2):
        """h2: [B,T,d_base2] residual hidden -> injected. No-op when no bank is set."""
        if self._bank is None:
            return h2
        if self.xlator == "affine":
            wdt = self.A.weight.dtype                          # fp32 compute (stable gate training)
            h2c = h2.to(wdt)
            u1 = self._tap_update(self.A(h2c))                 # [B,T,d_base1] frozen memory update
            u2 = self.B(u1)                                    # [B,T,d_base2] project back
            g = torch.tanh(self.gamma2)
            self.last_gate = g.abs().mean().detach()
            return h2 + (g * u2).to(h2.dtype)

        wdt = self.A_list[0].fc1.weight.dtype if self._mlp else self.A_list[0].weight.dtype
        h2c = h2.to(wdt)
        if not self._perpos:
            # mlp SHARED across positions: one map applied to every position.
            u1 = self._tap_update(self.A_list[0](h2c))
            u2 = self.B_list[0](u1)
            g = torch.tanh(self.gammas[0])
            self.last_gate = g.abs().mean().detach()
            return h2 + (g * u2).to(h2.dtype)

        # perpos / perpos-mlp: position t in [0,Kc) uses map t, applied at residual pos T-Kc+t.
        # Residual positions before the answer window use map 0 (loss-irrelevant, kept finite).
        T = h2c.shape[1]
        Kc = min(self.kc, T)
        out = torch.zeros_like(h2c)
        gate_acc = 0.0
        # earlier (non-answer) positions -> map 0
        if T > Kc:
            pre = h2c[:, : T - Kc]
            u1 = self._tap_update(self.A_list[0](pre))
            out[:, : T - Kc] = torch.tanh(self.gammas[0]) * self.B_list[0](u1)
        for t in range(Kc):
            j = T - Kc + t
            seg = h2c[:, j:j + 1]                              # [B,1,d_base2]
            u1 = self._tap_update(self.A_list[t](seg))
            g = torch.tanh(self.gammas[t])
            out[:, j:j + 1] = g * self.B_list[t](u1)
            gate_acc = gate_acc + g.abs().mean().detach()
        self.last_gate = (gate_acc / max(1, Kc))
        return h2 + out.to(h2.dtype)


class TranslatedInjector:
    """Registers a TranslatedTap forward-hook on one frozen base-2 decoder layer."""

    def __init__(self, base2, frozen_tap, tap_layer, xlator="affine", kc=1, mlp_mult=2.0):
        d_base2 = base2.config.get_text_config().hidden_size
        self.layers = decoder_layers(base2)
        self.tap_layer = tap_layer
        self.tt = TranslatedTap(frozen_tap, d_base2, xlator=xlator, kc=kc, mlp_mult=mlp_mult)
        self._handle = None

    def to(self, dev):
        self.tt.to(dev)
        return self

    def parameters(self):
        # train ONLY the translator; the wrapped tap is frozen.
        return self.A_params()

    def A_params(self):
        # generic across variants: every trainable TranslatedTap param (frozen tap excluded).
        return self.tt.trainable_params()

    def train(self, mode=True):
        self.tt.train(mode)
        return self

    def eval(self):
        self.tt.eval()
        return self

    def set_bank(self, bank):
        self.tt.set_bank(bank)

    def gate_stat(self):
        return float(self.tt.last_gate)

    def _hook(self):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (self.tt(out[0]),) + tuple(out[1:])
            return self.tt(out)
        return fn

    def attach(self):
        self._handle = self.layers[self.tap_layer].register_forward_hook(self._hook())
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return self
