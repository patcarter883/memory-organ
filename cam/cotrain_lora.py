"""Rung 2 of the memlayer-contrast ladder (docs/zaya-port/CAM_MEMLAYER_CONTRAST_SPEC.md):
co-train a low-rank delta on the composition-band attention JOINTLY with the memory tap, so the
(otherwise frozen) reasoning circuit learns to ROUTE ON the injected memory read — the Titans-MAC/LMM
mechanism (read/gate/projections trained WITH attention). This is the confirming positive contrast to
the frozen-base bolt-on tap, which delivers single-hop but cannot integrate an edit into multi-hop
reasoning (attendable/injected != composed; see cam-b4-ripple-shortcut-not-belief).

We add LoRA to the self-attention q/k/v/o projections of the LATE full-attention layers (where the
second hop fires — Biran "Hopping Too Late", CaKE circuit-mismatch). The base weights stay frozen; only
these low-rank adapters + the tap train. Gradient reaches the adapters for free: the ripple/edit losses
already backprop through the base (train_taps), so once the adapter params require grad and are in the
optimizer they co-train on the same objective. Adapters are LEFT ATTACHED after training so eval runs
the co-trained circuit.
"""
import torch
import torch.nn as nn

try:
    from .gated_tap import decoder_layers
except ImportError:
    from gated_tap import decoder_layers


class _LoRA(nn.Module):
    """y += scale * (x @ A^T) @ B^T ; B zero-init -> exact no-op at start (base output unchanged)."""

    def __init__(self, in_f, out_f, rank, alpha, device):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32, device=device))
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = float(alpha) / float(rank)
        self.enabled = True  # toggled off to get the frozen-base reference for the preservation anchor

    def delta(self, x):
        if not self.enabled:
            return x.new_zeros(())  # broadcast-add of 0 -> exact frozen-base forward
        x32 = x.to(torch.float32)
        return ((x32 @ self.A.t()) @ self.B.t()) * self.scale


def attach_cotrain_lora(base, layer_idx, targets, rank=8, alpha=16):
    """Attach LoRA forward-hooks to `targets` (proj names) on the given decoder layer indices.
    Returns (adapters, params, handles). Skips layers/projections that don't exist (e.g. a GDN/linear-
    attention layer has no self_attn.q_proj) with a printed note. Stores state on `base._cotrain_lora`
    so it survives until eval; DO NOT detach before evaluating the co-trained model."""
    layers = decoder_layers(base)
    adapters, params, handles, wired, skipped = [], [], [], [], []
    for li in layer_idx:
        if li < 0 or li >= len(layers):
            skipped.append(f"L{li}(oob)")
            continue
        attn = getattr(layers[li], "self_attn", None) or getattr(layers[li], "attn", None)
        if attn is None:
            skipped.append(f"L{li}(no-attn)")
            continue
        for tname in targets:
            mod = getattr(attn, tname, None)
            if not isinstance(mod, nn.Linear):
                skipped.append(f"L{li}.{tname}")
                continue
            dev = next(mod.parameters()).device
            lora = _LoRA(mod.in_features, mod.out_features, rank, alpha, dev)
            adapters.append(lora)
            params += list(lora.parameters())

            def _hook(module, inp, out, _l=lora):
                return out + _l.delta(inp[0]).to(out.dtype)

            handles.append(mod.register_forward_hook(_hook))
            wired.append(f"L{li}.{tname}")
    n_p = sum(p.numel() for p in params)
    print(f"[cotrain-lora] wired {len(wired)} adapters (rank={rank} alpha={alpha}, {n_p/1e6:.2f}M params): "
          f"{','.join(wired)}", flush=True)
    if skipped:
        print(f"[cotrain-lora] skipped: {','.join(skipped)}", flush=True)
    base._cotrain_lora = {"adapters": adapters, "handles": handles}
    return adapters, params, handles


def set_cotrain_lora_enabled(base, flag):
    """Toggle every attached LoRA adapter on/off (for the memory-OFF preservation-anchor reference)."""
    state = getattr(base, "_cotrain_lora", None)
    if not state:
        return
    for a in state["adapters"]:
        a.enabled = bool(flag)
