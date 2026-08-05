"""CPU-only validation that the residual MAGInjector tap fires correctly on pure-SSM donor blocks.
No weight download (tiny random configs), no GPU. Builds REAL HF SSM models (Mamba, Mamba2,
FalconMamba, RWKV), then for each:
  (1) decoder_layers() locates the block ModuleList -> report the path + count.
  (2) MAGInjector attaches; with NO bank (and with a bank but the zero-init gate) the model forward is
      BYTE-IDENTICAL to baseline (the tap is a no-op) -> untouched-base guarantee.
  (3) with a bank + an OPENED gate the output MOVES (the tap injects into the residual stream).
  (4) detach() restores byte-identical baseline.
  (5) block-level: the injected delta lands on the residual hidden (out[0] for RWKV's tuple return /
      the tensor for Mamba) and any other tuple elements (RWKV state) are PRESERVED — proving the
      existing post-hook composes on both the tensor and tuple block return shapes with no per-arch
      variant.

Run:  HIP_VISIBLE_DEVICES= <cpu-torch-python> tools/ssm_tap_selftest.py
"""
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cam.gated_tap import MAGInjector, decoder_layers  # noqa: E402


def _build_models():
    from transformers import (MambaConfig, MambaForCausalLM, Mamba2Config, Mamba2ForCausalLM,
                              FalconMambaConfig, FalconMambaForCausalLM, RwkvConfig, RwkvForCausalLM)
    H = 32
    models = {}
    models["mamba"] = MambaForCausalLM(MambaConfig(
        vocab_size=64, hidden_size=H, num_hidden_layers=4, state_size=8, intermediate_size=64))
    models["mamba2"] = Mamba2ForCausalLM(Mamba2Config(
        vocab_size=64, hidden_size=H, num_hidden_layers=4, state_size=8, num_heads=8, head_dim=8,
        n_groups=1, expand=2, intermediate_size=64, chunk_size=8))
    models["falcon_mamba"] = FalconMambaForCausalLM(FalconMambaConfig(
        vocab_size=64, hidden_size=H, num_hidden_layers=4, state_size=8, intermediate_size=64))
    models["rwkv"] = RwkvForCausalLM(RwkvConfig(
        vocab_size=64, hidden_size=H, num_hidden_layers=4, attention_hidden_size=H,
        intermediate_size=64, context_length=64))
    for m in models.values():
        m.eval()
    return models


@torch.no_grad()
def _logits(model, ids):
    return model(input_ids=ids).logits.clone()


def validate(name, model):
    torch.manual_seed(0)
    layers = decoder_layers(model)                       # (1) raises if not found
    n = len(layers)
    L = n // 2
    ids = torch.randint(0, 64, (1, 6))
    base = _logits(model, ids)

    inj = MAGInjector(model, [L], mem_dim=16, n_heads=2, conf_gate=False, n_rel=1)
    inj.attach().eval()

    # (2a) no bank -> exact no-op
    inj.set_bank(None)
    none_out = _logits(model, ids)
    assert torch.equal(none_out, base), f"{name}: bank=None must be a byte-identical no-op"

    # (2b) bank set but zero-init gate (gamma=0) -> still exact no-op
    bank = torch.randn(1, 3, 16)
    inj.set_bank(bank)
    gate0_out = _logits(model, ids)
    assert torch.equal(gate0_out, base), f"{name}: zero-init gate must be a byte-identical no-op even with a bank"

    # (3) open the gate -> the tap injects -> output MOVES
    with torch.no_grad():
        inj.taps[str(L)].gamma.fill_(0.8)                # tanh(0.8)~0.66 -> nonzero gated injection
    moved = _logits(model, ids)
    delta = (moved - base).abs().max().item()
    assert delta > 1e-4, f"{name}: opened gate + bank must move the output (got {delta})"

    # (5) block-level return-shape: injection lands on the residual hidden; other tuple elems preserved
    block = layers[L]
    h = torch.randn(1, 6, model.config.get_text_config().hidden_size)
    inj.set_bank(None)
    native = block(h)                                    # no-op (bank None) -> the block's native output
    inj.set_bank(bank)                                   # gate is open now
    hooked = block(h)
    if isinstance(native, tuple):
        assert isinstance(hooked, tuple) and len(hooked) == len(native), f"{name}: tuple shape not preserved"
        assert not torch.equal(hooked[0], native[0]), f"{name}: injection must change the residual hidden out[0]"
        for j in range(1, len(native)):
            a, b = native[j], hooked[j]
            same = (a is b) or (a is None and b is None) or (torch.is_tensor(a) and torch.equal(a, b))
            assert same, f"{name}: tuple element {j} (state/aux) must be preserved"
        ret = f"TUPLE(len={len(native)}) — tap on out[0], out[1:]=state preserved"
    else:
        assert torch.is_tensor(hooked) and not torch.equal(hooked, native), f"{name}: tensor injection must move output"
        ret = "TENSOR — tap on the residual-stream tensor"

    inj.detach()
    # (4) detach restores baseline
    restored = _logits(model, ids)
    assert torch.equal(restored, base), f"{name}: detach must restore the byte-identical baseline"

    path = _found_path(model)
    print(f"[ssmtest] {name:13s} OK | path={path:20s} n_layers={n} | tap L={L} | move max|Δ|={delta:.4f} | "
          f"block-return={ret}")
    return True


def _found_path(model):
    import torch.nn as nn
    for path in ("model.layers", "backbone.layers", "model.backbone.layers", "rwkv.blocks",
                 "backbone.blocks", "model.blocks", "transformer.h"):
        obj, ok = model, True
        for a in path.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return path
    return "?"


def main():
    models = _build_models()
    ok = 0
    for name, m in models.items():
        try:
            ok += int(validate(name, m))
        except Exception as e:
            print(f"[ssmtest] {name:13s} FAIL: {type(e).__name__}: {str(e)[:200]}")
    print(f"\n[ssmtest] {ok}/{len(models)} SSM archs validated for the residual tap")
    assert ok == len(models), "not all SSM archs validated"
    print("[ssmtest] ALL PASS")


if __name__ == "__main__":
    main()
