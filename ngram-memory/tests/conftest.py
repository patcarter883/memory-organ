"""Shared helpers: tiny random hosts (qwen2, qwen3_5, qwen3_5_moe) and a tiny key spec.

Tiny qwen3_5 hosts shrink the REAL configs (tests/fixtures/configs) as P/graft/test_graft.py did.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from ngram_memory.spec import KeySpec, graft_format  # noqa: E402

V = 512  # tiny host vocab
EOS, PAD = 1, 0
HOSTS = ("qwen2", "qwen3_5", "qwen3_5_moe")


def tiny_key(orders=(2, 3)) -> KeySpec:
    return KeySpec(vocab_size=V, eos_token_id=EOS, base=1000, orders=tuple(orders))


def tiny_fmt(arm="G-lin", orders=(2, 3), **kw):
    return graft_format(64, tiny_key(orders), arm=arm, **kw)


def _text_cfg(name: str) -> dict:
    c = json.loads((FIX / "configs" / name).read_text())
    return c.get("text_config", c)


def build_host(kind: str):
    torch.manual_seed(0)
    if kind == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, eos_token_id=EOS, bos_token_id=EOS,
                          pad_token_id=PAD, tie_word_embeddings=False)
        cfg._attn_implementation = "eager"
        return Qwen2ForCausalLM(cfg).eval()
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    moe = kind == "qwen3_5_moe"
    t = _text_cfg("qwen3.6-35b-a3b.config.json" if moe else "qwen3.8-27b.config.json")
    keep = {k: t[k] for k in ("attn_output_gate", "hidden_act", "rms_norm_eps", "full_attention_interval",
                              "partial_rotary_factor", "linear_conv_kernel_dim", "output_gate_type") if k in t}
    rope = dict(t["rope_parameters"])
    rope["mrope_section"] = [2, 1, 1]
    kw = dict(keep, vocab_size=V, hidden_size=64, num_hidden_layers=4, layer_types=t["layer_types"][:4],
              num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=16,
              linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, rope_parameters=rope,
              eos_token_id=EOS, bos_token_id=EOS, pad_token_id=PAD, tie_word_embeddings=False)
    if moe:
        kw.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
        cfg, cls = Qwen3_5MoeTextConfig(**kw), Qwen3_5MoeForCausalLM
    else:
        kw.update(intermediate_size=128)
        cfg, cls = Qwen3_5TextConfig(**kw), Qwen3_5ForCausalLM
    cfg._attn_implementation = "eager"
    return cls(cfg).eval()


_HOST_CACHE: dict = {}


@pytest.fixture(params=HOSTS)
def host(request):
    if request.param not in _HOST_CACHE:
        _HOST_CACHE[request.param] = build_host(request.param)
    return request.param, _HOST_CACHE[request.param]


def rand_ids(seed: int, B: int, T: int, lo: int = 3) -> torch.Tensor:
    return torch.randint(lo, V, (B, T), generator=torch.Generator().manual_seed(seed))


def logits(model, ids, **kw) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, **kw).logits.float()


def maxdiff(a, b) -> float:
    return (a - b).abs().max().item()


@torch.no_grad()
def decode_parity(model, ids, P, mask=None, pos=None):
    """Prefill ids[:, :P] with a cache, feed the rest one token at a time; max |dlogit| vs one full forward."""
    full = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    out = model(input_ids=ids[:, :P], attention_mask=None if mask is None else mask[:, :P],
                position_ids=None if pos is None else pos[:, :P], use_cache=True)
    pre = (out.logits.float() - full[:, :P]).abs()
    if mask is not None:
        pre = pre * mask[:, :P, None]
    cache, d = out.past_key_values, [pre.max().item()]
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=cache,
                  attention_mask=None if mask is None else mask[:, :t + 1],
                  position_ids=None if pos is None else pos[:, t:t + 1], use_cache=True)
        d.append(maxdiff(o.logits.float()[:, -1], full[:, t]))
    return max(d), full


def fill_random_rows(graft, ids: torch.Tensor, seed: int = 3, scale: float = 1.0):
    """Write random rows at every row `ids` reads (so everything fires)."""
    rid = graft.row_ids(ids).reshape(-1).unique()
    g = torch.Generator().manual_seed(seed)
    graft.write_rows(rid, torch.randn(rid.numel(), graft.fmt.row_width, generator=g) * scale)
    return rid
