"""Tiny random qwen4_exp and gemma4 hosts shrunk from the REAL configs in tests/fixtures/configs.

qwen4_exp: port of P/t1/src/tiny.py (tiny_config / tiny_model). gemma4: as P/graft/test_graft.py (tiny_gemma) and
P/t2-gemma4-ple/tiny_ple.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

FIX = Path(__file__).resolve().parent / "fixtures" / "configs"


def _text_cfg(name: str) -> dict:
    c = json.loads((FIX / name).read_text())
    return dict(c.get("text_config", c))


def tiny_qwen4exp_config(ngram_base: int = 1000, hidden: int = 128, layers: int = 4, vocab: int | None = 5000):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    tc = _text_cfg("qwen3.8-flash-next.config.json")
    tc.pop("model_type", None)
    tc.pop("mtp", None)
    tc.update(hidden_size=hidden, ple_embed_dim=hidden, num_hidden_layers=layers,
              layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
              head_dim=32, num_attention_heads=4, num_key_value_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
              num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
              hc_lowrank=32, indexer_n_heads=2, indexer_head_dim=32, indexer_budget=64, indexer_compress_ratio=4,
              ngram_vocab_size_base=ngram_base, vocab_size=vocab or tc["vocab_size"],
              eos_token_id=(vocab - 1 if vocab else tc["eos_token_id"]),
              bos_token_id=(vocab - 1 if vocab else tc["bos_token_id"]), split_ngram_parts=4,
              mtp_num_hidden_layers=0, rope_parameters=dict(tc["rope_parameters"], mrope_section=[2, 1, 1]))
    cfg = Qwen4ExpTextConfig(**tc)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return cfg


def tiny_qwen4exp(seed: int = 0, conv_scale: float = 0.3, dtype=torch.float32, **kw):
    """P/t1/src/tiny.py tiny_model: every PLE sub-path made non-trivial (HF init zeroes conv and norm weights)."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    torch.manual_seed(seed)
    cfg = tiny_qwen4exp_config(**kw)
    m = Qwen4ExpForCausalLM(cfg).float().eval()
    ple = m.model.layers[1].ple
    with torch.no_grad():
        ple.conv1d.weight.normal_(0, conv_scale)
        for n in (ple.norm_key, ple.norm_query, ple.norm_conv):
            n.weight.normal_(0, 0.1)
        ple.ple_embedding.ngram_embedding.weight.normal_(0, 0.5)
    return m.to(dtype)


def tiny_gemma_cfg(name: str, V: int = 512, ple_dim: int = 0, hidden: int = 64, layers: int = 6, **over):
    """Gemma4TextConfig shrunk from a real config (P/graft/test_graft.py tiny_gemma); ple_dim > 0 enables the
    per-layer-embedding path (hidden_size_per_layer_input)."""
    from transformers.models.gemma4 import Gemma4TextConfig
    t = _text_cfg(name)
    keep = {k: t[k] for k in ("hidden_activation", "rms_norm_eps", "final_logit_softcapping", "attention_k_eq_v",
                              "enable_moe_block", "use_double_wide_mlp", "rope_parameters") if k in t}
    lt = t["layer_types"][:layers]
    kw = dict(keep, vocab_size=V, vocab_size_per_layer_input=V, hidden_size=hidden, hidden_size_per_layer_input=ple_dim,
              num_hidden_layers=layers, layer_types=lt, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              global_head_dim=32, num_global_key_value_heads=1, intermediate_size=96, sliding_window=8,
              num_kv_shared_layers=0, eos_token_id=1, bos_token_id=2, pad_token_id=0, tie_word_embeddings=True,
              max_position_embeddings=512)
    if t.get("enable_moe_block"):
        kw.update(num_experts=8, top_k_experts=2, moe_intermediate_size=32)
    kw.update(over)
    cfg = Gemma4TextConfig(**kw)
    cfg._attn_implementation = "eager"
    return cfg


def tiny_gemma(name: str, seed: int = 0, **kw):
    from transformers.models.gemma4 import Gemma4ForCausalLM
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(tiny_gemma_cfg(name, **kw)).eval()


def record(name: str, **values) -> None:
    """Append a measurement to $NGRAM_M0_OUT/measurements.jsonl (no-op when unset), for reports that cite numbers."""
    import os
    out = os.environ.get("NGRAM_M0_OUT")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "measurements.jsonl"), "a") as f:
        f.write(json.dumps(dict(test=name, **values), default=float) + "\n")
