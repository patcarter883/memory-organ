"""Instantiate the REAL E4B text config on the meta device (no weights) and read KV-sharing / PLE structure."""
import json, sys, torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
c = json.load(open(sys.argv[1]))["text_config"]; c.pop("model_type", None)
cfg = Gemma4TextConfig(**c)
with torch.device("meta"):
    m = Gemma4ForCausalLM(cfg)
lay = m.model.layers
print("first_kv_shared_layer_idx", cfg.num_hidden_layers - cfg.num_kv_shared_layers)
print("store_full_length_kv", [i for i, l in enumerate(lay) if l.self_attn.store_full_length_kv])
print("kv_shared layers", [i for i, l in enumerate(lay) if l.self_attn.is_kv_shared_layer])
print("full attention layers", [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"])
print("embed_tokens_per_layer", tuple(m.model.embed_tokens_per_layer.weight.shape), "scale", m.model.embed_tokens_per_layer.scalar_embed_scale)
n = m.model.embed_tokens_per_layer.weight.numel(); tot = sum(p.numel() for p in m.parameters())
print("PLE table params", n, "of text-model params", tot, "frac", round(n / tot, 3), "bf16 GB", round(n * 2 / 1e9, 2))
ple_other = sum(p.numel() for nme, p in m.named_parameters() if "per_layer" in nme and "embed_tokens_per_layer" not in nme)
print("PLE non-table params", ple_other)
print("readable-by-later-tokens PLE layers: 0..", max(i for i, l in enumerate(lay) if l.self_attn.store_full_length_kv) - 1)
