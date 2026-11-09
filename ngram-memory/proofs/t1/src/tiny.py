"""Build a tiny randomly-initialised Qwen4Exp text model from the REAL Flash-Next config, shrunk."""
import json, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
P = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"


def tiny_config(ngram_base=1000, hidden=128, layers=4, vocab=None):
    tc = json.load(open(f"{P}/models/Qwen/Qwen3.8-Flash-Next/config.json"))["text_config"]
    tc.pop("model_type", None); tc.pop("mtp", None)
    tc.update(hidden_size=hidden, ple_embed_dim=hidden, num_hidden_layers=layers,
              layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
              head_dim=32, num_attention_heads=4, num_key_value_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
              num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
              hc_lowrank=32, indexer_n_heads=2, indexer_head_dim=32, indexer_budget=64, indexer_compress_ratio=4,
              ngram_vocab_size_base=ngram_base, vocab_size=vocab or tc["vocab_size"], eos_token_id=(4999 if vocab else tc["eos_token_id"]), bos_token_id=(4999 if vocab else tc["bos_token_id"]), split_ngram_parts=4, mtp_num_hidden_layers=0,
              rope_parameters=dict(tc["rope_parameters"], mrope_section=[2, 1, 1]))
    cfg = Qwen4ExpTextConfig(**tc)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return cfg


def tiny_model(seed=0, conv_scale=0.3, **kw):
    torch.manual_seed(seed)
    cfg = tiny_config(**kw)
    m = Qwen4ExpForCausalLM(cfg).float().eval()
    ple = m.model.layers[1].ple
    with torch.no_grad():
        # make every PLE sub-path non-trivial (init zeros conv & norm weights)
        ple.conv1d.weight.normal_(0, conv_scale)
        for n in (ple.norm_key, ple.norm_query, ple.norm_conv):
            n.weight.normal_(0, 0.1)
        ple.ple_embedding.ngram_embedding.weight.normal_(0, 0.5)
    return m
