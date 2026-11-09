import json, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
# 1. hc_count=1 is outside Qwen4Exp's validated config domain
tc=json.load(open(f"{P}/models/Qwen/Qwen3.8-Flash-Next/config.json"))["text_config"]; tc.pop("model_type",None); tc.pop("mtp",None)
try:
    Qwen4ExpTextConfig(**dict(tc, hc_count=1)); print("[1] hc_count=1 accepted")
except Exception as e: print("[1] hc_count=1 rejected:", type(e).__name__, str(e)[:90])
# 2. unigram-extension width divisibility (24 heads) with ple_embed_dim = host d
for n,d in [("27B",5120),("35B-A3B",2048),("gemma26B",2816),("gemma31B",5376),("FlashNext",2560)]:
    print(f"[2] {n}: d={d} d%16={d%16} d%24={d%24}")
# 3. Gemma-PLE graft with post_per_layer_input_norm.weight = 0 (as DESIGN 4.4 words it) -> writes are dead
def cfg():
    c=Gemma4TextConfig(vocab_size=128, vocab_size_per_layer_input=128, hidden_size=32, hidden_size_per_layer_input=8,
        intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        global_head_dim=16, num_kv_shared_layers=0, sliding_window=8, layer_types=["sliding_attention","sliding_attention","full_attention"],
        pad_token_id=0, bos_token_id=2, eos_token_id=1, tie_word_embeddings=True)
    c._attn_implementation="eager"; return c
torch.manual_seed(0); m=Gemma4ForCausalLM(cfg()).eval(); gm=m.model
with torch.no_grad():
    gm.embed_tokens_per_layer.weight.zero_(); gm.per_layer_model_projection.weight.zero_()
ids=torch.randint(3,128,(1,12))
for w in (0.0,1.0):
    with torch.no_grad():
        for L in gm.layers: L.post_per_layer_input_norm.weight.fill_(w)
        base=m(ids).logits.clone()
        gm.embed_tokens_per_layer.weight[ids[0,5]] += torch.randn(3*8)
        after=m(ids).logits
        gm.embed_tokens_per_layer.weight.zero_()
    W=gm.embed_tokens_per_layer.weight; W.grad=None
    with torch.no_grad(): W[ids[0,5]] += 1e-3*torch.randn(3*8)
    m(ids).logits[0,-1].logsumexp(-1).backward()
    g=W.grad[ids[0]].abs().max().item()
    with torch.no_grad(): W.zero_()
    print(f"[3] post_per_layer_input_norm.weight={w}: write changes logits by {(after-base).abs().max().item():.3e}; max|grad| on read rows {g:.3e}")
