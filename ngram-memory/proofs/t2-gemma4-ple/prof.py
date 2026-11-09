import time, torch, json, sys
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
torch.set_num_threads(int(sys.argv[1]))
E4B = json.load(open("../../models/google/gemma-4-E4B-it/config.json"))["text_config"]
L=12
cfg = Gemma4TextConfig(vocab_size=512, vocab_size_per_layer_input=512, hidden_size=128, hidden_size_per_layer_input=16,
    intermediate_size=512, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=1, head_dim=32, global_head_dim=64,
    num_kv_shared_layers=5, sliding_window=8, layer_types=["full_attention" if (i+1)%6==0 else "sliding_attention" for i in range(L)],
    final_logit_softcapping=30.0, rope_parameters=E4B["rope_parameters"], max_position_embeddings=512, pad_token_id=0)
cfg._attn_implementation = "eager"
m = Gemma4ForCausalLM(cfg).eval()
for p in m.parameters(): p.requires_grad_(False)
x = torch.randint(3, 512, (1, 7)); d = torch.zeros(1, requires_grad=True)
for i in range(3):
    t=time.time(); m(x).logits.sum().backward() if False else None; out=m(x); print("fwd", time.time()-t)
with torch.profiler.profile() as prof:
    m(x)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))
