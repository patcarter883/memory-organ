"""(a) With the as-coded dirac_ conv (one-step lag), can Stage-1's delta at anchor q affect logits at position q
(which predict the FIRST target token)? (b) ||e_q|| and max_delta_norm at default init (r_mem=512)."""
import sys, torch
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg
from transformers import Qwen2Config, Qwen2ForCausalLM
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
for conv in ["as_coded", "causal_identity"]:
    torch.manual_seed(0); model = Qwen2ForCausalLM(cfg).eval()
    eng = eg.GraftedEngramModule(64, 32, 50_021, 4, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        eng.v_proj.weight.normal_(0, 0.3)
        if conv == "causal_identity": eng.short_conv.weight.zero_(); eng.short_conv.weight[:, 0, -1] = 1
    eg.Layer2EngramGraft(model, model.model.layers[2], eng)
    ids = torch.randint(0, 1000, (1, 12)); q = 8
    delta = torch.zeros(32, requires_grad=True)
    eng._active_perturbation = (delta, [q])
    lg = model(input_ids=ids).logits[0]
    g_q = torch.autograd.grad(lg[q].sum() + 0*delta.sum(), delta, retain_graph=True, allow_unused=True)[0]
    g_q1 = torch.autograd.grad(lg[q+1].sum(), delta, allow_unused=True)[0]
    print(f"conv={conv}: ||d logits[q]/d delta||={g_q.norm():.3e} (predicts 1st target token) ; ||d logits[q+1]/d delta||={g_q1.norm():.3e}")
eng = eg.GraftedEngramModule(896, 512, 2_000_000, 4, device="cpu", dtype=torch.bfloat16)
ids = torch.randint(0, 150000, (8, 16)); eng.prefetch_from_host(ids)
en = eng._prefetched_e[:, 10].float().norm(dim=-1).mean().item()
print(f"default init r_mem=512 bf16: mean ||e_q||={en:.4f}; 0.4*||e_q||={0.4*en:.4f}; max_delta_norm={max(0.1, 0.4*en):.4f}")
