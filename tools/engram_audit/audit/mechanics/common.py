import sys, torch
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
# CPU-only torch: Tensor.pin_memory() raises "Cannot access accelerator device". Patch to identity so the
# verbatim module can be constructed for mechanics tests (documented as a finding).
_orig_pin = torch.Tensor.pin_memory
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg  # verbatim Gemini code, unmodified
from transformers import Qwen2Config, Qwen2ForCausalLM

QWEN = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

def tiny_model(seed=0, layers=4, d=64):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=d, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).eval()
    return m

def make_engram(d, r_mem=32, num_buckets=5000, conv_kernel=4):
    return eg.GraftedEngramModule(d_model=d, r_mem=r_mem, num_buckets=num_buckets, num_heads=4, orders=(2, 3),
                                  conv_kernel=conv_kernel, device="cpu", dtype=torch.float32)
