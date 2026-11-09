# Quick mechanics checks with the VERBATIM Gemini modules on a tiny random Qwen2 config:
# F4 (what Phase 1 trains, state_dict), F10 (leaf/grad), F13 (bf16 orthogonal_), F9 (bf16 rounding of table updates),
# F15 (teacher Engram OFF), the Phase-2 position-0 2-gram pad token.
import sys, hashlib, torch
torch.set_num_threads(2)
torch.Tensor.pin_memory = lambda self, *a, **k: self   # CPU-only build
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
import engram_graft as eg, subgraph_calibrator as sc
from transformers import Qwen2Config, Qwen2ForCausalLM
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024, attn_implementation="sdpa")
m = Qwen2ForCausalLM(cfg).eval()
for p in m.parameters(): p.requires_grad_(False)

# ---- F13: orthogonal_ on a bf16 tensor on CPU
try:
    w = torch.empty(64, 32, dtype=torch.bfloat16); torch.nn.init.orthogonal_(w, gain=0.01); print("F13 bf16 orthogonal_: OK (no crash)")
except Exception as e: print("F13 bf16 orthogonal_ raises:", type(e).__name__, str(e)[:90])

# ---- F4: Phase 1 trainables / tables
eng = eg.GraftedEngramModule(d_model=64, r_mem=32, num_buckets=5000, device="cpu", dtype=torch.float32)
print("F4 engram params:", [n for n, _ in eng.named_parameters()], "| any table in state_dict:", any("table" in k for k in eng.state_dict()))
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
md5 = lambda: hashlib.md5(b"".join(eng.cpu_tables[n].numpy().tobytes() for n in eng.orders)).hexdigest()
h0 = md5(); vb = eng.v_proj.weight.clone()
data = [torch.randint(0, 500, (2, 64)) for _ in range(10)]
cal.warmup_subspace_alignment(data)
print("F4 after Phase 1: tables changed?", md5() != h0, "| v_proj changed?", not torch.equal(vb, eng.v_proj.weight))

# ---- Phase 2 (fp32): do tables change at all? (tables do get the manual sparse SGD in Phase 2)
h1 = md5(); snap = {n: eng.cpu_tables[n].clone() for n in eng.orders}
cal.calibrate_context_distillation([torch.randint(0, 500, (2, 128)) for _ in range(3)], prefix_len=64)
print("Phase 2 (fp32 tables): tables changed?", md5() != h1, "| max |dtable| =", max((eng.cpu_tables[n]-snap[n]).abs().max().item() for n in eng.orders))
print("F10 _prefetched_e.grad populated:", eng._prefetched_e.grad is not None, "| is_leaf:", eng._prefetched_e.is_leaf)

# ---- F9: bf16 rounding arithmetic: smallest representable increment near |w|=0.0135 in bf16
w = torch.tensor([0.0135], dtype=torch.bfloat16)
for upd in [1e-10, 1e-6, 2e-5, 3e-5, 6e-5]:
    print(f"F9 bf16: 0.0135 + {upd:.0e} -> changed? {bool((w + torch.tensor([upd]).to(torch.bfloat16) != w).item())}")
print("F9 bf16 ulp at 0.0135 =", (torch.nextafter(torch.tensor(0.0135), torch.tensor(1.0)) - 0.0135).item(), "(fp32);",
      "bf16 spacing:", 2.0**(-7 - 7))  # 0.0135 in [2^-7, 2^-6): spacing 2^-7 * 2^-7

# ---- F9 zero-init: with v_proj == 0 the grad on e is exactly zero? (k path: alpha depends on k(e) but multiplies v=0)
eng0 = eg.GraftedEngramModule(d_model=64, r_mem=32, num_buckets=5000, device="cpu", dtype=torch.float32)
cal0 = sc.SubgraphGateCalibrator(m, eng0, 2, 6)
cal0.calibrate_context_distillation([torch.randint(0, 500, (2, 128))], prefix_len=64)
print("F9 zero-init Phase 2: max|grad e| =", eng0._prefetched_e.grad.abs().max().item(),
      "| grads on q/k/v:", {n: (p.grad.abs().max().item() if p.grad is not None else None) for n, p in eng0.named_parameters()})

# ---- Hasher pad: position-0 2-gram is keyed on (token_id 0, x). Is id 0 a real token? (Qwen: id 0 = '!')
from transformers import AutoTokenizer
tq = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
print("Qwen token id 0 =", repr(tq.convert_ids_to_tokens(0)), "| tokens of '!x':", tq("!x")["input_ids"])
