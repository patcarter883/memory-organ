"""F3 check: (a) what dirac_ builds on the depthwise conv; (b) does Gemini's VERBATIM calibrator
(Phase 1 warmup_subspace_alignment, Phase 2 calibrate_context_distillation) repair it?"""
import sys, torch
torch.set_num_threads(2)
GEM = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini"
sys.path.insert(0, GEM)
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg, subgraph_calibrator as sc
from transformers import Qwen2Config, Qwen2ForCausalLM

def mk(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = Qwen2ForCausalLM(cfg).eval()
    eng = eg.GraftedEngramModule(d_model=64, r_mem=32, num_buckets=50_021, num_heads=4, device="cpu", dtype=torch.float32)
    return model, eng

def report(tag, eng):
    w = eng.short_conv.weight.detach()
    ch = (w.abs().sum(dim=(1, 2)) > 0).sum().item()
    taps = w.abs().sum(dim=(0, 1)).tolist()
    vrows = (eng.v_proj.weight.detach().abs().sum(1) > 0).sum().item()
    print(f"{tag}: nonzero conv channels={ch}/64, |w| per tap (0..3, tap3=current token)={[round(t,4) for t in taps]}, "
          f"nonzero v_proj rows={vrows}/64, rank(v_proj)={torch.linalg.matrix_rank(eng.v_proj.weight.detach()).item()}")

model, eng = mk()
report("init (as coded)", eng)
w = eng.short_conv.weight.detach()
print("nonzero entries of conv weight (ch,in,tap):", w.nonzero().tolist())

torch.manual_seed(5)
batches = [torch.randint(0, 1000, (4, 64)) for _ in range(30)]
cal = sc.SubgraphGateCalibrator(model, eng, 2, 6)
cal.warmup_subspace_alignment(batches)
report("after Phase 1 (30 steps, verbatim)", eng)

# Phase 2 alone (skip Phase 1): v_proj at its zero init
model, eng = mk()
cal = sc.SubgraphGateCalibrator(model, eng, 2, 6)
fb = [torch.randint(0, 1000, (2, 64)) for _ in range(15)]
cal.calibrate_context_distillation(fb, prefix_len=32)
report("Phase 2 only, no Phase 1 (15 steps, verbatim)", eng)
