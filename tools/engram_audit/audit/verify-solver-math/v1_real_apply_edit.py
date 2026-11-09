"""Independent check: run Gemini's REAL EngramEditSolver.apply_edit (unmodified) on a tiny random Qwen2
and measure what Stage 2 actually writes, per order slice, for different order_penalties.
Only patch: pin_memory -> no-op (no CUDA)."""
import sys, copy, torch
torch.set_num_threads(2)
GEM = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini"
sys.path.insert(0, GEM)
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg
from transformers import Qwen2Config, Qwen2ForCausalLM

def build(vproj="random", conv="as_coded", seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).eval()
    eng = eg.GraftedEngramModule(d_model=64, r_mem=32, num_buckets=200_003, num_heads=4, device="cpu", dtype=torch.float32)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        if vproj == "random":
            eng.v_proj.weight.normal_(0, 0.3)
        if conv == "causal_identity":
            eng.short_conv.weight.zero_(); eng.short_conv.weight[:, 0, -1] = 1.0
    graft = eg.Layer2EngramGraft(model, model.model.layers[2], eng)
    return model, eng

def anchor_e(eng, ids, pos):
    eng.prefetch_from_host(ids)
    return eng._prefetched_e[:, pos].detach().clone() if isinstance(pos, int) else eng._prefetched_e.detach().clone()

torch.manual_seed(123)
M, T = 8, 12
# M paraphrases: varied prefix, same subject tokens [500, 7, 42] ending at anchor 8, then target tokens
ids = torch.randint(100, 900, (M, T))
ids[:, 6:9] = torch.tensor([500, 7, 42])  # "SpineGraph" "." "Step"
anchors = [8] * M
tmask = torch.zeros(M, T); tmask[:, 9:] = 1

def nll(model, ids, tmask):
    with torch.no_grad():
        lg = model(input_ids=ids).logits[:, :-1].float()
    tl = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction="none").view(M, T-1)
    m = tmask[:, 1:]
    return float((tl*m).sum()/m.sum())

for conv in ["as_coded", "causal_identity"]:
  for vp in ["zero", "random"]:
    print(f"\n=== v_proj={vp} conv={conv} ===")
    for pen in [{2: 1.0, 3: 1.0}, {2: 10.0, 3: 1.0}, {2: 1000.0, 3: 1.0}]:
        model, eng = build(vproj=vp, conv=conv)
        e0 = anchor_e(eng, ids[:1], 8)[0]
        pre = nll(model, ids, tmask)
        solver = eg.EngramEditSolver(model, eng)
        ret = solver.apply_edit(ids, anchors, tmask, order_penalties=pen)
        e1 = anchor_e(eng, ids[:1], 8)[0]
        d = e1 - e0; r = eng.r_per_order
        post = nll(model, ids, tmask)
        n2, n3 = d[:r].norm().item(), d[r:].norm().item()
        print(f"pen={pen}: written ||d_2g||={n2:.5f} ||d_3g||={n3:.5f} 3g share of ||d||^2={n3**2/max(n2**2+n3**2,1e-30):.3f} "
              f"| NLL pre={pre:.4f} returned={ret:.4f} post(table)={post:.4f}")
