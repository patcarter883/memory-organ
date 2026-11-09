# Independent check of the decode findings on Gemini's GraftedEngramModule + Layer2EngramGraft (verbatim import).
import sys, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2); torch.manual_seed(0)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
import engram_graft as eg
MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
try:
    eg.GraftedEngramModule(d_model=896, device="cpu", dtype=torch.float32, num_buckets=1000)
    print("constructor OK without patch")
except Exception as ex:
    print("constructor without patch raised:", type(ex).__name__, str(ex)[:120])
torch.Tensor.pin_memory = lambda self, *a, **k: self
eng = eg.GraftedEngramModule(d_model=896, device="cpu", dtype=torch.float32, num_buckets=100_000)
w = eng.short_conv.weight
print("short_conv weight", tuple(w.shape), "nonzero:", (w != 0).nonzero().tolist())
# what dirac_ WOULD do with groups=d_model: still the centre tap (index K//2 = 2) -> output[t] = input[t-1] under causal left-pad 3
w2 = torch.empty_like(w); nn.init.dirac_(w2, groups=896)
print("dirac_(groups=896) nonzero taps per channel:", sorted({tuple(x[1:]) for x in (w2 != 0).nonzero().tolist()}), "count", int((w2 != 0).sum()))
x = torch.randn(1, 896, 6)
y = F.conv1d(F.pad(x, (3, 0)), w2, groups=896)
print("with groups-correct dirac: y[t]==x[t]?", torch.allclose(y, x), " y[t]==x[t-1]?", torch.allclose(y[..., 1:], x[..., :-1]), " y[0]==0?", float(y[..., 0].abs().max()))
# hasher: single-token call vs full sequence
ids = torch.tensor([[791, 3974, 13876, 38835, 34208, 916, 279, 15678]])
full = eng.hasher.hash_sequence(ids); one = eng.hasher.hash_sequence(ids[:, 5:6])
padded = eng.hasher.hash_sequence(torch.tensor([[0, 0, int(ids[0, 5])]]))
for n in (2, 3):
    print(f"order {n}: full[pos5]==single-token? {bool((full[n][0,5]==one[n][0,0]).all())}; single-token == hash of zero-padded (0..,tok)? {bool((one[n][0,0]==padded[n][0,-1]).all())}")
# real model decode
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
seq = torch.tensor([tok("The quick brown fox jumps over the lazy dog while the Eiffel Tower stands tall in Paris today")["input_ids"][:20]])

def full_and_inc():
    with torch.no_grad():
        full = model(input_ids=seq).logits[0]
        o = model(input_ids=seq[:, :1], use_cache=True); inc = [o.logits[0, -1]]; pkv = o.past_key_values
        for t in range(1, seq.size(1)):
            o = model(input_ids=seq[:, t:t+1], past_key_values=pkv, use_cache=True); pkv = o.past_key_values; inc.append(o.logits[0, -1])
    return full, torch.stack(inc)

f0, i0 = full_and_inc()
gr = eg.Layer2EngramGraft(model, model.model.layers[2], eng)
with torch.no_grad(): eng.v_proj.weight.normal_(0, 2.0)
f1, i1 = full_and_inc()
print("v~N(0,2), shipped dirac: full-graft effect max|dlogit| =", float((f1 - f0).abs().max()),
      "| incremental graft vs incremental no-graft =", float((i1 - i0).abs().max()),
      "| incremental graft vs full graft =", float((i1 - f1).abs().max()))
with torch.no_grad(): eng.short_conv.weight.copy_(w2)   # groups-correct dirac (all channels, t-1 tap)
f2, i2 = full_and_inc()
print("v~N(0,2), groups-correct dirac: full-graft effect =", float((f2 - f0).abs().max()),
      "| incr graft vs incr no-graft =", float((i2 - i0).abs().max()), "| incr vs full graft =", float((i2 - f2).abs().max()))
w3 = torch.zeros_like(w); w3[:, 0, 3] = 1.0   # a TRUE identity (current-token tap): isolates the hashing bug from the conv-history bug
with torch.no_grad(): eng.short_conv.weight.copy_(w3)
f3, i3 = full_and_inc()
print("v~N(0,2), true-identity conv (tap 3): full-graft effect =", float((f3 - f0).abs().max()),
      "| incr graft vs incr no-graft =", float((i3 - i0).abs().max()), "| incr vs full graft =", float((i3 - f3).abs().max()),
      "| step0 diff (no history needed) =", float((i3[0] - f3[0]).abs().max()))
