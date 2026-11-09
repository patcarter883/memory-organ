import sys, torch, math
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
# pin_memory raises on CPU-only torch; check that first, then patch to identity for import.
try:
    torch.randn(3).pin_memory(); print("pin_memory OK")
except Exception as e:
    print("pin_memory raises:", type(e).__name__, str(e)[:80])
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg
import torch.nn as nn, torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=500, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()

def engram(d, nb=3000):
    return eg.GraftedEngramModule(d_model=d, r_mem=16, num_buckets=nb, num_heads=4, orders=(2,3), conv_kernel=4,
                                  device="cpu", dtype=torch.float32)

print("== M1: dirac_ on depthwise weight")
E = engram(32)
w = E.short_conv.weight
print("shape", tuple(w.shape), "nonzero idx", w.nonzero().tolist())
w2 = torch.zeros_like(w); nn.init.dirac_(w2, groups=32)
print("groups=d nonzero tap indices", sorted(set(w2.nonzero()[:,2].tolist())), "count", int((w2!=0).sum()))

print("== M1 dead-start: train engram params only on tiny model, 30 Adam steps")
m = tiny(); d = 32
for p in m.parameters(): p.requires_grad_(False)
E = engram(d); g = eg.Layer2EngramGraft(m, m.model.layers[2], E)
opt = torch.optim.Adam(E.parameters(), lr=1e-2)
ids = torch.randint(1, 500, (4, 24))
for s in range(30):
    opt.zero_grad(); out = m(input_ids=ids).logits
    loss = F.cross_entropy(out[:, :-1].reshape(-1, 500), ids[:, 1:].reshape(-1)); loss.backward(); opt.step()
print("v_proj nonzero rows:", (E.v_proj.weight.abs().sum(1) > 0).nonzero().flatten().tolist())
print("conv nonzero channels:", (E.short_conv.weight.abs().sum((1,2)) > 0).nonzero().flatten().tolist())
print("conv nonzero taps:", E.short_conv.weight.nonzero().tolist())
g.pre_hook.remove(); g.layer_hook.remove()

print("== Missed: EngramEdit Stage 1 with untrained (zero) v_proj -> delta gradient")
m = tiny(); E = engram(d); g = eg.Layer2EngramGraft(m, m.model.layers[2], E)
solver = eg.EngramEditSolver(m, E)
ids = torch.randint(1, 500, (3, 12)); tm = torch.zeros_like(ids); tm[:, 8:] = 1
tab_before = {n: E.cpu_tables[n].clone() for n in E.orders}
loss = solver.apply_edit(ids, [6, 6, 6], tm, steps=5)
print("final nll", loss, "| table max change", {n: float((E.cpu_tables[n]-tab_before[n]).abs().max()) for n in E.orders})
# reproduce stage1 grad directly
delta = torch.zeros(16, requires_grad=True)
E._active_perturbation = (delta, [6, 6, 6]); out = m(input_ids=ids).logits
l = F.cross_entropy(out[:, :-1].reshape(-1, 500), ids[:, 1:].reshape(-1)); l.backward()
print("|grad delta| with v_proj=0:", float(delta.grad.norm()))
# now nonzero v_proj (simulated warmup), verbatim dirac conv
with torch.no_grad(): E.v_proj.weight.normal_(0, 0.5)
delta.grad = None; out = m(input_ids=ids).logits
l = F.cross_entropy(out[:, :-1].reshape(-1, 500), ids[:, 1:].reshape(-1)); l.backward()
print("|grad delta| with v_proj!=0 (verbatim conv):", float(delta.grad.norm()))
# which residual channels does delta move? compare layer-2 hook output with/without delta
cap = {}
hh = m.model.layers[2].register_forward_hook(lambda mod, i, o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach()))
E._active_perturbation = None; m(input_ids=ids); h0 = cap['h']
E._active_perturbation = (torch.ones(16), [6, 6, 6]); m(input_ids=ids); h1 = cap['h']
dd = (h1 - h0).abs()
print("delta-induced residual change: channels", (dd.sum((0,1)) > 0).nonzero().flatten().tolist(),
      "positions", (dd.sum((0,2)) > 0).nonzero().flatten().tolist(), "(anchor=6)")
hh.remove(); E._active_perturbation = None; g.pre_hook.remove(); g.layer_hook.remove()

print("== M2: decode hashing under KV cache")
m = tiny(); E = engram(d); g = eg.Layer2EngramGraft(m, m.model.layers[2], E)
with torch.no_grad():
    E.v_proj.weight.normal_(0, 0.5)
    E.short_conv.weight.zero_(); E.short_conv.weight[:, 0, -1] = 1.0   # 'fixed' identity tap at current token
seen = []
orig = E.prefetch_from_host
E.prefetch_from_host = lambda x: (seen.append(tuple(x.shape)), orig(x))[1]
prompt = torch.randint(1, 500, (1, 10))
with torch.no_grad():
    o = m(input_ids=prompt, use_cache=True); pkv = o.past_key_values
    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    o2 = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
    full = m(input_ids=torch.cat([prompt, nxt], 1)).logits[:, -1]
print("shapes seen by prefetch:", seen)
print("max|cached - full| at decode step (identity-tap conv, v!=0):", float((o2.logits[:, -1] - full).abs().max()))
# edit the true-context n-gram rows for the decoded token and see if cached decode reads it
seq = torch.cat([prompt, nxt], 1)
idx = E.hasher.hash_sequence(seq)
with torch.no_grad():
    base_full = m(input_ids=seq).logits[:, -1].clone()
    o = m(input_ids=prompt, use_cache=True); base_dec = m(input_ids=nxt, past_key_values=o.past_key_values).logits[:, -1].clone()
    for n in (2, 3): E.cpu_tables[n][idx[n][0, -1]] += 3.0
    new_full = m(input_ids=seq).logits[:, -1]
    o = m(input_ids=prompt, use_cache=True); new_dec = m(input_ids=nxt, past_key_values=o.past_key_values).logits[:, -1]
print("edit effect full re-forward:", float((new_full - base_full).abs().max()), " cached decode:", float((new_dec - base_dec).abs().max()))
g.pre_hook.remove(); g.layer_hook.remove()

print("== M9: padding (graft v!=0, identity-tap conv)")
m = tiny(); E = engram(d); g = eg.Layer2EngramGraft(m, m.model.layers[2], E)
with torch.no_grad():
    E.v_proj.weight.normal_(0, 0.5); E.short_conv.weight.zero_(); E.short_conv.weight[:, 0, -1] = 1.0
x = torch.randint(1, 500, (1, 8)); P = 4
with torch.no_grad():
    ref = m(input_ids=x).logits[0]
    lp = torch.cat([torch.zeros(1, P, dtype=torch.long), x], 1); am = torch.cat([torch.zeros(1, P, dtype=torch.long), torch.ones(1, 8, dtype=torch.long)], 1)
    left = m(input_ids=lp, attention_mask=am).logits[0, P:]
    rp = torch.cat([x, torch.zeros(1, P, dtype=torch.long)], 1)
    right_nomask = m(input_ids=rp).logits[0, :8]
print("left-pad (with mask) max diff at real positions:", float((left - ref).abs().max()))
print("right-pad (NO mask, like solver) max diff at real positions:", float((right_nomask - ref).abs().max()))
g.pre_hook.remove(); g.layer_hook.remove()

print("== Missed: default ctor dtype/device")
try:
    m = tiny(); E2 = eg.GraftedEngramModule(d_model=32, r_mem=16, num_buckets=100, device="cpu")  # default bf16
    g = eg.Layer2EngramGraft(m, m.model.layers[2], E2); m(input_ids=torch.randint(1, 500, (1, 5)))
    print("fp32 model + default bf16 module: OK")
except Exception as e:
    print("fp32 model + default bf16 module ->", type(e).__name__, str(e)[:100])
