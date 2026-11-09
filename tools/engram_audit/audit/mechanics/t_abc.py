"""(a) identity at init, (b) dirac_ conv semantics, (c) gradient flow / dead start."""
from common import *
import torch.nn as nn, torch.nn.functional as F

print("=== (b) dirac_ on depthwise conv weight ===")
eng = make_engram(d=8, r_mem=8, num_buckets=100)
w = eng.short_conv.weight.detach()
print("weight shape", tuple(w.shape), "groups", eng.short_conv.groups)
nz = (w != 0).nonzero().tolist()
print("nonzero entries [out,in,k]:", nz, "values", w[w != 0].tolist())
# Numerical: apply the exact forward_fusion conv path to a ramp signal
T = 6
x = torch.arange(1, T + 1, dtype=torch.float32).view(1, 1, T).repeat(1, 8, 1) * torch.arange(1, 9).view(1, 8, 1)
y = eng.short_conv(F.pad(x, (eng.conv_kernel - 1, 0)))
print("input  ch0:", x[0, 0].tolist(), " ch1:", x[0, 1].tolist())
print("output ch0:", y[0, 0].tolist(), " ch1:", y[0, 1].tolist())
print("channels with any nonzero output:", (y.abs().sum(-1)[0] != 0).nonzero().flatten().tolist())
# what dirac_ with groups=d would have done
w2 = torch.empty(8, 1, 4); nn.init.dirac_(w2, groups=8)
print("dirac_(groups=d) nonzero:", (w2 != 0).nonzero().tolist()[:3], "... k-index", (w2 != 0).nonzero()[:, 2].unique().tolist())
y2 = F.conv1d(F.pad(x, (3, 0)), w2, groups=8)
print("groups=d version output ch1:", y2[0, 1].tolist(), "(still a 1-step delay since centre idx=k//2=2 of 0..3)")

print("\n=== (a) bit-exact identity at init (tiny Qwen2, fp32) ===")
m = tiny_model()
ids = torch.randint(0, 1000, (2, 16))
with torch.no_grad():
    ref = m(input_ids=ids).logits.clone()
d = m.config.hidden_size
eng = make_engram(d)
g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
with torch.no_grad():
    out = m(input_ids=ids).logits
print("max |diff| tiny:", (out - ref).abs().max().item(), "bit-exact:", torch.equal(out, ref))
# inputs_embeds path: prefetch never runs
try:
    with torch.no_grad():
        eng._prefetched_e = None
        m(inputs_embeds=m.model.embed_tokens(ids))
    print("inputs_embeds ok")
except Exception as e:
    print("inputs_embeds path ->", type(e).__name__, str(e)[:120])

print("\n=== (c) gradient flow at init (only engram params trainable, backbone frozen) ===")
for p in m.parameters(): p.requires_grad_(False)
for p in eng.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(eng.parameters(), lr=1e-2)
names = dict(eng.named_parameters())
for step in range(1, 6):
    opt.zero_grad()
    out = m(input_ids=ids).logits
    loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), ids[:, 1:].reshape(-1))
    loss.backward()
    rep = []
    for n, p in names.items():
        gr = p.grad
        if gr is None: rep.append(f"{n}:None"); continue
        if n == "v_proj.weight":
            rows = (gr.abs().sum(1) != 0).nonzero().flatten().tolist()
            rep.append(f"v_proj nz-grad rows={rows}")
        elif n == "short_conv.weight":
            ch = (gr.abs().sum((1, 2)) != 0).nonzero().flatten().tolist()
            rep.append(f"conv nz-grad channels={ch}")
        else:
            rep.append(f"{n}:|g|={gr.abs().sum().item():.2e}")
    print(f"step {step}: " + "; ".join(rep))
    opt.step()
print("after 5 steps: v_proj nonzero rows:", (eng.v_proj.weight.abs().sum(1) != 0).nonzero().flatten().tolist(),
      " conv nonzero channels:", (eng.short_conv.weight.abs().sum((1, 2)) != 0).nonzero().flatten().tolist())
# Does residual write live only in channel 0?
with torch.no_grad():
    h = torch.randn(1, 10, d)
    eng.prefetch_from_host(ids[:1, :10])
    dlt = eng.forward_fusion(h) - h
print("forward_fusion residual write: channels with nonzero delta:", (dlt.abs().sum((0, 1)) != 0).nonzero().flatten().tolist())
print("cpu_tables in parameters/state_dict?", any('table' in k for k in eng.state_dict().keys()),
      "| e_t requires_grad:", eng._prefetched_e.requires_grad)

print("\n=== (a') Qwen2.5-0.5B fp32 identity ===")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(QWEN)
M = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval()
ids = tok(["The capital of France is Paris, and the capital of Spain is"], return_tensors="pt").input_ids
with torch.no_grad():
    ref = M(input_ids=ids).logits.clone()
eng = make_engram(M.config.hidden_size, r_mem=512, num_buckets=20000)
g = eg.Layer2EngramGraft(M, M.model.layers[2], eng)
with torch.no_grad():
    out = M(input_ids=ids).logits
print("layer2 output type in transformers 5.x: tensor (hook non-tuple branch)")
print("Qwen max |diff|:", (out - ref).abs().max().item(), "bit-exact:", torch.equal(out, ref))
