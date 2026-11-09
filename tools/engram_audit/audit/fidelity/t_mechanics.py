"""Fidelity audit: mechanics of Gemini's engram_graft.py / subgraph_*.py vs the DeepSeek Engram reference.

Runs on a tiny random Qwen2 config (CPU). Imports Gemini's files read-only from ../../gemini.
"""
import sys, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(0)

# CPU-only box: pin_memory() needs an accelerator; make it a no-op for the audit.
torch.Tensor.pin_memory = lambda self, *a, **k: self

GEM = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gemini'))
sys.path.insert(0, GEM)
import engram_graft as eg
import subgraph_calibrator as sc
import subgraph_edit_solver as ses
from transformers import Qwen2Config, Qwen2ForCausalLM

D = 64
cfg = Qwen2Config(vocab_size=1000, hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


def mk_engram(**kw):
    return eg.GraftedEngramModule(d_model=D, r_mem=kw.pop('r_mem', 32), num_buckets=kw.pop('num_buckets', 50_000),
                                  device='cpu', dtype=torch.float32, **kw)


print('=' * 70)
print('T1  short_conv init: nn.init.dirac_(self.short_conv.weight) on a depthwise Conv1d')
eng = mk_engram()
w = eng.short_conv.weight.detach()
print('   weight shape', tuple(w.shape))
nz_channels = (w.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
print('   channels with any nonzero tap:', nz_channels, 'of', w.shape[0])
print('   nonzero taps:', [(int(i), int(j), int(k)) for i, j, k in w.nonzero().tolist()])
# what does the causal conv path do to an input?
x = torch.randn(1, 6, D)
y = F.pad(x.transpose(1, 2), (eng.conv_kernel - 1, 0))
y = eng.short_conv(y).transpose(1, 2)
print('   ch0: y[t] == x[t-1]?', torch.allclose(y[0, 1:, 0], x[0, :-1, 0]), ' y[0]==0?', float(y[0, 0, 0]) == 0.0)
print('   ch1..: max|y| =', float(y[..., 1:].abs().max()))
print('   (Engram reference: Y = SiLU(Conv1D(RMSNorm(V~))) + V~, conv ZERO-init, dilation=max n => Y=V~ at init, same position)')

print('=' * 70)
print('T2  graft site: how many decoder blocks run before the fusion?')
model = Qwen2ForCausalLM(cfg).eval()
order = []
for i, L in enumerate(model.model.layers):
    L.register_forward_pre_hook(lambda m, a, i=i: order.append(f'block{i}'))
eng = mk_engram()
orig_ff = eng.forward_fusion
def logged(h):
    order.append('ENGRAM')
    return orig_ff(h)
eng.forward_fusion = logged
graft = eg.Layer2EngramGraft(model, model.model.layers[2], eng)
ids = torch.randint(0, 1000, (1, 10))
with torch.no_grad():
    model(input_ids=ids)
print('   call order:', order[:6])
print('   blocks executed before Engram:', order.index('ENGRAM'))
print('   (Engram paper 6.2: Layer 2 is preceded by "one round of attention"; demo code adds Engram at the START of'
      ' block layer_id=1, i.e. after 1 block)')
graft.pre_hook.remove(); graft.layer_hook.remove()

print('=' * 70)
print('T3  identity at init and gradient flow with v_proj = 0')
model = Qwen2ForCausalLM(cfg).eval()
for p in model.parameters():
    p.requires_grad_(False)
eng = mk_engram()
with torch.no_grad():
    base = model(input_ids=ids).logits
graft = eg.Layer2EngramGraft(model, model.model.layers[2], eng)
out = model(input_ids=ids).logits
print('   bit-exact identity at init:', torch.equal(base, out))
out.float().pow(2).mean().backward()
for n, p in eng.named_parameters():
    print(f'   grad norm {n:22s} {float(p.grad.norm()) if p.grad is not None else None:.3e}')
print('   cpu_tables are registered parameters?', any('cpu_tables' in n for n, _ in eng.named_parameters()))
graft.pre_hook.remove(); graft.layer_hook.remove()

print('=' * 70)
print('T4  hasher: pad value 0 is a real token id (Qwen2.5 id 0 = "!")')
h = eg.MultiHeadNgramHasher(num_buckets=2_000_000, num_heads=4)
x = 4321
a = h.hash_sequence(torch.tensor([[x]]))
b = h.hash_sequence(torch.tensor([[0, x]]))
print('   bigram(<pad>,x) at pos0 == bigram("!",x) at pos1 :', torch.equal(a[2][0, 0], b[2][0, 1]))
c = h.hash_sequence(torch.tensor([[0, 0, x]]))
print('   trigram(<pad>,<pad>,x) == trigram("!","!",x)       :', torch.equal(a[3][0, 0], c[3][0, 2]))
print('   K heads (code default) =', h.num_heads, ' (CLAIMS.md says K=8; Engram-27B uses 8)')
print('   hash form: sum_i p_i*x_i mod 2,000,000 (linear, composite modulus); Engram: multiplicative-XOR mod per-head PRIME')
for n in (2, 3):
    P = h.primes[n]
    gcds = [[math.gcd(int(v), 2_000_000) for v in row] for row in P.tolist()]
    print(f'   order {n}: multipliers prime? {[[all(int(v) % q for q in range(2, int(math.isqrt(int(v))) + 1)) for v in row] for row in P.tolist()]}')
    print(f'   order {n}: gcd(multiplier, 2e6) per head/pos = {gcds}')
# structured collisions: trigrams sharing the last two tokens collide iff p0*(x-y) = 0 mod N
N = 2_000_000
for k in range(4):
    p0 = int(h.primes[3][k, 0]); g = math.gcd(p0, N)
    print(f'   head {k}: trigrams (x,".","Step") and (y,".","Step") collide iff (x-y) % {N // g} == 0')

print('=' * 70)
print('T5  EngramEdit clamp actually used vs paper (rho=32 x ||h_i(x_i)||, B.4.2)')
for r_mem, K in ((512, 4), (512, 8)):
    e = mk_engram(r_mem=r_mem, num_heads=K, num_buckets=20_000)
    e.prefetch_from_host(torch.randint(0, 150_000, (8, 16)))
    an = e._prefetched_e[:, 5].norm(dim=-1).mean()
    print(f'   r_mem={r_mem} K={K}: mean ||e_q|| = {float(an):.4f}; 0.4*||e_q|| = {float(0.4 * an):.4f}; '
          f'max_delta_norm after clamp(min=0.1) = {float((0.4 * an).clamp(min=0.1)):.4f}')

print('=' * 70)
print('T6  C2 check: does Gamma {2:10, 3:1} push 85-90% of ||Delta E|| into 3-gram buckets?')

def ridge(A, gamma, target):
    lhs = A.T @ A + torch.diag(gamma)
    rhs = A.sum(0, keepdim=True).T @ target.unsqueeze(0)
    return torch.linalg.solve(lhs, rhs)

torch.manual_seed(1)
r = 256
for M in (1, 4, 8):
    for K in (4, 8):
        res = {}
        for n, pen in ((2, 10.0), (3, 1.0)):
            # all M expressions share the anchor n-gram (same subject suffix) -> K buckets, A[m,u]=1/K
            A = torch.full((M, K), 1.0 / K)
            gamma = pen * (1.0 + torch.log1p(torch.ones(K))) * 1e-2  # bucket_counts start at 1
            tgt = torch.randn(r)
            dE = ridge(A, gamma, tgt)
            realized = (A @ dE)[0]
            res[n] = (float(dE.norm() ** 2), float(realized.norm() / tgt.norm()))
        share3 = res[3][0] / (res[2][0] + res[3][0])
        print(f'   M={M} K={K}: realized/target 2-gram={res[2][1]:.3f} 3-gram={res[3][1]:.3f} | '
              f'share of ||DeltaE||^2 in 3-gram rows = {share3:.3f}')
print('   NOTE: delta* is split into a 2-gram slice and a 3-gram slice (torch.split, line 277) and each is solved'
      ' independently, so Gamma cannot move mass between orders; it only shrinks each slice.')

print('=' * 70)
print('T7  subgraph edit: does delta at anchor q change the L6 state AT q? (window weight 2.0 on q)')
torch.manual_seed(2)
model = Qwen2ForCausalLM(cfg).eval()
eng = mk_engram()
with torch.no_grad():
    nn.init.normal_(eng.v_proj.weight, std=0.5)  # pretend calibrated
cal = sc.SubgraphGateCalibrator(model, eng, graft_layer_idx=2, target_layer_idx=6)
ids = torch.randint(0, 1000, (1, 12))
q = 5
with torch.no_grad():
    h0 = cal._run_subgraph(ids, use_engram=True)[1]
    eng._active_perturbation = (torch.randn(eng.r_mem) * 5.0, [q])
    h1 = cal._run_subgraph(ids, use_engram=True)[1]
    eng._active_perturbation = None
diff = (h1 - h0).norm(dim=-1)[0]
print('   ||h6(delta)-h6(0)|| per position:', [round(float(v), 4) for v in diff])
print(f'   at anchor q={q}: {float(diff[q]):.2e}  at q+1: {float(diff[q+1]):.2e}')

print('=' * 70)
print('T8  calibrator _run_subgraph (no attention_mask passed) vs full-model hidden_states[7]')
for impl in ('sdpa', 'eager'):
    c2 = Qwen2Config(**{**cfg.to_dict(), '_attn_implementation': impl})
    m = Qwen2ForCausalLM(c2).eval()
    m.config._attn_implementation = impl
    e = mk_engram()
    cal = sc.SubgraphGateCalibrator(m, e)
    ids = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        ref = m(input_ids=ids, output_hidden_states=True).hidden_states[7]
        _, sub, _ = cal._run_subgraph(ids, use_engram=False)
    print(f'   attn={impl:5s}: max|sub - ref| = {float((sub - ref).abs().max()):.3e}')
print('   (hidden_states[7] is post-block-6 output; final model norm only applies to the last entry)')
