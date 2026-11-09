"""Skeptic re-check of fidelity-audit mechanics (independent re-implementation, Gemini files imported read-only)."""
import sys, math, os, copy
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2); torch.manual_seed(0)
torch.Tensor.pin_memory = lambda self, *a, **k: self   # CPU box
GEM = '/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini'
sys.path.insert(0, GEM)
import engram_graft as eg, subgraph_calibrator as sc, subgraph_edit_solver as ses
from transformers import Qwen2Config, Qwen2ForCausalLM
D = 64
def cfg(impl='sdpa'):
    c = Qwen2Config(vocab_size=1000, hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    c._attn_implementation = impl
    return c
def mk(**kw): return eg.GraftedEngramModule(d_model=D, r_mem=kw.pop('r_mem',32), num_buckets=kw.pop('nb',50_000), device='cpu', dtype=torch.float32, **kw)

print('V1 dirac_ on depthwise conv')
e = mk(); w = e.short_conv.weight.detach()
print('  shape', tuple(w.shape), 'nonzero idx', w.nonzero().tolist())
x = torch.randn(1, 7, D)
y = e.short_conv(F.pad(x.transpose(1,2), (3,0))).transpose(1,2)
print('  y[:,t,0]==x[:,t-1,0]:', torch.allclose(y[0,1:,0], x[0,:-1,0]), '| max|y[...,1:]|', float(y[...,1:].abs().max()))

print('V2 graft site')
m = Qwen2ForCausalLM(cfg()).eval(); log = []
for i, L in enumerate(m.model.layers): L.register_forward_pre_hook(lambda mod, a, i=i: log.append(i))
e = mk(); ff = e.forward_fusion
e.forward_fusion = lambda h: (log.append('E'), ff(h))[1]
g = eg.Layer2EngramGraft(m, m.model.layers[2], e)
with torch.no_grad(): m(input_ids=torch.randint(0,1000,(1,6)))
print('  order', log[:6], '-> blocks before Engram =', log.index('E'))
g.pre_hook.remove(); g.layer_hook.remove()

print('V3 is cpu_tables a parameter / grads at init')
print('  params:', [n for n,_ in e.named_parameters()])

print('V4 does Phase-1 calibration break the one-token delay? (taps after training)')
torch.manual_seed(3)
m = Qwen2ForCausalLM(cfg()).eval(); e = mk()
cal = sc.SubgraphGateCalibrator(m, e, 2, 6)
batches = [torch.randint(0,1000,(2,32)) for _ in range(30)]
cal.warmup_subspace_alignment(batches, lr=1e-3)
w = e.short_conv.weight.detach()
print('  after 30 Phase-1 steps: mean|w| per tap (k=0..3):', [round(float(w[:,0,k].abs().mean()),5) for k in range(4)])
print('  channels with nonzero current-position tap k=3:', int((w[:,0,3].abs()>0).sum()), 'of', D)
for p in e.parameters(): p.requires_grad_(False)
ids = torch.randint(0,1000,(1,12)); q = 5
with torch.no_grad():
    h0 = cal._run_subgraph(ids, use_engram=True)[1]
    e._active_perturbation = (torch.randn(e.r_mem)*0.1, [q])
    h1 = cal._run_subgraph(ids, use_engram=True)[1]
    e._active_perturbation = None
d = (h1-h0).norm(dim=-1)[0]
print('  after calib: ||dh6|| at q-1, q, q+1:', [f'{float(v):.2e}' for v in d[q-1:q+2]])
# and with pristine dirac (uncalibrated conv but nonzero v_proj)
e2 = mk(); nn.init.normal_(e2.v_proj.weight, std=0.5)
cal2 = sc.SubgraphGateCalibrator(m, e2, 2, 6)
with torch.no_grad():
    h0 = cal2._run_subgraph(ids, use_engram=True)[1]
    e2._active_perturbation = (torch.randn(e2.r_mem)*0.1, [q]); h1 = cal2._run_subgraph(ids, use_engram=True)[1]
    e2._active_perturbation = None
d = (h1-h0).norm(dim=-1)[0]
print('  dirac conv (uncalibrated): ||dh6|| at q-1, q, q+1:', [f'{float(v):.2e}' for v in d[q-1:q+2]])

print('V5 hasher pad + structured collisions')
h = eg.MultiHeadNgramHasher(2_000_000, 4)
a = h.hash_sequence(torch.tensor([[777]])); b = h.hash_sequence(torch.tensor([[0,777]]))
print('  2-gram(pad,777)==2-gram(0,777):', torch.equal(a[2][0,0], b[2][0,1]), '| default K =', h.num_heads)
P = h.primes[3]
for k in range(4):
    g_ = math.gcd(int(P[k,0]), 2_000_000); per = 2_000_000//g_
    if per < 151_665:
        t1 = torch.tensor([[1000, 13, 8304]]); t2 = torch.tensor([[1000+per, 13, 8304]])
        print(f'  head {k}: period {per}; empirical collide ids 1000 vs {1000+per}:', bool(h.hash_sequence(t1)[3][0,2,k]==h.hash_sequence(t2)[3][0,2,k]),
              '| other heads equal?', [bool(h.hash_sequence(t1)[3][0,2,j]==h.hash_sequence(t2)[3][0,2,j]) for j in range(4) if j!=k])
print('  any multiplier prime?', [[all(int(v)%q for q in range(2,int(math.isqrt(int(v)))+1)) for v in row] for row in P.tolist()])

print('V6 C2 ridge share (closed form, as in solver lines 297-305)')
for M in (1,4,8):
  for K in (4,8):
    out = {}
    for n,pen in ((2,10.),(3,1.)):
        A = torch.full((M,K), 1./K); gam = pen*(1+math.log1p(1.))*1e-2
        lhs = A.T@A + gam*torch.eye(K); rhs = A.sum(0,keepdim=True).T  # unit target
        dE = torch.linalg.solve(lhs, rhs); c = float((A@dE)[0])
        out[n] = (c, float(dE.pow(2).sum()))
    sh = out[3][1]/(out[2][1]+out[3][1])
    # what 3-gram slice norm share in delta* would be needed for 85% of ||dE||^2 share?
    need = None
    r2, r3 = out[2][1], out[3][1]
    # share = r3*s/(r3*s + r2*(1-s)) = 0.85 -> s
    s = 0.85*r2/(r3-0.85*r3+0.85*r2)
    print(f'  shared-anchor M={M} K={K}: realized 2g={out[2][0]:.3f} 3g={out[3][0]:.3f}; share(equal slice norms)={sh:.3f}; delta* 3g-slice energy share needed for 85%: {s:.3f}')

print('V7 clamp floor')
for K in (4,8):
    e = mk(r_mem=512, num_heads=K, nb=20_000); e.prefetch_from_host(torch.randint(0,150_000,(8,16)))
    an = e._prefetched_e[:,5].norm(dim=-1).mean()
    print(f'  K={K}: mean||e_q||={float(an):.4f} 0.4x={float(0.4*an):.4f} -> clamp={float((0.4*an).clamp(min=0.1)):.4f}')

print('V8 sdpa vs eager subgraph without mask')
for impl in ('sdpa','eager'):
    mm = Qwen2ForCausalLM(cfg(impl)).eval(); c = sc.SubgraphGateCalibrator(mm, mk())
    ids = torch.randint(0,1000,(2,12))
    with torch.no_grad():
        ref = mm(input_ids=ids, output_hidden_states=True).hidden_states[7]; sub = c._run_subgraph(ids)[1]
    print(f'  {impl}: max|sub-ref|={float((sub-ref).abs().max()):.3e}  (config impl={mm.config._attn_implementation})')
