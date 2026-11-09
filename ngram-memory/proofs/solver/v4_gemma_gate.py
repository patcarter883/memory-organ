# V4: Gemma-4 PLE edit (modeling_gemma4.py l.1420-1427, l.1700-1780): per-layer unigram row E_l[tok] (256-d), multiplicative
# vector gate gelu(W_g h), projection W_p, post RMSNorm (real Gemma4RMSNorm), combined p = (ctxnorm + sqrt(D) E[tok]) / sqrt(2).
# One token's row is shared by every context of that token -> one row, P contexts. Linearised (Eq.28) vs Gauss-Newton vs direct.
import torch, math
from torch.func import jacfwd
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from transformers.activations import ACT2FN
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
d, D, P = 128, 16, 12
act = ACT2FN["gelu_pytorch_tanh"]
Wg = torch.randn(D, d) / math.sqrt(d); Wp = torch.randn(d, D) / math.sqrt(D)
norm = Gemma4RMSNorm(d).double()
with torch.no_grad(): norm.weight.fill_(0.5)              # real rms of post_per_layer_input_norm.weight ranges 0.05-8.6 (t2 real_stats)
row0 = torch.randn(D) / math.sqrt(D)                       # E[tok] (pre sqrt(D) scale); rms ~ 1/sqrt(D) -> scaled rms ~ 1
hbase = torch.randn(d) * 3
H = [hbase + 1.5 * torch.randn(d) for _ in range(P)]      # residual at the token's position in P contexts
C = [torch.randn(D) for _ in range(P)]                    # context part (per_layer_projection_norm(...)) - token-determined, ~const; vary a bit
C = [C[0] + 0.1 * c for c in C]
def out(u, p):
    pl = (C[p] + math.sqrt(D) * (row0 + u)) / math.sqrt(2)
    return norm(Wp @ (act(Wg @ H[p]) * pl))
base = [out(torch.zeros(D), p).detach() for p in range(P)]
bound = 2 * norm.weight.abs().pow(2).mean().sqrt().item() * math.sqrt(d)
print(f"|PLE out| per context ~ {base[0].norm().item():.3f} (fixed by post-norm: rms(w)*sqrt(d) = {0.5*math.sqrt(d):.3f}); |delta out| <= {bound:.3f}")
for s in (0.5, 2, 8, 32):
    u = torch.randn(D); u = u / u.norm() * s * row0.norm()
    print(f"  |du|/|row| {s:5.1f} -> |delta out|/bound {sum((out(u,p)-base[p]).norm().item() for p in range(P))/P/bound:.3f}")
def res(u, tgt): return math.sqrt(sum((out(u, p) - base[p] - tgt[p]).pow(2).sum().item() for p in range(P)) / sum(t.pow(2).sum().item() for t in tgt))
for kind in ("reachable", "generic"):
    for s in (0.1, 0.5, 2.0):
        torch.manual_seed(1)
        if kind == "reachable":
            ut = torch.randn(D); ut = ut / ut.norm() * s * row0.norm(); tgt = [(out(ut, p) - base[p]).detach() for p in range(P)]
        else:
            g = torch.randn(d); tgt = [g / g.norm() * s * base[p].norm() for p in range(P)]
        J = [jacfwd(lambda u: out(u, p))(torch.zeros(D)) for p in range(P)]
        lam = 1e-6
        u1 = torch.linalg.solve(sum(j.T @ j for j in J) + lam * torch.eye(D), sum(j.T @ t for j, t in zip(J, tgt)))
        u = u1.clone()
        for _ in range(10):
            Jk = [jacfwd(lambda v: out(v, p))(u) for p in range(P)]
            rk = [base[p] + tgt[p] - out(u, p) for p in range(P)]
            st = torch.linalg.solve(sum(j.T @ j for j in Jk) + lam * torch.eye(D), sum(j.T @ r for j, r in zip(Jk, rk)))
            a = 1.0; cur = res(u, tgt)
            while a > 1e-4 and res(u + a * st, tgt) > cur: a /= 2
            u = u + a * st
        ud = torch.zeros(D, requires_grad=True); opt = torch.optim.Adam([ud], lr=0.01 * max(s, 0.1))
        for _ in range(3000):
            opt.zero_grad(); L = sum((out(ud, p) - base[p] - tgt[p]).pow(2).sum() for p in range(P)); L.backward(); opt.step()
        print(f"{kind:9s} scale {s:4.1f} | rel.residual Eq28 one-shot {res(u1,tgt):.3f}  GN-10 {res(u,tgt):.3f}  direct-Adam {res(ud.detach(),tgt):.3f}")
# per-context Jacobians differ (vector gate): spread of the linear map across contexts
J = torch.stack([jacfwd(lambda u: out(u, p))(torch.zeros(D)) for p in range(P)])
print(f"context variation of J: |J_p - mean J| / |mean J| = {((J - J.mean(0)).norm() / J.mean(0).norm() / math.sqrt(P)).item():.3f}")
