# (e) How much of an arbitrary residual change at the anchor position can be written through row edits, using the REAL
# Flash-Next PLE block weights (key/value proj, 3 grouped norms, dilated conv) range-fetched from the checkpoint and real
# sampled table rows. Queries are NOT real layer-1 hidden states (not available on this box): Q1 = random Gaussian per
# stream; Q2 = real token-embedding rows repeated over the 4 streams + 30% noise (layer-0 output proxy).
import torch, math, sys
torch.manual_seed(0); torch.set_num_threads(2)
P = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
w = {k: v.float() for k, v in torch.load(f"{P}/nm/t1/real/real_ple_block.pt").items()}
emb = torch.load(f"{P}/nm/t1/real/embed_rows.pt")["E"]
d, hc, eps = 2560, 4, 1e-6
Wk, Wv = w["key_proj.weight"], w["value_proj.weight"]
nk, nq, nc = 1 + w["norm_key.weight"], 1 + w["norm_query.weight"], 1 + w["norm_conv.weight"]
cw = w["conv1d.weight"][:, 0, :]               # (4d, 4) taps for [t-9, t-6, t-3, t]
silu = torch.nn.functional.silu
def gnorm(x, wt):
    xs = x.view(*x.shape[:-1], hc, d); xs = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)
    return xs.flatten(-2) * wt
rows = [w[f"rows_head{h}_start"] for h in (0, 7, 8, 15)]
rand_e = lambda n: torch.cat([rows[h // 4][torch.randint(0, 2048, (n,))] for h in range(16)], -1)
def u_of(e, X):
    k = gnorm(e @ Wk.T, nk).view(*e.shape[:-1], hc, d); q = gnorm(X, nq).view(*X.shape[:-1], hc, d)
    g = (k * q).sum(-1, keepdim=True) / math.sqrt(d); g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
    a = torch.sigmoid(g)
    return (a * (e @ Wv.T).unsqueeze(-2)).flatten(-2), a.squeeze(-1)
L, t = 16, 3
HS = {"all16": list(range(16)), "3gram8": list(range(8, 16)), "2gram8": list(range(8))}
def setup(qmode):
    E0 = rand_e(L)
    X = torch.randn(L, 4 * d) if qmode == "Q1" else emb[:L].repeat(1, 4) + 0.3 * emb[:L].std() * torch.randn(L, 4 * d)
    u0, a0 = u_of(E0, X); n0 = gnorm(u0, nc)
    npad = torch.cat([torch.zeros(9, 4 * d), n0])
    pre0 = torch.stack([sum(cw[:, j] * npad[9 + p - 3 * (3 - j)] for j in range(4)) for p in range(L)])
    return E0, X, u0, a0, n0, pre0
def out_at(de, S, use_conv):
    E0, X, u0, a0, n0, pre0 = S
    u, a = u_of(E0[t] + de, X[t]); o = {}
    if not use_conv:
        return {t: u}, a
    n = gnorm(u, nc)
    for k in range(4):
        p = t + 3 * k
        o[p] = (u if k == 0 else u0[p]) + silu(pre0[p] + cw[:, 3 - k] * (n - n0[t]))
    return o, a
def base_out(S, use_conv):
    E0, X, u0, a0, n0, pre0 = S
    return {p: u0[p] + (silu(pre0[p]) if use_conv else 0) for p in range(L)}
def solve(S, dl, hs, use_conv, iters=60):
    mask = torch.zeros(16, 160); mask[hs] = 1; mask = mask.flatten()
    b = base_out(S, use_conv)
    de = torch.zeros(2560, requires_grad=True)
    opt = torch.optim.LBFGS([de], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); o, _ = out_at(de * mask, S, use_conv)
        loss = (o[t] - b[t] - dl).pow(2).sum() / dl.pow(2).sum(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        o, a = out_at(de * mask, S, use_conv)
        r2 = 1 - ((o[t] - b[t] - dl).pow(2).sum() / dl.pow(2).sum()).item()
        spill = [((o[t + 3 * k] - b[t + 3 * k]).norm() / dl.norm()).item() for k in (1, 2, 3)] if use_conv else [0, 0, 0]
        growth = (de * mask).view(16, 160)[hs].norm(dim=1).mean().item() / S[0][t].view(16, 160)[hs].norm(dim=1).mean().item()
    return r2, spill, a.tolist(), growth
def ridge_ls(S, dl, hs, use_conv, lam=1e-4):
    """frozen-gate regularised least squares (EngramEdit App. A.3.2 analogue): min sum_s||a_s W_v De - dl_s||^2 + lam||De||^2"""
    E0, X, u0, a0, n0, pre0 = S; a = a0[t]
    A = Wv[:, sum(([h * 160 + i for i in range(160)] for h in hs), [])]
    tgt = (a[:, None] * dl.view(4, d)).sum(0) / a.pow(2).sum()
    x = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ tgt)
    de = torch.zeros(2560); de[sum(([h * 160 + i for i in range(160)] for h in hs), [])] = x
    with torch.no_grad():
        o, _ = out_at(de, S, use_conv); b = base_out(S, use_conv)
        return 1 - ((o[t] - b[t] - dl).pow(2).sum() / dl.pow(2).sum()).item()
emb_norm = emb.norm(dim=1).mean().item()
S = setup("Q1"); E0, X, u0, a0, n0, pre0 = S
print(f"base magnitudes at t (Q1): |u_t| per stream {[round(x,3) for x in u0[t].view(4,d).norm(dim=1).tolist()]}, "
      f"|silu(conv)_t| per stream {[round(x,3) for x in silu(pre0[t]).view(4,d).norm(dim=1).tolist()]}, |embed row| {emb_norm:.3f}, gates {[round(x,2) for x in a0[t].tolist()]}")
print("query | conv | target | |dl|/|emb| per stream | heads | R2 ridge-LS(frozen gate) | R2 LBFGS | spill at t+3,6,9 / |dl| | gates after | edited-row-change / row norm")
for qmode in ("Q1", "Q2"):
    S = setup(qmode)
    for use_conv in (True, False):
        for tmode in ("shared", "free"):
            for scale in (0.2, 1.0, 5.0, 25.0):
                for hn in (["all16", "3gram8", "2gram8"] if scale == 1.0 and tmode == "shared" else ["all16"]):
                    torch.manual_seed(int(scale * 10) + (tmode == "free"))
                    dl = torch.randn(d).repeat(4) if tmode == "shared" else torch.randn(4 * d)
                    dl = dl / dl.view(4, d).norm(dim=1).mean() * scale * emb_norm
                    ls = ridge_ls(S, dl, HS[hn], use_conv); r2, sp, g, gr = solve(S, dl, HS[hn], use_conv)
                    print(f"{qmode} | {'on ' if use_conv else 'off'} | {tmode:6s} | {scale:5.1f} | {hn:6s} | {ls:7.3f} | {r2:6.3f} | "
                          f"{', '.join(f'{x:.2f}' for x in sp)} | {'/'.join(f'{x:.2f}' for x in g)} | {gr:8.1f}", flush=True)
