# V1: EngramEdit Eq.27 (nonlinear matching) vs Eq.28 (linearised ridge) on the REAL transformers Flash-Next PLE module
# (Qwen4ExpTextPLELayer, tiny dims), native-like weights, and on the graft 'editor' simplification (key_proj=0, conv=0).
import sys, torch, math, random
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
torch.set_num_threads(4); torch.manual_seed(0); random.seed(0)
m = tiny_model(vocab=5000, layers=4)
ple = m.model.layers[1].ple; cfg = m.config
for p in m.parameters(): p.requires_grad_(False)
E = ple.ple_embedding.ngram_embedding.weight
d, hc = cfg.hidden_size, cfg.hc_count; nh = 16; hd = E.shape[1]
def native_like():
    with torch.no_grad():
        E.normal_(0, 0.09 / math.sqrt(hd))                    # real per-head row L2 ~0.09 (t1/real_stats.out)
        w = ple.conv1d.weight; w.normal_(0, 0.0012); w[:, 0, -1].normal_(0, 0.03)   # real taps: lag0 >> lags 3,6,9
        for n in (ple.norm_key, ple.norm_query, ple.norm_conv): n.weight.normal_(-0.15, 0.1)
        ple.value_proj.weight.normal_(0, 12.3 / math.sqrt(d * hd * 16) * 4)  # roughly real ||W_v[:,head]||_F scale-free
        ple.key_proj.weight.normal_(0, 1 / math.sqrt(hd * 16))
def graft_editor():
    with torch.no_grad():
        E.zero_(); ple.key_proj.weight.zero_(); ple.conv1d.weight.zero_()
        for n in (ple.norm_key, ple.norm_query, ple.norm_conv): n.weight.zero_()
        q, _ = torch.linalg.qr(torch.randn(d, d)); ple.value_proj.weight.copy_(q[:, :E.shape[1] * 0 + ple.value_proj.weight.shape[1]])
R = lambda T: [random.randrange(4998) for _ in range(T)]
subj = R(3); P = 8
prompts = [R(random.randrange(2, 6)) + subj + R(3) for _ in range(P)]
anch = [len(x) - 4 for x in prompts]
hbase = torch.randn(d * hc)
H = [(hbase + 0.5 * torch.randn(len(x), d * hc)) for x in prompts]   # residual (hc streams) entering layer 1
fmt_rows = lambda ids, t: ple.ple_embedding(torch.tensor([ids]), None)  # unused helper
# candidate rows = the 16 rows read at the anchor (shared across prompts: subject has 3 tokens)
cap = {}
orig = ple.ple_embedding.ngram_embedding.forward
def rows_at(ids, t):
    out = {}
    def hook(mod, inp, outp): out['ids'] = inp[0]
    h = ple.ple_embedding.ngram_embedding.register_forward_hook(hook)
    ple.ple_embedding(torch.tensor([ids]), None); h.remove()
    return out['ids'][0, t].tolist()
rows = rows_at(prompts[0], anch[0]); assert all(rows_at(x, a) == rows for x, a in zip(prompts, anch))
rows_t = torch.tensor(rows)
IDS = []
for x in prompts:
    out = {}
    hk = ple.ple_embedding.ngram_embedding.register_forward_hook(lambda mod, inp, outp: out.__setitem__('ids', inp[0]))
    ple.ple_embedding(torch.tensor([x]), None); hk.remove(); IDS.append(out['ids'][0])
class _Emb(torch.nn.Module):          # stands in for Qwen4ExpTextNGramEmbedding: returns W[ids].flatten(-2) (same as l.1174)
    cur = None
    def forward(self, input_ids, pkv): return self.cur
real_emb = ple.ple_embedding; fake = _Emb(); ple.ple_embedding = fake
def F(U, p):   # PLE output (real Qwen4ExpTextPLELayer.forward, l.1229-1249) at anchor and conv tap t+3, rows += U
    W = E0.index_add(0, rows_t, U.view(16, hd))
    fake.cur = W[IDS[p]].flatten(-2)[None]
    out = ple(H[p][None], torch.tensor([prompts[p]]), None)[0]
    return out[[anch[p], anch[p] + 3]].flatten()
from torch.func import jacfwd
def run(tag, scales):
    global E0
    E0 = real_emb.ngram_embedding.weight.detach().clone()
    n = 16 * hd; Z = torch.zeros(n)
    base = [F(Z, p).detach() for p in range(P)]
    rown = E0[rows_t].norm(dim=1).mean().item() if E0[rows_t].abs().sum() > 0 else 0.09
    J = [jacfwd(lambda u: F(u, p))(Z) for p in range(P)]
    for kind in ("reachable", "generic"):
        for s in scales:
            torch.manual_seed(1)
            if kind == "reachable":
                Ut = torch.randn(n); Ut = Ut / Ut.view(16, hd).norm(dim=1).mean() * s * rown
                tgt = [(F(Ut, p) - base[p]).detach() for p in range(P)]
            else:   # one shared free delta per anchor (EngramEdit delta_i*), norm = s * |PLE base out at anchor| (or 1)
                dl = torch.randn(base[0].numel() // 2); tgt = []
                for p in range(P):
                    dd = torch.cat([dl, torch.zeros_like(dl)]); tgt.append(dd / dd.norm() * s * max(base[p].norm().item(), 1.0))
            Tn = math.sqrt(sum(t.pow(2).sum().item() for t in tgt))
            res = lambda U: math.sqrt(sum((F(U, p).detach() - base[p] - tgt[p]).pow(2).sum().item() for p in range(P))) / Tn
            lam = 1e-4
            A = sum(j.T @ j for j in J) + lam * torch.eye(n); b = sum(j.T @ t for j, t in zip(J, tgt))
            U1 = torch.linalg.solve(A, b); r1 = res(U1)
            U = U1.clone()   # Gauss-Newton with step halving, 10 iters
            for _ in range(8):
                Jk = [jacfwd(lambda u: F(u, p))(U) for p in range(P)]
                rk = [base[p] + tgt[p] - F(U, p).detach() for p in range(P)]
                step = torch.linalg.solve(sum(j.T @ j for j in Jk) + lam * torch.eye(n), sum(j.T @ r for j, r in zip(Jk, rk)))
                cur = res(U); a = 1.0
                while a > 1e-3 and res(U + a * step) > cur: a /= 2
                if a > 1e-3: U = U + a * step
            rgn = res(U)
            Ud = torch.zeros(n, requires_grad=True); opt = torch.optim.Adam([Ud], lr=0.02 * max(s, 0.05) * rown)
            for it in range(500):
                opt.zero_grad(); L = sum((F(Ud, p) - base[p] - tgt[p]).pow(2).sum() for p in range(P)) / Tn**2 + lam * Ud.pow(2).sum() / Tn**2
                L.backward(); opt.step()
            rd = res(Ud.detach())
            print(f"{tag:7s} {kind:9s} scale {s:5.2f} | rel.residual: Eq28 one-shot {r1:.3f}  GN-8 {rgn:.3f}  direct-Adam {rd:.3f} | |U*|/|row| {U1.view(16,hd).norm(dim=1).mean().item()/rown:.2f}")
ple.ple_embedding = real_emb; native_like(); ple.ple_embedding = fake; run("native", [0.1, 1.0, 5.0])
ple.ple_embedding = real_emb; graft_editor(); ple.ple_embedding = fake; run("graft", [0.1, 1.0, 5.0])
