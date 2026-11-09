# V3: tiny random Qwen4Exp (real transformers classes): optimisation variable choice.
#  (A) direct row-space: optimise the anchor's 16 rows for final-position NLL (+ KL essence term)
#  (B) EngramEdit two-stage: free delta at the PLE output (hc space, 4 x d) at the anchor for final NLL, then allocate
#      by nonlinear matching (Gauss-Newton on Eq.27)       (C) as B but delta restricted to the reachable family g_c * v
import sys, torch, math, random
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
from torch.func import jacfwd
torch.set_num_threads(4); torch.manual_seed(0); random.seed(3)
m = tiny_model(vocab=5000, layers=4); cfg = m.config
for p in m.parameters(): p.requires_grad_(False)
ple = m.model.layers[1].ple; E = ple.ple_embedding.ngram_embedding.weight; hd = E.shape[1]; d = cfg.hidden_size; hc = cfg.hc_count
with torch.no_grad():
    E.normal_(0, 0.09 / math.sqrt(hd)); w = ple.conv1d.weight; w.normal_(0, 0.0012); w[:, 0, -1].normal_(0, 0.03)
    for n in (ple.norm_key, ple.norm_query, ple.norm_conv): n.weight.normal_(-0.15, 0.1)
E0 = E.detach().clone()
R = lambda T: [random.randrange(4998) for _ in range(T)]
subj, rel, tgt = R(3), R(3), 777
train = [R(random.randrange(2, 6)) + subj + rel for _ in range(6)]
held = [R(random.randrange(2, 6)) + subj + rel for _ in range(6)]
ess = [R(3) + subj + R(3) for _ in range(4)]        # same subject, other continuation ("{s} is a" stand-in)
anc = lambda x: len(x) - 4
def rows_at(x, t):
    out = {}
    hk = ple.ple_embedding.ngram_embedding.register_forward_hook(lambda mod, inp, o: out.__setitem__('i', inp[0]))
    ple.ple_embedding(torch.tensor([x]), None); hk.remove(); return out['i'][0, t]
rows = rows_at(train[0], anc(train[0]))
DELTA = {}
def hook(mod, inp, out):
    if 'd' in DELTA: out = out.clone(); out[0, DELTA['t']] += DELTA['d']
    return out
ple.register_forward_hook(hook)
def logits(x, U=None):
    with torch.no_grad(): E.copy_(E0)
    if U is not None:
        del ple.ple_embedding.ngram_embedding.weight
        ple.ple_embedding.ngram_embedding.weight = E0.index_add(0, rows, U)
    lo = m(input_ids=torch.tensor([x]), use_cache=False).logits[0, -1]
    if U is not None: ple.ple_embedding.ngram_embedding.weight = torch.nn.Parameter(E0.clone(), requires_grad=False)
    return lo
base_lp = {id(x): torch.log_softmax(logits(x), -1).detach() for x in train + held + ess}
def evaluate(tag, fn):
    with torch.no_grad():
        lt = sum(torch.log_softmax(fn(x), -1)[tgt].item() - base_lp[id(x)][tgt].item() for x in train) / len(train)
        lh = sum(torch.log_softmax(fn(x), -1)[tgt].item() - base_lp[id(x)][tgt].item() for x in held) / len(held)
        kl = sum(torch.nn.functional.kl_div(base_lp[id(x)], torch.log_softmax(fn(x), -1), log_target=True, reduction='sum').item() for x in ess) / len(ess)
        lpe = sum(torch.log_softmax(fn(x), -1)[tgt].item() - base_lp[id(x)][tgt].item() for x in ess) / len(ess)
    print(f"{tag:42s} | dlogP(tgt) train {lt:+.2f} held-out {lh:+.2f} | essence KL {kl:.3f} dlogP(tgt) {lpe:+.2f}")
def nll_loss(fn, xs): return sum(-torch.log_softmax(fn(x), -1)[tgt] for x in xs) / len(xs)
def kl_loss(fn): return sum(torch.nn.functional.kl_div(base_lp[id(x)], torch.log_softmax(fn(x), -1), log_target=True, reduction='sum') for x in ess) / len(ess)
STEPS = 120
# hc-space norm at the anchor (trust-region reference)
cap = {}
hk = m.model.layers[1].register_forward_pre_hook(lambda mod, a, k: cap.__setitem__('h', (a[0] if a else k['hidden_states'])), with_kwargs=True)
m(input_ids=torch.tensor([train[0]])); hk.remove(); hn = cap['h'][0, anc(train[0])].norm().item()
print(f"|h| at layer-1 input (hc {hc}x{d}) at anchor: {hn:.2f}; |PLE out| there {ple(cap['h'], torch.tensor([train[0]]), None)[0, anc(train[0])].norm().item():.3f}")
for lam_kl in (0.0, 1.0):
    # (A) direct row space
    U = torch.zeros(16, hd, requires_grad=True); opt = torch.optim.Adam([U], lr=0.05)
    for s in range(STEPS):
        opt.zero_grad(); f = lambda x: logits(x, U); L = nll_loss(f, train) + lam_kl * kl_loss(f); L.backward(); opt.step()
    Ua = U.detach().clone(); evaluate(f"(A) direct rows, lam_KL={lam_kl}", lambda x: logits(x, Ua))
    # (B)/(C) free / reachable delta at anchor, then allocate
    for kind in ("free", "reachable"):
        DELTA.clear()
        g = None
        if kind == "reachable":   # current per-stream gates at the anchor of each prompt are ~ shared; use prompt 0's
            gcap = {}
            def gh(mod, a, o): gcap['o'] = o
            v = torch.zeros(d, requires_grad=True); par = [v]
        else:
            dl = torch.zeros(hc * d, requires_grad=True); par = [dl]
        opt = torch.optim.Adam(par, lr=0.05)
        def fd(x):
            DELTA['t'] = anc(x); DELTA['d'] = (dl if kind == "free" else v.repeat(hc) * 0.5); return logits(x)
        for s in range(STEPS):
            opt.zero_grad(); L = nll_loss(fd, train) + lam_kl * kl_loss(fd); L.backward(); opt.step()
            with torch.no_grad():
                for p_ in par:
                    nr = p_.norm() * (1 if kind == "free" else 0.5 * math.sqrt(hc))
                    if nr > 1.0 * hn: p_.mul_(1.0 * hn / nr)        # clamp rho = 1 relative to |h| (hc space)
        delta = (dl if kind == "free" else v.repeat(hc) * 0.5).detach().clone()
        evaluate(f"(B/C) stage-1 delta* {kind}, lam_KL={lam_kl}", lambda x: (DELTA.update(t=anc(x), d=delta), logits(x))[1])
        DELTA.clear()
        # stage 2: allocate delta into rows by Gauss-Newton matching of the PLE output at the anchor (Eq.27), all train prompts
        Hs = []
        for x in train:
            cap = {}; hk = m.model.layers[1].register_forward_pre_hook(lambda mod, a, k: cap.__setitem__('h', a[0] if a else k['hidden_states']), with_kwargs=True)
            m(input_ids=torch.tensor([x])); hk.remove(); Hs.append(cap['h'])
        def Fm(Uf, i):
            x = train[i]
            del ple.ple_embedding.ngram_embedding.weight
            ple.ple_embedding.ngram_embedding.weight = E0.index_add(0, rows, Uf.view(16, hd))
            o = ple(Hs[i], torch.tensor([x]), None)[0, anc(x)]
            ple.ple_embedding.ngram_embedding.weight = torch.nn.Parameter(E0.clone(), requires_grad=False)
            return o
        b0 = [Fm(torch.zeros(16 * hd), i).detach() for i in range(len(train))]
        Uf = torch.zeros(16 * hd)
        for it in range(8):
            Js = [jacfwd(lambda u: Fm(u, i))(Uf) for i in range(len(train))]
            rs = [b0[i] + delta - Fm(Uf, i).detach() for i in range(len(train))]
            Uf = Uf + torch.linalg.solve(sum(j.T @ j for j in Js) + 1e-4 * torch.eye(16 * hd), sum(j.T @ r for j, r in zip(Js, rs)))
        frac = 1 - sum((b0[i] + delta - Fm(Uf, i).detach()).pow(2).sum().item() for i in range(len(train))) / (len(train) * delta.pow(2).sum().item())
        Ub = Uf.view(16, hd).clone()
        evaluate(f"(B/C) allocated {kind} (R2 match {frac:.2f}), lam_KL={lam_kl}", lambda x: logits(x, Ub))
