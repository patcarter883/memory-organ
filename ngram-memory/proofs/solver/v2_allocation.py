# V2: allocation of a module-output target across shared hashed rows (linear sum aggregation = graft 'editor' init,
# or the Stage-2 subproblem). Checks: closed form (EngramEdit Eq.10) == gradient solution; per-n-gram (A.3.3) vs
# per-ROW collateral-load penalty; generalisation by subject length and key set (subject-internal / context-crossing /
# unigram superset for grafts); collateral on a synthetic Zipfian corpus hashed with a Flash-Next-style hash.
import torch, math, random, collections, itertools
torch.manual_seed(0); random.seed(0); torch.set_default_dtype(torch.float64)
V_SZ, HPN, R, D = 5000, 8, 8, 64             # vocab, heads per order, row width, residual width
ORDERS_NATIVE, ORDERS_GRAFT = (2, 3), (1, 2, 3)
def _np(x):
    x += 1
    while any(x % q == 0 for q in range(2, int(x**0.5) + 1)): x += 1
    return x
PR = []
_p = 19999
for _ in range(24): _p = _np(_p); PR.append(_p)
MULT = [2 * random.randrange(1 << 40) + 1 for _ in range(3)]
def hrow(ng, n, h):   # Flash-Next style: XOR of token*multiplier, mod a per-head prime (rows are per (order, head))
    x = 0
    for k, tok in enumerate(reversed(ng)): x ^= tok * MULT[k]
    return (n, h, x % PR[(n - 1) * HPN + h])
# Zipfian corpus with some bigram structure
zipf = [1 / (i + 1) ** 1.07 for i in range(V_SZ)]; Z = sum(zipf); zipf = [z / Z for z in zipf]
corpus = random.choices(range(V_SZ), weights=zipf, k=400_000)
cnt = collections.Counter(); load = collections.Counter()
for n in (1, 2, 3):
    for i in range(n - 1, len(corpus)):
        ng = tuple(corpus[i - n + 1:i + 1]); cnt[ng] += 1
        for h in range(HPN): load[hrow(ng, n, h)] += 1
def pct(values):   # within-order percentile
    s = sorted(values); return lambda v: sum(1 for x in s if x <= v) / len(s)
pct_ng = {n: pct([c for g, c in cnt.items() if len(g) == n]) for n in (1, 2, 3)}
pct_row = {n: pct([c for k, c in load.items() if k[0] == n]) for n in (1, 2, 3)}
V = torch.linalg.qr(torch.randn(D, D))[0][:, :D]
Vh = {}  # value map per (order, head): D x R slice of a fixed random projection (0.5 * W_v in the graft)
Wv = 0.5 * torch.randn(D, 3 * HPN * R) / math.sqrt(R * 2)
for i, (n, h) in enumerate(itertools.product((1, 2, 3), range(HPN))): Vh[(n, h)] = Wv[:, i * R:(i + 1) * R]
def keys_at(seq, t, orders):
    return [hrow(tuple(seq[t - n + 1:t + 1]), n, h) for n in orders for h in range(HPN) if t - n + 1 >= 0]
eta = {1: 32, 2: 8, 3: 2}   # EngramEdit B.4: eta2=8, eta3=2 (eta1 extrapolated, UNVERIFIED choice)
def lam_ngram(key_ng, n):   # EngramEdit A.3.3, Eq.29 + freq weight, lambda = 0.05 w + 0.01
    wf = min(1 + 9 * pct_ng[n](cnt.get(key_ng, 0)), 10); return 0.05 * min(eta[n] + wf - 1, 64) + 0.01
def lam_row(row, own):      # proposed: penalise the row's COLLATERAL corpus load (other n-grams hashing there)
    coll = max(load.get(row, 0) - own, 0); wf = min(1 + 9 * pct_row[row[0]](coll), 10) if coll > 0 else 1
    return 0.05 * min(eta[row[0]] + wf - 1, 64) + 0.01
def solve(train, tgt, orders, pen, keep=None):
    cols = sorted({k for s, t in train for k in keys_at(s, t, orders) if keep is None or keep(s, t, k)})
    ix = {k: i for i, k in enumerate(cols)}
    A = torch.zeros(len(train) * D, len(cols) * R)
    for p, (s, t) in enumerate(train):
        for k in keys_at(s, t, orders):
            if k not in ix: continue
            A[p * D:(p + 1) * D, ix[k] * R:(ix[k] + 1) * R] += Vh[k[0], k[1]]
    B = tgt.repeat(len(train))
    lam = torch.zeros(len(cols) * R)
    for k, i in ix.items():
        ngs = {tuple(s[t - k[0] + 1:t + 1]) for s, t in train if k in keys_at(s, t, orders) and tuple(s[t - k[0] + 1:t + 1]) and hrow(tuple(s[t - k[0] + 1:t + 1]), k[0], k[1]) == k}
        own = sum(cnt.get(g, 0) for g in ngs)
        lam[i * R:(i + 1) * R] = (lam_ngram(next(iter(ngs)), k[0]) if pen == "ngram" else lam_row(k, own) if pen == "row" else 0.01)
    U = torch.linalg.solve(A.T @ A + torch.diag(lam), A.T @ B)           # Eq.10
    return {k: U[i * R:(i + 1) * R] for k, i in ix.items()}, (A, B, lam, U)
def realised(seq, t, U, orders, tgt):
    out = sum((Vh[k[0], k[1]] @ U[k] for k in keys_at(seq, t, orders) if k in U), torch.zeros(D))
    return (out @ tgt / tgt.dot(tgt)).item()
def collateral(U, own_ngrams):   # expected squared spill per corpus token, relative to |tgt|^2: sum_j collload_j |V u_j|^2 / N
    tot = 0.0
    for k, u in U.items():
        own = sum(cnt.get(g, 0) for g in own_ngrams if len(g) == k[0]); coll = max(load.get(k, 0) - own, 0)
        tot += coll * (Vh[k[0], k[1]] @ u).pow(2).sum().item()
    return tot / len(corpus)
# -- check 1: closed form == gradient descent on the same weighted ridge objective
ctx = lambda: random.choices(range(V_SZ), weights=zipf, k=1)[0]
subj = [random.randrange(500, 5000) for _ in range(3)]
train = [([ctx(), ctx()] + subj, 4) for _ in range(8)]; tgt = torch.randn(D); tgt /= tgt.norm()
U, (A, B, lam, Ucf) = solve(train, tgt, ORDERS_NATIVE, "ngram")
u = torch.zeros_like(Ucf, requires_grad=True); opt = torch.optim.LBFGS([u], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
def clo():
    opt.zero_grad(); L = (A @ u - B).pow(2).sum() + (lam * u * u).sum(); L.backward(); return L
opt.step(clo)
print(f"[check1] closed-form vs LBFGS on weighted ridge: max|dU| = {(u.detach() - Ucf).abs().max().item():.2e} (|U| max {Ucf.abs().max().item():.2e})")
# -- check 2: generalisation + collateral by subject length, key set, penalty
print("subjL orders  penalty | train-fit  heldout-fit(new ctx)  heldout-min | order share (2g/3g/1g of |Vu|^2) | collateral/edit")
for L in (1, 2, 3):
    rows_out = collections.defaultdict(list)
    for trial in range(12):
        subj = [random.randrange(300, 5000) for _ in range(L)]
        train = [([ctx(), ctx(), ctx()] + subj, 2 + L) for _ in range(8)]
        held = [([ctx(), ctx(), ctx()] + subj, 2 + L) for _ in range(16)]
        tgt = torch.randn(D); tgt /= tgt.norm()
        own = {tuple(s[t - n + 1:t + 1]) for s, t in train for n in (1, 2, 3)}
        for orders, pen in ((ORDERS_NATIVE, "ridge"), (ORDERS_NATIVE, "ngram"), (ORDERS_NATIVE, "row"), (ORDERS_GRAFT, "ngram"), (ORDERS_GRAFT, "row")):
            U, _ = solve(train, tgt, orders, pen)
            tr = sum(realised(s, t, U, orders, tgt) for s, t in train) / len(train)
            hv = [realised(s, t, U, orders, tgt) for s, t in held]
            en = collections.Counter()
            for k, u in U.items(): en[k[0]] += (Vh[k[0], k[1]] @ u).pow(2).sum().item()
            tot = sum(en.values())
            rows_out[(orders, pen)].append((tr, sum(hv) / len(hv), min(hv), en[2] / tot, en[3] / tot, en[1] / tot, collateral(U, own)))
    for (orders, pen), v in rows_out.items():
        mean = [sum(x[i] for x in v) / len(v) for i in range(7)]
        print(f"  {L}    {''.join(map(str, orders)):6s} {pen:6s}  | {mean[0]:.3f}      {mean[1]:.3f}                 {mean[2]:.3f}      | {mean[3]:.2f}/{mean[4]:.2f}/{mean[5]:.2f}                    | {mean[6]:.2e}")
# -- check 3: does the per-row penalty steer an edit away from loaded heads?  (same n-gram, 8 heads with different loads)
subj = [random.randrange(300, 5000) for _ in range(3)]
train = [([ctx(), ctx()] + subj, 4) for _ in range(8)]; tgt = torch.randn(D); tgt /= tgt.norm()
own = {tuple(s[t - n + 1:t + 1]) for s, t in train for n in (2, 3)}
for pen in ("ngram", "row"):
    U, _ = solve(train, tgt, ORDERS_NATIVE, pen)
    for n in (2, 3):
        ks = sorted(k for k in U if k[0] == n)
        print(f"[check3] {pen:5s} {n}-gram heads: collateral load " + " ".join(f"{max(load[k]-sum(cnt.get(g,0) for g in own if len(g)==n),0):3d}" for k in ks)
              + " | |V u| " + " ".join(f"{(Vh[k[0],k[1]] @ U[k]).norm().item():.3f}" for k in ks))
    print(f"[check3] {pen:5s} collateral/edit {collateral(U, own):.2e}  train-fit {sum(realised(s,t,U,ORDERS_NATIVE,tgt) for s,t in train)/8:.3f}")

# -- check 4: context diversity in the Stage-2 system (EngramEdit uses 5 prefix contexts per expression) and
#    subject-internal-only key sets. internal(k) <=> the n-gram ending at t lies inside the subject span.
print("[check4] subjL orders keys      #train-ctx | train-fit heldout-fit heldout-min collateral")
for L in (1, 2):
    for orders, keyset in ((ORDERS_NATIVE, "all"), (ORDERS_NATIVE, "internal"), (ORDERS_GRAFT, "all"), (ORDERS_GRAFT, "internal")):
        for K in (8, 40):
            acc = []
            for trial in range(8):
                subj = [random.randrange(300, 5000) for _ in range(L)]
                train = [([ctx(), ctx(), ctx()] + subj, 2 + L) for _ in range(K)]
                held = [([ctx(), ctx(), ctx()] + subj, 2 + L) for _ in range(16)]
                tgt = torch.randn(D); tgt /= tgt.norm()
                own = {tuple(s[t - n + 1:t + 1]) for s, t in train for n in (1, 2, 3)}
                keep = None if keyset == "all" else (lambda s, t, k: k[0] <= L)
                U, _ = solve(train, tgt, orders, "row", keep)
                if not U: acc.append((0, 0, 0, 0)); continue
                tr = sum(realised(s, t, U, orders, tgt) for s, t in train) / len(train)
                hv = [realised(s, t, U, orders, tgt) for s, t in held]
                acc.append((tr, sum(hv) / len(hv), min(hv), collateral(U, own)))
            mean = [sum(x[i] for x in acc) / len(acc) for i in range(4)]
            print(f"[check4]  {L}    {''.join(map(str, orders)):6s} {keyset:9s} {K:3d}        | {mean[0]:.3f}     {mean[1]:.3f}       {mean[2]:.3f}       {mean[3]:.2e}")
