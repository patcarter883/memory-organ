# (a) Phase 1 "zero-backprop subspace alignment": what is trained, what cosine is achieved, double-counting.
import sys, time, hashlib; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
torch.manual_seed(0)
tok, m = load_qwen()
stream, _ = repo_token_stream(tok)
B, T = 4, 128
NTRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 120
train, p = batches(stream, B, T, NTRAIN)
held, _ = batches(stream, B, T, 12, start=len(stream) - 12*B*T)   # last ~6k tokens, disjoint
eng = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, num_heads=4, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)

print("named_parameters of engram:", [n for n, _ in eng.named_parameters()])
print("cpu_tables is nn.Parameter?", isinstance(eng.cpu_tables[2], torch.nn.Parameter), "| in state_dict?", any('table' in k for k in eng.state_dict()))
def tsum(): return hashlib.md5(eng.cpu_tables[2].numpy().tobytes() + eng.cpu_tables[3].numpy().tobytes()).hexdigest()
h0 = tsum()

def eval_cos(batch_list, tag):
    cs, cs_no0, l1, nr = [], [], [], []
    with torch.no_grad():
        for ids in batch_list:
            h2, _, delta = cal._run_subgraph(ids, use_engram=False, record_ffn_accumulation=True)
            eng.prefetch_from_host(ids)
            inj = eng.forward_fusion(h2) - h2
            c = F.cosine_similarity(inj, 0.25*delta, dim=-1)
            cs.append(c.mean()); cs_no0.append(c[:, 1:].mean()); l1.append(F.smooth_l1_loss(inj, 0.25*delta)); nr.append((inj[:,1:].norm(dim=-1)/(0.25*delta[:,1:]).norm(dim=-1)).mean())
    print(f"  [{tag}] cos(all)={torch.stack(cs).mean():.4f}  cos(pos>=1)={torch.stack(cs_no0).mean():.4f}  smoothL1={torch.stack(l1).mean():.4e}  mean ||inj||/||target|| (pos>=1)={torch.stack(nr).mean():.4f}")

t0 = time.time()
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cal.warmup_subspace_alignment(train*EP, lr=1e-3, lookahead_scale=0.25)   # VERBATIM Phase 1, one pass over NTRAIN batches
print(f"Phase 1 ran {NTRAIN}x{EP} steps of [{B},{T}] in {time.time()-t0:.0f}s; cpu_tables changed? {tsum()!=h0}")
eval_cos(train[-12:], "Phase-1 module, TRAIN batches")
eval_cos(held, "Phase-1 module, HELD-OUT")
# fraction of injection energy outside channel 0 (dirac conv init only passes channel 0 initially)
with torch.no_grad():
    w = eng.short_conv.weight; print(f"  short_conv weight: channels with |w|>1e-6: {(w.abs().sum((1,2))>1e-6).sum().item()}/896")

# ---- Baselines on the same target (positions >=1) ----
def collect(batch_list):
    H2, DL, IDS = [], [], []
    with torch.no_grad():
        for ids in batch_list:
            h2, _, delta = cal._run_subgraph(ids, use_engram=False, record_ffn_accumulation=True)
            H2.append(h2[:, 1:].reshape(-1, 896)); DL.append(delta[:, 1:].reshape(-1, 896)); IDS.append(ids)
    return torch.cat(H2), torch.cat(DL), IDS
Htr, Dtr, IDtr = collect(train); Hte, Dte, IDte = collect(held)
def cosm(a, b): return F.cosine_similarity(a, b, dim=-1).mean().item()
mu = Dtr.mean(0, keepdim=True)
print(f"  baseline constant mean-delta direction: held-out cos={cosm(mu.expand_as(Dte), Dte):.4f}")
# ridge h2 -> delta (what the frozen model's own function supports; full-rank linear)
X = torch.cat([Htr, torch.ones(len(Htr), 1)], 1); lam = 1e-2 * X.pow(2).mean()*X.shape[1]
W = torch.linalg.solve(X.T@X + lam*torch.eye(X.shape[1]), X.T@Dtr)
Xte = torch.cat([Hte, torch.ones(len(Hte), 1)], 1)
print(f"  baseline ridge linear(h_2)->delta: train cos={cosm(X@W, Dtr):.4f} held-out cos={cosm(Xte@W, Dte):.4f}")
# per-bigram mean-delta table (what a *trained* table could store); evaluated on held-out tokens whose bigram was seen
from collections import defaultdict
def bigrams(IDS):
    out = []
    for ids in IDS:
        for b in range(ids.shape[0]):
            for t in range(1, ids.shape[1]): out.append((ids[b, t-1].item(), ids[b, t].item()))
    return out
btr, bte = bigrams(IDtr), bigrams(IDte)
acc = defaultdict(lambda: [torch.zeros(896), 0])
for k, d in zip(btr, Dtr): acc[k][0] += d; acc[k][1] += 1
pred, seen = [], []
for k in bte:
    if k in acc: pred.append(acc[k][0]/acc[k][1]); seen.append(True)
    else: pred.append(mu[0]); seen.append(False)
pred = torch.stack(pred); seen = torch.tensor(seen)
print(f"  baseline trained-bigram-table (mean delta per seen bigram, else mean): held-out cos={cosm(pred, Dte):.4f}; on seen bigrams ({seen.float().mean():.2%}) cos={cosm(pred[seen], Dte[seen]):.4f}")
print(f"  distinct train bigrams {len(acc)}, train tokens {len(btr)}, r_mem=512")

# ---- Double counting: inject the EXACT Phase-1 target at L2 and keep running L3..L23 ----
def nll_with_l2_add(ids, add):
    hook = m.model.layers[2].register_forward_hook(lambda mod, i, o: (o + add) if not isinstance(o, tuple) else (o[0] + add,) + o[1:])
    try:
        out = m(input_ids=ids, labels=ids, output_hidden_states=True)
    finally: hook.remove()
    return out.loss.item(), out.hidden_states
print("Double-counting test (held-out, oracle injection = s * (L3..L6 block-delta sum) added at L2 output):")
with torch.no_grad():
    for s in [0.0, 0.25, 1.0]:
        nl, rel6, cosov = [], [], []
        for ids in held[:6]:
            _, h6, delta = cal._run_subgraph(ids, record_ffn_accumulation=True)
            nll, hs = nll_with_l2_add(ids, s*delta)
            # compare the new L6 state with the base L6 state and with base+ s*delta (pure overshoot)
            new6 = hs[7]
            nl.append(nll); rel6.append(((new6 - h6)[:, 1:].norm() / h6[:, 1:].norm()).item())
            cosov.append(F.cosine_similarity((new6 - h6)[:, 1:], delta[:, 1:], dim=-1).mean().item())
        print(f"  s={s:4.2f}: LM NLL={sum(nl)/len(nl):.4f}  ||h6_new-h6||/||h6|| (pos>=1)={sum(rel6)/len(rel6):.4f}  cos(h6_new-h6, delta)={sum(cosov)/len(cosov):.4f}")
    # the trained Phase-1 module grafted into the full model
    g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
    nl = [m(input_ids=ids, labels=ids).loss.item() for ids in held[:6]]
    g.pre_hook.remove(); g.layer_hook.remove()
    print(f"  Phase-1-trained graft in full model: LM NLL={sum(nl)/len(nl):.6f}")
