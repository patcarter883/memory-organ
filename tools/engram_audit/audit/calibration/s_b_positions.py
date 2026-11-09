# (b) Phase 2 context-drop target: how much of the teacher-vs-student L6 gap is positional (sink / pos-0) artefact?
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
tok, m = load_qwen()
stream, _ = repo_token_stream(tok)
P, C, NB = 256, 256, 6
eng = eg.GraftedEngramModule(d_model=896, r_mem=64, num_buckets=1000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
def h6(ids):
    with torch.no_grad(): return cal._run_subgraph(ids)[1]
def tokloss(a, b):  # the Phase-2 loss, per token: LN-MSE + 0.5*(1-cos) on LN'd states
    na, nb = F.layer_norm(a, (896,)), F.layer_norm(b, (896,))
    return (na - nb).pow(2).mean(-1) + 0.5 * (1 - F.cosine_similarity(na, nb, dim=-1))
buckets = {"pos 0": slice(0, 1), "pos 1-3": slice(1, 4), "pos 4-15": slice(4, 16), "pos 16-255": slice(16, C)}
acc = {k: {"true_vs_student": [], "rand_vs_student": [], "true_vs_rand": []} for k in buckets}
tot = {"true_vs_student": [], "rand_vs_student": [], "true_vs_rand": []}
norm0 = []
for i in range(NB):
    base = 20000 + i * 3000
    prefix = stream[base:base+P]; cont = stream[base+P:base+P+C]
    rprefix = stream[90000 + i*700: 90000 + i*700 + P]          # unrelated text from a different file region
    T_true = h6(torch.cat([prefix, cont]).view(1, -1))[:, P:]
    T_rand = h6(torch.cat([rprefix, cont]).view(1, -1))[:, P:]
    S = h6(cont.view(1, -1))
    norm0.append((S[0, 0].norm() / S[0, 1:].norm(dim=-1).median()).item())
    L = {"true_vs_student": tokloss(T_true, S)[0], "rand_vs_student": tokloss(T_rand, S)[0], "true_vs_rand": tokloss(T_true, T_rand)[0]}
    for k, v in L.items():
        tot[k].append(v.mean().item())
        for bk, sl in buckets.items(): acc[bk][k].append(v[sl].sum().item())
print(f"student L6 ||h[pos0]|| / median ||h[pos>=1]|| = {sum(norm0)/len(norm0):.1f}x (massive-activation sink at student pos 0)")
for k in tot:
    print(f"mean per-token Phase-2 loss  {k:16s}: {sum(tot[k])/len(tot[k]):.4f}")
print("share of the summed per-token loss by position bucket (true_vs_student = the actual Phase-2 target gap):")
for k in ["true_vs_student", "rand_vs_student", "true_vs_rand"]:
    totsum = sum(sum(acc[bk][k]) for bk in buckets)
    print(f"  {k:16s}: " + "  ".join(f"{bk}={sum(acc[bk][k])/totsum:.1%} (mean/tok {sum(acc[bk][k])/NB/(sl.stop-sl.start):.4f})" for bk, sl in buckets.items()))
