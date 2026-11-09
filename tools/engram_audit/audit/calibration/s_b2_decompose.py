# (b2) Decompose the Phase-2 gap: S -> E_rand[T] (generic "being after some prefix": position/sink) vs E_rand[T] -> T_true (prefix-content specific)
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
tok, m = load_qwen()
stream, _ = repo_token_stream(tok)
P, C, NB, KR = 256, 256, 5, 4
eng = eg.GraftedEngramModule(d_model=896, r_mem=64, num_buckets=1000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
def h6(ids):
    with torch.no_grad(): return F.layer_norm(cal._run_subgraph(ids)[1][0], (896,))
def tl(na, nb): return (na - nb).pow(2).mean(-1) + 0.5 * (1 - F.cosine_similarity(na, nb, dim=-1))
B = {"pos0": slice(0,1), "pos1-15": slice(1,16), "pos16+": slice(16,C), "all": slice(0,C)}
R = {k: {b: [] for b in B} for k in ["true_vs_S", "generic(S->meanRand)", "content(meanRand->true)", "SshiftedOne_vs_true"]}
for i in range(NB):
    base = 20000 + i*3000
    prefix = stream[base:base+P]; cont = stream[base+P:base+P+C]
    Tt = h6(torch.cat([prefix, cont]).view(1,-1))[P:]
    Tr = torch.stack([h6(torch.cat([stream[90000+(i*KR+k)*600:90000+(i*KR+k)*600+P], cont]).view(1,-1))[P:] for k in range(KR)]).mean(0)
    S = h6(cont.view(1,-1))
    # cheapest positional fix: give the student ONE dummy token (newline) before the continuation, drop it at the loss
    S1 = h6(torch.cat([torch.tensor(tok("\n")["input_ids"]), cont]).view(1,-1))[1:]
    for k, v in {"true_vs_S": tl(Tt,S), "generic(S->meanRand)": tl(Tr,S), "content(meanRand->true)": tl(Tt,Tr), "SshiftedOne_vs_true": tl(Tt,S1)}.items():
        for b, sl in B.items(): R[k][b].append(v[sl].mean().item())
for k in R: print(f"{k:26s} " + "  ".join(f"{b}={sum(v)/len(v):.4f}" for b, v in R[k].items()))
