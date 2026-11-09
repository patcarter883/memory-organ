# (b3) Run VERBATIM Phase 1 then VERBATIM Phase 2 (fp32) briefly; where does the Phase-2 loss reduction land (by position)?
import sys, time; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
torch.manual_seed(0)
tok, m = load_qwen(); stream, _ = repo_token_stream(tok)
eng = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
cal.warmup_subspace_alignment(batches(stream, 4, 128, 120)[0])
NP2 = int(sys.argv[1]) if len(sys.argv) > 1 else 60
p2 = batches(stream, 2, 512, NP2, start=0)[0]
held = batches(stream, 2, 512, 4, start=len(stream) - 4*1024)[0]
Bk = {"pos0": slice(0,1), "pos1-15": slice(1,16), "pos16+": slice(16,256), "all": slice(0,256)}
def eval_loss(tag):
    acc = {b: [] for b in Bk}
    with torch.no_grad():
        for full in held:
            Tt = cal._run_subgraph(full)[1][:, 256:]
            S = cal._run_subgraph(full[:, 256:], use_engram=True)[1]
            nt, ns = F.layer_norm(Tt, (896,)), F.layer_norm(S, (896,))
            tl = (ns - nt).pow(2).mean(-1) + 0.5*(1 - F.cosine_similarity(ns, nt, dim=-1))
            for b, sl in Bk.items(): acc[b].append(tl[:, sl].mean().item())
        g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
        nll = sum(m(input_ids=x, labels=x).loss.item() for x in held) / len(held)
        g.pre_hook.remove(); g.layer_hook.remove()
    print(f"[{tag}] held-out Phase-2 loss " + "  ".join(f"{b}={sum(v)/len(v):.4f}" for b, v in acc.items()) + f" | graft LM NLL={nll:.4f}")
with torch.no_grad(): base_nll = sum(m(input_ids=x, labels=x).loss.item() for x in held) / len(held)
print(f"base model LM NLL (no graft) = {base_nll:.4f}")
eval_loss("after Phase 1")
snap = {n: eng.cpu_tables[n].clone() for n in eng.orders}
t0 = time.time()
for i in range(0, NP2, 20):
    cal.calibrate_context_distillation(p2[i:i+20], prefix_len=256, lr_gates=float(sys.argv[2]))  # verbatim code, lr_gates raised (audit deviation) (note: AdamW state re-created per call)
    eval_loss(f"after Phase 2 x{i+20} steps of [2,512], {time.time()-t0:.0f}s")
print("max |table change| after Phase 2 (fp32 tables):", max((eng.cpu_tables[n]-snap[n]).abs().max().item() for n in eng.orders))
