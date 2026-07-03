# P0 GATE: quantization-aware proxy = product-key top-k SLOT OVERLAP within disjoint banks.
# Validate it reproduces the KNOWN delivery ordering: raw input-embed B-sweep (0.255->0.655 as B up)
# and GTE death (0.000). If the proxy's collision DROPS as B rises for raw, and is catastrophic for GTE,
# it correlates with delivery -> trustworthy for the combinatorial screen.
import os, pickle, numpy as np, hashlib
b = pickle.load(open("/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl","rb")); kept=b["kept"]
rng0=np.random.default_rng(20260625)
sel=rng0.permutation(len(kept))[:137]                       # ~edit-set size
kept=[kept[i] for i in sel]; subs=[r.subject for r in kept]; tids=[list(r.subject_tids) for r in kept]
N=len(kept); MEM=512; NSUB=32; SUBTOPK=4
def bankid(t,B):
    if B<=1: return 0
    return int(hashlib.md5(",".join(map(str,t)).encode()).hexdigest(),16)%B
def whiten(X,eps=0.05):
    mu=X.mean(0);Xc=X-mu;C=(Xc.T@Xc)/len(X);U,S,_=np.linalg.svd(C)
    return Xc@(U@np.diag(1/np.sqrt(S+eps))@U.T)
def proj(X):                                                # generic in_proj -> MEM dims
    P=np.random.default_rng(0).standard_normal((X.shape[1],MEM)).astype(np.float32)/np.sqrt(X.shape[1])
    return X@P
def pk_slots(K,bn):
    if bn: K=(K-K.mean(0))/(K.std(0)+1e-6)                  # query BatchNorm model
    h=MEM//2; rg=np.random.default_rng(1)
    C1=rg.standard_normal((NSUB,h));C2=rg.standard_normal((NSUB,h))
    s1=K[:,:h]@C1.T; s2=K[:,h:2*h]@C2.T
    t1=np.argpartition(-s1,SUBTOPK,1)[:,:SUBTOPK]; t2=np.argpartition(-s2,SUBTOPK,1)[:,:SUBTOPK]
    return [set(int(i)*NSUB+int(j) for i in t1[n] for j in t2[n]) for n in range(N)]
def collision(K,B,bn):
    slots=pk_slots(K,bn); bids=[bankid(t,B) for t in tids]; tot=cnt=0.0
    for bk in range(B):
        ix=[n for n in range(N) if bids[n]==bk]
        for a in range(len(ix)):
            for c in range(a+1,len(ix)):
                s,t=slots[ix[a]],slots[ix[c]]; tot+=len(s&t)/len(s|t); cnt+=1
    return tot/max(cnt,1)
# --- encoders ---
E={}
from transformers import AutoModelForCausalLM
import torch
W=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",dtype=torch.float32,low_cpu_mem_usage=True).get_input_embeddings().weight.detach().numpy()
X=np.stack([W[t].mean(0) for t in tids]).astype(np.float32); del W
E["inembed raw"]=proj(X); E["inembed whitened"]=proj(whiten(X))
from pylate import models
m=models.ColBERT("lightonai/GTE-ModernColBERT-v1")
G=np.stack([np.asarray(d,np.float32).mean(0) for d in m.encode(subs,is_query=False,show_progress_bar=False)])
E["GTE raw"]=proj(G); E["GTE whitened"]=proj(whiten(G))
print("[proxy] mean pairwise SLOT-OVERLAP within banks (lower=better; want: raw drops as B up, GTE high)",flush=True)
print(f"[proxy] {'encoder':22s} {'BN':3s} | B=1     B=8     B=16    B=32",flush=True)
for name,K in E.items():
    for bn in (False,True):
        row=[collision(K,B,bn) for B in (1,8,16,32)]
        print(f"[proxy] {name:22s} {'on' if bn else 'off':3s} | "+"  ".join(f"{v:.3f}" for v in row),flush=True)
print("[proxy] KNOWN delivery (raw, N=137): B1 0.255  B8 0.421  B16 0.526  B32 0.655 | GTE 0.000",flush=True)
print("[proxy] DONE",flush=True)
