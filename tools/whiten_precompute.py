import os, pickle, numpy as np, torch
b=pickle.load(open("/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl","rb")); kept=b["kept"]
tids=[tuple(r.subject_tids) for r in kept]; subs=[r.subject for r in kept]
def fit_whiten(X,eps=0.05):
    mu=X.mean(0); Xc=X-mu; C=(Xc.T@Xc)/len(X); U,S,_=np.linalg.svd(C)
    W=U@np.diag(1/np.sqrt(S+eps))@U.T; return mu,W
def save(d,path): pickle.dump(d,open(path,"wb")); print(f"[white] saved {len(d)} -> {path} dim {next(iter(d.values())).shape}",flush=True)
# 1. whitened input-embeddings (soft-ZCA, eps=0.05)
from transformers import AutoModelForCausalLM
E=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",dtype=torch.float32,low_cpu_mem_usage=True).get_input_embeddings().weight.detach().numpy()
X=np.stack([E[list(t)].mean(0) for t in tids]).astype(np.float32); del E
mu,W=fit_whiten(X); Xw=(X-mu)@W
save({t:Xw[i] for i,t in enumerate(tids)}, "/engine/.probe_cache/whiten_inembed_keys.pkl")
# 2. whitened GTE (the revival)
from pylate import models
m=models.ColBERT("lightonai/GTE-ModernColBERT-v1")
G=np.stack([np.asarray(d,np.float32).mean(0) for d in m.encode(subs,is_query=False,show_progress_bar=False)])
mu2,W2=fit_whiten(G); Gw=(G-mu2)@W2
save({t:Gw[i] for i,t in enumerate(tids)}, "/engine/.probe_cache/whiten_gte_keys.pkl")
print("[white] DONE",flush=True)
