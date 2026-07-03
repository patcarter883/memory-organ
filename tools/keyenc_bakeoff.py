import os, pickle, numpy as np, torch
b = pickle.load(open("/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl","rb"))
kept = b["kept"]; subs=[r.subject for r in kept]; tids=[list(r.subject_tids) for r in kept]
N=len(subs); print(f"[bake] {N} subjects", flush=True)
rng = np.random.default_rng(0)
idx = rng.permutation(N)[:1200]           # subsample for O(N^2) metrics speed
subsS=[subs[i] for i in idx]; tidsS=[tids[i] for i in idx]; M=len(idx)

def whiten(X):                            # ZCA whitening: center + decorrelate -> isotropy
    mu=X.mean(0); Xc=X-mu; C=(Xc.T@Xc)/len(X)
    U,S,_=np.linalg.svd(C); W=U@np.diag(1.0/np.sqrt(S+1e-5))@U.T
    return Xc@W

def rep_metrics(X, tag):                  # X [M,d] dense -> separability metrics
    Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
    S=Xn@Xn.T
    pair=S[np.triu_indices(M,1)].mean()
    np.fill_diagonal(S,-9); nn=S.max(1).mean()
    top1=int((S.argmax(1)==np.arange(M)).sum())/M   # (self is -9'd, so this is nearest-OTHER; report collision instead)
    print(f"[bake] {tag:34s} mean-pair-cos {pair:+.3f} | nearest-neighbor-cos {nn:+.3f}  (lower=better sep)", flush=True)
    return nn

results={}
# --- 1. Qwen3.5-4B INPUT EMBEDDINGS (incumbent) ---
try:
    from transformers import AutoModelForCausalLM
    E=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",dtype=torch.float32,low_cpu_mem_usage=True).get_input_embeddings().weight.detach().numpy()
    X=np.stack([E[t].mean(0) for t in tidsS]).astype(np.float32)
    results["Qwen-inembed raw"]=rep_metrics(X,"Qwen input-embed pooled RAW")
    results["Qwen-inembed whitened"]=rep_metrics(whiten(X),"Qwen input-embed pooled WHITENED")
    del E
except Exception as e: print("[bake] Qwen-inembed FAILED",repr(e),flush=True)
# --- 2/3. GTE-ModernColBERT pooled (raw+white) and MaxSim ---
try:
    from pylate import models
    m=models.ColBERT("lightonai/GTE-ModernColBERT-v1")
    docs=[np.asarray(d,np.float32) for d in m.encode(subsS,is_query=False,show_progress_bar=False)]
    X=np.stack([d.mean(0) for d in docs])
    results["GTE pooled raw"]=rep_metrics(X,"GTE-ColBERT pooled RAW")
    results["GTE pooled whitened"]=rep_metrics(whiten(X),"GTE-ColBERT pooled WHITENED")
    qs=[np.asarray(q,np.float32) for q in m.encode(subsS,is_query=True,show_progress_bar=False)]
    Sm=np.zeros((M,M),np.float32)
    for i in range(M):
        for j in range(M): Sm[i,j]=(qs[i]@docs[j].T).max(1).sum()
    Sn=Sm.copy(); np.fill_diagonal(Sn,-1e9)
    diag=Sm[np.arange(M),np.arange(M)]; margin=(diag-Sn.max(1)).mean()
    # normalize maxsim to a pseudo-cos for comparability: nn = best-other / self
    nn=(Sn.max(1)/(diag+1e-9)).mean()
    print(f"[bake] {'GTE-ColBERT MaxSim (multi)':34s} nn/self ratio {nn:+.3f} | margin {margin:+.3f}  (lower ratio=better)",flush=True)
    results["GTE MaxSim (ratio)"]=nn
except Exception as e: print("[bake] GTE FAILED",repr(e),flush=True)
# --- 4. Qwen3-Embedding-0.6B (dense semantic MTEB) ---
try:
    from sentence_transformers import SentenceTransformer
    st=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    X=st.encode(subsS,show_progress_bar=False,convert_to_numpy=True).astype(np.float32)
    results["Qwen3-Emb raw"]=rep_metrics(X,"Qwen3-Embedding-0.6B RAW")
    results["Qwen3-Emb whitened"]=rep_metrics(whiten(X),"Qwen3-Embedding-0.6B WHITENED")
except Exception as e: print("[bake] Qwen3-Emb FAILED",repr(e),flush=True)
# --- 5. SPLADE-v3 (learned sparse lexical) ---
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tk=AutoTokenizer.from_pretrained("naver/splade-v3"); sp=AutoModelForMaskedLM.from_pretrained("naver/splade-v3").eval()
    def splade(txts):
        out=[]
        for t in txts:
            enc=tk(t,return_tensors="pt")
            with torch.no_grad(): lo=sp(**enc).logits[0]
            v=torch.max(torch.log1p(torch.relu(lo))*enc.attention_mask[0].unsqueeze(-1),dim=0).values
            out.append(v.numpy())
        return np.stack(out).astype(np.float32)
    X=splade(subsS)
    results["SPLADE-v3 sparse"]=rep_metrics(X,"SPLADE-v3 sparse-lexical")
except Exception as e: print("[bake] SPLADE FAILED",repr(e),flush=True)

print("\n[bake] === RANKING by nearest-neighbor confusability (lower = more separable) ===",flush=True)
for k,v in sorted(results.items(),key=lambda kv:kv[1]):
    print(f"[bake]   {v:+.3f}  {k}",flush=True)
print("[bake] DONE",flush=True)
