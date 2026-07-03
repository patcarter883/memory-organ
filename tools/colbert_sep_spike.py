import os, pickle, numpy as np, torch
# --- 1. the 137 subjects from the probe cache (realedit must be importable) ---
b = pickle.load(open("/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl","rb"))
kept = b["kept"]; subs = [r.subject for r in kept]; tids = [list(r.subject_tids) for r in kept]
N = len(subs); print(f"[spike] {N} subjects; multi-token: {sum(len(t)>1 for t in tids)}", flush=True)

def self_retrieval(score_fn, tag):
    S = score_fn()                                   # [N,N] score matrix (query i vs doc j)
    top1 = int((S.argmax(1) == np.arange(N)).sum())
    # margin: self-score minus best-other, averaged
    diag = S[np.arange(N), np.arange(N)]
    Sm = S.copy(); Sm[np.arange(N), np.arange(N)] = -1e9
    margin = float((diag - Sm.max(1)).mean())
    print(f"[spike] {tag:28s} self-retrieval top-1 = {top1}/{N} ({top1/N:.3f}) | mean margin {margin:+.3f}", flush=True)

# --- 2. GTE-ModernColBERT: MaxSim (multi-vector) AND mean-pool (single-vector) ---
try:
    from pylate import models
    m = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
    docs = m.encode(subs, is_query=False, show_progress_bar=False)   # list of [T,128]
    qs   = m.encode(subs, is_query=True,  show_progress_bar=False)
    docs = [np.asarray(d, dtype=np.float32) for d in docs]; qs=[np.asarray(q,dtype=np.float32) for q in qs]
    def maxsim():
        S=np.zeros((N,N),np.float32)
        for i in range(N):
            for j in range(N):
                S[i,j]=(qs[i]@docs[j].T).max(1).sum()   # sum over query toks of max over doc toks
        return S
    self_retrieval(maxsim, "GTE-ColBERT MaxSim (multi)")
    # mean-pool single vector (L2-normalized) cosine
    dv=np.stack([d.mean(0)/ (np.linalg.norm(d.mean(0))+1e-9) for d in docs])
    qv=np.stack([q.mean(0)/ (np.linalg.norm(q.mean(0))+1e-9) for q in qs])
    self_retrieval(lambda: qv@dv.T, "GTE-ColBERT mean-pool (single)")
except Exception as e:
    print("[spike] GTE-ColBERT FAILED:", repr(e), flush=True)

# --- 3. baseline: served-model (Qwen3.5-4B) INPUT EMBEDDINGS, mean-pooled (current key encoder) ---
try:
    from transformers import AutoModelForCausalLM
    mm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.float32, low_cpu_mem_usage=True)
    E = mm.get_input_embeddings().weight.detach()    # [vocab, 2560]
    def pool(t): 
        v = E[torch.tensor(t)].mean(0); return (v/ (v.norm()+1e-9)).numpy()
    X = np.stack([pool(t) for t in tids])
    self_retrieval(lambda: X@X.T, "Qwen input-embed mean-pool")
except Exception as e:
    print("[spike] Qwen-embed baseline FAILED:", repr(e), flush=True)
print("[spike] DONE", flush=True)
