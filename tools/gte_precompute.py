import os, pickle, numpy as np
b = pickle.load(open("/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl","rb"))
kept = b["kept"]
from pylate import models
m = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
subs = [r.subject for r in kept]
tids = [tuple(r.subject_tids) for r in kept]
embs = m.encode(subs, is_query=False, show_progress_bar=False)   # list [T,128]
d = {}
for t, e in zip(tids, embs):
    e = np.asarray(e, dtype=np.float32)
    d[t] = e.mean(0)                     # pooled 128-d key (single-vector store mode)
pickle.dump(d, open("/engine/.probe_cache/gte_keys.pkl","wb"))
print(f"[gte] saved {len(d)} pooled GTE keys, dim {next(iter(d.values())).shape}", flush=True)
