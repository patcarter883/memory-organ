import torch, math
torch.manual_seed(0)
d=64; K=4; Ntot=8; M=8
delta=torch.randn(d)
pen={2:15.,3:1.}
def g(n,c): return pen[n]*(1+math.log1p(c))*1e-3
def run(case):
    # columns: list of (order, bucket_id); each paraphrase hits K buckets per order
    cols={}; rows=[]
    for m in range(M):
        hits=[]
        for n in (2,3):
            key = case[n](m)
            for k in range(K): hits.append((n,key,k))
        rows.append(hits)
    allc=sorted({h for r in rows for h in r}); idx={c:i for i,c in enumerate(allc)}
    A=torch.zeros(M,len(allc))
    for m,r in enumerate(rows):
        for h in r: A[m,idx[h]]+=1/Ntot
    G=torch.diag(torch.tensor([g(c[0], 1e5 if c[0]==2 else 1) for c in allc]))
    dE=torch.linalg.solve(A.T@A+G, A.sum(0,keepdim=True).T@delta[None])
    train=float((A@dE)[0].norm()/delta.norm())
    # held-out paraphrase with a NEW preceding context token
    new = [(n,case[n](999),k) for n in (2,3) for k in range(K)]
    e=sum((dE[idx[h]]/Ntot if h in idx else torch.zeros(d)) for h in new)
    return train, float(e.norm()/delta.norm())
cases={
 "subject>=3 tokens (anchor 2/3-grams subject-internal)": {2:lambda m:"s", 3:lambda m:"s"},
 "subject=2 tokens (3-gram includes preceding ctx)":       {2:lambda m:"s", 3:lambda m:f"ctx{m}"},
 "subject=1 token (both n-grams include ctx)":             {2:lambda m:f"c2_{m}", 3:lambda m:f"c3_{m}"},
}
for name,c in cases.items():
    tr,ho=run(c); print(f"{name:55s} train-realised={tr:.3f}  heldout-new-context-realised={ho:.3f}")

# collision exposure: fraction of ordinary tokens whose 8 lookups hit >=1 edited row (rows/edit/order ~= K)
for E in (1e3,1e4,1e5):
    R=K*E; p=1-(1-R/2e6)**8
    print(f"edits={int(E):>6}: P(unrelated token gets a stray ~(1/8)*1.9*delta*) ~ {p:.3%}")
# memory of dense zero tables
for dm in (896,5120):
    print(f"d_model={dm}: 2 orders x 2M x d fp32 pinned = {2*2e6*dm*4/1e9:.1f} GB")
