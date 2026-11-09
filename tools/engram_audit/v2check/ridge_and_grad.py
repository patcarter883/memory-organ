import torch, math
torch.manual_seed(0)
r=512; K=4; Ntot=8; M=8
delta=torch.randn(r)
def solve(c2, c3, same_rows=True, pen={2:15.,3:1.}):
    # same_rows: all M paraphrases share the anchor n-grams (subject >= 3 tokens)
    A2=torch.zeros(M,K); A3=torch.zeros(M,K)
    A2[:]=1/Ntot; A3[:]=1/Ntot
    g2=pen[2]*(1+math.log1p(c2))*1e-3; g3=pen[3]*(1+math.log1p(c3))*1e-3
    A=torch.cat([A2,A3],1); G=torch.diag(torch.tensor([g2]*K+[g3]*K))
    dE=torch.linalg.solve(A.T@A+G, A.sum(0,keepdim=True).T@delta[None])
    e2=(A2[0]@dE[:K]); e3=(A3[0]@dE[K:])
    real=e2+e3
    return dict(c2=c2,c3=c3,realised=float(real.norm()/delta.norm()),
                share3=float(e3.norm()/(e2.norm()+e3.norm())),
                row_norm_ratio=float(dE[K:].norm(dim=1).mean()/delta.norm()))
for c2,c3 in [(1e5,1),(1e5,1e3),(1e5,1e5),(1,1)]:
    print(solve(c2,c3))

# Stage-1 gradient with v_proj zero-init (as v2 ships), tables random 0.02
import torch.nn as nn
d=64
q=nn.Linear(d,r,bias=False); k=nn.Linear(r,r,bias=False); v=nn.Linear(r,d,bias=False); nn.init.zeros_(v.weight)
qn=nn.RMSNorm(r); kn=nn.RMSNorm(r)
h=torch.randn(1,5,d); e=torch.randn(1,5,r)*0.02
dl=torch.zeros(r,requires_grad=True)
ee=e.clone(); ee[0,3]=ee[0,3]+dl
a=torch.sigmoid((qn(q(h))*kn(k(ee))).sum(-1,keepdim=True)*r**-0.5)
out=h+a*v(ee)
out.pow(2).sum().backward()
print("grad norm on delta with v_proj=0:", float(dl.grad.norm()))
# trust region scale: ||e_q|| for mean of 8 random 0.02 rows of width 512
print("||e_q|| ~", float((torch.randn(8,r)*0.02).mean(0).norm()), "-> clamp 0.4x, floor 0.1")
