# Statistics of the REAL PLE block weights + sampled table rows.
import torch, json
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
w=torch.load(f"{P}/nm/t1/real/real_ple_block.pt")
f=lambda t:t.float()
for k in ["norm_key.weight","norm_query.weight","norm_conv.weight"]:
    t=f(w[k]).view(4,2560); print(k,"(1+w) mean per stream",(1+t).mean(1).tolist(),"absmax",t.abs().max().item())
c=f(w["conv1d.weight"]).view(4,2560,4); print("conv taps [t-9,t-6,t-3,t] mean|w| per stream:",c.abs().mean(1).tolist())
print("conv zero frac",(c==0).float().mean().item())
Wv=f(w["value_proj.weight"]); Wk=f(w["key_proj.weight"])
S=torch.linalg.svdvals(Wv); print("W_v svals max/min/cond",S[0].item(),S[-1].item(),(S[0]/S[-1]).item(), "eff rank(>1e-3*max)",(S>1e-3*S[0]).sum().item())
for h in [0,8]:
    Sh=torch.linalg.svdvals(Wv[:,h*160:(h+1)*160]); print(f"W_v head{h} slice svals max/min",Sh[0].item(),Sh[-1].item())
print("||W_v[:,head]||_F per head",[round(Wv[:,h*160:(h+1)*160].norm().item(),2) for h in range(16)])
print("||W_k[:,head]||_F per head",[round(Wk[:,h*160:(h+1)*160].norm().item(),2) for h in range(16)])
for k in [k for k in w if k.startswith("rows_")]:
    R=f(w[k]); n=R.norm(dim=1)
    print(k,"row L2 mean/median/max",n.mean().item(),n.median().item(),n.max().item(),"zero rows",(n==0).sum().item(),"/",len(n))
# typical e_t magnitude and value magnitude with 16 random real rows
rows=[f(w[f"rows_head{h}_start"]) for h in [0,7,8,15]]
g=torch.Generator().manual_seed(0)
E=torch.cat([rows[(h//4)][torch.randint(0,2048,(256,),generator=g)] for h in range(16)],1)  # 256 synthetic e_t
v=E@Wv.T; print("synthetic |e|",E.norm(dim=1).mean().item()," |v|=|W_v e|",v.norm(dim=1).mean().item(), "rms(v)",(v.pow(2).mean(1).sqrt()).mean().item())
