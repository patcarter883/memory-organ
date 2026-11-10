# Direct term vs conv branch magnitude in the REAL Flash-Next PLE block, on sequences of real (random-bucket) rows.
# Gate is unknown without layer-0 states; the conv branch is per-(token,stream) RMS-normalised, so sigmoid(g) cancels
# there except via eps. We sweep a constant gate s in {0.1,0.5,0.9}.
import torch, torch.nn.functional as F
torch.set_num_threads(1)
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
w=torch.load(f"{P}/nm/t1/real/real_ple_block.pt"); f=lambda t:t.float()
Wv=f(w["value_proj.weight"]); nc=f(w["norm_conv.weight"]).view(4,2560); C=f(w["conv1d.weight"]).view(4,2560,4)
rows=[f(w[f"rows_head{h}_start"]) for h in [0,7,8,15]]
g=torch.Generator().manual_seed(0); T=256
E=torch.cat([rows[h//4][torch.randint(0,2048,(T,),generator=g)] for h in range(16)],1)
v=E@Wv.T                                  # (T,2560)
print("rms(v) mean", v.pow(2).mean(1).sqrt().mean().item())
for s in [0.1,0.5,0.9]:
    gv=(s*v).unsqueeze(1).expand(T,4,2560)                       # (T,4,d)
    nrm=gv*torch.rsqrt(gv.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nc)
    x=nrm.permute(1,2,0)                                          # (4,d,T)
    xp=F.pad(x,(9,0))
    full=sum(C[:,:,k:k+1]*xp[:,:,3*k:3*k+T] for k in range(4))    # taps k=0..3 -> t-9,t-6,t-3,t
    cur=C[:,:,3:4]*xp[:,:,9:9+T]
    conv_full=F.silu(full); conv_cur=F.silu(cur)
    d=gv.permute(1,2,0)
    r=lambda t:t.pow(2).mean((1,2)).sqrt()
    print(f"gate={s}: direct rms/stream",[round(a,5) for a in r(d).tolist()],
          " conv rms/stream",[round(a,5) for a in r(conv_full).tolist()],
          " conv/direct",[round(a,1) for a in (r(conv_full)/r(d)).tolist()],
          " |conv_full-conv_cur|/|conv_full|",[round(a,2) for a in (r(conv_full-conv_cur)/r(conv_full)).tolist()])
    # silu linearity: |silu(x) - x/2| / |silu(x)|
    print("   silu nonlinearity share",[round(a,3) for a in (r(conv_full-full/2)/r(conv_full)).tolist()])
