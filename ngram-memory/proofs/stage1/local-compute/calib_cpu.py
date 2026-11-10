"""CPU calibration of HF end-to-end efficiency for ROWS-ONLY training (frozen weights, grad only to a
sparse delta-row table injected at decoder layer 1 = Flash-Next's PLE site; per-layer checkpointing;
no weight grads) on Qwen2.5-0.5B, vs a bf16 GEMM roofline measured in the same process.
Reports achieved model FLOP/s = (6*P_mm + attn) * tok/s and the ratio to the GEMM rate.
Usage: python -I calib_cpu.py <threads>"""
import sys, time, json, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
T=int(sys.argv[1]); torch.set_num_threads(T); torch.manual_seed(0)
# roofline: bf16 GEMM at layer-like shapes
def gemm(M,K,N,it=5):
    a=torch.randn(M,K,dtype=torch.bfloat16); b=torch.randn(K,N,dtype=torch.bfloat16); a@b
    t=time.perf_counter()
    for _ in range(it): a@b
    return 2*M*K*N*it/(time.perf_counter()-t)/1e9
roof={f"{M}x{K}x{N}":round(gemm(M,K,N),1) for M,K,N in [(2048,896,4864),(2048,4864,896),(2048,896,896),(512,896,4864)]}
P="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
m=AutoModelForCausalLM.from_pretrained(P,dtype=torch.bfloat16).eval()
for p in m.parameters(): p.requires_grad_(False)
cfg=m.config; d=cfg.hidden_size; L=cfg.num_hidden_layers; V=cfg.vocab_size
H=8; R=100_003; w=d//H
rows_store=torch.zeros(H*R,w)                               # sparse rows; gather -> leaf -> scatter
mult=torch.tensor([23703573157769,20109073645365])
def rowids(ids):
    prev=torch.cat([torch.zeros_like(ids[:,:1]),ids[:,:-1]],1); mix=(ids*mult[0])^(prev*mult[1])
    return torch.stack([h*R+(mix+h*7919)%R for h in range(H)],-1)
lay=m.model.layers; rot=m.model.rotary_emb
def run(ids,train):
    B,S=ids.shape; h=m.model.embed_tokens(ids); pos=torch.arange(S)[None].expand(B,S); pe=rot(h,pos)
    mask=create_causal_mask(config=cfg,inputs_embeds=h,attention_mask=torch.ones(B,S,dtype=torch.long),past_key_values=None,position_ids=pos)
    f=lambda i,x: lay[i](x,attention_mask=mask,position_ids=pos,position_embeddings=pe)
    h=f(0,h); h=h[0] if isinstance(h,tuple) else h
    leaf=None
    if train:
        rid=rowids(ids); uniq,inv=torch.unique(rid,return_inverse=True)
        leaf=rows_store[uniq].clone().requires_grad_(True)
        h=h+leaf[inv].reshape(B,S,d).to(h.dtype)
    for i in range(1,L):
        h=checkpoint(f,i,h,use_reentrant=False) if train else f(i,h)
        h=h[0] if isinstance(h,tuple) else h
    h=m.model.norm(h); tgt=ids[:,1:].reshape(-1); hs=h[:,:-1].reshape(-1,d); loss=0.
    for c in range(0,hs.shape[0],1024):
        lg=m.lm_head(hs[c:c+1024]).float(); loss=loss+nn.functional.cross_entropy(lg,tgt[c:c+1024],reduction="sum")
    return loss/tgt.numel(), leaf
P_mm=sum(p.numel() for n,p in m.named_parameters() if "layers" in n and p.dim()==2)+V*d  # + tied lm_head
P_l0=sum(p.numel() for n,p in m.named_parameters() if "layers.0." in n and p.dim()==2)
nh,hd=cfg.num_attention_heads,d//cfg.num_attention_heads
res={"threads":T,"gemm_GFLOPs":roof,"P_mm_G":round(P_mm/1e9,4)}
for S,B in [(1024,4),(2048,2)]:
    ids=torch.randint(0,V,(B,S))
    attn_fwd=2*hd*S*nh*L                          # causal QK+AV per token, all layers
    f_train=2*P_mm+2*(P_mm-P_l0)*2+attn_fwd*(1+1+2.5)   # fwd + recompute(L1..) + dX-only bwd(L1..)
    f_eval=2*P_mm+attn_fwd
    for train in (True,False):
        ts=[]
        for it in range(3):
            t=time.perf_counter()
            if train: l,leaf=run(ids,True); l.backward(); g=leaf.grad; rows_store.index_add_(0,torch.unique(rowids(ids)),-1e-3*g)
            else:
                with torch.no_grad(): run(ids,False)
            ts.append(time.perf_counter()-t)
        s=min(ts[1:]); tps=B*S/s; fl=(f_train if train else f_eval)*tps/1e9
        r=dict(S=S,B=B,train=train,s=round(s,2),tok_s=round(tps,1),model_GFLOPs=round(fl,1),
               frac_of_best_gemm=round(fl/max(roof.values()),3))
        print(json.dumps(r),flush=True); res.setdefault("runs",[]).append(r)
print(json.dumps(res))
