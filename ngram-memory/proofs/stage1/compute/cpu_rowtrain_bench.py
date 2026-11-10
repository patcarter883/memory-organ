"""Calibration: tokens/s for TABLE-ROWS-ONLY training (no weight grads) on frozen Qwen2.5-0.5B, CPU.
A hashed 2-gram delta table (8 heads x rows x d/8) is added to the residual entering decoder layer 1
(Flash-Next's PLE site). Per-layer activation checkpointing (non-reentrant). Measures:
 (1) fwd+recompute+bwd tok/s, (2) forward-only (no_grad) tok/s = held-out eval rate, at S in {512,1024,2048}.
Usage: python -I cpu_rowtrain_bench.py <threads>"""
import sys, time, json, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
torch.set_num_threads(int(sys.argv[1])); torch.manual_seed(0)
P="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
m=AutoModelForCausalLM.from_pretrained(P,dtype=torch.bfloat16).eval()
for p in m.parameters(): p.requires_grad_(False)
cfg=m.config; d=cfg.hidden_size; L=cfg.num_hidden_layers; V=cfg.vocab_size
H=8; R=200_003; w=d//H
delta=nn.Parameter(torch.zeros(H*R, w, dtype=torch.float32))   # dense here; real impl is sparse rows
mult=torch.tensor([23703573157769, 20109073645365])
def rows(ids):
    prev=torch.cat([torch.zeros_like(ids[:,:1]), ids[:,:-1]],1)
    mix=(ids*mult[0])^(prev*mult[1])
    return torch.stack([h*R + (mix+h*7919)%R for h in range(H)],-1)       # [B,S,H]
lay=m.model.layers; rot=m.model.rotary_emb
def run(ids, train):
    B,S=ids.shape
    h=m.model.embed_tokens(ids)
    pos=torch.arange(S)[None].expand(B,S); pe=rot(h,pos)
    mask=None
    from transformers.masking_utils import create_causal_mask
    mask=create_causal_mask(config=cfg,inputs_embeds=h,attention_mask=torch.ones(B,S,dtype=torch.long),past_key_values=None,position_ids=pos)
    def f(i,x): return lay[i](x,attention_mask=mask,position_ids=pos,position_embeddings=pe)
    h=f(0,h)  # layer 0 below the site: no grad needed
    if train:
        e=delta[rows(ids)].reshape(B,S,d).to(h.dtype); h=h+e
    for i in range(1,L):
        h=checkpoint(f,i,h,use_reentrant=False) if train else f(i,h)
        if isinstance(h,tuple): h=h[0]
    h=m.model.norm(h)
    # chunked CE on 2-D rows (avoid [B,S,V] fp32 blowup)
    tgt=ids[:,1:].reshape(-1); hs=h[:,:-1].reshape(-1,d); loss=0.
    for c in range(0,hs.shape[0],1024):
        lg=m.lm_head(hs[c:c+1024]).float(); loss=loss+nn.functional.cross_entropy(lg,tgt[c:c+1024],reduction="sum")
    return loss/tgt.numel()
res=[]
for S,B in [(512,4),(1024,2),(2048,1)]:
    ids=torch.randint(0,V,(B,S))
    for train in (True,False):
        ts=[]
        for it in range(3):
            t=time.perf_counter()
            if train:
                delta.grad=None; l=run(ids,True); l.backward()
            else:
                with torch.no_grad(): l=run(ids,False)
            ts.append(time.perf_counter()-t)
        best=min(ts[1:]); r=dict(S=S,B=B,train=train,s=round(best,3),tok_s=round(B*S/best,1))
        if train: r["delta_grad_nonzero_rows"]=int((delta.grad.abs().sum(1)>0).sum())
        print(json.dumps(r),flush=True); res.append(r)
act=sum(p.numel() for n,p in m.named_parameters() if "embed" not in n)+m.lm_head.weight.numel()*0  # tied
print(json.dumps({"active_params_excl_embed_G":round(act/1e9,4),"lm_head_tied_G":round(m.lm_head.weight.numel()/1e9,4),"threads":int(sys.argv[1])}))
