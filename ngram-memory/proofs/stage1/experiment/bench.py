import torch, time
from transformers import AutoModelForCausalLM
torch.set_num_threads(3)
M="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
for dt in (torch.bfloat16, torch.float32):
    m=AutoModelForCausalLM.from_pretrained(M,dtype=dt); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    x=torch.randint(0,150000,(1,512))
    emb=m.model.embed_tokens
    for it in range(3):
        t=time.time()
        with torch.no_grad(): m(input_ids=x,use_cache=False)
        tf=time.time()-t
        t=time.time()
        h=emb(x).detach().requires_grad_(True)
        out=m(inputs_embeds=h,labels=x,use_cache=False); out.loss.backward()
        tb=time.time()-t
        print(dt,"fwd",round(tf,2),"fwd+bwd",round(tb,2),flush=True)
    del m
