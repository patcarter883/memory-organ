# Token-embedding RMS on Flash-Next vs the two graft hosts (sets the natural scale a graft's layer-1 output competes with).
# 256 random vocab rows per model via HTTP range; compares with Flash-Next PLE RMS(v)=0.0058 from P/t1/real_stats.out.
import json, struct, ssl, urllib.request, random, concurrent.futures as cf, torch
torch.set_num_threads(1); random.seed(1)
ctx=ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
op=urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
def rng(u,a,b): return op.open(urllib.request.Request(u,headers={"Range":f"bytes={a}-{b}"}),timeout=60)
for repo,f,d in [("Qwen/Qwen3.8-Flash-Next","model-00130-of-00131.safetensors",2560),("Qwen/Qwen3.8-27B","model-00003-of-00018.safetensors",5120),("Qwen/Qwen3.6-35B-A3B","model-00001-of-00026.safetensors",2048)]:
    r=rng(f"https://huggingface.co/{repo}/resolve/main/{f}",0,7); u=r.geturl(); hl=struct.unpack("<Q",r.read())[0]
    h=json.loads(rng(u,8,8+hl-1).read()); v=h["model.language_model.embed_tokens.weight"]; assert v["shape"][1]==d and v["dtype"]=="BF16"
    off=8+hl+v["data_offsets"][0]; ids=random.sample(range(248000),256)
    with cf.ThreadPoolExecutor(32) as ex: raw=list(ex.map(lambda i:rng(u,off+i*2*d,off+(i+1)*2*d-1).read(),ids))
    E=torch.stack([torch.frombuffer(bytearray(b),dtype=torch.bfloat16).float() for b in raw])
    rms=E.pow(2).mean(1).sqrt()
    print(f"{repo:26s} d={d} embed RMS median {rms.median():.5f}  L2 median {E.norm(dim=1).median():.4f}")
