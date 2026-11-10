# Token-embedding RMS, uniform-vocab vs token-frequency-weighted (minisgl Python, ~200k tokens), on the 3 models.
import json, struct, ssl, urllib.request, random, glob, collections, concurrent.futures as cf, torch
torch.set_num_threads(1); random.seed(2)
from tokenizers import Tokenizer
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
tok=Tokenizer.from_file(f"{P}/nm/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json")
c=collections.Counter(); n=0
for p in sorted(glob.glob("/home/user/patcarter883/minisglang-rdna4/python/minisgl/**/*.py",recursive=True)):
    ids=tok.encode(open(p,errors="ignore").read()).ids; c.update(ids); n+=len(ids)
    if n>200_000: break
top=[t for t,_ in c.most_common(256)]; cov=sum(c[t] for t in top)/n
ctx=ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
op=urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
def rng(u,a,b): return op.open(urllib.request.Request(u,headers={"Range":f"bytes={a}-{b}"}),timeout=60)
print(f"corpus {n} tok; top-256 tokens cover {cov:.2f}")
for repo,f,d in [("Qwen/Qwen3.8-Flash-Next","model-00130-of-00131.safetensors",2560),("Qwen/Qwen3.8-27B","model-00003-of-00018.safetensors",5120),("Qwen/Qwen3.6-35B-A3B","model-00001-of-00026.safetensors",2048)]:
    r=rng(f"https://huggingface.co/{repo}/resolve/main/{f}",0,7); u=r.geturl(); hl=struct.unpack("<Q",r.read())[0]
    h=json.loads(rng(u,8,8+hl-1).read()); v=h["model.language_model.embed_tokens.weight"]; off=8+hl+v["data_offsets"][0]
    with cf.ThreadPoolExecutor(32) as ex: raw=list(ex.map(lambda i:rng(u,off+i*2*d,off+(i+1)*2*d-1).read(),top))
    E=torch.stack([torch.frombuffer(bytearray(b),dtype=torch.bfloat16).float() for b in raw]); rms=E.pow(2).mean(1).sqrt()
    w=torch.tensor([c[t] for t in top],dtype=torch.float); wr=((w*rms.pow(2)).sum()/w.sum()).sqrt()
    print(f"{repo:26s} top-256 frequent tokens: embed RMS median {rms.median():.5f}, freq-weighted {wr:.5f}")
