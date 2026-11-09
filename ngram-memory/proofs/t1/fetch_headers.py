# Fetch only safetensors JSON headers (range requests) for files holding layers.1.ple.* tensors.
import json, struct, subprocess, sys, os
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
out=f"{P}/nm/t1/hdrs"
idx=json.load(open(f"{P}/models/Qwen/Qwen3.8-Flash-Next/model.safetensors.index.json"))["weight_map"]
files=sorted({f for k,f in idx.items() if ".ple." in k})
url="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
res={}
for f in files:
    def rng(a,b):
        return subprocess.run(["curl","-sS","-L","-r",f"{a}-{b}",url+f],capture_output=True,check=True).stdout
    n=struct.unpack("<Q",rng(0,7))[0]
    h=json.loads(rng(8,8+n-1))
    for k,v in h.items():
        if ".ple." in k: res[k]=dict(file=f,**v)
json.dump(res,open(f"{out}/ple_tensors.json","w"),indent=1)
print(len(res))
