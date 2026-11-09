# Fetch the real int64 hashing buffers (24 + 128 + 128 bytes) and value-check against code-derived values.
import json, struct, subprocess, sys
sys.path.insert(0,"")
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
r=json.load(open(f"{P}/nm/t1/hdrs/ple_tensors.json"))
url="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
def rng(f,a,b): return subprocess.run(["curl","-sS","-L","-r",f"{a}-{b}",url+f],capture_output=True,check=True).stdout
out={}
for name in ["layer_multipliers","ngram_heads_offsets","ngram_heads_vocab_sizes"]:
    k=f"model.language_model.layers.1.ple.ple_embedding.{name}"; v=r[k]
    n=struct.unpack("<Q",rng(v['file'],0,7))[0]
    a,b=v['data_offsets']; raw=rng(v['file'],8+n+a,8+n+b-1)
    out[name]=list(struct.unpack(f"<{(b-a)//8}q",raw))
    print(name,out[name])
json.dump(out,open(f"{P}/nm/t1/hdrs/real_buffers.json","w"))
from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers,_find_nth_prime_after
cfg=json.load(open(f"{P}/models/Qwen/Qwen3.8-Flash-Next/config.json"))["text_config"]
m=_build_layer_multipliers(cfg["vocab_size"],3,0,1234).tolist()
sizes=[_find_nth_prime_after(cfg["ngram_vocab_size_base"]-1,h+1) for h in range(16)]
offs=[sum(sizes[:h]) for h in range(16)]
print("code multipliers (ple_layer_index=0, seed=1234):",m, "MATCH" if m==out["layer_multipliers"] else "MISMATCH")
print("code sizes match:",sizes==out["ngram_heads_vocab_sizes"]," offsets match:",offs==out["ngram_heads_offsets"])
print("total",sum(sizes),"padded",-(-sum(sizes)//128)*128)
