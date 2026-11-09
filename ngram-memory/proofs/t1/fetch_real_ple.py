# Range-fetch the REAL Qwen3.8-Flash-Next PLE block (key/value proj, norms, conv) + sampled table rows. ~70 MB total.
import json, struct, subprocess, torch
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
r=json.load(open(f"{P}/nm/t1/hdrs/ple_tensors.json"))
url="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
hl={}
def rng(f,a,b): return subprocess.run(["curl","-sS","-L","-r",f"{a}-{b}",url+f],capture_output=True,check=True).stdout
def hlen(f):
    if f not in hl: hl[f]=struct.unpack("<Q",rng(f,0,7))[0]
    return hl[f]
pre="model.language_model.layers.1.ple."
out={}
for n in ["key_proj.weight","value_proj.weight","norm_key.weight","norm_query.weight","norm_conv.weight","conv1d.weight"]:
    v=r[pre+n]; a,b=v["data_offsets"]; raw=rng(v["file"],8+hlen(v["file"])+a,8+hlen(v["file"])+b-1)
    out[n]=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).view(v["shape"]).clone(); print(n,out[n].shape)
ROWS=2500012; RB=160*2
def rows(g0,cnt):
    s,l=divmod(g0,ROWS); assert l+cnt<=ROWS
    v=r[pre+f"ple_embedding.ngram_embedding.shard_{s}.weight"]; a=v["data_offsets"][0]+l*RB
    raw=rng(v["file"],8+hlen(v["file"])+a,8+hlen(v["file"])+a+cnt*RB-1)
    return torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).view(cnt,160).clone()
offs=json.load(open(f"{P}/nm/t1/hdrs/real_buffers.json"))["ngram_heads_offsets"]
for h in [0,7,8,15]:
    out[f"rows_head{h}_start"]=rows(offs[h]+1000,2048)   # 2048 contiguous rows inside head h
out["rows_pad_tail"]=rows(320001446-16,16+90)            # last 16 real rows + 90 padding rows
torch.save(out,f"{P}/nm/t1/real/real_ple_block.pt"); print("saved")
