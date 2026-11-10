"""Token count + distinct n-gram/row census of the fork's python/ tree at df29117, 3 tokenizers.
Usage: python -I tokcount.py"""
import os, json, sys, hashlib
from collections import Counter
from tokenizers import Tokenizer
S="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm"
ROOT="/home/user/patcarter883/minisglang-rdna4/python"
TOK={"flashnext":f"{S}/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json",
     "qwen2.5":"/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json",
     "gemma4":f"{S}/t2-gemma4-ple/hf/tokenizer.json"}
files=[]
for dp,dn,fn in os.walk(ROOT):
    dn[:]=[d for d in dn if d!="__pycache__"]
    for f in sorted(fn):
        p=os.path.join(dp,f)
        if f.endswith((".pyc",".gz",".so")): continue
        try: files.append((p,open(p,encoding="utf-8").read()))
        except UnicodeDecodeError: pass
out={"root":ROOT,"files":len(files),"py_files":sum(p.endswith(".py") for p,_ in files),
     "bytes":sum(len(t.encode()) for _,t in files)}
for name,path in TOK.items():
    tk=Tokenizer.from_file(path)
    enc=tk.encode_batch([t for _,t in files],add_special_tokens=False)
    lens=[len(e.ids) for e in enc]; n=sum(lens)
    d1=set(); d2=set(); d3=set(); c2=Counter(); c3=Counter()
    for e in enc:
        ids=e.ids; d1.update(ids)
        for i in range(len(ids)):
            a=ids[i-1] if i>=1 else -1; b=ids[i-2] if i>=2 else -1   # -1 = eos/segment start
            c2[(a,ids[i])]+=1; c3[(b,a,ids[i])]+=1
    r={"tokens":n,"bytes_per_token":round(out["bytes"]/n,3),"max_file_tokens":max(lens),
       "files_gt_4096":sum(l>4096 for l in lens),"files_gt_16384":sum(l>16384 for l in lens),
       "distinct_tokens":len(d1),"distinct_2gram":len(c2),"distinct_3gram":len(c3),
       "distinct_2gram_ge2":sum(v>=2 for v in c2.values()),"distinct_3gram_ge2":sum(v>=2 for v in c3.values())}
    if name!="gemma4":  # Flash-Next KeySpec: 8 heads per order, 16 heads; collisions negligible at 2% fill
        r["flashnext_rows_upper"]=8*(r["distinct_2gram"]+r["distinct_3gram"])
    out[name]=r
print(json.dumps(out,indent=1))
