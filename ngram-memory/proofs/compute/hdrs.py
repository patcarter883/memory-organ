# Fetch ONLY safetensors JSON headers (HTTP range) for every shard of each repo; no weights downloaded.
import json, sys, os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import get_safetensors_metadata, HfApi
OUT=os.path.dirname(os.path.abspath(__file__))+"/hdr"; os.makedirs(OUT,exist_ok=True)
repos=sys.argv[1:]
def one(r):
    try:
        m=get_safetensors_metadata(r)
        d={}
        for fn,fm in m.files_metadata.items():
            for k,t in fm.tensors.items(): d[k]=dict(file=fn,dtype=t.dtype,shape=list(t.shape))
        json.dump(d,open(f"{OUT}/{r.replace('/','__')}.json","w"))
        return r,len(d),len(m.files_metadata)
    except Exception as e: return r,"ERR",repr(e)[:300]
with ThreadPoolExecutor(4) as ex:
    for x in ex.map(one,repos): print(x,flush=True)
