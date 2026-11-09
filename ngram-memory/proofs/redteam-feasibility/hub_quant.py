from huggingface_hub import HfApi
import re, collections
api=HfApi()
for r in ["RadixArk/Qwen3.8-Flash-Next-NVFP4","tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"]:
    try:
        info=api.model_info(r, files_metadata=True)
        st=[s for s in info.siblings if s.rfilename.endswith(".safetensors")]
        print(r, info.sha[:10], "nfiles", len(st), "safetensors_GB", round(sum(s.size or 0 for s in st)/1e9,2), "dtypes", info.safetensors.parameters if info.safetensors else None)
    except Exception as e:
        print(r,"ERR",type(e).__name__,str(e)[:150])
# Flash-Next: are table shards in table-only files?
import json
idx=json.load(open("models/Qwen/Qwen3.8-Flash-Next/model.safetensors.index.json"))["weight_map"]
byf=collections.defaultdict(lambda:[0,0])
for k,f in idx.items():
    byf[f][0 if "ngram_embedding.shard_" in k else 1]+=1
mixed=[f for f,(a,b) in byf.items() if a and b]; tab=[f for f,(a,b) in byf.items() if a and not b]
print("table-only files",len(tab),"mixed files",len(mixed), sorted(mixed)[:5])
info=api.model_info("Qwen/Qwen3.8-Flash-Next", files_metadata=True)
sz={s.rfilename:s.size for s in info.siblings}
print("bytes in files with ANY table shard GB", round(sum(sz[f] for f,(a,b) in byf.items() if a)/1e9,2), " non-table files GB", round(sum(sz[f] for f,(a,b) in byf.items() if not a)/1e9,2))
