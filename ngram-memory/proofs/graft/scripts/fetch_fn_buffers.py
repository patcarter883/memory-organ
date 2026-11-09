# Range-read ONLY the safetensors header + the three tiny hash buffers of Flash-Next's native PLE (no full shard download).
import json, struct, sys, numpy as np
from huggingface_hub import get_hf_file_metadata, hf_hub_url, get_session
repo='Qwen/Qwen3.8-Flash-Next'; out={}
s=get_session()
for shard, names in [('model-00005-of-00131.safetensors',['layer_multipliers']),('model-00037-of-00131.safetensors',['ngram_heads_offsets','ngram_heads_vocab_sizes'])]:
    url=hf_hub_url(repo, shard)
    r=s.get(url, headers={'Range':'bytes=0-7'}, follow_redirects=True); n=struct.unpack('<Q', r.content[:8])[0]
    r=s.get(url, headers={'Range':f'bytes=8-{8+n-1}'}, follow_redirects=True); hdr=json.loads(r.content)
    ple={k:v for k,v in hdr.items() if 'layers.1.ple' in k}
    for k,v in ple.items(): print(shard, k, v['dtype'], v['shape'])
    for nm in names:
        k='model.language_model.layers.1.ple.ple_embedding.'+nm; v=hdr[k]; a,b=v['data_offsets']
        r=s.get(url, headers={'Range':f'bytes={8+n+a}-{8+n+b-1}'}, follow_redirects=True)
        arr=np.frombuffer(r.content, dtype={'I64':np.int64}[v['dtype']]); out[nm]=arr.tolist()
json.dump(out, open(sys.argv[1],'w')); print(json.dumps(out))
