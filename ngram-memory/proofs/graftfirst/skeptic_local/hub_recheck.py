# Re-check repo sizes + per-dtype tensor bytes for the recommended checkpoints via HF API (headers only, no weights).
import json, sys
from huggingface_hub import HfApi, get_safetensors_metadata
api = HfApi()
repos = ["cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", "pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4",
         "sakamakismile/Qwen3.8-27B-MTP-NVFP4", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B"]
out = {}
for r in repos:
    try:
        info = api.model_info(r, files_metadata=True)
        tot = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        md = get_safetensors_metadata(r)
        cat = {}
        for fn, fm in md.files_metadata.items():
            for name, ti in fm.tensors.items():
                import math
                n = math.prod(ti.shape)
                bpe = {"BF16":2,"F16":2,"F32":4,"I32":4,"U8":1,"I8":1,"F8_E4M3":1,"I64":8,"U32":4}.get(ti.dtype, 0)
                k = ("visual" if "visual" in name else "mtp" if "mtp" in name else "embed" if "embed_tokens" in name else
                     "lm_head" if "lm_head" in name else "experts" if (".experts." in name and "shared" not in name) else "other")
                cat.setdefault(k, [0, {}]); cat[k][0] += n * bpe; cat[k][1][ti.dtype] = cat[k][1].get(ti.dtype, 0) + n*bpe
        out[r] = dict(safetensors_GB=round(tot/1e9, 2), by_part_GB={k: round(v[0]/1e9, 3) for k, v in cat.items()},
                      dtypes={k: {d: round(b/1e9, 3) for d, b in v[1].items()} for k, v in cat.items()})
    except Exception as e:
        out[r] = repr(e)[:300]
    print(r, json.dumps(out[r]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
