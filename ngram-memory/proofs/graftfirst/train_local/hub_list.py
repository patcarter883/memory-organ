import json, sys
from huggingface_hub import HfApi
api = HfApi()
out = {}
for q in ["Qwen3.6-35B-A3B", "Qwen3.8-27B"]:
    ms = list(api.list_models(search=q, sort="downloads", limit=400, expand=["downloads","tags","library_name","createdAt"]))
    out[q] = [dict(id=m.id, downloads=m.downloads, tags=[t for t in (m.tags or []) if t in ("gguf","awq","gptq","fp8","mlx","safetensors","compressed-tensors","8-bit","4-bit","bitsandbytes","exl2","exl3","quark","modelopt")]) for m in ms]
    print(q, len(ms))
json.dump(out, open("hub_list.json","w"), indent=1)
