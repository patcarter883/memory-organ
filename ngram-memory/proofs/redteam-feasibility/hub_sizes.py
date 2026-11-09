from huggingface_hub import HfApi
api=HfApi()
repos=["Qwen/Qwen3.8-Flash-Next","google/gemma-4-E4B","google/gemma-4-E4B-it","google/gemma-4-E2B-it","Qwen/Qwen3.8-27B","Qwen/Qwen3.6-35B-A3B","google/gemma-4-26B-A4B-it","google/gemma-4-31B-it"]
for r in repos:
    try:
        info=api.model_info(r, files_metadata=True)
        tot=sum((s.size or 0) for s in info.siblings)
        st=sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        sp = info.safetensors.total if info.safetensors else None
        params = info.safetensors.parameters if info.safetensors else None
        print(r, "sha", info.sha[:10], "all_files_GB", round(tot/1e9,3), "safetensors_GB", round(st/1e9,3), "params_total", sp, params, "gated", info.gated)
    except Exception as e:
        print(r, "ERR", type(e).__name__, str(e)[:200])
