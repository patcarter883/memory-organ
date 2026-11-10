import json, re, collections, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repos = {
 "35B": ["Qwen/Qwen3.6-35B-A3B","Qwen/Qwen3.6-35B-A3B-FP8","nvidia/Qwen3.6-35B-A3B-NVFP4","RedHatAI/Qwen3.6-35B-A3B-NVFP4",
         "RedHatAI/Qwen3.6-35B-A3B-FP8","RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic","unsloth/Qwen3.6-35B-A3B-NVFP4","QuantTrio/Qwen3.6-35B-A3B-AWQ",
         "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit","pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4","palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
         "btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit","Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound","amd/Qwen3.6-35B-A3B-w4a16-llmcompressor",
         "lokeshe09/Qwen3.6-35B-A3B-INT8",
         "unsloth/Qwen3.6-35B-A3B-GGUF","bartowski/Qwen_Qwen3.6-35B-A3B-GGUF","ggml-org/Qwen3.6-35B-A3B-GGUF","noctrex/Qwen3.6-35B-A3B-MXFP4_MOE-GGUF"],
 "27B": ["Qwen/Qwen3.8-27B","Qwen/Qwen3.8-27B-FP8","nvidia/Qwen3.8-27B-NVFP4","RadixArk/Qwen3.8-27B-NVFP4","unsloth/Qwen3.8-27B-NVFP4",
         "sakamakismile/Qwen3.8-27B-MTP-NVFP4","cyankiwi/Qwen3.8-27B-AWQ-INT4","amd/Qwen3.8-27B-Quark-AWQ-MXFP4","amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16",
         "RedHatAI/Qwen3.8-27B-INT4","RedHatAI/Qwen3.8-27B-NVFP4","RedHatAI/Qwen3.8-27B-MXFP4","unsloth/Qwen3.8-27B-FP8","btbtyler09/Qwen3.8-27B-GPTQ-4bit",
         "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
         "unsloth/Qwen3.8-27B-GGUF","bartowski/Qwen3.8-27B-GGUF","ggml-org/Qwen3.8-27B-GGUF"],
}
out = {}
for grp, rs in repos.items():
    for r in rs:
        rec = {"group": grp}
        try:
            info = api.model_info(r, files_metadata=True)
            rec["sha"] = info.sha[:10]
            sib = [(s.rfilename, s.size or 0) for s in info.siblings]
            st = [(f, z) for f, z in sib if f.endswith(".safetensors")]
            gg = [(f, z) for f, z in sib if f.endswith(".gguf")]
            rec["safetensors_GB"] = round(sum(z for _, z in st)/1e9, 2)
            # exclude mtp / visual separately if file names hint
            if info.safetensors:
                rec["st_dtypes"] = {k: v for k, v in info.safetensors.parameters.items()}
            if gg:
                by = collections.defaultdict(int)
                for f, z in gg:
                    base = f.split("/")[-1]
                    m = re.search(r'((?:UD-)?(?:IQ\d_[A-Z0-9_]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|Q\d_K|BF16|F16|MXFP4_MOE|MXFP4|F32|TQ\d_\d))', base, re.I)
                    key = (("mmproj-" if "mmproj" in base.lower() else "") + (m.group(1) if m else base))
                    by[key] += z
                rec["gguf_GB"] = {k: round(v/1e9, 2) for k, v in sorted(by.items(), key=lambda kv: kv[1])}
            if st:
                try:
                    p = hf_hub_download(r, "config.json", local_dir=f"cfg/{r.replace('/','__')}")
                    c = json.load(open(p))
                    qc = c.get("quantization_config") or c.get("text_config", {}).get("quantization_config")
                    if qc:
                        rec["quant_method"] = qc.get("quant_method")
                        for k in ("bits","group_size","fmt","format","quant_algo","zero_point","sym","version","kv_cache_scheme"):
                            if k in qc: rec[k] = qc[k] if k!="kv_cache_scheme" else bool(qc[k])
                        cg = qc.get("config_groups")
                        if cg:
                            rec["groups"] = {g: {"w": {kk: v.get("weights", {}).get(kk) for kk in ("num_bits","type","strategy","group_size")} if v.get("weights") else None,
                                                 "a": ({kk: v["input_activations"].get(kk) for kk in ("num_bits","type","strategy","dynamic")} if v.get("input_activations") else None),
                                                 "fmt": v.get("format")} for g, v in cg.items()}
                        ign = qc.get("ignore") or qc.get("modules_to_not_convert") or qc.get("exclude") or []
                        rec["n_ignore"] = len(ign); rec["ignore_sample"] = ign[:6]
                        if qc.get("quant_method") == "quark":
                            gq = qc.get("global_quant_config", {})
                            rec["quark_w"] = gq.get("weight"); rec["quark_a"] = gq.get("input_tensors")
                except Exception as e:
                    rec["cfg_err"] = str(e)[:120]
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {str(e)[:150]}"
        out[r] = rec
        print(r, json.dumps({k: v for k, v in rec.items() if k not in ("groups","ignore_sample","quark_w","quark_a")})[:600], flush=True)
json.dump(out, open("hub_detail.json", "w"), indent=1, default=str)
