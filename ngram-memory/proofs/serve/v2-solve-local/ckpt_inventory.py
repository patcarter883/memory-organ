"""Inventory of tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ (the repack the fork calls q4e-mxfp4),
from safetensors HEADERS only (fetched with HTTP range requests into ./dl). Groups bytes by what the
edit solve would have to touch per iteration."""
import json, glob, os, re, sys, collections
D = sys.argv[1]
G = 2**30
groups = collections.defaultdict(int); per_layer_dense = collections.defaultdict(int)
dtypes = collections.defaultdict(set); shapes = {}
for p in sorted(glob.glob(os.path.join(D, "hdr_*.json"))):
    f = os.path.basename(p)[4:-5]
    for k, v in json.load(open(p)).items():
        if k == "__metadata__": continue
        sz = v["data_offsets"][1] - v["data_offsets"][0]
        kk = re.sub(r"layers\.\d+", "layers.N", k); kk = re.sub(r"shard_\d+", "shard_S", kk)
        dtypes[kk].add(v["dtype"]); shapes[kk] = (v["shape"], f)
        m = re.search(r"layers\.(\d+)\.", k)
        if k.startswith("model.visual"): g = "visual (unused)"
        elif k.startswith("mtp"): g = "mtp (unused by solve)"
        elif "ngram_embedding.shard_" in k: g = "PLE rows (int6 codes + f16 scales)"
        elif ".mlp.experts." in k: g = "routed experts MXFP4"
        elif k in ("lm_head.weight",): g = "lm_head bf16"
        elif "embed_tokens" in k: g = "embed_tokens bf16"
        else:
            g = "dense per-layer (attn/GDN/hc/router/shared/PLE-proj/norms)"
            if m: per_layer_dense[int(m.group(1))] += sz
        groups[g] += sz
tot = sum(groups.values())
for g, s in sorted(groups.items(), key=lambda x: -x[1]): print(f"{s/G:9.3f} GiB  {g}")
print(f"{tot/G:9.3f} GiB  TOTAL")
pl = list(per_layer_dense.values())
print(f"dense per layer: min {min(pl)/2**20:.1f} MiB max {max(pl)/2**20:.1f} MiB  (layers {len(pl)})")
print("\nselected tensors (dtype, shape, file):")
for kk in sorted(shapes):
    if kk.startswith(("model.visual", "mtp")): continue
    if re.search(r"(experts|shared_expert|weight_scale_inv|ple\.|hyper|norm|gate\.weight|A_log|dt_bias|conv1d|in_proj_a|in_proj_b|indexer|q_scale|k_scale|v_scale|routing)", kk):
        print(f"  {sorted(dtypes[kk])} {shapes[kk][0]} {shapes[kk][1]}  {kk}")
