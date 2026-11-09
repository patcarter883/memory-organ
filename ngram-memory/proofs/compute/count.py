# Exact param/byte accounting from safetensors headers (hdr/*.json) + real configs.
import json, math, re, sys
M="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/models"
B={"BF16":2,"F16":2,"F32":4,"I64":8,"F8_E4M3":1,"U8":1}
ROWS_FN=320001536; ROWS_FN_REAL=320001446  # padded / sum of 16 primes (from hdr + t1/hdrs)
targets=[("T1 Qwen3.8-Flash-Next","Qwen/Qwen3.8-Flash-Next","Qwen/Qwen3.8-Flash-Next"),
 ("T2 gemma-4-E4B-it","google/gemma-4-E4B-it","google/gemma-4-E4B-it"),
 ("(ref) gemma-4-E2B-it","google/gemma-4-E2B-it",None),
 ("T3 Qwen3.8-27B","Qwen/Qwen3.8-27B","Qwen/Qwen3.8-27B"),
 ("T3 Qwen3.6-35B-A3B","Qwen/Qwen3.6-35B-A3B","Qwen/Qwen3.6-35B-A3B"),
 ("T3 gemma-4-26B-A4B-it","google/gemma-4-26B-A4B-it","google/gemma-4-26B-A4B-it"),
 ("T3 gemma-4-31B-it","google/gemma-4-31B-it","google/gemma-4-31B-it")]
def cat(k):
    if re.search(r'visual|vision|audio',k): return "mm"
    if k.startswith("mtp."): return "mtp"
    if "ngram_embedding" in k or "embed_tokens_per_layer" in k: return "table"
    if ".ple." in k or "per_layer" in k: return "table_aux"
    if k.endswith("embed_tokens.weight"): return "embed"
    if k.startswith("lm_head"): return "lm_head"
    if ".experts." in k: return "routed"
    if "shared_expert" in k: return "shared"
    return "dense"
res={}
for name,repo,cfgrepo in targets:
    h=json.load(open(f"hdr/{repo.replace('/','__')}.json"))
    cfgf=f"{M}/{cfgrepo or 'google/gemma-4-E4B-it'}/config.json"
    c=json.load(open(cfgf)); t=c.get("text_config",c)
    s={}; b={}
    for k,v in h.items():
        n=math.prod(v["shape"]); g=cat(k); s[g]=s.get(g,0)+n; b[g]=b.get(g,0)+n*B[v["dtype"]]
    tot=sum(s.values()); totb=sum(b.values())
    d=t["hidden_size"]; V=t["vocab_size"]; L=t["num_hidden_layers"]
    E=t.get("num_experts") or 0; k=t.get("num_experts_per_tok") or t.get("top_k_experts") or 0
    if repo.endswith("E2B-it"): d=1536; L=35
    tied = "lm_head" not in s
    routed_active = s.get("routed",0)*(k/E) if E else 0
    # active per token (text path, no MTP/mm): dense+shared+routed*k/E+lm_head(tied counts vocab*d)+aux; input embed & tables = rows read
    lm = V*d if tied else s["lm_head"]
    table_rows_read = 0
    if "table" in s:
        table_rows_read = (16*160) if "Flash" in repo else L*t["hidden_size_per_layer_input"] if "E4B" in repo else 35*256
    active = s.get("dense",0)+s.get("shared",0)+routed_active+lm+s.get("table_aux",0)
    text = tot - s.get("mm",0) - s.get("mtp",0)
    r=dict(total=tot,total_bytes=totb,text=text,text_bf16_GB=text*2/1e9,mm=s.get("mm",0),mtp=s.get("mtp",0),
           table=s.get("table",0),table_aux=s.get("table_aux",0),embed=s.get("embed",0),lm_head=s.get("lm_head",0),tied=tied,
           routed=s.get("routed",0),shared=s.get("shared",0),dense=s.get("dense",0),
           active_compute=active, active_lookup=d+table_rows_read, d=d, L=L, E=E, k=k)
    res[name]=r
# graft table sizing (Flash-Next format: 2 n-gram orders x 8 heads = 16 head-slots, ~20M prime rows each)
def graft(d, ple_dim):
    w=ple_dim//16; rows=ROWS_FN
    table=rows*w
    aux= ple_dim*d + ple_dim*d + 3*d + d*4   # key_proj(hc=1), value_proj, 3 RMSNorm, depthwise conv k=4
    return dict(row_width=w,table=table,table_bf16_GB=table*2/1e9,table_fp8_GB=table/1e9,aux=aux)
for name,r in res.items():
    r["graft_A_dimmatched"]=graft(r["d"],r["d"])     # ple_embed_dim = host hidden (row width d/16)
    r["graft_B_rowmatched"]=graft(r["d"],2560)       # ple_embed_dim = 2560 (row width 160, byte-identical rows to Flash-Next)
json.dump(res,open("counts.json","w"),indent=1)
G=lambda x:f"{x/1e9:.3f}"
print("| target | total params (G) | ckpt bytes (GB) | text-only (G) | native table (G / bf16 GB) | table aux (G) | routed experts (G) | active compute/token (G) | lookup elems/token |")
print("|---|---|---|---|---|---|---|---|---|")
for n,r in res.items():
    print(f"| {n} | {G(r['total'])} | {G(r['total_bytes'])} | {G(r['text'])} | {G(r['table'])} / {r['table']*2/1e9:.2f} | {r['table_aux']/1e6:.1f}M | {G(r['routed'])} (top-{r['k']}/{r['E']}) | {G(r['active_compute'])} | {r['active_lookup']} |")
print()
print("| host | d | graft A: ple_dim=d, row width | A table G / bf16 GB / fp8 GB | B: ple_dim=2560 row width 160, table G / bf16 GB | aux params A / B (M) |")
print("|---|---|---|---|---|---|")
for n,r in res.items():
    a=r["graft_A_dimmatched"]; bb=r["graft_B_rowmatched"]
    print(f"| {n} | {r['d']} | {a['row_width']} | {G(a['table'])} / {a['table_bf16_GB']:.1f} / {a['table_fp8_GB']:.1f} | {G(bb['table'])} / {bb['table_bf16_GB']:.1f} | {a['aux']/1e6:.1f} / {bb['aux']/1e6:.1f} |")
for n,r in res.items(): print(n, {k:(round(v/1e9,4) if isinstance(v,(int,float)) and v>1e5 else v) for k,v in r.items() if not isinstance(v,dict)})
