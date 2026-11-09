# Solve-memory + FLOP estimates at real depth from measured per-layer saved bytes (act.out) and exact counts (counts.json).
import json
M="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/models"
C=json.load(open("counts.json"))
A={}
for l in open("act.out"):
    if l.startswith("{"): r=json.loads(l); A[r["which"]]=r["bytes_per_token"]
A["q27"]={"L2:linear_attention":631516,"L3:full_attention":227968,"head":20692}  # from first run (same script)
def lt(repo):
    c=json.load(open(f"{M}/{repo}/config.json")); return c.get("text_config",c)["layer_types"]
rows=[("T1 Qwen3.8-Flash-Next","flash","Qwen/Qwen3.8-Flash-Next",1),("T2 gemma-4-E4B-it","e4b","google/gemma-4-E4B-it",0),
      ("T3 Qwen3.8-27B","q27","Qwen/Qwen3.8-27B",1),("T3 Qwen3.6-35B-A3B","q35","Qwen/Qwen3.6-35B-A3B",1),
      ("T3 gemma-4-26B-A4B-it","g26","google/gemma-4-26B-A4B-it",1),("T3 gemma-4-31B-it","g31","google/gemma-4-31B-it",1)]
print("| target | site | saved act/token, no ckpt (MB) | N=320 no ckpt (GB) | N=320 per-layer ckpt (GB) | body weights excl. table: bf16 / fp8 / ~4.5-bit (GB) | FLOP/iter N=320 (TFLOP) |")
print("|---|---|---|---|---|---|---|")
out={}
for name,w,repo,site in rows:
    a=A[w]; types=lt(repo); c=C[name]
    def per(t, i):
        full = [v for k,v in a.items() if ("full" in k or "indexed" in k)]
        if t=="full_attention": return full[0]
        lin=[v for k,v in a.items() if ("linear" in k or "sliding" in k) and not k.startswith("L0") ]
        return min(lin) if lin else 0
    per_layer=[per(t,i) for i,t in enumerate(types)][site:]
    if w=="flash": per_layer[0]=a["L1:linear_attention"]  # the PLE layer itself
    tot_tok=sum(per_layer)+a["head"]
    hid = c["d"]*2*(4 if w=="flash" else 1)
    ck_tok = hid*len(per_layer) + (42*256*2 if w=="e4b" else 0)
    ck_peak_tok = ck_tok + max(per_layer)
    N=320
    body = c["text"]-c["table"]
    emb = c["embed"]+c["lm_head"]
    q4 = (body-emb)*4.5/8 + emb*2
    flop = 4*c["active_compute"]*N
    out[name]=dict(act_MB_tok=tot_tok/1e6, nockpt_GB=tot_tok*N/1e9, ckpt_GB=ck_peak_tok*N/1e9, body_bf16=body*2/1e9, body_fp8=body/1e9, body_q4=q4/1e9, tflop=flop/1e12)
    o=out[name]
    print(f"| {name} | layer idx {site} input | {o['act_MB_tok']:.1f} | {o['nockpt_GB']:.1f} | {o['ckpt_GB']:.2f} | {o['body_bf16']:.0f} / {o['body_fp8']:.0f} / {o['body_q4']:.0f} | {o['tflop']:.1f} |")
json.dump(out,open("plan_calc.json","w"),indent=1)
