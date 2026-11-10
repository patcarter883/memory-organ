"""Stage-1 LOCAL compute model: tokens/s for TABLE-ROWS-ONLY training (fwd + checkpoint recompute + input-grad
backward down to the n-gram/PLE site; no weight grads) and forward-only held-out eval, on the owner's box.
Every rate is an ESTIMATE from the bracketed inputs below; each input cites its source.
Usage: python -I local_plan.py  (writes local_plan.json next to itself)"""
import json, os
GB=1e9
# ---- model shapes (config.json in scratchpad/models; active params from P/compute/counts.json) ----
# A = active matmul params/token incl. lm_head (counts.json 'active_compute'); lmh = lm_head params
# layers: (n, kind, heads, head_dim, window) ; site = decoder index where rows enter (layers below: fwd only)
M={
 "Flash-Next (native)": dict(A=6.6713,lmh=0.6357,L=48,site=1,hc=4,d=2560,
     lay=[(36,"gdn",48,128,0),(12,"full",24,256,0)], routed=120.796, streamed_GB=64.18,  # experts MXFP4 59.77 GiB (INTEGRATION §4.1)
     core_GB=5.7),
 "Gemma-4-E4B (native)": dict(A=4.6995,lmh=0.6711,L=42,site=0,hc=1,d=2560,
     lay=[(35,"slide",8,256,512),(7,"full",8,512,0)], routed=0, streamed_GB=0, core_GB=9.4),
 "Qwen3.6-35B-A3B (graft)": dict(A=2.9464,lmh=0.5086,L=40,site=1,hc=1,d=2048,
     lay=[(30,"gdn",32,128,0),(10,"full",16,256,0)], routed=32.212, streamed_GB=32.212*4.25/8, core_GB=2.2),
 "gemma-4-26B-A4B (graft)": dict(A=3.8225,lmh=0.7382,L=30,site=1,hc=1,d=2816,
     lay=[(25,"slide",16,256,1024),(5,"full",16,512,0)], routed=22.838, streamed_GB=22.838*4.25/8, core_GB=2.6),
 "Qwen3.8-27B (graft)": dict(A=25.6246,lmh=1.2714,L=64,site=1,hc=1,d=5120,
     lay=[(48,"gdn",48,128,0),(16,"full",24,256,0)], routed=0, streamed_GB=24.353*4.25/8, core_GB=2.6),
 "gemma-4-31B (graft)": dict(A=30.6973,lmh=1.4093,L=60,site=1,hc=1,d=5376,
     lay=[(50,"slide",32,256,1024),(10,"full",32,512,0)], routed=0, streamed_GB=29.288*4.25/8, core_GB=2.9),
}
def mix_fwd(m,S):
    """non-param FLOP/token, forward: causal attention 4*ctx*H*hd (QK^T+PV), GDN chunked ~H*(6 dk dv + 4*64*(dk+dv))."""
    f=0.
    for n,k,H,hd,w in m["lay"]:
        if k=="full": f+=n*4*(S/2)*H*hd
        elif k=="slide": f+=n*4*min(S/2,w)*H*hd
        else: f+=n*H*(6*hd*hd+4*64*2*hd)
    return f
def flops(m,S):
    body=(m["A"]-m["lmh"])*1e9; per=body/m["L"]; mixf=mix_fwd(m,S)
    below=m["site"]*per
    # fwd all layers (2/param) ; recompute + input-grad bwd for layers >= site (2+2/param); attention bwd ~2.5x fwd
    train=2*m["A"]*1e9 + 4*(body-below) + 4*m["lmh"]*1e9*0 + 2*m["lmh"]*1e9 + mixf*(1+1+2.5)*(1-m["site"]/m["L"]) + mixf*m["site"]/m["L"]
    evalf=2*m["A"]*1e9 + mixf
    return train, evalf
# ---- GPU bracket (2 cards, serve stopped) ----
GPU=dict(
 E0=(11e12,50e12),      # whole-model effective FLOP/s on card 0 (RX 9070 XT). low = the fork's measured serve
                        # prefill (Flash-Next 1406-1429 tok/s 2 cards, serve.sh:1145-1148; 27B 449 tok/s,
                        # QWEN38_27B_LIVE_TRAFFIC:20) ~= 11-12 TF/s/card; high = ~43% of measured dense bf16
                        # hipBLASLt 115.3 TF/s at M=2048 (BLOCKWISE_FP8_BRIEF:167). Peak bf16 matrix 194.6.
 f1=(0.74,1.0),         # card 1 vs card 0: RX 9070 56CU@2.52 vs 64CU@2.97 (CLAUDE.md:'GPU 1 (RX 9070)'); 1.0 if 2x XT
 bw0=28.93e9, bw1=(14.48e9,28.93e9), ddr=57.0e9,   # cpu_tier.py:6 measured; serve.sh:1137 says both Gen5 x8; DDR wall cpu_tier.py:63
 Tload=(4096,8192),     # tokens per card per expert/weight load (token-major 4k in VRAM; layer-major 8k w/ host ckpts)
 ov=(0.0,1.0),          # overlap of H2D streaming with compute (0 = serial, 1 = perfect double-buffer)
 hbm=(429e9,674e9),     # achieved HBM GB/s (27B journal:§3 429; PERFORMANCE.md 674 via BRIEF:258) for unfused dequant
 deq_passes=(3,0),      # unfused bf16 dequant per fwd/recompute/bwd (low) vs fused-in-GEMM (high)
)
CPU=dict(R=(0.15e12,0.6e12),  # Zen 4, 4 cores: sandbox measured 1.32 TF/s whole-model (cpu_rowtrain_bench.T4.out);
                              # Zen4 scale 0.15-0.45x (no AMX; ASSUMED); low also covers qwen4_exp HF layers measured
                              # 3.5 ms/token/layer at 256 tok (realdim_layers.T4.txt)
         deq_ns=(0.69,0.32),  # ns/param MXFP4->bf16: 2.26 ms / 4.92M-param expert x (1.5 .. 0.7) (INTEGRATION §4.2)
         T=(2048,4096), passes=3, nvme=(3.0e9,4.5e9))
def gpu(m,S,lo):
    i=0 if lo else 1
    tr,ev=flops(m,S)
    out={}
    for mode,F,npass in (("train",tr,2),("eval",ev,1)):
        rates=[]
        for card,(E,bw) in enumerate(((GPU["E0"][i],GPU["bw0"]),(GPU["E0"][i]*GPU["f1"][i],GPU["bw1"][i]))):
            tc=F/E
            tio=npass*m["streamed_GB"]*GB/(bw*GPU["Tload"][i])
            nparam=(m["routed"] if m["routed"] else (m["A"]-m["lmh"]))*1e9
            if m["streamed_GB"]==0: nparam=0
            dq=(GPU["deq_passes"][i] if mode=="train" else min(1,GPU["deq_passes"][i]))
            tdq=dq*nparam*2.53/(GPU["hbm"][i]*GPU["Tload"][i])
            t=max(tc,tio)+(1-GPU["ov"][i])*min(tc,tio)+tdq
            rates.append(1/t)
        # DDR wall: both cards streaming concurrently
        tot=sum(rates)
        if m["streamed_GB"]:
            ddr_cap=GPU["ddr"]*GPU["Tload"][i]/(npass*m["streamed_GB"]*GB)
            tot=min(tot,ddr_cap)
        out[mode]=tot; out[mode+"_1card"]=rates[0]
    return out
def cpu(m,S,lo):
    i=0 if lo else 1
    tr,ev=flops(m,S); out={}
    for mode,F,np_ in (("train",tr,CPU["passes"]),("eval",ev,1)):
        nparam=(m["routed"] if m["routed"] else (m["A"]-m["lmh"]))*1e9
        quant=m["routed"]>0 or "graft" in name
        tdq=(np_*nparam*CPU["deq_ns"][i]*1e-9/CPU["T"][i]) if quant else 0
        out[mode]=1/(F/CPU["R"][i]+tdq)
    return out
res={}
S=4096
for name,m in M.items():
    tr,ev=flops(m,S)
    g_lo,g_hi=gpu(m,S,True),gpu(m,S,False); c_lo,c_hi=cpu(m,S,True),cpu(m,S,False)
    res[name]=dict(train_GFLOP_tok=tr/1e9,eval_GFLOP_tok=ev/1e9,
        gpu_train=(g_lo["train"],g_hi["train"]),gpu_eval=(g_lo["eval"],g_hi["eval"]),
        gpu1_train=(g_lo["train_1card"],g_hi["train_1card"]),
        cpu_train=(c_lo["train"],c_hi["train"]),cpu_eval=(c_lo["eval"],c_hi["eval"]),
        ckpt_MB_tok=m["L"]*m["hc"]*m["d"]*2/1e6, streamed_GB=m["streamed_GB"])
TOK_EPOCH=1_440_935   # tokcount.out, Flash-Next tokenizer, python/ at df29117
HELD=200_897          # 15% by-file held-out (P/../stage1/design/walltime.out)
def hms(s):
    return f"{s/3600:.1f} h" if s>=5400 else f"{s/60:.0f} min"
print(f"S={S}; epoch={TOK_EPOCH} tok; held-out={HELD} tok. tok/s low-high (ESTIMATES)")
for name,r in res.items():
    print(f"\n{name}: train {r['train_GFLOP_tok']:.1f} GFLOP/tok, eval {r['eval_GFLOP_tok']:.1f}; ckpt {r['ckpt_MB_tok']:.2f} MB/tok; streamed {r['streamed_GB']:.1f} GB")
    for k in ("gpu_train","gpu1_train","gpu_eval","cpu_train","cpu_eval"):
        lo,hi=r[k]
        print(f"  {k:10s} {lo:8.1f} - {hi:8.1f} tok/s | 1M: {hms(1e6/hi)} - {hms(1e6/lo)} | 10M: {hms(1e7/hi)} - {hms(1e7/lo)} | epoch: {hms(TOK_EPOCH/hi)} - {hms(TOK_EPOCH/lo)} | held-out: {hms(HELD/hi)} - {hms(HELD/lo)}")
json.dump(dict(S=S,GPU=GPU,CPU=CPU,res=res),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"local_plan.json"),"w"),indent=1,default=str)
