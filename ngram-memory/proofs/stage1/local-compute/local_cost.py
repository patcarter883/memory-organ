"""LOCAL wall-clock model for Stage-1 table-rows-only training (fwd + per-layer-checkpoint recompute +
input-grad-only backward down to the table's injection layer; NO weight grads) on the owner's box.
Every output is an ESTIMATE; inputs are cited inline. Usage: python -I local_cost.py > local_cost.out"""
import json, math
G=1e9; T_=1e12
# ---------------- measured / published inputs ----------------
R0_GEMM=115.3e12   # card0 RX 9070 XT dense bf16 hipBLASLt @M=2048 (fork BLOCKWISE_FP8_BRIEF_2026-09-22.md:167 <- PERFORMANCE.md)
R0_PEAK=194.6e12   # card0 bf16 matrix peak (fork CONTINUANCE_gemm_efficiency.md:37)
R1_PEAK=145e12     # card1 RX 9070, AMD spec 145 TFLOPS FP16 matrix dense (amd.com product page)
R1_GEMM=R0_GEMM*R1_PEAK/R0_PEAK   # ASSUMED same GEMM efficiency as card0 -> 85.9 TF/s
PCIE0,PCIE1=28.93e9,14.48e9       # host->GPU, fork weights/cpu_tier.py:6 (card1 loaded 12.36, :53-54)
DDR_WALL=57.0e9                   # fork cpu_tier.py:420
VRAM_BW=0.60e12                   # ~94% of 640 GB/s GDDR6 (674 measured, fork BLOCKWISE brief :258) -- est.
NVME_OD=(3.12e9,5.21e9)           # O_DIRECT 2979-4967 MiB/s, INTEGRATION.md 4.1 (ckpt_read.py:60-63)
# end-to-end fraction of the measured GEMM rate (ASSUMED brackets; CPU calibration calib_cpu.T4.out gave
# 0.24-0.29 contended, prior uncontended bench ~0.56 on Qwen2.5-0.5B)
EFF_DENSE=(0.30,0.65); EFF_MOE=(0.15,0.45)
# CPU (owner Zen 4, 4 cores pinned beside the serve): end-to-end model FLOP/s bracket, ASSUMED
CPU_RATE=(0.15e12,0.60e12)
CPU_DEQ_NS_PER_ELEM=(2.26e-3/4.915e6*0.7, 2.26e-3/4.915e6*1.5)   # 2.26 ms/expert 4 thr (INTEGRATION 4.2) x0.7-1.5
S=4096             # training window (90/261 files exceed 4096 tok; packed, EOS between files)
# ---------------- targets (count.out exact active params; configs in scratchpad/models) ----------------
def attn_fwd(full,H,hd,S,slide=0,Hs=0,hds=0,W=0,gdn=0,gH=0,dk=128,dv=128):
    a=full*2*hd*S*H
    if slide: ctx=W-W*W/(2*S); a+=slide*4*hds*ctx*Hs
    if gdn: a+=gdn*gH*(4*dk*dv+2*64*(dk+dv))   # chunked gated-delta (C=64), rough
    return a
T={
 "Flash-Next (native)":dict(P=6.6713*G,routed=120.796*G*10/512,L=48,site=1,attn=attn_fwd(12,24,256,S,gdn=36,gH=48),
     ckpt=4*2560*2*48, stream_bytes=59.77*2**30, stream_elems=120.796*G, tok=1440935),
 "Gemma-4-E4B (native PLE)":dict(P=4.6995*G,routed=0,L=42,site=0,attn=attn_fwd(7,8,512,S,slide=35,Hs=8,hds=256,W=512),
     ckpt=2560*2*42, resident=9.4e9, tok=1585835),
 "Qwen3.6-35B-A3B (graft)":dict(P=2.9464*G,routed=32.212*G*8/256,L=40,site=1,attn=attn_fwd(10,16,256,S,gdn=30,gH=32),
     ckpt=2048*2*40, w4=34.66*G-0.5086*G, tok=1440935, alt_stream=32.212*G*4.25/8),
 "gemma-4-26B-A4B (graft)":dict(P=3.8225*G,routed=22.838*G*8/128,L=30,site=1,attn=attn_fwd(5,16,512,S,slide=25,Hs=16,hds=256,W=1024),
     ckpt=2816*2*30, w4=25.233*G-0.7382*G, tok=1585835, alt_stream=22.838*G*4.25/8),
 "Qwen3.8-27B (graft)":dict(P=25.6246*G,routed=0,L=64,site=1,attn=attn_fwd(16,24,256,S,gdn=48,gH=48),
     ckpt=5120*2*64, w4=26.896*G-1.2714*G, tok=1440935),
 "gemma-4-31B (graft)":dict(P=30.6973*G,routed=0,L=60,site=1,attn=attn_fwd(10,32,512,S,slide=50,Hs=32,hds=256,W=1024),
     ckpt=5376*2*60, w4=30.697*G-1.4093*G, tok=1585835),
}
HELD=144_000; RETAIN=500_000   # eval set: 10% held-out files (~0.1 x 1.44M) + 0.5M-token OOD retention set (ASSUMED sizes)
def flops(t):
    P=t["P"]; Pbw=P*(1-t["site"]/t["L"])
    tr=2*P+4*Pbw+t["attn"]*4.5   # fwd + recompute + dX-only bwd; attention bwd ~2.5x fwd
    ev=2*P+t["attn"]
    return tr,ev
def gpu_rate(t,tr,eff_d,eff_m,cards):
    fm=t["routed"]/t["P"]
    r=[R0_GEMM,R1_GEMM][:cards] if cards else None
    def card(R): return 1/((1-fm)/(R*eff_d)+fm/(R*eff_m))
    return [card(R) for R in r]
def hfmt(s):
    return f"{s/60:.0f} min" if s<5400 else (f"{s/3600:.1f} h" if s<172800 else f"{s/86400:.1f} d")
rows=[]
def add(name,venue,tps_lo,tps_hi,ntok,ev_lo,ev_hi,note):
    r=dict(target=name,venue=venue,train_tok_s=[round(tps_lo,1),round(tps_hi,1)],
      per_1M=[hfmt(1e6/tps_hi),hfmt(1e6/tps_lo)],per_10M=[hfmt(1e7/tps_hi),hfmt(1e7/tps_lo)],
      per_epoch=[hfmt(ntok/tps_hi),hfmt(ntok/tps_lo)],eval_tok_s=[ev_lo if math.isnan(ev_lo) else round(ev_lo),ev_hi if math.isnan(ev_hi) else round(ev_hi)],
      eval_point=["-","-"] if math.isnan(ev_lo) else [hfmt((HELD+RETAIN)/ev_hi),hfmt((HELD+RETAIN)/ev_lo)],note=note)
    rows.append(r)
for name,t in T.items():
    tr,ev=flops(t)
    print(f"\n## {name}: train {tr/G:.1f} GFLOP/tok, eval {ev/G:.1f} GFLOP/tok, attn fwd {t['attn']/G:.2f} GFLOP/tok @S={S}, "
          f"ckpt {t['ckpt']/1e6:.2f} MB/tok, epoch {t['tok']} tok")
    lo=gpu_rate(t,tr,EFF_DENSE[0],EFF_MOE[0],2); hi=gpu_rate(t,tr,EFF_DENSE[1],EFF_MOE[1],2)
    if "stream_bytes" in t:   # Flash-Next: core in VRAM, MXFP4 experts in pinned host RAM, streamed per layer
        for Tb in (8192,16384):
            # fast: 2 batches interleaved across a 2-stage layer pipeline, experts split 2:1, PCIe concurrent (0.85 eff)
            io_f=2*t["stream_bytes"]/(min(PCIE0+PCIE1,DDR_WALL)*0.85)
            CK_VRAM=12e9   # ASSUMED VRAM left for checkpoints across both cards after core 6.4 + staging 5.4 + working ~8
            spill_f=max(0,2*Tb*t["ckpt"]-CK_VRAM)   # 2 batches in flight
            io_f+=2*spill_f/(min(PCIE0+PCIE1,DDR_WALL)*0.85)
            comp_f=Tb*tr/(hi[0]+hi[1]); deq_f=2*t["stream_elems"]*4.5/(2*VRAM_BW); ov_f=0.3
            st_f=max(io_f,comp_f+deq_f+ov_f)
            # slow: serial pipeline (layers split evenly), each card streams its half alone, no overlap
            io_s=2*t["stream_bytes"]*(0.5/(PCIE0*0.85)+0.5/(PCIE1*0.85))
            spill_s=max(0,Tb*t["ckpt"]-CK_VRAM); io_s+=2*spill_s*(0.5/(PCIE0*0.85)+0.5/(PCIE1*0.85))
            comp_s=Tb*tr*(0.5/lo[0]+0.5/lo[1]); deq_s=3*t["stream_elems"]*4.5/VRAM_BW; ov_s=512*48*3*30e-6
            st_s=io_s+comp_s+deq_s+ov_s
            # eval: one expert pass, no ckpt, Tb_eval=32768
            Te=32768; e_f=Te/max(t["stream_bytes"]/(min(PCIE0+PCIE1,DDR_WALL)*0.85), Te*ev/(hi[0]+hi[1])+deq_f/2)
            e_s=Te/(t["stream_bytes"]*(0.5/(PCIE0*.85)+0.5/(PCIE1*.85))+Te*ev*(0.5/lo[0]+0.5/lo[1])+deq_s/3+ov_s/3)
            print(f"  GPUx2 T={Tb}: fast step {st_f:.1f}s (io {io_f:.1f}, comp {comp_f:.1f}, deq {deq_f:.2f}) | "
                  f"slow step {st_s:.1f}s (io {io_s:.1f}, comp {comp_s:.1f}, deq {deq_s:.2f}, launch {ov_s:.1f}) | ckpt {Tb*t['ckpt']/1e9:.1f} GB/batch, host-spilled fast {spill_f/1e9:.1f} slow {spill_s/1e9:.1f} GB")
            add(name,f"2x RDNA4, serve down, experts host-streamed, T={Tb}",Tb/st_s,Tb/st_f,t["tok"],e_s,e_f,
                "IO-bound at fast end" if io_f>=comp_f+deq_f else "compute-bound")
        # (e2) skip routed-expert Jacobian: backward needs no experts -> 1 expert pass/step
        Tb=8192; io_f=t["stream_bytes"]/(min(PCIE0+PCIE1,DDR_WALL)*0.85); fm=t["routed"]/t["P"]
        tr2=tr-2*t["routed"]*2   # drop recompute+bwd of experts
        comp_f=Tb*tr2/(hi[0]+hi[1]); st_f=max(io_f,comp_f+0.3)
        io_s=t["stream_bytes"]*(0.5/(PCIE0*.85)+0.5/(PCIE1*.85)); comp_s=Tb*tr2*(0.5/lo[0]+0.5/lo[1])
        st_s=io_s+comp_s+t["stream_elems"]*4.5/VRAM_BW+512*48*30e-6
        add(name,"2x RDNA4, (e2) no expert Jacobian, T=8192 [approx. gradient]",Tb/st_s,Tb/st_f,t["tok"],float('nan'),float('nan'),"gradient approx; cosine unmeasured")
        # CPU beside the serve (INTEGRATION.md loader, MemoryMax=15G, 4 Zen4 cores)
        for Tb in (2048,3072):
            comp=[Tb*tr/CPU_RATE[1],Tb*tr/CPU_RATE[0]]
            deq=[2*t["stream_elems"]*CPU_DEQ_NS_PER_ELEM[0], 3*t["stream_elems"]*CPU_DEQ_NS_PER_ELEM[1]]
            io=[2*t["stream_bytes"]/NVME_OD[1], 2*t["stream_bytes"]/NVME_OD[0]]
            st=[max(io[0],comp[0]+deq[0]), max(io[1],comp[1]+deq[1])]
            ev_=[32768/max(t["stream_bytes"]/NVME_OD[0], 32768*ev/CPU_RATE[0]+1.5*t["stream_elems"]*CPU_DEQ_NS_PER_ELEM[1]),
                 32768/max(t["stream_bytes"]/NVME_OD[1], 32768*ev/CPU_RATE[1]+t["stream_elems"]*CPU_DEQ_NS_PER_ELEM[0])]
            print(f"  CPU T={Tb}: step {st[0]:.0f}-{st[1]:.0f}s (comp {comp[0]:.0f}-{comp[1]:.0f}, deq {deq[0]:.0f}-{deq[1]:.0f}, io {io[0]:.0f}-{io[1]:.0f}); RAM ckpt {Tb*t['ckpt']/2**30:.1f} GiB on ~10.9 GiB fixed")
            add(name,f"CPU beside serve, 4 cores, 15G cap, T={Tb}",Tb/st[1],Tb/st[0],t["tok"],ev_[0],ev_[1],"overnight only")
        continue
    # resident-weight targets
    if name.startswith("Gemma-4-E4B"):
        for cards,lab in ((1,"1 card (9070 XT)"),(2,"2 cards, data-parallel replicas")):
            hi_=sum(hi[:cards]); lo_=sum(lo[:cards]) if cards==1 else lo[0]+lo[1]
            if cards==1: hi_,lo_=hi[0],lo[0]
            add(name,lab,lo_/tr,hi_/tr,t["tok"],lo_/ev,hi_/ev,f"bf16 body {t['resident']/1e9:.1f} GB/card; table subset on GPU, rest CPU")
        add(name,"1 card (9070, slower)",lo[1]/tr,hi[1]/tr,t["tok"],lo[1]/ev,hi[1]/ev,"")
        cpu=[CPU_RATE[0]/tr,CPU_RATE[1]/tr]
        add(name,"CPU beside serve (bf16 9.4 GB in RAM)",cpu[0],cpu[1],t["tok"],CPU_RATE[0]/ev,CPU_RATE[1]/ev,"fits 15G cap barely; overnight")
        continue
    wbytes=t["w4"]*4.25/8 + 1.0e9   # 4-bit matmul weights (~4.25 b/w incl. scales) + bf16 lm_head chunking slack
    deq_elems=t["w4"]
    vram_free=32e9-wbytes-2*2.5e9   # minus ~2.5 GB/card working set + allocator slack (ASSUMED)
    Tb=int(min(16384, max(1024, vram_free/t["ckpt"]))//1024*1024)
    # fast: 2-stage interleaved pipeline (both cards busy); slow: serial pipeline + 3 dequant passes
    comp_f=Tb*tr/(hi[0]+hi[1]); deq_f=2*deq_elems*4.5/(2*VRAM_BW); st_f=comp_f+deq_f+0.2
    comp_s=Tb*tr*(0.5/lo[0]+0.5/lo[1]); deq_s=3*deq_elems*4.5/VRAM_BW; st_s=comp_s+deq_s+(0.5 if t["routed"] else 0.1)
    Te=Tb*4; e_f=Te/(Te*ev/(hi[0]+hi[1])+deq_f/2); e_s=Te/(Te*ev*(0.5/lo[0]+0.5/lo[1])+deq_s/3)
    print(f"  GPUx2 resident 4-bit: weights {wbytes/1e9:.1f} GB, ckpt room {vram_free/1e9:.1f} GB -> T={Tb}; "
          f"fast step {st_f:.1f}s slow {st_s:.1f}s")
    add(name,f"2x RDNA4 serve down, ~4-bit resident ({wbytes/1e9:.0f} GB), T={Tb}",Tb/st_s,Tb/st_f,t["tok"],e_s,e_f,"")
    if "alt_stream" in t:  # one card, experts streamed from host (other card free)
        Tb1=8192; io=2*t["alt_stream"]/(PCIE0*0.85)
        comp=[Tb1*tr/hi[0], Tb1*tr/lo[0]]
        st=[max(io,comp[0]+deq_f), io+comp[1]+deq_s]
        add(name,f"1 card (9070 XT), 4-bit experts host-streamed, T={Tb1}",Tb1/st[1],Tb1/st[0],t["tok"],float('nan'),float('nan'),
            f"io {io:.1f}s/step")
    # CPU-streamed beside the serve: 4-bit weights O_DIRECT per layer, dequant on CPU
    Tc=2048
    comp=[Tc*tr/CPU_RATE[1],Tc*tr/CPU_RATE[0]]
    deq=[2*deq_elems*CPU_DEQ_NS_PER_ELEM[0],3*deq_elems*CPU_DEQ_NS_PER_ELEM[1]]
    st=[comp[0]+deq[0],comp[1]+deq[1]]
    evc=[32768/(32768*ev/CPU_RATE[0]+1.5*deq_elems*CPU_DEQ_NS_PER_ELEM[1]),32768/(32768*ev/CPU_RATE[1]+deq_elems*CPU_DEQ_NS_PER_ELEM[0])]
    add(name,"CPU-streamed beside serve, 4 cores, T=2048",Tc/st[1],Tc/st[0],t["tok"],evc[0],evc[1],"overnight only")
print("\n## TABLE (all ESTIMATES)")
print("| target | venue | train tok/s | per 1M tok | per 10M tok | per epoch | eval tok/s | per eval point (644k tok) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    ev="-" if math.isnan(r["eval_tok_s"][0]) else f"{r['eval_tok_s'][0]}-{r['eval_tok_s'][1]}"
    ep="-" if ev=="-" else f"{r['eval_point'][0]} - {r['eval_point'][1]}"
    print(f"| {r['target']} | {r['venue']} | {r['train_tok_s'][0]}-{r['train_tok_s'][1]} | {r['per_1M'][0]} - {r['per_1M'][1]} | "
          f"{r['per_10M'][0]} - {r['per_10M'][1]} | {r['per_epoch'][0]} - {r['per_epoch'][1]} | {ev} | {ep} |")
json.dump(rows,open("local_cost.json","w"),indent=1,default=str)
print(f"\nR1_GEMM (scaled) = {R1_GEMM/T_:.1f} TF/s")
