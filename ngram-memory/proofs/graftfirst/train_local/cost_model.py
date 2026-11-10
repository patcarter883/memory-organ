# Throughput + memory model for graft-row training ("cartridge ingestion") on the two hosts. All outputs are ESTIMATES
# built from: exact param counts (nm/compute/count.out), configs, measured GDN-vs-linear CPU ratio (gdn_cpu_bench.json),
# HF checkpoint sizes (hub_detail.json), fork-measured PCIe/HBM/GEMM figures, AMD-published peaks.
import json
G = 1e9
bench = json.load(open("gdn_cpu_bench.json"))
H = {
 "Qwen3.6-35B-A3B": dict(A=2.9464*G, lm=0.5086*G, L=40, d=2048, Lfa=10, Lgdn=30, Hq=16, hd=256, Hv=32, ff=None,
                         routed=32.2123*G, dense=1.3053*G, w=128,
                         ckpt4="pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4", ckpt4_GB=23.12, text_q4_GB=21.5,
                         bf16_text_GB=69.3),
 "Qwen3.8-27B":     dict(A=25.6246*G, lm=1.2714*G, L=64, d=5120, Lfa=16, Lgdn=48, Hq=24, hd=256, Hv=48, ff=17408,
                         routed=0, dense=24.3532*G, w=320,
                         ckpt4="amd/Qwen3.8-27B-Quark-AWQ-MXFP4", ckpt4_GB=19.80, text_q4_GB=18.2,
                         bf16_text_GB=53.8),
}
T = 2048                      # training window (tokens per sequence)
TOK_GPU, TOK_CPU = 16384, 4096 # tokens per optimizer step
PEAK = {"RX 9070 XT (card0)": 194.6e12, "RX 9070 (card1)": 144.5e12}   # dense bf16 matrix
PEAK2 = sum(PEAK.values())
PIPE = 0.8                    # 2-stage pipeline with >=4 microbatches (bubble ~1/(m+1)), est.
MFU = {"Qwen3.6-35B-A3B": (0.07, 0.25), "Qwen3.8-27B": (0.20, 0.45)}   # est.; fork single large GEMM = 115.3/194.6 = 0.59
GDN_GPU = {"Qwen3.6-35B-A3B": (1.5, 1.1), "Qwen3.8-27B": (1.2, 1.05)}  # (torch fallback, fla) multiplicative, est.
HBM = 600e9                   # B/s per card, fork measured 674 (PERFORMANCE.md) less margin
CPU_RATE = (0.3e12, 1.0e12)   # 4 Zen-4 cores, whole-step bf16, est. (sandbox: 2.2 TF/s GEMM 4 thr AMX; 0.55-0.71 1 thr)
CPU_DEQ = (1.2e9, 2.6e9)      # params/s, 4 threads: sandbox 1.73 Gparam/s (2.26 ms per 3.93M-param expert) x0.7-1.5
NVME = 3.0e9                  # O_DIRECT, fork ckpt_read.py 2,979-4,967 MiB/s
BPP4 = 0.53                   # bytes/param 4-bit incl. scales
out = {}
for n, h in H.items():
    attn_fwd = 4 * h["Hq"] * h["hd"] * (T / 2) * h["Lfa"]          # QK^T + AV, causal mean context
    flop_tok = 6 * h["A"] + 4 * attn_fwd                             # fwd + recompute + input-grad bwd (no weight grads)
    # measured CPU 1-thread GDN-core / projection ratio -> per-token GDN-core time relative to the linear work
    b = bench["35B-A3B" if "35B" in n else "27B"]
    gdn_cpu_s_tok = b["gdn_core_fwd_bwd_s"] / b["T"] * h["Lgdn"] * 1.33   # + recompute of the core
    lin_cpu_s_tok_1thr = flop_tok / (b["proj_GFLOPs_rate"] * G)
    gdn_cpu_factor = 1 + gdn_cpu_s_tok / lin_cpu_s_tok_1thr
    r = dict(flop_per_tok_GFLOP=round(flop_tok / G, 1), attn_share=round(4 * attn_fwd / flop_tok, 3),
             TFLOP_per_Mtok=round(flop_tok * 1e6 / 1e12), gdn_cpu_factor=round(gdn_cpu_factor, 2))
    # ---- GPU, both cards, serve stopped
    lo = PEAK2 * MFU[n][0] * PIPE; hi = PEAK2 * MFU[n][1] * PIPE
    steps = 1e6 / TOK_GPU
    deq_bytes = (h["routed"] + h["dense"]) * (BPP4 + 2) * 3          # read packed + write bf16, 3 passes/step
    deq_s = deq_bytes / (2 * HBM) * steps
    gpu_hi_s = flop_tok * 1e6 / lo * GDN_GPU[n][0] + deq_s
    gpu_lo_s = flop_tok * 1e6 / hi * GDN_GPU[n][1] + deq_s
    r["gpu_eff_TFLOPs"] = (round(lo / 1e12), round(hi / 1e12))
    r["gpu_h_per_Mtok"] = (round(gpu_lo_s / 3600, 2), round(gpu_hi_s / 3600, 2))
    r["gpu_dequant_s_per_Mtok"] = round(deq_s, 1)
    # bf16 weights streamed from pinned RAM instead (serve stopped): per pass every layer's weights cross PCIe
    pcie = {"card0": 28.93e9, "card1": 14.48e9}       # fork cpu_tier.py:6
    half = h["bf16_text_GB"] * G / 2
    stream_s_step = max(half * 3 / pcie["card0"], half * 3 / pcie["card1"])   # 3 passes; slow card bounds it
    comp_s_step = flop_tok * TOK_GPU / hi, flop_tok * TOK_GPU / lo
    r["bf16_stream_s_per_step"] = round(stream_s_step, 1)
    r["compute_s_per_step_16k"] = tuple(round(x, 1) for x in comp_s_step)
    # ---- CPU, 4 cores beside the serve, <=15 GiB
    steps_c = 1e6 / TOK_CPU
    io_s = h["text_q4_GB"] * G * 2 / NVME * steps_c                 # 2 reads/step (fwd; recompute+bwd share a buffer)
    deq_c = ((h["routed"] + h["dense"]) * 2 / CPU_DEQ[1] * steps_c, (h["routed"] + h["dense"]) * 2 / CPU_DEQ[0] * steps_c)
    comp_c = (flop_tok * 1e6 / CPU_RATE[1] * gdn_cpu_factor, flop_tok * 1e6 / CPU_RATE[0] * gdn_cpu_factor)
    cpu = (max(comp_c[0] + deq_c[0], io_s), max(comp_c[1] + deq_c[1], io_s))   # IO overlapped by prefetch
    r["cpu_h_per_Mtok"] = tuple(round(x / 3600, 1) for x in cpu)
    r["cpu_io_h_per_Mtok"] = round(io_s / 3600, 2)
    # ---- memory, GPU (per 2 cards) at 16k tokens/step = 8 x 2048, 4 microbatches of 4096
    mb = 4096
    ckpt_act = mb * h["d"] * 2 * h["L"] * 4 / G       # layer inputs for all in-flight microbatches (4) -- worst case
    work = (mb * (h["ff"] or 9 * 512) * 2 * 4 + mb * h["d"] * 2 * 12) / G   # MLP intermediates + misc inside one layer
    gdn_work = mb / 64 * h["Hv"] * 64 * 64 * 4 * 6 / G + h["Hv"] * 128 * 128 * 4 * (mb / 64) / G  # chunk mats + chunk states
    deq_buf = (h["dense"] + h["routed"]) / h["L"] * 2 * 2 / G      # double-buffered bf16 layer
    lmhead_chunk = 1024 * 248320 * 4 * 2 / G                       # chunked CE: 1024 positions fp32 logits + grad
    rows = mb * 16 * h["w"] * 4 * 2 / G                             # gathered rows fp32 + grad
    r["vram_GB"] = dict(weights_q4_text=h["text_q4_GB"], ckpt_activations=round(ckpt_act, 2), layer_work=round(work + gdn_work, 2),
                        dequant_bufs=round(deq_buf, 2), lm_head_ce_chunk=round(lmhead_chunk, 2), rows=round(rows, 3))
    r["vram_total_GB"] = round(sum(v for v in r["vram_GB"].values()), 1)
    out[n] = r
print(json.dumps(out, indent=1))
json.dump(out, open("cost_model.json", "w"), indent=1)
