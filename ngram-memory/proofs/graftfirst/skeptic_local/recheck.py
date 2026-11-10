# Skeptic re-derivation of the train_local cost/memory claims. All outputs are ESTIMATES unless noted.
import json
G = 1e9
hub = json.load(open("hub_recheck.json"))                      # measured header bytes (HF API)
H = {"35B": dict(A=2.9464*G, exp=32.2123*G, dense=1.3053*G, L=40, d=2048, E=256, k=8, I=512,
                 ck="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", bpp_exp=0.578),     # 18.623 GB / 32.21 G
     "27B": dict(A=25.6246*G, exp=0, dense=24.3532*G, L=64, d=5120, E=0, k=0, I=17408,
                 ck="amd/Qwen3.8-27B-Quark-AWQ-MXFP4", bpp_exp=0)}
HBM2 = 2*600e9; out = {}
for n, h in H.items():
    p = hub[h["ck"]]["by_part_GB"]
    resident = p["other"] + p.get("experts", 0) + p["lm_head"]           # text body w/o embed, visual, mtp
    r = dict(resident_weights_GB=round(resident, 2))
    # dequant traffic: packed read + bf16 write + bf16 read by the GEMM, 3 passes (fwd, recompute, input-grad)
    qparams = h["exp"] + (h["dense"] if n == "27B" else 0)              # 35B AWQ: dense path is already F16
    per_pass = qparams * (0.55 + 2 + 2)
    for sched, mb in (("report: once per 16k step", 16384), ("1F1B 2048-token microbatches", 2048), ("layer-wise 8k halves", 8192)):
        r[f"dequant_s_per_Mtok [{sched}]"] = round(per_pass*3/HBM2 * 1e6/mb, 0)
    if h["E"]:
        for mb in (2048, 8192, 16384):
            r[f"tokens_per_expert_at_{mb}"] = mb*h["k"]/h["E"]
        # one expert's gate_up+down at M tokens: FLOP / bf16-weight bytes -> HBM-bound ceiling at 640 GB/s
        for M in (64, 256, 512):
            ai = 2*M/2; r[f"expert_gemm_ceiling_TFs_M{M}"] = round(min(ai*640e9, 194.6e12)/1e12, 0)
    # per-card VRAM, 57:43 layer split, lm_head + CE chunk on card 1, 16.82 GB per card (15.67 GiB)
    per_layer = (resident - p["lm_head"]) / h["L"]
    l0 = round(h["L"]*0.574); l1 = h["L"] - l0
    deq_layer = (h["exp"] if h["E"] else h["dense"]) / h["L"] * 2 / G  # one layer bf16, GB
    act = 2*2048*h["d"]*2                                             # 2 in-flight microbatch inputs per layer
    c0 = l0*per_layer + deq_layer*(0.5 if h["E"] else 2) + l0*act/G + 1.5
    c1 = l1*per_layer + p["lm_head"] + deq_layer*(0.5 if h["E"] else 2) + l1*act/G + 1.02 + 1.5
    r["card0_GB(est)"] = round(c0, 1); r["card1_GB(est)"] = round(c1, 1); r["card_cap_GB"] = 16.82
    out[n] = r
# CPU cartridge wall time from the report's own 11-35 / 62-205 h per Mtok at 1.44M tokens x 3 epochs
out["cpu_cartridge_days_3ep"] = {"35B": [round(1.44*3*x/24, 1) for x in (11.1, 35.0)], "27B": [round(1.44*3*x/24, 1) for x in (61.8, 204.5)]}
# report's upper MFU (45%) vs best measured single GEMM (115.3/194.6)
out["27B_upper_MFU_as_frac_of_best_GEMM"] = round(0.45/(115.3/194.6), 2)
print(json.dumps(out, indent=1)); json.dump(out, open("recheck.json", "w"), indent=1)
