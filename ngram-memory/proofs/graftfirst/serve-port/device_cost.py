"""Device-side cost / footprint of the grafted PLE block, from config.json alone. Arithmetic, not measured on GPU:
every time below is an ESTIMATE from bytes / bandwidth or FLOPs / throughput, with the assumptions printed."""
import json
M = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/models/Qwen"
BW = {"RX 9070 XT-class 640 GB/s": 640e9, "RX 9060 XT-class 322 GB/s": 322e9}
EFF_BW = 0.6          # assumed achieved fraction of peak for a skinny bf16 GEMV
TFLOPS = 50e12        # assumed achieved bf16 GEMM throughput per card for prefill (est.)
BASE_MS = {"Qwen3.8-27B": 31.4, "Qwen3.6-35B-A3B": 20.4}  # tools/serve.sh:384-385 (31.8 tok/s), docs/journal/GLM_CONTINUANCE.md:158
out = {}
for m in ("Qwen3.6-35B-A3B", "Qwen3.8-27B"):
    c = json.load(open(f"{M}/{m}/config.json")); t = c.get("text_config", c)
    d = t["hidden_size"]; w = d // 16; E = 16 * w
    assert t["layer_types"][1] == "linear_attention"
    wv = d * E * 2; wk = d * E * 2
    r = {"d": d, "row_width_w": w, "ple_embed_dim": E, "layer1_type": t["layer_types"][1],
         "W_v_bf16_MiB": round(wv / 2**20, 1), "general_path_extra_W_k_MiB": round(wk / 2**20, 1),
         "conv_state_per_slot_KiB_general": round(d * 9 * 2 / 1024, 1), "token_history_per_slot_B": 16}
    for T in (2048, 8192):
        r[f"staging_dev_MiB_T{T}_current_fp32+bf16"] = round(T * E * 6 / 2**20, 1)
        r[f"staging_dev_MiB_T{T}_proposed_bf16_only"] = round(T * E * 2 / 2**20, 1)
    for name, bw in BW.items():
        lin_us = wv / (bw * EFF_BW) * 1e6; gen_us = (wv + wk) / (bw * EFF_BW) * 1e6
        r[f"decode_GEMV_us_linear[{name}]"] = round(lin_us)
        r[f"decode_GEMV_us_general[{name}]"] = round(gen_us)
        r[f"decode_overhead_pct_linear[{name}]"] = round(100 * lin_us / 1e3 / BASE_MS[m], 2)
    r["prefill_8192_dense_GEMM_ms_linear"] = round(2 * 8192 * d * E / TFLOPS * 1e3, 2)
    r["prefill_8192_backbone_ms_rough"] = round(2 * 8192 * (27e9 if "27B" in m else 3e9) / (2 * TFLOPS) * 1e3)
    r["baseline_decode_ms"] = BASE_MS[m]
    out[m] = r
out["assumptions"] = {"EFF_BW": EFF_BW, "TFLOPS_per_card": TFLOPS, "W_v replicated on every TP rank": True,
                      "backbone prefill rough": "2*T*active_params / (2 cards * TFLOPS); 35B uses 3B active"}
print(json.dumps(out, indent=1))
