"""Skeptic recompute of graft-serve §3/§6 numbers with the fork's CURRENT baselines (arithmetic + bench .out reuse).
Every GPU time here is an ESTIMATE. Host times are the sandbox single-thread medians from serve-port/bench_host_path.out."""
import json
B = json.load(open("../serve-port/bench_host_path.out"))
hosts = {  # d, w, decode baselines ms (source), prefill tok/s measured or rough
 "35B-A3B": dict(d=2048, w=128, dec={"EP SPEC=none 2026-09-29 (serve.sh:148-149)": 1000/102.6,
                                     "TP bs=1 ROCm10 2026-09-16 (ROCM10_UPGRADE.md:59)": 10.50,
                                     "report's 20.4 ms (GLM_CONTINUANCE.md:158, stale)": 20.4},
                 prefill_tps={"measured EP 7014 (serve.sh:149)": 7014, "report rough": 8192/0.492}),
 "27B": dict(d=5120, w=320, dec={"MXFP4 W4A16 SPEC=none 46.5 tok/s 2026-10-04 (serve.sh:416)": 1000/46.5,
                                 "MXFP4 W4A16 47.86 tok/s (serve.sh:~398)": 1000/47.86,
                                 "report's INT4 31.4 ms (2026-08-15)": 31.4},
             prefill_tps={"report rough (unmeasured)": 8192/4.424}),
}
BW = {"gfx1201 674 GB/s x 0.50 (LM-head GEMV 50.4% roofline, COUNTER_SCORECARD.md:341-343)": 674e9*0.50,
      "gfx1201 640 GB/s x 0.60 (report)": 640e9*0.60}
PCIE = (12.4e9, 14.5e9)  # repo-measured host->device cap (CONTINUANCE_gpu_tune_and_customar.md:24)
out = {}
for h, c in hosts.items():
    d, w = c["d"], c["w"]; r = {}
    wv = d*d*2
    host_us = (9.9 + 3.3 + 2.0, 11.5 + 3.4 + 3.1)  # vectorised hash + lookup + dense stage, bs1
    h2d_lat_us = (10, 25)  # ESTIMATE: one small non_blocking H2D + the `active` copy, not measured on ROCm
    for bn, bw in BW.items():
        g = wv/bw*1e6
        tot = (host_us[0] + h2d_lat_us[0] + g, host_us[1] + h2d_lat_us[1] + g)
        r[f"decode_linear_us[{bn}]"] = [round(x, 1) for x in tot]
        r[f"decode_pct[{bn}]"] = {k: [round(100*x/1000/v, 2) for x in tot] for k, v in c["dec"].items()}
    # prefill, per chunk; host from bench (anchors10000..100000 envelope), 100% and 0% hit
    for T in (2048, 8192):
        rows = []
        for a in ("anchors10000", "anchors100000"):
            k = f"w{w}/{a}/prefill_{T}"; b = B.get(k)
            if b is None: continue
            h2d = T*d*2
            for hit in ("0pct", "100pct"):
                host_ms = (b["hash"] + b[f"lookup_{hit}_hit"] + b[f"stage_dense_{hit}"])/1e3
                dev_ms = 2*T*d*d/50e12*1e3
                rows.append((hit, a, round(host_ms, 1), round(h2d/PCIE[1]*1e3, 1), round(h2d/PCIE[0]*1e3, 1), round(dev_ms, 2)))
        r[f"prefill_{T}"] = {"rows(hit,anchors,host_ms,h2d_ms_lo,h2d_ms_hi,gemm_ms)": rows}
        for k, tps in c["prefill_tps"].items():
            bb = T/tps*1e3
            worst = max(x[2]+x[4]+x[5] for x in rows); best = min(x[2]+x[3]+x[5] for x in rows)
            r[f"prefill_{T}"][f"pct_vs[{k}: {bb:.0f} ms/chunk]"] = [round(100*best/bb, 1), round(100*worst/bb, 1)]
    # RAM (shared /dev/shm, one copy per box, not per rank)
    r["cartridge_GiB_bf16+i64"] = {n: round(n_*(2*w+8)/2**30, 2) for n, n_ in (("6.72M", 6721239), ("2.04M", 2040221), ("0.55M", 550891))}
    r["Wv_MiB"] = wv/2**20
    out[h] = r
out["bitmap_MB"] = 320001446/8/1e6
out["row_id_lt_2^29"] = 320001446 < 2**29
print(json.dumps(out, indent=1))
