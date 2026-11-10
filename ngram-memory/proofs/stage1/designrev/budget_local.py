"""Arithmetic for DESIGN.md 2026-11-10 revision (local-first compute). All inputs cited; outputs are estimates.
Inputs:
  rows: ingest-algo/count_rows.out (7,555,967 all rows; 2,040,221 count>=2 on train; 550,891 count>=5)
  per-1M-token GPU/CPU rates: graftfirst/REPORTS.md skeptic verdict on graft-hosts-local (35B 0.1-0.45 h, 27B 0.5-1.0 h;
     CPU >=11-35 h and 62-205 h); Flash-Next: stage1/local-compute/local_cost.out (9-59 min /1M, unreviewed)
  per-fact: REPORTS.md table (35B GPU 10-30 s/edit, CPU 12-38 min; 27B GPU 10-40 s, CPU 20-125 min);
     INTEGRATION.md 4.2(A) Flash-Next CPU 19-40 min/edit, batched 3.2-6.9 min/edit
  old cloud table: DESIGN.md 7 (iterations 100-300 assumed)
"""
rows = {"all": 7_555_967, "count>=2": 2_040_221, "count>=5": 550_891}
W = {"Flash-Next": 160, "35B-A3B": 128, "27B": 320, "gemma-26B": 176, "gemma-31B": 336}
print("== trainer state, fp32 delta + 1 fp32 Adagrad acc + int64 id = 4W+12 B/row (GB, 1e9) ==")
for h, w in W.items():
    b = 4 * w + 12
    print(f"{h:11s} W={w:3d} {b:5d} B/row  " + "  ".join(f"{k}: {v*b/1e9:5.2f}" for k, v in rows.items()))
print("== served cartridge, bf16 + int64 id = 2W+8 B/row (GiB) ==")
for h, w in W.items():
    b = 2 * w + 8
    print(f"{h:11s} {b:4d} B/row  " + "  ".join(f"{k}: {v*b/2**30:5.2f}" for k, v in rows.items()))

tok = 1.44e6  # minisgl python/ tree, Flash-Next tokenizer (local-compute/tokcount.out)
train_tok = 1.22e6  # 85% by-file split (ingest-algo walltime)
print("== cartridge wall-clock, 3 epochs over the train split + 0.5x anchor KL tokens (h) ==")
for h, (lo, hi) in {"35B-A3B GPU": (0.10, 0.45), "27B GPU": (0.5, 1.0), "Flash-Next GPU": (9/60, 59/60),
                    "35B-A3B CPU beside serve": (11, 35), "27B CPU beside serve": (62, 205)}.items():
    t = 3 * train_tok * 1.5 / 1e6
    boot = 2 * 300 / 3600 if "GPU" in h else 0
    print(f"{h:26s} {t*lo+boot:7.2f} - {t*hi+boot:7.2f} h  (incl. {boot:.2f} h serve stop+boot)")

print("== Stage-3 per-fact cohort wall-clock (2,000 facts, 20-30 iterations/solve) ==")
for h, (lo, hi) in {"35B GPU (10-30 s/edit)": (10, 30), "27B GPU (10-40 s/edit)": (10, 40),
                    "35B CPU beside serve (12-38 min)": (12*60, 38*60),
                    "Flash-Next CPU batched (3.2-6.9 min)": (3.2*60, 6.9*60),
                    "Flash-Next GPU downtime (3.6-30 s/iter x 20-30)": (3.6*20, 30*30)}.items():
    print(f"{h:48s} {2000*lo/3600:7.1f} - {2000*hi/3600:7.1f} h")

print("== old cloud table rescaled from 100-300 to 20-30 iterations/solve (low x0.2, high x0.1) ==")
old = {"M2-lite": (19, 64), "M2": (43, 513), "M2g": (4, 90), "M3 native": (102, 1275), "M3 graft": (42, 1075),
       "M3 native-vs-graft": (15, 146), "M4 E4B": (5, 120), "M5": (10, 240), "M6 seq 5k": (826, 10585)}
tl = th = 0
for k, (lo, hi) in old.items():
    tl += lo * 0.2; th += hi * 0.1
    print(f"{k:20s} ${lo*0.2:7.0f} - ${hi*0.1:7.0f}")
print(f"{'total':20s} ${tl:7.0f} - ${th:7.0f}  (setup hours not rescaled; old total $1.1k-14k)")
