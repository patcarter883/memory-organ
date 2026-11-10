"""Local wall-clock for one Stage-1 cartridge build (owner's box, serve stopped, both RDNA4 cards).

MEASURED inputs (cited):
  project tokens 1,418,418; by-file train 85% (count_rows.json: split_by_file.train_tokens)
  active params/token: DESIGN.md §7 table (P/compute/count.out)
  Flash-Next routed experts 59.77 GiB MXFP4 (P/serve/v2-solve-local/ckpt_inventory.txt)
  PCIe 28.93 / 14.48 GB/s card0 / card1 (R/weights/cpu_tier.py:6)
  serve boot 300 s (R/weights/ckpt_read.py:5-10)
ASSUMED (labelled): effective bf16 TFLOPS per card in torch-ROCm 25..60 (RX 9070 XT / RX 9070;
  the spec-sheet dense WMMA peak is several times higher, unverified here); 2-card pipeline
  efficiency 0.7; FLOP = 6*active*N (fwd + recompute + bwd, DESIGN §7); KL-anchor tokens = 0.5x
  train tokens per epoch; 3 epochs; eval forward-only = 2*active*N.
"""
import json, sys

cr = json.load(open(sys.argv[1]))
train = cr["split_by_file"]["train_tokens"]
held = cr["split_by_file"]["heldout_tokens"]
timeh = cr["split_by_time"]["added_positions"]
EPOCHS, ANCHOR = 3, 0.5
hosts = {  # active params / token (G), experts streamed from host RAM (GB) or None
    "Qwen3.8-Flash-Next (native)": (6.67, 59.77 * 1.0737),
    "Qwen3.6-35B-A3B (graft)": (2.95, None),
    "gemma-4-26B-A4B (graft)": (3.82, None),
    "Qwen3.8-27B (graft)": (25.6, None),
    "gemma-4-31B (graft)": (30.7, None),
}
MB = 8192  # tokens per microbatch
print(f"train tokens {train}, held-out {held}, by-time {timeh}; epochs {EPOCHS}, anchor x{ANCHOR}")
out = {}
for name, (act, xb) in hosts.items():
    row = []
    for tf in (25, 60):
        eff = 2 * tf * 0.7 * 1e12
        t_mb = 6 * act * 1e9 * MB / eff
        if xb:  # two expert sweeps per microbatch (fwd; recompute+bwd share one), split 2:1 by PCIe speed
            h2d = max(2 * xb * (2 / 3) / 28.93, 2 * xb * (1 / 3) / 14.48)
            t_mb = max(t_mb, h2d) * 1.15     # +15% for imperfect overlap / dequant (ASSUMED)
        tok_s = MB / t_mb
        n_train = EPOCHS * train * (1 + ANCHOR)
        n_eval = EPOCHS * (held + timeh + ANCHOR * train) / 3   # fwd-only ~1/3 of fwd+bwd
        n_base = ANCHOR * train / 3                               # base logits for KL, once
        hours = (n_train + n_eval + n_base) / tok_s / 3600 + 2 * 300 / 3600
        row.append((round(tok_s), round(hours, 2)))
    out[name] = row
    print(f"{name:32s} tok/s fwd+bwd {row[1][0]:>5}-{row[0][0]:<5}  build {row[1][1]:.1f}-{row[0][1]:.1f} h (incl. 2x300 s serve stop/boot)")
# incremental: median / p90 commit -> windows to retrain
inc = cr["incremental"]
for q in ("median", "p90"):
    dirty = inc["dirty_rows_per_commit"][q]
    print(f"incremental {q}: {dirty:.0f} dirty rows/commit")
