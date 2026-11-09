"""Overlay apply cost vs hit fraction (gap flagged by the skeptic: prototype forced only 4 hits)."""
import sys, time, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("ov", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng = np.random.default_rng(0); N = 320_001_536
for K in (10_000, 200_000):
    ov = m.PLEOverlay(160, N); ov.put("e", rng.choice(N, K, replace=False), rng.standard_normal((K, 160)).astype(np.float32) * 1e-3)
    ids = ov._compiled[0]
    for n_tok in (8, 2048):
        n = n_tok * 16
        for frac in (0.0, 0.1, 0.5, 1.0):
            flat = rng.integers(0, N, n); h = int(frac * n)
            flat[:h] = rng.choice(ids, h); rows = np.zeros((n, 160), np.float32)
            reps = 2000 if n_tok < 100 else 30
            t = time.perf_counter()
            for _ in range(reps): ov.apply(flat, rows)
            dt = (time.perf_counter() - t) / reps
            print(f"K={K:>7} tokens={n_tok:>5} rows={n:>6} hit={frac:>4.0%}  {dt*1e6:9.1f} us")
    print(f"K={K}: overlay store {sum(v.nbytes+i.nbytes for i,v in ov._edits.values())/2**20:.1f} MiB + compiled {(ov._compiled[0].nbytes+ov._compiled[1].nbytes)/2**20:.1f} MiB (per TP rank)")
