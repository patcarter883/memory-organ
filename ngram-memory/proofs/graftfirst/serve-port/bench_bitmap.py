"""Prefill-chunk membership prefilter: a 1-bit-per-logical-row bitmap (320,001,536 rows -> 40 MB) of 'some loaded
cartridge holds this row', checked before the composite-key searchsorted. numpy only, 1 thread. Median us."""
import time, json, numpy as np
rng = np.random.default_rng(0)
NROWS = 320_001_536
bm = np.zeros(NROWS // 8 + 1, dtype=np.uint8)
out = {}
for U in (160_000, 1_600_000, 16_000_000):
    ids = rng.integers(0, NROWS, size=U, dtype=np.int64)
    bm[:] = 0; np.bitwise_or.at(bm, ids >> 3, (1 << (ids & 7)).astype(np.uint8))
    keys = np.sort(ids)
    for N in (32_768, 131_072):
        q = rng.integers(0, NROWS, size=N, dtype=np.int64)
        def f(): return (bm[q >> 3] >> (q & 7).astype(np.uint8)) & 1
        def s(): return np.searchsorted(keys, q)
        tf = []; ts = []
        for _ in range(9):
            t0 = time.perf_counter(); f(); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); s(); ts.append(time.perf_counter() - t0)
        out[f"U{U}/N{N}"] = {"bitmap_us": round(np.median(tf) * 1e6), "searchsorted_us": round(np.median(ts) * 1e6),
                             "frac_maybe_hit": float(f().mean())}
print(json.dumps(out, indent=1))
