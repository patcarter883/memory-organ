"""Cost of the hook vs the dequant it follows (torch blocked). 200k edited rows, 0.1% hit rate."""
import sys, time, importlib.util
sys.modules["torch"] = None
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/serve/v2-editpath")
from ple_overlay_np import PLEOverlayNP, f32_to_bf16_bits
s = importlib.util.spec_from_file_location("rt", "/home/user/patcarter883/minisglang-rdna4/python/minisgl/weights/row_table.py")
rt = importlib.util.module_from_spec(s); sys.modules["rt"] = rt; s.loader.exec_module(rt)
rng = np.random.default_rng(0); N = 320_001_536; K = 200_000
eids = np.unique(rng.integers(0, N, K)); K = eids.size
for mode in ("bf16", "delta"):
    ov = PLEOverlayNP(160, N, mode)
    pay = f32_to_bf16_bits(rng.standard_normal((K, 160)).astype(np.float32) * 1e-2) if mode == "bf16" else rng.standard_normal((K, 160)).astype(np.float32)
    ov.publish(eids, pay)
    for n in (128, 32768):
        flat = rng.integers(0, N, n).astype(np.int64); flat[: max(1, n // 1000)] = eids[: max(1, n // 1000)]
        fused = rng.integers(0, 256, (n, 130), dtype=np.uint8); fused[:, 120:] = np.frombuffer(np.full(n * 5, 1e-3, np.float16).tobytes(), np.uint8).reshape(n, 10)
        host = np.empty((n, 160), np.float32)
        def dq():
            host[:] = rt.dequant_int6_g32(fused[:, :120], np.ascontiguousarray(fused[:, 120:]).view("<f2"))
        def ap(): ov.apply(flat, host)
        for f, name in ((dq, "int6 dequant"), (ap, f"overlay[{mode}]")):
            f(); t = time.perf_counter(); r = 20 if n > 1000 else 300
            for _ in range(r): f()
            print(f"{name:16s} rows/call {n:6d}: {(time.perf_counter() - t) / r * 1e6:9.1f} us")
