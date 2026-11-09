"""Expected DISTINCT routed experts per layer touched by N tokens under top-10/512 routing.
uniform: closed form 512(1-(1-10/512)^N).
profile: the checkpoint's own measured usage profile (model-expertprofile.safetensors,
`expert_routing_counts` F32 [48,512], 2751 prompts; weights.py:1607-1619 at df29117 documents it),
tokens drawn INDEPENDENTLY by Gumbel-top-10 on log(frequency). Real prompts are more correlated than
independent draws (same topic, adjacent tokens), so 'profile' is still an upper bound on D."""
import json, struct, sys, numpy as np
p = sys.argv[1]
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
cnt = np.frombuffer(f.read(), dtype=np.float32).reshape(48, 512).astype(np.float64)
freq = cnt / cnt.sum(1, keepdims=True)
logf = np.log(np.maximum(freq, 1e-12))
rng = np.random.default_rng(0)
top = np.sort(freq, 1)[:, ::-1]
print(f"profile: per-layer share of routing mass in the top 64 experts: median {np.median(top[:, :64].sum(1)):.2f}, "
      f"top 128: {np.median(top[:, :128].sum(1)):.2f}; experts with <0.01% share: median {np.median((freq < 1e-4).sum(1)):.0f}/512")
print(f"{'N tok':>6} {'uniform':>8} {'profile mean':>13} {'min-max layer':>14}  {'MXFP4 GB/pass (48 L, profile)':>30}")
out = {}
for N in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048):
    reps = 6 if N <= 256 else 2
    Ds = np.zeros((reps, 48))
    for r in range(reps):
        g = rng.gumbel(size=(48, N, 512)) + logf[:, None, :]
        idx = np.argpartition(-g, 10, axis=-1)[..., :10]
        for L in range(48):
            Ds[r, L] = np.unique(idx[L]).size
    D = Ds.mean(0); u = 512 * (1 - (1 - 10 / 512) ** N)
    out[N] = (u, D.mean())
    print(f"{N:6d} {u:8.1f} {D.mean():13.1f} {D.min():7.0f}-{D.max():<6.0f}  {48 * D.mean() * 2_611_200 / 1e9:30.1f}")
json.dump({str(k): v for k, v in out.items()}, open(sys.argv[2], "w"))
