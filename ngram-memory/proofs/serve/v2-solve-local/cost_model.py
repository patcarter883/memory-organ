"""Per-iteration / per-edit projection for the four venues, from MEASURED inputs where we have them.

Measured in this directory (4-vCPU SPR sandbox, quiet box, threads=4):
  realdim_layers.T4.txt  real widths, 4 layers, HF modules + our MXFP4 experts, fwd+bwd:
      per touched expert (2 dequants + fwd + dX) 5.01 / 5.09 / 5.49 ms at 32 / 64 / 256 tokens
      everything else per layer: 0.30 / 0.41 / 0.90 s at 32 / 64 / 256 tokens
  distinct_experts.txt   D(N) from the checkpoint's own routing profile (independent tokens -> upper bound)
From the fork at df29117 (cited in the report): O_DIRECT 2.98-4.97 GB/s, QD1 3.3 MiB reads 1.22 GB/s
(weights/ckpt_read.py:60-63,89-98,546-550); PCIe 28.93 / 14.48 GB/s card0/card1 (weights/cpu_tier.py:6);
serve boot 300.1 s (ckpt_read.py:5-10). Everything else is labelled ASSUMED.
"""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
D = {int(k): v[1] for k, v in json.load(open(os.path.join(here, "distinct_experts.json"))).items()}
L, EXPERT_B = 48, 2_611_200

def c_expert(N):            # ms per touched expert per fwd+bwd, measured 5.0-5.5; M per expert grows with N
    return 5.0 if N <= 64 else 5.5 if N <= 256 else 6.5      # 1024: ASSUMED from M=32 matmul bench (+1 ms)
def rest_layer(N):          # s per layer, linear fit through the measured 32/64/256 points
    return 0.25 + 0.0025 * N

def A(N, ckpt):             # local CPU, 4 cores, sandbox-equivalent
    e = L * D[N] * c_expert(N) / 1e3
    r = L * rest_layer(N)
    if ckpt:                # per-layer recompute: +1 dequant+fwd per expert (x1.4), +1 dense fwd (x1.55)
        e, r = e * 1.4, r * 1.55
    io = 2 * L * D[N] * EXPERT_B / 1e9 / 3.0                  # 2 reads per touched expert, 3 GB/s O_DIRECT
    return e + r, io

def B(N):                   # serve down, our torch process on both cards, compact experts in host RAM
    per_pass = L * D[N] * EXPERT_B / 1e9
    h2d = 2 * (per_pass / 2 / 28.93 + per_pass / 2 / 14.48)   # layer-split: half the layers per card
    compute = (1.0, 4.0)                                       # ASSUMED: batched per-layer expert kernels .. eager loop
    return h2d + compute[0], h2d + compute[1]

def C(N):                   # one H200, everything compact-resident in HBM; ASSUMED launch/compute bound
    return (1.0 + N / 1024, 3.0 + 3 * N / 1024)

STEPS = 20 + 1              # Stage-1 Adam steps + ~2 forward-only checks (~0.5 iteration each)
print(f"{'case':38s} {'N':>5} {'D/layer':>7} | {'A s/it (x0.7-1.5)':>18} {'A IO s':>6} | {'B s/it':>9} | {'C s/it':>9}")
# GDN prefix state the prefix-cached driver must hold: 36 layers x 48 heads x 128 x 128 per sequence, bf16
STATE_GIB_PER_SEQ = 36 * 48 * 128 * 128 * 2 / 2**30
for name, N, K, seqs, ck in (("1 edit x 8 prompts, prefix-cached", 24, 1, 8, True),
                             ("1 edit x 8 prompts, full 32-tok seqs", 256, 1, 8, True),
                             ("8 edits x 8 prompts, prefix-cached", 192, 8, 64, True),
                             ("32 edits x 4 prompts, prefix-cached", 384, 32, 128, True),
                             ("32 edits x 8 prompts, prefix-cached", 768, 32, 256, True)):
    print(f"  [{name}: GDN prefix state {seqs * STATE_GIB_PER_SEQ:.1f} GiB]")
    if N not in D:
        lo = max(k for k in D if k < N); hi = min(k for k in D if k > N)
        D[N] = D[lo] + (D[hi] - D[lo]) * (N - lo) / (hi - lo)
    a, io = A(N, ck); b0, b1 = B(N); c0, c1 = C(N)
    print(f"{name:38s} {N:5d} {D[N]:7.0f} | {0.7*a:7.0f}-{1.5*a:<7.0f}    {io:6.0f} | {b0:4.1f}-{b1:<4.1f} | {c0:4.1f}-{c1:<4.1f}"
          f"   per edit: A {STEPS*0.7*a/K/60:5.1f}-{STEPS*1.5*a/K/60:<5.1f} min, B {STEPS*b0/K/60:4.2f}-{STEPS*b1/K/60:.2f} min "
          f"(+ ~10 min downtime per session), C {STEPS*c0/K:4.0f}-{STEPS*c1/K:.0f} s (+ ~10 min setup per session)")
