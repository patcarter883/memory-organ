"""Per-iteration cost model for the solve venues. Every constant is an ASSUMPTION or a cited
measurement; change them here, not in prose. One iteration = forward + backward (input grads only)
through decoder layers 1..47 (layer 0's output is constant across iterations and cached once).
Routed experts are touched TWICE per iteration (forward; backward re-reads + re-dequantises)."""
L = 47                       # layers 1..47
EXP_B = 2_611_200            # compact MXFP4 bytes/expert (tools/offload/p9_elastic_slot_growth.py:10)
EXP_EL = 4_915_200
DENSE_PARAMS_TOK = 3.6e9     # non-routed params touched per token, layers 1-47, no lm_head (GDN/attn/HC/shared/router)
def distinct(tokens, skew=0.9):          # uniform-routing expectation x skew factor (real routing is skewed)
    return 512 * (1 - (1 - 10 / 512) ** tokens) * (skew if tokens < 400 else 1.0)

def cpu_iter(tokens, disk_gbs, deq_ms, native_gbs, dense_gflops):
    D = distinct(tokens)
    uses = 2 * L * D                                   # expert touches per iteration
    io = uses * EXP_B / disk_gbs / 1e9
    comp = uses * deq_ms / 1e3 if native_gbs is None else uses * EXP_B / native_gbs / 1e9
    dense = tokens * DENSE_PARAMS_TOK * 2 * 2 / (dense_gflops * 1e9)   # fwd + bwd-input, no recompute
    return D, io, comp, dense, max(io, comp) + dense   # I/O overlapped with expert compute

def gpu_iter(tokens, h2d_gbs, overhead_s):
    D = distinct(tokens)
    return D, 2 * L * D * EXP_B / h2d_gbs / 1e9 + overhead_s

print("tokens/iter  D/layer | A-torch(compile dequant)        | A-native(fused GEMV+GEMV^T)    | B local GPUs | C cloud")
for tok in (64, 256, 1024, 2048):
    rows = []
    for (disk, deq, nat, gf) in ((4.5, 1.0, None, 500), (3.0, 2.5, None, 300)):
        rows.append(cpu_iter(tok, disk, deq, nat, gf))
    nat = [cpu_iter(tok, 4.5, None, 35, 500), cpu_iter(tok, 3.0, None, 25, 300)]
    D = rows[0][0]
    b = [gpu_iter(tok, 43.4, 1.0)[1], gpu_iter(tok, 28.0, 3.0)[1]]
    c = (1.0 + tok / 2048, 3.0 + 2 * tok / 2048)
    print(f"{tok:10d} {D:8.0f} | io {rows[0][1]:4.0f}-{rows[1][1]:3.0f}s deq {rows[0][2]:4.0f}-{rows[1][2]:4.0f}s dense {rows[0][3]:3.0f}-{rows[1][3]:3.0f}s"
          f" => {rows[0][4]:4.0f}-{rows[1][4]:4.0f}s | io {nat[0][1]:3.0f}-{nat[1][1]:3.0f}s => {nat[0][4]:3.0f}-{nat[1][4]:3.0f}s"
          f" | {b[0]:4.1f}-{b[1]:4.1f}s | {c[0]:.1f}-{c[1]:.1f}s")
print("\nAssumptions: disk = O_DIRECT from the checkpoint NVMe 4.5 / 3.0 GB/s (ckpt_read.py:60-63, 89-98);")
print(" deq = torch.compile'd MXFP4 dequant 1.0 / 2.5 ms per expert on 4 Zen4 cores (sandbox: 2.52 ms at 2 SPR vCPUs);")
print(" native = tools/cpu_moe MXFP4 policy at 35 / 25 GB/s effective under the serve's DDR load (cpu_tier.py:271, 420);")
print(" dense bf16 GEMM 500 / 300 GFLOP/s on 4 Zen4 cores (ASSUMED, not measured); skew 0.9 on distinct experts;")
print(" B: H2D 43.4 GB/s both cards concurrently (28.93+14.48, cpu_tier.py:6) or 28 GB/s, + 1-3 s eager overhead;")
print(" C: weights resident in HBM, Python/launch-bound; 1-3 s + size term (ESTIMATE).")
