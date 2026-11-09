"""CPU micro-bench for option (A): MXFP4 expert dequant in torch + expert fwd/bwd (input grad only).

Geometry = Qwen3.8-Flash-Next routed expert in the q4e-mxfp4 (tcclaviger MXFP4-FP8-GPTQ) checkpoint:
  gate_up_proj_packed [1280, 1280] u8 + scale [1280, 80] u8 E8M0  (gate_up: N=1280, K=2560)
  down_proj_packed    [2560,  320] u8 + scale [2560, 20] u8 E8M0  (down:    N=2560, K=640)
  = 2,611,200 B/expert compact, 4,915,200 weights.
Dequant convention = minisgl quant/mxfp4.py:262-275 (dequant_reference): low nibble = even K,
E2M1 LUT, scale 2^(s-127). Checked bit-exact against that reference below (imported by path).
"""
import sys, time, math, importlib.util, os
import torch, torch.nn.functional as F

REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl/quant/mxfp4.py"
spec = importlib.util.spec_from_file_location("mxfp4_ref", REPO)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

torch.manual_seed(0)
H, I = 2560, 640
THREADS = int(os.environ.get("T", torch.get_num_threads()))
torch.set_num_threads(THREADS)

def bench(f, n=5, warm=1):
    for _ in range(warm): f()
    s = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - s) / n

E2M1 = torch.tensor(ref.FP4_E2M1_LUT, dtype=torch.float32)
# byte -> (even, odd) pair LUT: one gather per BYTE instead of two nibble gathers
PAIR = torch.stack([E2M1[torch.arange(256) & 0xF], E2M1[torch.arange(256) >> 4]], -1)  # [256,2] f32

def deq_fast(packed, scale, dtype=torch.bfloat16):
    """packed [N, K/2] u8, scale [N, K/32] u8 -> [N, K] dtype. Exact for E8M0 1..254."""
    N, Kh = packed.shape
    lut = PAIR.to(dtype)
    w = lut[packed.long()].view(N, Kh * 2 // 32, 32)            # [N, G, 32]
    s = (scale.to(torch.int32) << 23).view(torch.float32).to(dtype)  # 2^(s-127) as fp32 bit pattern
    return (w * s.unsqueeze(-1)).view(N, Kh * 2)

def rand_mx(N, K):
    p = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    s = torch.randint(115, 125, (N, K // 32), dtype=torch.uint8)  # realistic 2^-12..2^-3 window
    return p, s

gu_p, gu_s = rand_mx(2 * I, H)
dn_p, dn_s = rand_mx(H, I)

# --- parity vs the fork's reference -----------------------------------------------------------
r = ref.dequant_reference(gu_p, gu_s)
f32 = deq_fast(gu_p, gu_s, torch.float32)
assert torch.equal(r, f32), "fast dequant != quant/mxfp4.py::dequant_reference"
print(f"parity: deq_fast(fp32) == mxfp4.dequant_reference bit-exact on [{2*I},{H}]")
print(f"threads={THREADS}")

# --- calibration: this box's GEMM and stream rates ---------------------------------------------
A = torch.randn(2048, 2048); B = torch.randn(2048, 2048)
s = bench(lambda: A @ B, n=3); print(f"sgemm 2048^3: {2*2048**3/s/1e9:.1f} GFLOP/s")
Ab, Bb = A.bfloat16(), B.bfloat16()
s = bench(lambda: Ab @ Bb, n=3); print(f"bf16 gemm 2048^3: {2*2048**3/s/1e9:.1f} GFLOP/s")
big = torch.empty(256 * 2**20 // 4); dst = torch.empty_like(big)
s = bench(lambda: dst.copy_(big), n=5); print(f"stream copy 256 MiB: {2*big.numel()*4/s/1e9:.1f} GB/s (r+w)")

# --- dequant of one whole expert --------------------------------------------------------------
for dt in (torch.bfloat16, torch.float32):
    s = bench(lambda: (deq_fast(gu_p, gu_s, dt), deq_fast(dn_p, dn_s, dt)), n=5)
    print(f"MXFP4 dequant 1 expert -> {str(dt)[6:]}: {s*1e3:.2f} ms  ({4.9152e6/s/1e9:.2f} Gel/s, "
          f"{2.6112e6/s/1e9:.2f} GB/s compact in)")
s = bench(lambda: (ref.dequant_reference(gu_p, gu_s), ref.dequant_reference(dn_p, dn_s)), n=3)
print(f"  (fork dequant_reference, fp32: {s*1e3:.2f} ms)")

# --- expert fwd + bwd(input grad only), dequant included in BOTH passes ------------------------
class DeqLinear(torch.autograd.Function):
    """y = x W^T with W re-dequantised in backward (nothing weight-sized saved)."""
    @staticmethod
    def forward(ctx, x, p, s):
        ctx.save_for_backward(p, s)
        return x @ deq_fast(p, s, x.dtype).t()
    @staticmethod
    def backward(ctx, gy):
        p, s = ctx.saved_tensors
        return gy @ deq_fast(p, s, gy.dtype), None, None

def expert(x):
    g, u = DeqLinear.apply(x, gu_p, gu_s).chunk(2, -1)
    return DeqLinear.apply(F.silu(g) * u, dn_p, dn_s)

for dt in (torch.bfloat16, torch.float32):
    for m in (1, 2, 4, 16, 64):
        x = torch.randn(m, H, dtype=dt, requires_grad=True)
        def step():
            x.grad = None
            expert(x).float().square().sum().backward()
        s = bench(step, n=4)
        fl = 2 * 2 * m * 4.9152e6
        print(f"expert fwd+bwd {str(dt)[6:]:8s} m={m:3d}: {s*1e3:7.2f} ms  ({fl/s/1e9:6.1f} GFLOP/s)")

# --- a whole layer's routed experts the torch way (HF Experts loop pattern), fwd+bwd -----------
for n_exp, tok in ((128, 2), (330, 2)):
    # one expert's weights reused n_exp times = compute-only cost (disk is costed separately)
    xs = [torch.randn(tok, H, dtype=torch.bfloat16, requires_grad=True) for _ in range(min(n_exp, 32))]
    def lay():
        for i in range(n_exp):
            x = xs[i % len(xs)]; x.grad = None
            expert(x).float().sum().backward()
    s = bench(lay, n=1, warm=0)
    print(f"layer: {n_exp} experts x {tok} tok, bf16 fwd+bwd: {s:.2f} s  ({s/n_exp*1e3:.2f} ms/expert)")
