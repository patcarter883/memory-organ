"""Faster MXFP4 / FP8-block dequant spellings for OUR solve process (torch.compile, no custom kernel).

E2M1 -> bf16 by INTEGER bit assembly instead of a table gather: code c = s|e1e0|m;
  value = 0                     if (c&7)==0
        = 0.5 * 2^x             if e==0, m==1            -> bf16 exponent 126 + x, mantissa 0
        = (1+m/2) * 2^(e-1+x)   if e>=1                  -> bf16 exponent 126 + e + x, mantissa bit6 = m
with x = s_e8m0 - 127 folded straight into the exponent field (E8M0's bias IS fp32/bf16's bias,
quant/mxfp4.py:101-114 at df29117). Exact; checked against the fork's dequant_reference.
Assumes 1 <= 126+e+x <= 254 (no bf16 subnormal/overflow), i.e. block exponents in [-125, 126]; a real
loader asserts that per tensor once (cheap: min/max of the scale bytes)."""
import importlib.util, os, time
import torch
T = int(os.environ.get("T", "1")); torch.set_num_threads(T)
spec = importlib.util.spec_from_file_location(
    "fork_mxfp4", "/home/user/patcarter883/minisglang-rdna4/python/minisgl/quant/mxfp4.py")
fork = importlib.util.module_from_spec(spec); spec.loader.exec_module(fork)

def deq_bits(p, s):                        # (N, K/2) u8, (N, K/32) u8 -> (N, K) bf16
    N, K2 = p.shape
    c = torch.stack([p & 0xF, p >> 4], -1).view(N, K2 * 2 // 32, 32).to(torch.int16)
    x = s.to(torch.int16).unsqueeze(-1) - 127
    e = (c >> 1) & 3; m = c & 1
    exp = torch.where(e == 0, 126 + x, 126 + e + x)
    man = torch.where(e == 0, 0, m << 6)
    bits = ((c & 8) << 12) | (exp << 7) | man
    bits = torch.where((c & 7) == 0, (c & 8) << 12, bits)
    return bits.view(torch.bfloat16).view(N, K2 * 2)

def fp8_block_deq(w8, sinv):               # quant/method.py:1117-1121 arithmetic, 128x128 tiles
    N, K = w8.shape
    w = w8.float().view(N // 128, 128, K // 128, 128)
    return (w * sinv.float()[:, None, :, None]).view(N, K).to(torch.bfloat16)

def best(fn, reps=7):
    fn(); fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

g = torch.Generator().manual_seed(0)
gu_p = torch.randint(0, 256, (1280, 1280), dtype=torch.uint8, generator=g)
gu_s = torch.randint(117, 125, (1280, 80), dtype=torch.uint8, generator=g)
dn_p = torch.randint(0, 256, (2560, 320), dtype=torch.uint8, generator=g)
dn_s = torch.randint(117, 125, (2560, 20), dtype=torch.uint8, generator=g)
dc = torch.compile(deq_bits, dynamic=False)
for p, s in ((gu_p, gu_s), (dn_p, dn_s)):
    assert torch.equal(dc(p, s).float(), fork.dequant_reference(p, s)), "bit assembly != fork reference"
    assert torch.equal(dc(p, s).view(torch.int16), fork.dequant_reference(p, s).to(torch.bfloat16).view(torch.int16))
t_e = best(lambda: (deq_bits(gu_p, gu_s), deq_bits(dn_p, dn_s)), 3)
t_c = best(lambda: (dc(gu_p, gu_s), dc(dn_p, dn_s)))
print(f"threads={T}  MXFP4 expert -> bf16 by bit assembly: eager {t_e*1e3:.2f} ms, compiled {t_c*1e3:.2f} ms "
      f"({4_915_200 / t_c / 1e9:.2f} Gel/s, {2_611_200 / t_c / 1e9:.2f} GB/s compact in); bit-exact vs fork incl. sign of zero")

w8 = (torch.randn(10240, 2560) * 0.05).to(torch.float8_e4m3fn); sinv = torch.rand(80, 20).to(torch.bfloat16)
fc = torch.compile(fp8_block_deq, dynamic=False)
ref = (w8.float() * sinv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.bfloat16)
assert torch.equal(fc(w8, sinv), ref)
print(f"threads={T}  FP8-block 10240x2560 -> bf16: eager {best(lambda: fp8_block_deq(w8, sinv), 3)*1e3:.1f} ms, "
      f"compiled {best(lambda: fc(w8, sinv))*1e3:.1f} ms")

# the table spelling from bench_solve_cpu.py (256-entry (lo,hi) bf16 pair table in an int32), compiled
LUT = torch.tensor(fork.FP4_E2M1_LUT, dtype=torch.float32)
PAIR_I32 = torch.stack([LUT[torch.arange(256) & 0xF], LUT[torch.arange(256) >> 4]], -1).to(torch.bfloat16).contiguous().view(torch.int32).view(256)
def deq_pair(p, s):
    N, K2 = p.shape
    w = PAIR_I32[p.to(torch.int64)].view(torch.bfloat16).view(N, K2 * 2 // 32, 32)
    sc = (s.to(torch.int32) << 23).view(torch.float32).to(torch.bfloat16).unsqueeze(-1)
    return (w * sc).view(N, K2 * 2)
dp = torch.compile(deq_pair, dynamic=False)
assert torch.equal(dp(gu_p, gu_s).float(), fork.dequant_reference(gu_p, gu_s))
t_p = best(lambda: (dp(gu_p, gu_s), dp(dn_p, dn_s)))
print(f"threads={T}  MXFP4 expert -> bf16 by pair table, compiled {t_p*1e3:.2f} ms ({4_915_200 / t_p / 1e9:.2f} Gel/s)")
if t_p < t_c: dc = dp

# a whole layer's touched experts, the way the solve would run them: dequant + fwd + dX-bwd per expert
import torch.nn.functional as F
def layer_cost(D, tok_per_expert):
    t = time.perf_counter()
    for _ in range(D):
        Wgu, Wdn = dc(gu_p, gu_s), dc(dn_p, dn_s)
        x = torch.randn(tok_per_expert, 2560, dtype=torch.bfloat16, requires_grad=True)
        gt, up = (x @ Wgu.t()).chunk(2, -1)
        y = (F.silu(gt) * up) @ Wdn.t(); y.backward(torch.ones_like(y))
    return time.perf_counter() - t
for D, m in ((40, 2), (40, 6)):
    layer_cost(4, m)
    tl = min(layer_cost(D, m) for _ in range(2))
    print(f"threads={T}  per touched expert, dequant(compiled)+fwd+dX-bwd at {m} tok: {tl / D * 1e3:.2f} ms")
