"""CPU microbenchmarks for option (A): a separate, torch-based edit-solve process on the serve host.

Everything here is OUR code (the solve process), not engine code. The fork is used only as the
FORMAT reference: quant/mxfp4.py::dequant_reference is imported by path to prove our fast dequant
is bit-identical to the fork's golden decode.

Geometry is the real q4e-mxfp4 one (headers: ../v2-solve-local/ckpt_inventory.txt):
  experts.gate_up_proj_packed U8 (512, 1280, 1280)   gate_up_proj_scale U8 (512, 1280, 80)
  experts.down_proj_packed    U8 (512, 2560,  320)   down_proj_scale    U8 (512, 2560, 20)
  -> one expert = 2,611,200 B compact, 4,915,200 weights.

Measures (median of reps, all threads of this box):
  1. dequant one expert -> bf16: fork reference / LUT-pair eager / torch.compile
  2. expert forward + input-gradient backward at M routed tokens, weights materialised once
  3. a fused torch.compile "dequant-inside-the-GEMV" (no bf16 weight materialised) at small M
  4. dense GEMM rates at the solve's token count (M=256) and FP8-block dequant of a 10240x2560 matrix
  5. HF's torch_chunk_gated_delta_rule fwd+bwd at real GDN dims
"""
import importlib.util, os, statistics, sys, time
import torch
import torch.nn.functional as F

THREADS = int(os.environ.get("T", torch.get_num_threads()))
torch.set_num_threads(THREADS)
spec = importlib.util.spec_from_file_location(
    "fork_mxfp4", "/home/user/patcarter883/minisglang-rdna4/python/minisgl/quant/mxfp4.py")
fork = importlib.util.module_from_spec(spec); spec.loader.exec_module(fork)

H, I = 2560, 640
g = torch.Generator().manual_seed(0)
def rnd_expert():
    gu_p = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, generator=g)
    gu_s = torch.randint(117, 125, (2 * I, H // 32), dtype=torch.uint8, generator=g)
    dn_p = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, generator=g)
    dn_s = torch.randint(117, 125, (H, I // 32), dtype=torch.uint8, generator=g)
    return gu_p, gu_s, dn_p, dn_s

CPU_TIMES = []
def timeit(fn, reps=7, warm=2):
    """MIN wall over reps (the box is shared with another agent's job, so min, not median, is the
    uncontended estimate); the matching per-thread CPU time is kept in CPU_TIMES (exact at T=1)."""
    for _ in range(warm): fn()
    ts, cs = [], []
    for _ in range(reps):
        c = time.thread_time(); t = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t); cs.append(time.thread_time() - c)
    CPU_TIMES.append(min(cs))
    return min(ts)

# ---- our dequant: one 256-entry table of (lo, hi) bf16 PAIRS packed in an int32 ---------------------
LUT = torch.tensor(fork.FP4_E2M1_LUT, dtype=torch.float32)
pair_bf16 = torch.stack([LUT[torch.arange(256) & 0xF], LUT[torch.arange(256) >> 4]], -1).to(torch.bfloat16)
PAIR_I32 = pair_bf16.contiguous().view(torch.int32).view(256)          # little-endian: lo = even k

def e8m0_to_f32(s):                       # 2^(s-127) exactly: shift the byte into the fp32 exponent
    return (s.to(torch.int32) << 23).view(torch.float32)               # s=0 -> 0.0 (true 2^-127; harmless)

def deq_pair(p, s):                       # (N, K/2) u8, (N, K/32) u8 -> (N, K) bf16, exact
    N, K2 = p.shape
    w = PAIR_I32[p.to(torch.int64)].view(torch.bfloat16).view(N, K2 * 2 // 32, 32)
    return (w * e8m0_to_f32(s).to(torch.bfloat16).unsqueeze(-1)).view(N, K2 * 2)

def deq_simple(p, s):                     # the plain spelling, for torch.compile
    lo = (p & 0xF).to(torch.int64); hi = (p >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], -1).view(p.shape[0], -1)
    w = LUT[codes].view(p.shape[0], -1, 32) * e8m0_to_f32(s).unsqueeze(-1)
    return w.view(p.shape[0], -1).to(torch.bfloat16)

out = []
def say(x):
    print(x, flush=True); out.append(x)

say(f"threads={THREADS}  torch {torch.__version__}  cpu_capability {torch.backends.cpu.get_cpu_capability()}")
gu_p, gu_s, dn_p, dn_s = rnd_expert()
ref = fork.dequant_reference(gu_p, gu_s)
assert torch.equal(deq_pair(gu_p, gu_s).float(), ref) and torch.equal(deq_simple(gu_p, gu_s).float(), ref)
assert torch.equal(deq_pair(dn_p, dn_s).float(), fork.dequant_reference(dn_p, dn_s))
say("parity: our LUT-pair and simple dequant == fork quant/mxfp4.py::dequant_reference, bit-exact (bf16 holds E2M1*2^e exactly)")

def deq_expert(f):
    return f(gu_p, gu_s), f(dn_p, dn_s)
t_ref = timeit(lambda: (fork.dequant_reference(gu_p, gu_s), fork.dequant_reference(dn_p, dn_s)), reps=3)
t_pair = timeit(lambda: deq_expert(deq_pair))
deq_c = torch.compile(deq_simple, dynamic=False)
t0 = time.perf_counter(); deq_expert(deq_c); t_comp_first = time.perf_counter() - t0
assert torch.equal(deq_c(gu_p, gu_s).float(), ref)
t_comp = timeit(lambda: deq_expert(deq_c))
say(f"[1] dequant 1 expert -> bf16:  fork reference(fp32) {t_ref*1e3:7.2f} ms | LUT-pair eager {t_pair*1e3:6.2f} ms | "
    f"torch.compile {t_comp*1e3:6.2f} ms (first call incl. compile {t_comp_first:.1f} s)")

# ---- [2] expert fwd + input-grad bwd, weights materialised ONCE per expert per phase ----------------
Wgu, Wdn = deq_expert(deq_c)
def expert_fwd_bwd(x):                    # x (M, H) bf16, requires grad; weights are constants
    gt, up = (x @ Wgu.t()).chunk(2, -1)
    y = (F.silu(gt) * up) @ Wdn.t()
    y.backward(torch.ones_like(y))
for dt in (torch.bfloat16, torch.float32):
    Wgu, Wdn = (w.to(dt) for w in deq_expert(deq_c))
    row = []
    for M in (1, 2, 4, 8, 32):
        x = torch.randn(M, H, dtype=dt, requires_grad=True)
        row.append(f"M={M}: {timeit(lambda: expert_fwd_bwd(x))*1e3:6.2f}")
    say(f"[2] expert fwd+dX-bwd matmuls only ({str(dt)[6:]}, W pre-dequantised) ms: " + "  ".join(row))

# ---- [3] fused: dequant inside the reduction, never materialise W -------------------------------------
def gemv_fused(x, p, s):                  # x (M, K) f32; y = x @ deq(p,s).T without a (N,K) weight in memory
    N = p.shape[0]
    lo = LUT[(p & 0xF).to(torch.int64)]; hi = LUT[(p >> 4).to(torch.int64)]
    sc = e8m0_to_f32(s).repeat_interleave(16, -1)                    # per packed byte
    xe, xo = x[:, 0::2], x[:, 1::2]                                  # (M, K/2)
    return ((lo * sc)[None] * xe[:, None, :] + (hi * sc)[None] * xo[:, None, :]).sum(-1)
def gemv_fused_T(gy, p, s):               # dX = gy @ deq(p,s): reduction over N
    lo = LUT[(p & 0xF).to(torch.int64)]; hi = LUT[(p >> 4).to(torch.int64)]
    sc = e8m0_to_f32(s).repeat_interleave(16, -1)
    ge = (gy[:, :, None] * (lo * sc)[None]).sum(1); go = (gy[:, :, None] * (hi * sc)[None]).sum(1)
    return torch.stack([ge, go], -1).flatten(1)
fz, fzT = torch.compile(gemv_fused, dynamic=False), torch.compile(gemv_fused_T, dynamic=False)
for M in (1, 4):
    x = torch.randn(M, H); Wf = ref.float()
    t0 = time.perf_counter(); yv = fz(x, gu_p, gu_s); tc = time.perf_counter() - t0
    err = (yv - x @ Wf.t()).abs().max() / (x @ Wf.t()).abs().max()
    gy = torch.randn(M, 2 * I); gx = fzT(gy, gu_p, gu_s)
    errT = (gx - gy @ Wf).abs().max() / (gy @ Wf).abs().max()
    tf = timeit(lambda: fz(x, gu_p, gu_s)); tfT = timeit(lambda: fzT(gy, gu_p, gu_s))
    say(f"[3] fused compiled MXFP4 GEMV gate_up (1280x2560) M={M}: fwd {tf*1e3:6.2f} ms, dX {tfT*1e3:6.2f} ms "
        f"(rel err {err:.1e}/{errT:.1e}; first call {tc:.0f} s)")

# ---- [4] dense parts ------------------------------------------------------------------------------
for dt in (torch.bfloat16, torch.float32):
    a = torch.randn(256, 2560, dtype=dt); b = torch.randn(10240, 2560, dtype=dt)
    t = timeit(lambda: a @ b.t())
    say(f"[4] dense GEMM 256x2560 @ 2560x10240 {str(dt)[6:]}: {t*1e3:6.2f} ms = {2*256*2560*10240/t/1e9:6.1f} GFLOP/s")
w8 = torch.randn(10240, 2560).to(torch.float8_e4m3fn); sinv = torch.rand(80, 20).to(torch.bfloat16)
def fp8_block_deq():                      # quant/method.py:1117-1121 arithmetic: w_fp8 * scale_inv per 128x128
    sc = sinv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (w8.float() * sc).to(torch.bfloat16)
say(f"[4] FP8-block [128,128] dequant 10240x2560 -> bf16: {timeit(fp8_block_deq, reps=5)*1e3:6.1f} ms")

# ---- [5] GDN core at real dims (48 v-heads, dk=dv=128) -----------------------------------------------
import transformers.models.qwen4_exp.modeling_qwen4_exp as M_
fn = M_.torch_chunk_gated_delta_rule
fn = getattr(fn, "__wrapped__", fn)
for B, T in ((8, 32), (4, 16)):
    if os.environ.get("NOGDN"): break
    q = torch.randn(B, T, 48, 128, requires_grad=True); k = torch.randn(B, T, 48, 128, requires_grad=True)
    v = torch.randn(B, T, 48, 128, requires_grad=True)
    gg = (-torch.rand(B, T, 48)).requires_grad_(); be = torch.rand(B, T, 48).requires_grad_()
    def gdn():
        o, _ = fn(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
        o.sum().backward()
    say(f"[5] HF torch_chunk_gated_delta_rule fwd+bwd B={B} T={T}: {timeit(gdn, reps=3)*1e3:6.1f} ms")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bench_solve_cpu.T{THREADS}.txt"), "w").write("\n".join(out) + "\n")
