"""Small CPU micro-benchmarks to scale option (a)/(b) estimates. 4-core sandbox; owner box is 8C Zen AVX-512."""
import time, torch, math
torch.manual_seed(0)
H, I = 2560, 640
def t(f, n=5):
    f(); s=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-s)/n
# expert fwd+bwd(input only) for k tokens, bf16 and fp32
for dt in (torch.bfloat16, torch.float32):
    Wgu = torch.randn(2*I, H, dtype=dt)*0.02; Wd = torch.randn(H, I, dtype=dt)*0.02
    for k in (5, 20):
        x = torch.randn(k, H, dtype=dt, requires_grad=True)
        def f():
            g,u = (x@Wgu.T).chunk(2,-1); y=(torch.nn.functional.silu(g)*u)@Wd.T; y.sum().backward()
        s = t(f); fl = 3*2*k*3*I*H  # fwd + bwd-input ~ 2x fwd flops
        print(f"expert fwd+bwd {dt} k={k}: {s*1e3:.2f} ms  ({fl/s/1e9:.1f} GFLOP/s)")
# NVFP4 dequant of one expert (codes uint8 packed 2/byte + e4m3 scale g16) -> bf16
n_el = 3*I*H
codes = torch.randint(0,256,(n_el//2,),dtype=torch.uint8)
scales = torch.randint(0,120,(n_el//16,),dtype=torch.uint8).view(torch.float8_e4m3fn)
lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0.,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.bfloat16)
def deq():
    lo = lut[(codes & 0xF).long()]; hi = lut[(codes >> 4).long()]
    w = torch.stack([lo,hi],-1).reshape(-1,16) * scales.to(torch.bfloat16).unsqueeze(-1)
    return w
s = t(deq)
print(f"nvfp4 dequant 1 expert ({n_el/1e6:.2f}M el): {s*1e3:.1f} ms -> {n_el/s/1e9:.2f} Gel/s; 512x48 experts = {512*48*s:.0f} s/pass")
# uniform-routing distinct experts
for N in (16, 32, 64, 128, 256, 512, 1024):
    d = 512*(1-(1-10/512)**N)
    print(f"N={N:5d} tokens: E[distinct experts/layer] uniform = {d:6.1f}  ({d/512:.0%})")
print("threads", torch.get_num_threads())
