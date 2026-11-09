import torch, time
torch.manual_seed(0)
print("threads", torch.get_num_threads(), "cpu cap", torch.backends.cpu.get_cpu_capability())
for dt in (torch.bfloat16, torch.float32):
    a=torch.randn(320,2560,dtype=dt); w=torch.randn(2560,10240,dtype=dt)
    for _ in range(3): a@w
    n=10; t=time.perf_counter()
    for _ in range(n): a@w
    s=(time.perf_counter()-t)/n
    print(dt, "GEMM 320x2560x10240 %.1f ms = %.1f GFLOP/s"%(s*1e3, 2*320*2560*10240/s/1e9))
# E2B-like dense MLP block fwd + input-grad bwd (weights frozen), bf16
d,ff,N=1536,12288,320
x=torch.randn(N,d,dtype=torch.bfloat16,requires_grad=True)
W1=torch.randn(d,ff,dtype=torch.bfloat16)*0.02; W2=torch.randn(ff,d,dtype=torch.bfloat16)*0.02; W3=torch.randn(d,ff,dtype=torch.bfloat16)*0.02
def blk(x): return (torch.nn.functional.gelu(x@W1)*(x@W3))@W2
for _ in range(2): blk(x).sum().backward()
n=5; t=time.perf_counter()
for _ in range(n): x.grad=None; blk(x).sum().backward()
s=(time.perf_counter()-t)/n; fl=4*N*3*d*ff
print("gated MLP fwd+inputgrad bf16 N=320 d=1536 ff=12288: %.1f ms = %.1f GFLOP/s (4*N*params)"%(s*1e3, fl/s/1e9))
