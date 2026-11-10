# 1-thread CPU timing of HF's pure-torch chunked gated-delta rule (fwd + input-grad bwd) at real head dims,
# vs a bf16 GEMM of the same layer's projections, to size the GDN share of a CPU training step.
import time, torch, json
torch.set_num_threads(1)
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as gdr
res = {}
for host, Hv, d, Hk in (("35B-A3B", 32, 2048, 16), ("27B", 48, 5120, 16)):
    T = 512; dk = dv = 128
    q = torch.randn(1, T, Hv, dk, requires_grad=True); k = torch.randn(1, T, Hv, dk, requires_grad=True)
    v = torch.randn(1, T, Hv, dv, requires_grad=True)
    g = (-torch.rand(1, T, Hv)*0.1).requires_grad_(); b = torch.rand(1, T, Hv).requires_grad_()
    def run():
        o, _ = gdr(q, k, v, g=g, beta=b, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        o.float().sum().backward()
    run()
    t = time.perf_counter(); run(); tg = time.perf_counter() - t
    # same layer's linear projections (in_proj_qkvz+ba ~ d*(2*Hk*dk + 2*Hv*dv) , out_proj Hv*dv*d), fwd + input-grad
    nproj = d*(2*Hk*dk + 2*Hv*dv + 2*Hv) + Hv*dv*d
    x = torch.randn(T, d, dtype=torch.bfloat16); W = torch.randn(nproj // d, d, dtype=torch.bfloat16)
    y = x @ W.T; t = time.perf_counter(); y = x @ W.T; gx = y @ W; tl = time.perf_counter() - t
    res[host] = dict(T=T, gdn_core_fwd_bwd_s=round(tg, 3), proj_fwd_inputgrad_s=round(tl, 3),
                     proj_GFLOP=round(4*T*nproj/1e9, 2), proj_GFLOPs_rate=round(4*T*nproj/tl/1e9, 1))
    print(host, res[host], flush=True)
json.dump(res, open("gdn_cpu_bench.json", "w"), indent=1)
