"""Same op count for HF's pure-torch fallback torch_chunk_gated_delta_rule (q35moe:289-422), which the
design names as the reference and as the prefix-cached path. 35B-A3B shapes, HV=32 (q,k repeated), K=V=128.
Also re-counts the fork's batched reference at larger B (fold), for the 16k-64k-token per-layer chunks."""
import time, torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
torch.set_num_threads(4)
f = M.torch_chunk_gated_delta_rule
f = getattr(f, "__wrapped__", f)
class Count(TorchDispatchMode):
    def __init__(s): super().__init__(); s.n = 0
    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        s.n += 1; return func(*args, **(kwargs or {}))
HV, K = 32, 128
for B, T in [(1, 1024), (4, 1024)]:
    q = torch.randn(B, T, HV, K, requires_grad=True); k = torch.randn(B, T, HV, K, requires_grad=True)
    v = torch.randn(B, T, HV, K, requires_grad=True)
    g = (-torch.rand(B, T, HV)).requires_grad_(); beta = torch.rand(B, T, HV).requires_grad_()
    m = Count(); t0 = time.time()
    with m:
        o, _ = f(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        torch.autograd.grad(o, (q, k, v, g, beta), torch.randn_like(o))
    print(f"HF torch_chunk B={B} T={T}: aten ops fwd+bwd = {m.n:,} ({m.n/(B*T):.2f} per token) cpu {time.time()-t0:.1f}s", flush=True)
