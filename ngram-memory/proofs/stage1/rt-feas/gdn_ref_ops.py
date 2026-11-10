"""Count aten ops (≈ GPU kernel launches) and CPU time of the fork's GDN training backward, which is a
pure-torch reference recompute (tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py:305-347, 480-535;
no native GDN-core backward op exists, only causal_conv1d_bwd / rmsnorm_gated_bwd at :369, :403).
Shapes = Qwen3.6-35B-A3B / Qwen3.5-4B GDN: H=16 key heads, HV=32 value heads, K=V=128.
Usage: python -I gdn_ref_ops.py
"""
import re, sys, time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
SRC = "/home/user/patcarter883/minisglang-rdna4/tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py"
src = open(SRC).read()
src = src.replace("from ._ops import add_op_namespace_prefix, ops", "ops = None")
cut = src.index("def _native_bwd_ok")
ns = {"__name__": "gdnref"}
exec(compile(src[:cut], SRC, "exec"), ns)
ref = ns["ref_gdn_prefill_core"]
torch.set_num_threads(4)

class Count(TorchDispatchMode):
    def __init__(s): super().__init__(); s.n = 0
    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        s.n += 1
        return func(*args, **(kwargs or {}))

H, HV, K = 16, 32, 128
def run(B, T, count=True):
    # batched path folds B into heads with chunk = max(4, 64//B) (autograd.py:516-517)
    cw = max(4, 64 // max(1, B)) if B > 1 else 64
    Tt = T
    q = torch.randn(Tt, H * B, K, requires_grad=True); k = torch.randn(Tt, H * B, K, requires_grad=True)
    v = torch.randn(Tt, HV * B, K, requires_grad=True)
    a = torch.randn(Tt, HV * B, requires_grad=True); b = torch.randn(Tt, HV * B, requires_grad=True)
    A = torch.randn(HV * B, requires_grad=True); d = torch.randn(HV * B, requires_grad=True)
    m = Count()
    t0 = time.time()
    with m:
        o = ref(q, k, v, a, b, A, d, K ** -0.5, True, chunk=cw)
        g = torch.autograd.grad(o, (q, k, v, a, b, A, d), torch.randn_like(o))
    return m.n, time.time() - t0, cw

import os
CASES = eval(os.environ.get("CASES", "[(1, 512), (1, 1024), (4, 256), (4, 512)]"))
for B, T in CASES:
    n, dt, cw = run(B, T)
    print(f"B={B} T={T} chunk={cw}: aten ops fwd+bwd = {n:,}  ({n/(B*T):.1f} per token)  cpu {dt:.1f}s ({dt/(B*T)*1e3:.2f} ms/token)", flush=True)
