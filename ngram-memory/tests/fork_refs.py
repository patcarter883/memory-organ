"""Read-only loaders for the fork's dequant code (R/quant/mxfp4.py, R/quant/kernels.py at HEAD df29117) without
importing the minisgl package (whose __init__ pulls serve-only dependencies). Returns None if the clone is absent."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch

R = Path("/home/user/patcarter883/minisglang-rdna4/python/minisgl")


def fork_mxfp4_dequant():
    """minisgl.quant.mxfp4.dequant_reference (pure torch module)."""
    p = R / "quant" / "mxfp4.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("fork_mxfp4_readonly", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dequant_reference


def fork_awq_to_op_layout():
    """minisgl.quant.kernels.awq_to_op_layout, extracted from the source by AST (the module itself imports
    engine-side helpers)."""
    p = R / "quant" / "kernels.py"
    if not p.exists():
        return None
    src = p.read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "awq_to_op_layout")
    order = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_REVERSE_AWQ_PACK_ORDER" for t in n.targets))
    ns: dict = {"torch": torch}
    exec(compile("from __future__ import annotations\n" + ast.get_source_segment(src, order) + "\n"
                 + ast.get_source_segment(src, fn), str(p), "exec"), ns)
    return ns["awq_to_op_layout"]


def fork_awq_dequant(qweight, qzeros, scales, group_size):
    """Fork's AWQ path: awq_to_op_layout, then the op's documented decode (code j of word at bits 4j..4j+3
    along K; zero of channel n = nibble n%8 of word n//8 in group-major zeros; w = s * (q - z)). -> fp32 [N, K]."""
    conv = fork_awq_to_op_layout()
    if conv is None:
        return None
    w_packed, s_op, z_op = conv(qweight, scales, qzeros, bits=4)
    N, Kp = w_packed.shape
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = ((w_packed.unsqueeze(-1) >> sh) & 0xF).reshape(N, Kp * 8)          # [N, K]
    G = z_op.shape[0]
    z = ((z_op.unsqueeze(-1) >> sh) & 0xF).reshape(G, N)                       # [G, N]
    g = torch.arange(Kp * 8) // group_size
    return (codes - z[g].T).to(torch.float32) * s_op.float()[g].T
