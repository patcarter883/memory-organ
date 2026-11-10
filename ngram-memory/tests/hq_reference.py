"""Independent numpy reference for the two 4-bit weight formats, written from their specs and sharing no code
with ngram_memory.backends.graft.host_quant (used by tests/test_host_quant.py and scripts/host_quant_validate.py).

AWQ "gemm" (AutoAWQ WQLinear_GEMM.from_linear): intweight [K, N] (input-major), packed per row into N/8 int32
with ``for col in range(N // 8): for i in range(8): qweight[:, col] |= intweight[:, col*8 + order_map[i]] << 4*i``,
order_map = [0, 2, 4, 6, 1, 3, 5, 7]; zeros are packed the same way over [K/g, N]; scales fp16 [K/g, N];
dequant w = (q - z) * s, linear weight = w^T.

MXFP4 (OCP Microscaling spec v1.0): FP4 E2M1 element = 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit;
exponent 0 is subnormal (value = m * 0.5); otherwise 2^(e-1) * (1 + m/2). Scale E8M0 = 2^(x - 127), x = 255
is NaN. Packing (compressed-tensors / Quark 'reorder'): byte j of a row holds element 2j in its low nibble and
element 2j+1 in its high nibble; one scale per 32 consecutive elements of a row.
"""
from __future__ import annotations

import math

import numpy as np

ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def awq_pack_ref(intweight: np.ndarray) -> np.ndarray:
    """[R, N] ints in 0..15 -> [R, N/8] int32 (AutoAWQ's loop, verbatim structure)."""
    R, N = intweight.shape
    out = np.zeros((R, N // 8), dtype=np.uint32)
    for col in range(N // 8):
        for i in range(8):
            out[:, col] |= intweight[:, col * 8 + ORDER_MAP[i]].astype(np.uint32) << np.uint32(4 * i)
    return out.view(np.int32)


def awq_unpack_ref(packed: np.ndarray) -> np.ndarray:
    """Inverse of awq_pack_ref, element by element: channel col*8 + ORDER_MAP[i] = nibble i of word col."""
    p = packed.view(np.uint32)
    R, Np = p.shape
    out = np.zeros((R, Np * 8), dtype=np.int64)
    for col in range(Np):
        for i in range(8):
            out[:, col * 8 + ORDER_MAP[i]] = (p[:, col] >> np.uint32(4 * i)) & np.uint32(0xF)
    return out


def awq_dequant_ref(qweight: np.ndarray, qzeros: np.ndarray, scales_f16: np.ndarray, group_size: int) -> np.ndarray:
    """-> fp32 linear weight [N, K]; every element computed as float32((q - z)) * float32(s)."""
    q = awq_unpack_ref(qweight)                     # [K, N]
    z = awq_unpack_ref(qzeros)                      # [G, N]
    K, N = q.shape
    w = np.empty((K, N), dtype=np.float32)
    s = scales_f16.astype(np.float32)
    for kk in range(K):
        gi = kk // group_size
        w[kk] = (q[kk] - z[gi]).astype(np.float32) * s[gi]
    return w.T.copy()


def e2m1_value(code: int) -> float:
    sign = -1.0 if (code >> 3) & 1 else 1.0
    e = (code >> 1) & 3
    m = code & 1
    mag = m * 0.5 if e == 0 else 2.0 ** (e - 1) * (1.0 + m / 2.0)
    return sign * mag


def e8m0_value(x: int) -> float:
    return float("nan") if x == 255 else math.ldexp(1.0, x - 127)


def mxfp4_dequant_ref(weight_u8: np.ndarray, scale_u8: np.ndarray, group_size: int = 32) -> np.ndarray:
    """uint8 [N, K/2] + uint8 [N, K/32] -> fp32 [N, K] (python-float math, exact in fp32)."""
    N, Kh = weight_u8.shape
    K = 2 * Kh
    lut = np.array([e2m1_value(c) for c in range(16)], dtype=np.float64)
    sc = np.array([e8m0_value(x) for x in range(256)], dtype=np.float64)
    codes = np.empty((N, K), dtype=np.int64)
    codes[:, 0::2] = weight_u8 & 0x0F
    codes[:, 1::2] = weight_u8 >> 4
    vals = lut[codes]
    scale = sc[scale_u8.astype(np.int64)]
    out = vals * np.repeat(scale, group_size, axis=1)
    with np.errstate(over="ignore"):               # 6 * 2^127 overflows fp32 to inf, as it must
        return out.astype(np.float32)


def mxfp4_pack_ref(codes: np.ndarray) -> np.ndarray:
    """[N, K] codes 0..15 -> uint8 [N, K/2] (element 2j low nibble)."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def mxfp4_quantize_ref(w: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest MXFP4 of an fp32 [N, K] matrix (for synthetic checkpoints in tests; not Quark's rounding)."""
    N, K = w.shape
    g = w.reshape(N, K // group_size, group_size)
    amax = np.abs(g).max(-1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30))) - 2, -127)   # 6 = 1.5 * 2^2
    x = np.clip(e + 127, 0, 254).astype(np.uint8)
    scale = np.array([e8m0_value(int(v)) for v in range(256)])[x.astype(np.int64)]
    lut = np.array([e2m1_value(c) for c in range(16)])
    v = g / scale[..., None]
    codes = np.abs(v[..., None] - lut).argmin(-1).reshape(N, K)
    return mxfp4_pack_ref(codes), x


def awq_quantize_ref(w_nk: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric int4 RTN of a linear weight [N, K] in AWQ gemm layout (synthetic checkpoints in tests)."""
    w = w_nk.T                                         # [K, N]
    K, N = w.shape
    g = w.reshape(K // group_size, group_size, N)
    lo, hi = g.min(1), g.max(1)
    s = np.maximum(hi - lo, 1e-6) / 15.0
    z = np.clip(np.round(-lo / s), 0, 15)
    q = np.clip(np.round(g / s[:, None, :] + z[:, None, :]), 0, 15).reshape(K, N).astype(np.int64)
    return awq_pack_ref(q), awq_pack_ref(z.astype(np.int64)), np.ascontiguousarray(s.astype(np.float16))
