"""numpy-only row codecs shared by the ledger, patches and cartridges (no torch).

* bf16: round-to-nearest-even fp32 -> bf16, as torch's `.to(torch.bfloat16)` and the serve's device cast
  (`R/ple/source.py:245`) do; carried as uint16 bit patterns.
* int6 group-32 "fused" rows, the served Flash-Next n-gram codec (`serve/INTEGRATION.md` §0: 120 B of 6-bit codes
  + 5 fp16 scales = 130 B per 160-wide row). Layout (R/weights/row_table.py:109-151, 880-884):
      fused row = packed codes (w*6/8 bytes) || fp16 LE group scales (w/32 * 2 bytes)
      word = b0 | b1<<8 | b2<<16 ; code_i = ((word >> 6i) & 0x3F) - 32 ; value = code * scale[group of 32]
  The encoder is the one P/serve/v2-editpath/requant_delta.py used: scale = fp16(group absmax / 31) (exact zero
  groups get 6e-8), code = clip(rint(x / scale), -31, 31). It is ours, not the checkpoint author's (the shipped
  table was GPTQ-quantised); the DECODE side is what must match the serve, and tests pin it to the fork's
  `dequant_int6_g32` bit for bit when the fork is present.
"""
from __future__ import annotations

import numpy as np

INT6_GROUP = 32
INT6_CODE_OFFSET = 32
INT6_ZERO_SCALE = np.float16(6e-8)


# ----------------------------------------------------------------------------------------------------- bf16
def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit patterns (uint16), round-to-nearest-even. NaN payloads are not preserved."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    if nan.any():
        out[nan] = np.uint16(0x7FC0)
    return out


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> nearest bf16 value, returned as fp32."""
    return bf16_bits_to_f32(bf16_bits(x))


# ----------------------------------------------------------------------------------------------- int6 g32
def int6_row_bytes(row_width: int) -> int:
    if row_width % INT6_GROUP:
        raise ValueError(f"int6_g32 needs row_width % 32 == 0, got {row_width}")
    return row_width * 6 // 8 + (row_width // INT6_GROUP) * 2


def int6_scales(x: np.ndarray) -> np.ndarray:
    """fp16 group scales [n, w/32]: group absmax / 31 (exact-zero groups get 6e-8, as the reference encoder)."""
    x = np.asarray(x, dtype=np.float32)
    n, w = x.shape
    s = (np.abs(x.reshape(n, w // INT6_GROUP, INT6_GROUP)).max(-1) / np.float32(31.0)).astype(np.float16)
    s[s == 0] = INT6_ZERO_SCALE
    return s


def int6_pack(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """fp32 [n, w] + fp16 scales [n, w/32] -> packed codes uint8 [n, w*6/8]."""
    x = np.asarray(x, dtype=np.float32)
    n, w = x.shape
    q = np.rint(x.reshape(n, w // INT6_GROUP, INT6_GROUP) / scales.astype(np.float32)[..., None])
    codes = (np.clip(q, -31, 31).astype(np.int32) + INT6_CODE_OFFSET).astype(np.uint32).reshape(n, -1, 4)
    word = codes[..., 0] | (codes[..., 1] << 6) | (codes[..., 2] << 12) | (codes[..., 3] << 18)
    return np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, -1)


def int6_unpack(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """packed uint8 [n, w*6/8] + fp16 scales [n, w/32] -> fp32 [n, w] (the serve's decode convention)."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    n = packed.shape[0]
    if packed.shape[1] % 3:
        raise ValueError("int6 packed rows must be whole 3-byte words")
    words = packed.reshape(n, -1, 3).astype(np.uint32)
    word = words[:, :, 0] | (words[:, :, 1] << 8) | (words[:, :, 2] << 16)
    codes = np.stack([(word >> (6 * i)) & 0x3F for i in range(4)], axis=-1).reshape(n, -1).astype(np.int16)
    codes = codes - np.int16(INT6_CODE_OFFSET)
    g = np.asarray(scales).shape[1]
    if codes.shape[1] != g * INT6_GROUP:
        raise ValueError(f"{codes.shape[1]} codes vs {g} scales x {INT6_GROUP}")
    vals = codes.reshape(n, g, INT6_GROUP).astype(np.float32)
    vals *= np.asarray(scales).astype(np.float32).reshape(n, g, 1)
    return vals.reshape(n, g * INT6_GROUP)


def int6_encode_fused(x: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """fp32 [n, w] -> fused rows uint8 [n, int6_row_bytes(w)] = packed || fp16 scales."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("int6_encode_fused takes [n, w]")
    int6_row_bytes(x.shape[1])
    s = int6_scales(x) if scales is None else np.asarray(scales, dtype=np.float16)
    packed = int6_pack(x, s)
    return np.concatenate([packed, np.ascontiguousarray(s, dtype="<f2").view(np.uint8).reshape(x.shape[0], -1)], 1)


def int6_split_fused(fused: np.ndarray, row_width: int) -> tuple[np.ndarray, np.ndarray]:
    fused = np.ascontiguousarray(fused, dtype=np.uint8)
    split = row_width * 6 // 8
    if fused.shape[1] != int6_row_bytes(row_width):
        raise ValueError(f"fused row is {fused.shape[1]} B, expected {int6_row_bytes(row_width)}")
    return fused[:, :split], np.ascontiguousarray(fused[:, split:]).view("<f2")


def int6_decode_fused(fused: np.ndarray, row_width: int) -> np.ndarray:
    packed, s = int6_split_fused(fused, row_width)
    return int6_unpack(packed, s)


# ---------------------------------------------------------------------------------------------- dispatch
CODECS = ("fp32", "bf16", "int6_g32")


def row_bytes(codec: str, row_width: int) -> int:
    if codec == "fp32":
        return 4 * row_width
    if codec == "bf16":
        return 2 * row_width
    if codec == "int6_g32":
        return int6_row_bytes(row_width)
    if codec == "none":
        return 0
    raise ValueError(f"unknown codec {codec!r}")


def encode_rows(x: np.ndarray, codec: str) -> np.ndarray:
    """fp32 [n, w] -> raw row bytes uint8 [n, row_bytes]."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    if codec == "fp32":
        return x.view(np.uint8).reshape(n, -1)
    if codec == "bf16":
        return bf16_bits(x).view(np.uint8).reshape(n, -1)
    if codec == "int6_g32":
        return int6_encode_fused(x)
    if codec == "none":
        return np.zeros((n, 0), np.uint8)
    raise ValueError(f"unknown codec {codec!r}")


def decode_rows(raw: np.ndarray, codec: str, row_width: int) -> np.ndarray:
    """raw row bytes uint8 [n, row_bytes] -> fp32 [n, w]."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    n = raw.shape[0]
    if codec == "fp32":
        return raw.view("<f4").reshape(n, row_width).astype(np.float32)
    if codec == "bf16":
        return bf16_bits_to_f32(raw.view("<u2").reshape(n, row_width))
    if codec == "int6_g32":
        return int6_decode_fused(raw, row_width)
    if codec == "none":
        return np.zeros((n, row_width), np.float32)
    raise ValueError(f"unknown codec {codec!r}")


def roundtrip_stats(x: np.ndarray, codec: str, served_cast: str | None = "bf16") -> dict:
    """Round-trip error of `codec` on fp32 rows x [n, w] (and, with served_cast='bf16', after the serve's cast).

    Keys: max_abs, rms, median_rel_elem (|err|/|x| over x != 0), median_row_rel_l2 / max_row_rel_l2
    (||err||/||x|| per nonzero row), frac_exact, median_step (int6: median group scale), row_bytes."""
    x = np.asarray(x, dtype=np.float32)
    dec = decode_rows(encode_rows(x, codec), codec, x.shape[1])
    if served_cast == "bf16":
        dec = bf16_round(dec)
    err = (dec.astype(np.float64) - x.astype(np.float64))
    nz = x != 0
    rn = np.linalg.norm(x.astype(np.float64), axis=1)
    rel_rows = np.linalg.norm(err, axis=1)[rn > 0] / rn[rn > 0]
    out = dict(codec=codec, served_cast=served_cast, n_rows=int(x.shape[0]), row_width=int(x.shape[1]),
               row_bytes=row_bytes(codec, x.shape[1]), max_abs=float(np.abs(err).max(initial=0.0)),
               rms=float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0,
               median_rel_elem=float(np.median(np.abs(err[nz]) / np.abs(x[nz]))) if nz.any() else 0.0,
               median_row_rel_l2=float(np.median(rel_rows)) if rel_rows.size else 0.0,
               max_row_rel_l2=float(rel_rows.max()) if rel_rows.size else 0.0,
               frac_exact=float(np.mean(err == 0)) if err.size else 1.0)
    if codec == "int6_g32":
        out["median_step"] = float(np.median(int6_scales(x).astype(np.float32)))
    return out


__all__ = ["bf16_bits", "bf16_bits_to_f32", "bf16_round", "int6_row_bytes", "int6_scales", "int6_pack",
           "int6_unpack", "int6_encode_fused", "int6_decode_fused", "int6_split_fused", "encode_rows", "decode_rows",
           "row_bytes", "roundtrip_stats", "CODECS", "INT6_GROUP"]
