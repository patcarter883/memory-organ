"""Frozen 4-bit host layers for graft training (DESIGN §4.6, §8 M2-0): weights stay packed, each matrix is
dequantised to bf16 just in time inside an ``autograd.Function`` that returns only dX (no weight gradient).

Formats (read from the checkpoints' ``quantization_config`` and safetensors tensor names):

* **AWQ int4 "gemm"** (``cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit``: ``quant_method=awq, bits=4, group_size=32,
  zero_point=true, version=gemm``; routed experts only, everything else F16). Per linear with K inputs and
  N outputs: ``qweight`` int32 [K, N/8], ``qzeros`` int32 [K/g, N/8], ``scales`` fp16 [K/g, N]. Each int32
  packs 8 consecutive output channels in AutoAWQ's order: nibble i (bits 4i..4i+3) holds channel
  ``8c + AWQ_ORDER[i]``, AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7). Dequant: ``w[k, n] = (q[k, n] - z[k // g, n]) *
  s[k // g, n]``; the nn.Linear weight is w^T [N, K].
* **MXFP4** (``amd/Qwen3.8-27B-Quark-AWQ-MXFP4``: Quark, ``weight.dtype=fp4, scale_format=e8m0,
  group_size=32, pack_method=reorder``; every body linear incl. the GDN projections). ``weight`` uint8
  [N, K/2] (low nibble = even k), ``weight_scale`` uint8 [N, K/32]. Dequant: ``E2M1[code] * 2^(e - 127)``
  (e = 255 is NaN). The AWQ smoothing of this checkpoint is already folded into the stored weights.

Both dequants are exact in fp32 and then rounded once to the compute dtype (bf16 by default).

Modules: ``QuantLinear`` (any nn.Linear), ``QuantMoEExperts`` (drop-in for HF ``Qwen3_5MoeExperts``; AWQ
experts stacked [E, ...] and dequantised in expert groups; hand-written backward giving dX and the gradient of
the routing weights, so dX through the router is exact). Loaders read safetensors directly, from a local
checkpoint or by HTTP range read (``CheckpointReader``), so neither gptqmodel nor HF's AWQ integration is used.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import ssl
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "AWQ_ORDER", "AWQ_REVERSE", "E2M1_VALUES", "awq_unpack", "awq_dequant", "mxfp4_dequant", "e8m0_to_fp32",
    "QuantLinear", "QuantMoEExperts", "fake_quant_fp8", "CheckpointReader", "SafetensorsFile", "QuantScheme",
    "scheme_from_config", "load_quant_linear", "load_awq_experts", "build_decoder_layer", "grouped_mm_probe",
    "packed_layer_bytes",
]

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)     # nibble i holds channel 8c + AWQ_ORDER[i]
AWQ_REVERSE = (0, 4, 1, 5, 2, 6, 3, 7)   # channel 8c + j sits in nibble AWQ_REVERSE[j]
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# ------------------------------------------------------------------------------------------------------------
# dequantisation (vectorised torch; device-preserving)
# ------------------------------------------------------------------------------------------------------------

def awq_unpack(packed: torch.Tensor) -> torch.Tensor:
    """int32 [..., R, Np] AWQ-packed along the last axis -> int32 [..., R, 8*Np] in natural channel order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    rev = torch.tensor(AWQ_REVERSE, dtype=torch.long, device=packed.device)
    u = (packed.unsqueeze(-1) >> shifts) & 0xF               # [..., R, Np, 8] nibble i
    return u.index_select(-1, rev).reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def awq_dequant(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int,
                dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """AWQ gemm [..., K, N/8] / [..., K/g, N/8] / [..., K/g, N] -> nn.Linear weight [..., N, K] in ``dtype``.
    (q - z) is an exact small integer and the product with an fp16 scale is exact in fp32."""
    q = awq_unpack(qweight)                                   # [..., K, N]
    z = awq_unpack(qzeros)                                    # [..., G, N]
    s = scales.to(torch.float32)
    K = q.shape[-2]
    G = z.shape[-2]
    if G * group_size != K:
        raise ValueError(f"AWQ shapes: K={K}, groups={G}, group_size={group_size}")
    w = (q.view(*q.shape[:-2], G, group_size, q.shape[-1]) - z.unsqueeze(-2)).to(torch.float32) * s.unsqueeze(-2)
    return w.reshape(*q.shape).transpose(-1, -2).to(dtype)


def e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """E8M0 exponent byte -> 2^(e-127) as fp32, built from the bit pattern (exact; 0 -> 2^-127 subnormal,
    255 -> NaN)."""
    e = scale_u8.to(torch.int32)
    bits = torch.where(e == 0, torch.full_like(e, 0x00400000),
                       torch.where(e == 255, torch.full_like(e, 0x7FC00000), e << 23))
    return bits.view(torch.float32)


def mxfp4_dequant(weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16,
                  group_size: int = 32) -> torch.Tensor:
    """MXFP4 uint8 [..., N, K/2] codes + uint8 [..., N, K/32] E8M0 -> [..., N, K] in ``dtype``."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=weight.device)
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).reshape(*weight.shape[:-1], weight.shape[-1] * 2)
    vals = lut[codes.long()]
    K = vals.shape[-1]
    if weight_scale.shape[-1] * group_size != K:
        raise ValueError(f"MXFP4 shapes: K={K}, scale groups={weight_scale.shape[-1]}")
    s = e8m0_to_fp32(weight_scale)
    w = vals.view(*vals.shape[:-1], K // group_size, group_size) * s.unsqueeze(-1)
    return w.reshape(vals.shape).to(dtype)


def fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    """Per-token dynamic fp8 e4m3 round trip (the W4A8 serve's activation path, est.); no autograd here."""
    amax = x.abs().amax(-1, keepdim=True).float().clamp_min(1e-12)
    s = amax / 448.0
    return ((x.float() / s).to(torch.float8_e4m3fn).to(torch.float32) * s).to(x.dtype)


# ------------------------------------------------------------------------------------------------------------
# frozen linear
# ------------------------------------------------------------------------------------------------------------

class _FrozenLinearFn(torch.autograd.Function):
    """y = x W^T (+ b) with W dequantised just in time; backward returns dX = dY W only (W re-dequantised,
    nothing saved but a reference to the module)."""

    @staticmethod
    def forward(ctx, x, layer: "QuantLinear"):
        ctx.layer = layer
        ctx.xdtype = x.dtype
        cdt = layer.compute_dtype
        W = layer.dequant()
        xin = fake_quant_fp8(x.to(cdt)) if layer.act_quant == "fp8_e4m3" else x.to(cdt)
        return F.linear(xin, W, None if layer.bias is None else layer.bias.to(cdt))

    @staticmethod
    def backward(ctx, dy):
        layer = ctx.layer
        W = layer.dequant()
        dx = dy.to(layer.compute_dtype) @ W      # straight-through for the optional fp8 fake-quant
        return dx.to(ctx.xdtype), None


class QuantLinear(nn.Module):
    """Frozen packed linear. fmt 'awq' (buffers qweight/qzeros/scales) or 'mxfp4' (weight/weight_scale).
    AWQ fp16 scales are stored as an int16 bit view so a later ``module.to(dtype)`` cannot round them."""

    def __init__(self, in_features: int, out_features: int, fmt: str, tensors: dict[str, torch.Tensor], *,
                 group_size: int = 32, compute_dtype: torch.dtype = torch.bfloat16,
                 bias: torch.Tensor | None = None, act_quant: str | None = None):
        super().__init__()
        self.in_features, self.out_features, self.fmt = int(in_features), int(out_features), fmt
        self.group_size, self.compute_dtype, self.act_quant = int(group_size), compute_dtype, act_quant
        if fmt == "awq":
            qw, qz, sc = tensors["qweight"], tensors["qzeros"], tensors["scales"]
            if tuple(qw.shape) != (in_features, out_features // 8) or tuple(sc.shape) != (
                    in_features // group_size, out_features):
                raise ValueError(f"AWQ tensor shapes {tuple(qw.shape)}, {tuple(sc.shape)} do not match "
                                 f"K={in_features}, N={out_features}, g={group_size}")
            self.register_buffer("qweight", qw.to(torch.int32).contiguous())
            self.register_buffer("qzeros", qz.to(torch.int32).contiguous())
            self.register_buffer("scales_bits", sc.to(torch.float16).contiguous().view(torch.int16))
        elif fmt == "mxfp4":
            w, s = tensors["weight"], tensors["weight_scale"]
            if tuple(w.shape) != (out_features, in_features // 2) or tuple(s.shape) != (
                    out_features, in_features // group_size):
                raise ValueError(f"MXFP4 tensor shapes {tuple(w.shape)}, {tuple(s.shape)} do not match "
                                 f"N={out_features}, K={in_features}")
            self.register_buffer("weight_packed", w.to(torch.uint8).contiguous())
            self.register_buffer("weight_scale", s.to(torch.uint8).contiguous())
        else:
            raise ValueError(f"unknown 4-bit format {fmt!r}")
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def dequant(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dt = dtype or self.compute_dtype
        if self.fmt == "awq":
            return awq_dequant(self.qweight, self.qzeros, self.scales_bits.view(torch.float16), self.group_size, dt)
        return mxfp4_dequant(self.weight_packed, self.weight_scale, dt, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FrozenLinearFn.apply(x, self)

    def packed_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, fmt={self.fmt}, g={self.group_size}"


# ------------------------------------------------------------------------------------------------------------
# MoE experts (AWQ, stacked)
# ------------------------------------------------------------------------------------------------------------

def _silu_grad(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


class _MoEFn(torch.autograd.Function):
    """Routed experts: out[t] = sum_j w[t,j] * down_e(silu(gate_e(x_t)) * up_e(x_t)), e = idx[t,j].
    Saves only (x, idx, w); backward recomputes per expert group and returns dX and dW_routing."""

    @staticmethod
    def forward(ctx, x, top_k_index, top_k_weights, mod: "QuantMoEExperts"):
        ctx.mod = mod
        ctx.save_for_backward(x, top_k_index, top_k_weights)
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        for e, tok, pos, Wgu, Wd in mod._iter_experts(top_k_index):
            xe = x[tok].to(mod.compute_dtype)
            xe = fake_quant_fp8(xe) if mod.act_quant == "fp8_e4m3" else xe
            gate, up = F.linear(xe, Wgu).chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = fake_quant_fp8(h) if mod.act_quant == "fp8_e4m3" else h
            ye = F.linear(h, Wd)
            out.index_add_(0, tok, ye.float() * top_k_weights[tok, pos, None].float())
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, dout):
        x, idx, w = ctx.saved_tensors
        mod = ctx.mod
        cdt = mod.compute_dtype
        dx = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        dw = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
        for e, tok, pos, Wgu, Wd in mod._iter_experts(idx):
            xe = x[tok].to(cdt)
            xe = fake_quant_fp8(xe) if mod.act_quant == "fp8_e4m3" else xe
            gu = F.linear(xe, Wgu)
            gate, up = gu.chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = fake_quant_fp8(h) if mod.act_quant == "fp8_e4m3" else h
            ye = F.linear(h, Wd)
            de = dout[tok].to(cdt)
            dw[tok, pos] = (de.float() * ye.float()).sum(-1)
            dye = (de.float() * w[tok, pos, None].float()).to(cdt)
            dh = dye @ Wd                                          # [n, I]
            gf, uf, dhf = gate.float(), up.float(), dh.float()
            dgu = torch.cat((dhf * uf * _silu_grad(gf), dhf * F.silu(gf)), dim=-1).to(cdt)
            dx.index_add_(0, tok, (dgu @ Wgu).float())
        return dx.to(x.dtype), None, dw.to(w.dtype), None


class QuantMoEExperts(nn.Module):
    """Drop-in for HF ``Qwen3_5MoeExperts`` (forward(hidden [n, d], top_k_index [n, k], top_k_weights [n, k]))
    over AWQ experts kept packed: gate and up stacked along N (so dequant gives HF's gate_up_proj layout
    [E, 2I, d]), down [E, d, I]. ``experts_per_group`` experts are dequantised together (peak transient
    = group x (2I*d + d*I) x 2 bytes in bf16)."""

    def __init__(self, gate_up: dict[str, torch.Tensor], down: dict[str, torch.Tensor], *, hidden: int,
                 intermediate: int, group_size: int = 32, compute_dtype: torch.dtype = torch.bfloat16,
                 experts_per_group: int = 32, act_quant: str | None = None):
        super().__init__()
        self.num_experts = int(gate_up["qweight"].shape[0])
        self.hidden_dim, self.intermediate_dim = int(hidden), int(intermediate)
        self.group_size, self.compute_dtype = int(group_size), compute_dtype
        self.experts_per_group, self.act_quant = int(experts_per_group), act_quant
        E, d, I = self.num_experts, self.hidden_dim, self.intermediate_dim
        exp = {"gu": ((E, d, 2 * I // 8), (E, d // group_size, 2 * I // 8), (E, d // group_size, 2 * I)),
               "dn": ((E, I, d // 8), (E, I // group_size, d // 8), (E, I // group_size, d))}
        for tag, t in (("gu", gate_up), ("dn", down)):
            shapes = (tuple(t["qweight"].shape), tuple(t["qzeros"].shape), tuple(t["scales"].shape))
            if shapes != exp[tag]:
                raise ValueError(f"{tag} AWQ expert shapes {shapes} != {exp[tag]}")
            self.register_buffer(f"{tag}_qweight", t["qweight"].to(torch.int32).contiguous())
            self.register_buffer(f"{tag}_qzeros", t["qzeros"].to(torch.int32).contiguous())
            self.register_buffer(f"{tag}_scales_bits", t["scales"].to(torch.float16).contiguous().view(torch.int16))

    def dequant_experts(self, ids: torch.Tensor, dtype: torch.dtype | None = None):
        """(gate_up [len(ids), 2I, d], down [len(ids), d, I]) for expert ids."""
        dt = dtype or self.compute_dtype
        sel = lambda b: b.index_select(0, ids)  # noqa: E731
        gu = awq_dequant(sel(self.gu_qweight), sel(self.gu_qzeros), sel(self.gu_scales_bits).view(torch.float16),
                         self.group_size, dt)
        dn = awq_dequant(sel(self.dn_qweight), sel(self.dn_qzeros), sel(self.dn_scales_bits).view(torch.float16),
                         self.group_size, dt)
        return gu, dn

    def _iter_experts(self, top_k_index: torch.Tensor):
        k = top_k_index.shape[1]
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.num_experts).tolist()
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)
        hit = [e for e in range(self.num_experts) if counts[e] > 0]
        for g0 in range(0, len(hit), self.experts_per_group):
            grp = hit[g0:g0 + self.experts_per_group]
            gu, dn = self.dequant_experts(torch.tensor(grp, device=top_k_index.device))
            for j, e in enumerate(grp):
                sl = order[starts[e]:starts[e + 1]]
                yield e, sl // k, sl % k, gu[j], dn[j]
            del gu, dn

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        return _MoEFn.apply(hidden_states, top_k_index, top_k_weights, self)

    def packed_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def extra_repr(self) -> str:
        return f"E={self.num_experts}, d={self.hidden_dim}, I={self.intermediate_dim}, awq g={self.group_size}"


def grouped_mm_probe(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Does ``torch._grouped_mm`` run here (bf16, 3 groups) and match a per-group matmul? (DESIGN §7.5 #2)"""
    f = getattr(torch, "_grouped_mm", None)
    if f is None:
        return {"available": False, "reason": "torch._grouped_mm missing"}
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(48, 64, generator=g).to(device=device, dtype=torch.bfloat16)
        w = torch.randn(3, 64, 32, generator=g).to(device=device, dtype=torch.bfloat16)
        offs = torch.tensor([16, 32, 48], dtype=torch.int32, device=device)
        y = f(x, w, offs=offs)
        ref = torch.cat([x[i * 16:(i + 1) * 16].float() @ w[i].float() for i in range(3)])
        err = ((y.float() - ref).norm() / ref.norm()).item()
        return {"available": True, "rel_err": err, "ok": err < 1e-2}
    except Exception as e:  # noqa: BLE001 - this is a capability probe
        return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}


# ------------------------------------------------------------------------------------------------------------
# safetensors readers: local files or HTTP range reads
# ------------------------------------------------------------------------------------------------------------

_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
              "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn}


def _opener():
    ca = next((os.environ[v] for v in ("NGM_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")
               if os.environ.get(v) and Path(os.environ[v]).exists()), None)
    ctx = ssl.create_default_context(cafile=ca)
    return urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))


class SafetensorsFile:
    """One safetensors file, local (``pread``) or remote (HTTP ``Range``). Never reads more than asked."""

    def __init__(self, location: str, *, timeout: float = 120.0, retries: int = 4):
        self.location = location
        self.remote = location.startswith("http://") or location.startswith("https://")
        self.timeout, self.retries = timeout, retries
        self.bytes_read = 0
        self._op = _opener() if self.remote else None
        self._fd = None if self.remote else os.open(location, os.O_RDONLY)
        n = struct.unpack("<Q", self.read(0, 8))[0]
        self.header: dict = json.loads(self.read(8, 8 + n))
        self.metadata = self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def read(self, start: int, end: int) -> bytes:
        """Bytes [start, end)."""
        if end <= start:
            return b""
        if not self.remote:
            out = os.pread(self._fd, end - start, start)
        else:
            last: Exception | None = None
            for attempt in range(self.retries):
                try:
                    req = urllib.request.Request(self.location, headers={"Range": f"bytes={start}-{end - 1}",
                                                                         "User-Agent": "ngram-memory/0.0.1"})
                    with self._op.open(req, timeout=self.timeout) as r:
                        out = r.read()
                    if len(out) != end - start:
                        raise IOError(f"short range read {len(out)} != {end - start}")
                    break
                except Exception as e:  # noqa: BLE001 - retried, then re-raised
                    last = e
            else:
                raise IOError(f"range read failed for {self.location} [{start},{end}): {last}")
        self.bytes_read += len(out)
        return out

    def info(self, name: str) -> tuple[torch.dtype, list[int], int, int]:
        h = self.header[name]
        a, b = h["data_offsets"]
        return _ST_DTYPES[h["dtype"]], list(h["shape"]), self.data_start + a, self.data_start + b

    def get(self, name: str) -> torch.Tensor:
        dt, shape, a, b = self.info(name)
        return self._decode(self.read(a, b), dt, shape)

    def get_rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Rows [start, stop) along dim 0 (one contiguous range read)."""
        dt, shape, a, b = self.info(name)
        if not 0 <= start <= stop <= shape[0]:
            raise IndexError(f"{name}: rows [{start}, {stop}) outside [0, {shape[0]})")
        row = (b - a) // shape[0]
        return self._decode(self.read(a + start * row, a + stop * row), dt, [stop - start] + shape[1:])

    @staticmethod
    def _decode(buf: bytes, dt: torch.dtype, shape: list[int]) -> torch.Tensor:
        if not buf:
            return torch.empty(shape, dtype=dt)
        return torch.frombuffer(bytearray(buf), dtype=dt).reshape(shape)

    def get_many(self, names: Iterable[str], *, max_gap: int = 1 << 20, piece: int = 32 << 20,
                 workers: int = 16) -> dict[str, torch.Tensor]:
        """Fetch several tensors with coalesced, parallel range reads (gaps up to ``max_gap`` are read through)."""
        items = sorted((self.info(n)[2], self.info(n)[3], n) for n in names)
        spans: list[list] = []
        for a, b, n in items:
            if spans and a - spans[-1][1] <= max_gap:
                spans[-1][1] = max(spans[-1][1], b)
                spans[-1][2].append(n)
            else:
                spans.append([a, b, [n]])
        jobs = [(s, min(s + piece, b)) for a, b, _ in spans for s in range(a, b, piece)]
        if self.remote and len(jobs) > 1:
            with cf.ThreadPoolExecutor(workers) as ex:
                parts = dict(zip(jobs, ex.map(lambda j: self.read(*j), jobs)))
        else:
            parts = {j: self.read(*j) for j in jobs}
        out = {}
        for a, b, ns in spans:
            blob = b"".join(parts[(s, min(s + piece, b))] for s in range(a, b, piece))
            for n in ns:
                dt, shape, x, y = self.info(n)
                out[n] = self._decode(blob[x - a:y - a], dt, shape)
        return out

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class CheckpointReader:
    """Tensor-name -> file map over one or more safetensors files (``model.safetensors.index.json`` or a single
    ``model.safetensors``), local directory or HF hub repo (range reads; the index may come from the HF cache)."""

    def __init__(self, files: dict[str, SafetensorsFile], weight_map: dict[str, str]):
        self.files, self.weight_map = files, weight_map

    @classmethod
    def local(cls, ckpt_dir: str | Path) -> "CheckpointReader":
        d = Path(ckpt_dir)
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            f = SafetensorsFile(str(d / "model.safetensors"))
            return cls({"model.safetensors": f}, {k: "model.safetensors" for k in f.header})
        files = {fn: SafetensorsFile(str(d / fn)) for fn in sorted(set(wm.values()))}
        return cls(files, wm)

    @classmethod
    def hub(cls, repo: str, revision: str = "main", index_json: str | Path | None = None,
            files: Iterable[str] | None = None) -> "CheckpointReader":
        """Remote reader. With ``index_json`` (e.g. the cached index) only the shard headers listed in ``files``
        (default: all shards) are fetched; without it a single ``model.safetensors`` is assumed."""
        base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        if index_json is None:
            f = SafetensorsFile(base + "model.safetensors")
            return cls({"model.safetensors": f}, {k: "model.safetensors" for k in f.header})
        wm = json.loads(Path(index_json).read_text())["weight_map"]
        want = sorted(set(files) if files is not None else set(wm.values()))
        return cls({fn: SafetensorsFile(base + fn) for fn in want}, {k: v for k, v in wm.items() if v in want})

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self):
        return self.weight_map.keys()

    def get(self, name: str) -> torch.Tensor:
        return self.files[self.weight_map[name]].get(name)

    def get_rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self.files[self.weight_map[name]].get_rows(name, start, stop)

    def get_many(self, names: Iterable[str], **kw) -> dict[str, torch.Tensor]:
        by: dict[str, list[str]] = {}
        for n in names:
            by.setdefault(self.weight_map[n], []).append(n)
        out: dict[str, torch.Tensor] = {}
        for fn, ns in by.items():
            out.update(self.files[fn].get_many(ns, **kw))
        return out

    def dtype_of(self, name: str) -> torch.dtype:
        return self.files[self.weight_map[name]].info(name)[0]

    @property
    def bytes_read(self) -> int:
        return sum(f.bytes_read for f in self.files.values())


# ------------------------------------------------------------------------------------------------------------
# schemes and loaders
# ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantScheme:
    """How a checkpoint stores its body linears. method 'awq' (AWQ gemm; only names under ``quantized_substr``
    are packed, the rest dense) or 'mxfp4' (Quark: any linear whose ``.weight`` is U8 with ``.weight_scale``)."""
    method: str
    group_size: int
    bits: int = 4
    quantized_substr: tuple[str, ...] = ()


def scheme_from_config(cfg: dict) -> QuantScheme:
    """Read ``quantization_config`` of an HF config.json dict (AWQ gemm 4-bit, or Quark fp4/e8m0 'reorder')."""
    q = cfg.get("quantization_config") or {}
    m = q.get("quant_method")
    if m == "awq":
        if q.get("version", "gemm") != "gemm" or int(q.get("bits", 4)) != 4 or not q.get("zero_point", True):
            raise ValueError(f"unsupported AWQ config {q}")
        return QuantScheme("awq", int(q["group_size"]), 4, (".mlp.experts.",))
    if m == "quark":
        w = q["global_quant_config"]["weight"]
        pack = (q.get("export") or {}).get("pack_method", "reorder")
        if (w["dtype"], w["scale_format"], pack) != ("fp4", "e8m0", "reorder") or q.get("layer_quant_config") \
                or q.get("layer_type_quant_config"):
            raise ValueError("only Quark MXFP4 (fp4 + e8m0, pack_method=reorder, no per-layer overrides)")
        return QuantScheme("mxfp4", int(w["group_size"]), 4)
    raise ValueError(f"no 4-bit reader for quant_method={m!r}")


def load_quant_linear(reader: CheckpointReader, prefix: str, scheme: QuantScheme, *,
                      compute_dtype: torch.dtype = torch.bfloat16, act_quant: str | None = None,
                      tensors: dict[str, torch.Tensor] | None = None) -> QuantLinear:
    """``prefix`` = module path without the parameter suffix (e.g. '...linear_attn.in_proj_qkv')."""
    t = tensors if tensors is not None else {}
    if scheme.method == "awq":
        names = [f"{prefix}.{s}" for s in ("qweight", "qzeros", "scales")]
        t = {n: t[n] for n in names} if all(n in t for n in names) else reader.get_many(names)
        qw, qz, sc = (t[n] for n in names)
        return QuantLinear(qw.shape[0], sc.shape[1], "awq", {"qweight": qw, "qzeros": qz, "scales": sc},
                           group_size=scheme.group_size, compute_dtype=compute_dtype, act_quant=act_quant)
    names = [f"{prefix}.weight", f"{prefix}.weight_scale"]
    t = {n: t[n] for n in names} if all(n in t for n in names) else reader.get_many(names)
    w, s = t[names[0]], t[names[1]]
    return QuantLinear(w.shape[1] * 2, w.shape[0], "mxfp4", {"weight": w, "weight_scale": s},
                       group_size=scheme.group_size, compute_dtype=compute_dtype, act_quant=act_quant)


def _expert_names(prefix: str, E: int) -> list[str]:
    return [f"{prefix}.{e}.{p}.{s}" for e in range(E) for p in ("gate_proj", "up_proj", "down_proj")
            for s in ("qweight", "qzeros", "scales")]


def load_awq_experts(reader: CheckpointReader, prefix: str, num_experts: int, scheme: QuantScheme, *,
                     hidden: int, intermediate: int, compute_dtype: torch.dtype = torch.bfloat16,
                     experts_per_group: int = 32, act_quant: str | None = None,
                     tensors: dict[str, torch.Tensor] | None = None) -> QuantMoEExperts:
    """Stack per-expert AWQ tensors '{prefix}.{e}.{gate,up,down}_proj.{qweight,qzeros,scales}' (packed)."""
    t = tensors if tensors is not None else reader.get_many(_expert_names(prefix, num_experts))
    g = lambda e, p, s: t[f"{prefix}.{e}.{p}.{s}"]  # noqa: E731
    gate_up = {s: torch.stack([torch.cat((g(e, "gate_proj", s), g(e, "up_proj", s)), dim=-1)
                               for e in range(num_experts)]) for s in ("qweight", "qzeros", "scales")}
    down = {s: torch.stack([g(e, "down_proj", s) for e in range(num_experts)]) for s in ("qweight", "qzeros", "scales")}
    return QuantMoEExperts(gate_up, down, hidden=hidden, intermediate=intermediate, group_size=scheme.group_size,
                           compute_dtype=compute_dtype, experts_per_group=experts_per_group, act_quant=act_quant)


def _text_config(cfg: dict):
    t = dict(cfg.get("text_config", cfg))
    mt = t.get("model_type", cfg.get("model_type"))
    if mt in ("qwen3_5_moe", "qwen3_5_moe_text"):
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as C
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer as L
    elif mt in ("qwen3_5", "qwen3_5_text"):
        from transformers.models.qwen3_5 import Qwen3_5TextConfig as C
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer as L
    else:
        raise ValueError(f"unsupported host model_type {mt!r}")
    t.pop("model_type", None)
    t.pop("dtype", None)
    c = C(**t)
    c._attn_implementation = "sdpa"
    return c, L


def build_decoder_layer(cfg: dict, layer_idx: int, reader: CheckpointReader, *, compute_dtype=torch.bfloat16,
                        device: str | torch.device = "cpu", prefix: str = "model.language_model.layers",
                        experts_per_group: int = 32, act_quant: str | None = None,
                        dense_dtype: torch.dtype | None = None,
                        parts: tuple[str, ...] | None = None) -> tuple[nn.Module, Any]:
    """One frozen HF decoder layer of a 4-bit host, loaded straight from safetensors. Quantised linears become
    ``QuantLinear`` / ``QuantMoEExperts`` (packed), every other tensor is read and cast to ``dense_dtype``
    (default ``compute_dtype``). Returns (layer, text_config). Built on the meta device first, so the bf16
    expert stacks HF would allocate are never materialised. ``parts`` (e.g. ('input_layernorm', 'linear_attn'))
    materialises only those submodules; the rest stay on the meta device and must not be called."""
    scheme = scheme_from_config(cfg)
    tcfg, Layer = _text_config(cfg)
    dense_dtype = dense_dtype or compute_dtype
    with torch.device("meta"):
        layer = Layer(tcfg, layer_idx)
    base = f"{prefix}.{layer_idx}"
    want = lambda n: parts is None or any(n == p or n.startswith(p + ".") for p in parts)  # noqa: E731
    mods = dict(layer.named_modules())
    replaced: set[str] = set()
    if scheme.method == "awq" and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and want("mlp.experts"):
        ex = layer.mlp.experts
        q = load_awq_experts(reader, f"{base}.mlp.experts", ex.num_experts, scheme, hidden=ex.hidden_dim,
                             intermediate=ex.intermediate_dim, compute_dtype=compute_dtype,
                             experts_per_group=experts_per_group, act_quant=act_quant)
        layer.mlp.experts = q.to(device)
        replaced.add("mlp.experts")
    for name, m in mods.items():
        if not isinstance(m, nn.Linear) or not want(name) or any(name == r or name.startswith(r + ".")
                                                                  for r in replaced):
            continue
        ck = f"{base}.{name}"
        packed = (scheme.method == "mxfp4" and f"{ck}.weight_scale" in reader) or \
                 (scheme.method == "awq" and f"{ck}.qweight" in reader)
        if packed:
            parent, _, leaf = name.rpartition(".")
            ql = load_quant_linear(reader, ck, scheme, compute_dtype=compute_dtype, act_quant=act_quant)
            if (ql.in_features, ql.out_features) != (m.in_features, m.out_features):
                raise ValueError(f"{ck}: packed {ql.in_features}->{ql.out_features} vs HF {m.in_features}->{m.out_features}")
            setattr(mods[parent] if parent else layer, leaf, ql.to(device))
            replaced.add(name)
    names = [n for n, p in layer.named_parameters() if p.is_meta and want(n)]
    t = reader.get_many([f"{base}.{n}" for n in names])
    for n in names:
        parent, _, leaf = n.rpartition(".")
        mod = layer.get_submodule(parent) if parent else layer
        old = getattr(mod, leaf)
        v = t[f"{base}.{n}"]
        if tuple(v.shape) != tuple(old.shape):
            raise ValueError(f"{base}.{n}: checkpoint {tuple(v.shape)} vs HF {tuple(old.shape)}")
        mod._parameters[leaf] = nn.Parameter(v.to(device=device, dtype=dense_dtype), requires_grad=False)
    left = [n for n, b in layer.named_buffers() if b.is_meta and want(n)]
    if left:
        raise ValueError(f"unmaterialised buffers: {left}")
    return layer.eval(), tcfg


def packed_layer_bytes(reader: CheckpointReader, layer_idx: int, prefix: str = "model.language_model.layers") -> int:
    """On-disk bytes of every tensor of one decoder layer (what stays resident when weights are kept packed)."""
    base = f"{prefix}.{layer_idx}."
    tot = 0
    for n in reader.keys():
        if n.startswith(base):
            _, _, a, b = reader.files[reader.weight_map[n]].info(n)
            tot += b - a
    return tot
