"""Native gdn_hip forward + HF torch-chunk recompute backward, and the class-name widening adapter for the fork's
shim (DESIGN §4.6 "Fix before M2's build", §7.5 #1, §8 M2-0 / M2; review items F1, N9).

The fork's training path (``R/gdn/hf_patch.py`` -> ``R/gdn/layer.py:378-405`` -> ``gdn_hip.autograd``) runs the
native ``gdn_prefill`` forward and, in the backward, recomputes a pure-torch reference with a Python
forward-substitution loop per chunk of width ``max(4, 64 // B)`` (24.1 aten ops per token at B=4). It also patches
only layers whose class is exactly ``Qwen3_5GatedDeltaNet`` and so finds 0 layers on the 35B-A3B
(``Qwen3_5MoeGatedDeltaNet``).

This module adds, beside ``gdn.py`` (which already holds our hand-written chunked backward ``RawGDN``):

* ``NativeFwdHFBwd``: the fix DESIGN §4.6 names first. One ``torch.autograd.Function`` whose forward is the native
  op (one varlen launch, the fork's ``_GDNPrefillBatchFn`` layout) and whose backward recomputes HF's own GDN prep
  (sigmoid, softplus gate, GQA ``repeat_interleave``) and HF's pure-torch ``torch_chunk_gated_delta_rule`` (the
  ``__wrapped__`` function, never fla, which hangs on gfx1201) under ``enable_grad`` and takes one
  ``autograd.grad`` to all seven raw inputs. It saves only the raw q, k, v, a, b (activation dtype, before GQA
  expansion) and A_log / dt_bias. ``bwd_group`` recomputes that many sequences at a time (bounds the transient
  memory of the HF recompute, which keeps every chunk intermediate; gradients are identical).
* ``NativeFwdForkRefBwd``: the fork's own pairing (native forward + its reference recompute), for timings.
* ``gdn_core``: one entry point over the forward x backward arms (``fwd`` in native / hf / ours, ``bwd`` in hf /
  ours / fork_ref). ``fwd='hf'`` with a custom backward runs ``hf_standin_op`` (HF's torch forward behind the
  native op's exact signature), which is how every native-forward arm is exercised on CPU.
* ``patch_gdn_native(model, fwd=..., bwd=...)``: per-instance forward override of every HF GDN mixer (both class
  names), as ``gdn.patch_gdn`` but over ``gdn_core``. A prefill with a fresh HF cache (``use_cache=True``, HF's
  default) runs the patched kernel and fills the cache; a cache that already holds state or varlen metadata goes
  to HF's forward, and raises instead while gradients flow (``strict='grad'``).
* ``patch_fork_shims(model)``: the fork's ``patch_qwen3_5_gdn`` loop widened to ``GDN_CLASS_NAMES`` (builds the
  fork's ``NativeGDNShim`` per layer, imported lazily from the fork; GPU-only in practice). The fork is not edited.

Layouts. Raw layer tensors: q, k [B, T, H, K] (after conv + split, before l2norm), v [B, T, HV, V], a, b [B, T, HV],
A_log, dt_bias [HV]. Native op (``autograd.py:480-500``): ``op(q[BT,H,K], k, v[BT,HV,V], a[BT,HV], b[BT,HV],
A_log fp32, dt_bias fp32, cu int32[B+1], state_idx long[B], has_init u8[B], ssm_state fp32[B+1,HV,V,K], scale,
use_l2norm) -> [BT, HV, V]`` in v's dtype; its state layout is [HV, V, K], HF's is [HV, K, V].

This file is torch code; it lives in the graft backend subpackage, not in the torch-free core engine.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from . import gdn as _gdn
from .gdn import GDN_CLASS_NAMES, gdn_modules

__all__ = [
    "FWD_ARMS", "BWD_ARMS", "hf_chunk_torch", "hf_torch_fn", "hf_prep", "hf_core", "hf_standin_op",
    "recurrent_standin_op", "NativeFwdHFBwd", "NativeFwdForkRefBwd", "gdn_core", "resolve_native_op",
    "load_fork_reference", "fork_paths", "NativePatchConfig", "patch_gdn_native", "unpatch_gdn_native",
    "ForkShimPatch", "patch_fork_shims", "GDNFallbackError",
]

FWD_ARMS = ("native", "hf", "ours")
BWD_ARMS = ("hf", "ours", "fork_ref")
FORK_ROOT_ENV = "NGM_MINISGL_FORK"
_DEFAULT_FORK_ROOT = "/home/user/patcarter883/minisglang-rdna4"
_HF_MODULES = ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "transformers.models.qwen3_5.modeling_qwen3_5")


# ------------------------------------------------------------------------------------------------------------
# HF's pure-torch functions (never the hub / fla kernels)
# ------------------------------------------------------------------------------------------------------------

def hf_torch_fn(name: str, module: Any = None) -> Callable:
    """HF's pure-torch ``name`` (e.g. ``torch_chunk_gated_delta_rule``, ``causal_conv1d_fn``). transformers wraps
    these with ``use_kernel_func_from_hub_with_fallback``, which calls fla / causal-conv1d when installed (fla hangs
    on gfx1201); ``__wrapped__`` is the torch body (``functools.wraps``)."""
    mods = [module] if module is not None else [m for m in _HF_MODULES]
    for m in mods:
        try:
            M = importlib.import_module(m) if isinstance(m, str) else m
        except ImportError:
            continue
        f = getattr(M, name, None)
        if f is not None:
            return getattr(f, "__wrapped__", f)
    raise ImportError(f"transformers' Qwen3.5 modeling module with {name!r} not importable")


def hf_chunk_torch() -> Callable:
    """HF's ``torch_chunk_gated_delta_rule`` torch body (qwen3_5_moe, falling back to qwen3_5)."""
    return hf_torch_fn("torch_chunk_gated_delta_rule")


def hf_prep(q, k, v, a, b, A_log, dt_bias, beta_fp32: bool = False):
    """HF's GDN forward between the split and the delta rule (``Qwen3_5MoeGatedDeltaNet.forward``):
    beta = sigmoid(b) in the activation dtype, g = -exp(A_log) * softplus(a + dt_bias) in fp32, GQA by
    ``repeat_interleave``. Returns (q, k, v, g, beta) for ``torch_chunk_gated_delta_rule``.
    ``beta_fp32``: sigmoid(b) in fp32 instead, as the native kernel computes it. For bf16 b, HF's bf16 rounding of
    beta alone moves the 35B layer-0 mixer output by about 3e-3 relative (experiments/m2_0/gdn_validate.py)."""
    beta = b.float().sigmoid() if beta_fp32 else b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    rep = v.shape[2] // q.shape[2]
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    return q, k, v, g, beta


def hf_core(q, k, v, a, b, A_log, dt_bias, *, chunk_size: int = 64, initial_state=None,
            output_final_state: bool = False, beta_fp32: bool = False):
    """HF's GDN core from the raw layer tensors, differentiable by plain autograd (the reference arm, and the
    function ``NativeFwdHFBwd`` recomputes). Returns core [B, T, HV, V] in q's dtype (and the fp32 final state
    [B, HV, K, V] when asked)."""
    qx, kx, vx, g, beta = hf_prep(q, k, v, a, b, A_log, dt_bias, beta_fp32)
    o, sf = hf_chunk_torch()(qx, kx, vx, g=g, beta=beta, chunk_size=chunk_size, initial_state=initial_state,
                             output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    return (o, sf) if output_final_state else o


# ------------------------------------------------------------------------------------------------------------
# CPU stand-ins with gdn_hip.gdn_prefill's exact signature
# ------------------------------------------------------------------------------------------------------------

def _varlen(cu: torch.Tensor, state_idx: torch.Tensor, has_init: torch.Tensor, n_tok: int):
    cu_l = [int(x) for x in cu.tolist()]
    if cu_l[0] != 0 or cu_l[-1] != n_tok or any(b < a for a, b in zip(cu_l, cu_l[1:])):
        raise ValueError(f"bad cu_seqlens {cu_l[:4]}... for {n_tok} tokens")
    if state_idx.numel() != len(cu_l) - 1 or has_init.numel() != len(cu_l) - 1:
        raise ValueError("state_indices / has_initial_state must have one entry per sequence")
    return cu_l, [int(x) for x in state_idx.tolist()], [bool(x) for x in has_init.tolist()]


def hf_standin_op(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices, has_initial_state, ssm_state,
                  scale: float, use_l2norm: int):
    """HF's torch delta rule behind the native op's interface (no grad; writes the final state of each sequence
    into ``ssm_state[state_indices[i]]`` in the native [HV, V, K] layout, reads it as the initial state where
    ``has_initial_state``). Equal-length sequences run as one batch. HF hard-codes scale = K**-0.5."""
    Kk = q.shape[-1]
    if abs(float(scale) - Kk ** -0.5) > 1e-6 * Kk ** -0.5:
        raise ValueError(f"hf_standin_op: HF's torch delta rule uses scale K**-0.5, got {scale}")
    cu, sidx, hinit = _varlen(cu_seqlens, state_indices, has_initial_state, q.shape[0])
    lens = [e - s for s, e in zip(cu, cu[1:])]
    f = hf_chunk_torch()
    out = v.new_empty(v.shape)
    groups = [list(range(len(lens)))] if len(set(lens)) == 1 else [[i] for i in range(len(lens))]
    with torch.no_grad():
        for grp in groups:
            L = lens[grp[0]]
            if L == 0:
                continue
            take = lambda x: torch.stack([x[cu[i]:cu[i + 1]] for i in grp])  # noqa: E731
            qq, kk, vv, aa, bb = (take(x) for x in (q, k, v, a, b))
            init = None
            if any(hinit[i] for i in grp):
                init = torch.stack([ssm_state[sidx[i]].transpose(-1, -2) if hinit[i]
                                    else torch.zeros_like(ssm_state[sidx[i]]).transpose(-1, -2) for i in grp]).float()
            qx, kx, vx, g, beta = hf_prep(qq, kk, vv, aa, bb, A_log, dt_bias)
            o, sf = f(qx, kx, vx, g=g, beta=beta, initial_state=init, output_final_state=True,
                      use_qk_l2norm_in_kernel=bool(use_l2norm))
            for j, i in enumerate(grp):
                out[cu[i]:cu[i + 1]] = o[j].to(v.dtype)
                ssm_state[sidx[i]] = sf[j].transpose(-1, -2).to(ssm_state.dtype)
    return out


def recurrent_standin_op(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices, has_initial_state, ssm_state,
                         scale: float, use_l2norm: int):
    """Exact per-token fp32 recurrence with the native op's signature and the recurrent kernel's math
    (``autograd.py`` docstring): q, k l2-normed with rsqrt(sumsq + 1e-6), q *= scale, beta = sigmoid(b) in fp32
    (fp64 for fp64 inputs),
    g = -exp(A_log) * softplus(a + dt_bias); per token S *= exp(g); v' = (v - S k) beta; S += v' k^T; o = S q, with
    S [HV, V, K]. Independent of HF's chunked code; one Python step per token (slow; for checks)."""
    cu, sidx, hinit = _varlen(cu_seqlens, state_indices, has_initial_state, q.shape[0])
    lens = [e - s for s, e in zip(cu, cu[1:])]
    H, HV = q.shape[1], v.shape[1]
    rep = HV // H
    out = v.new_empty(v.shape)
    groups = [list(range(len(lens)))] if len(set(lens)) == 1 else [[i] for i in range(len(lens))]
    cdt = torch.float64 if q.dtype == torch.float64 else torch.float32     # fp64 only for gradchecks
    with torch.no_grad():
        for grp in groups:
            L = lens[grp[0]]
            take = lambda x: torch.stack([x[cu[i]:cu[i + 1]] for i in grp]).to(cdt)  # noqa: E731
            qq, kk, vv, aa, bb = (take(x) for x in (q, k, v, a, b))       # [G, L, ...]
            if use_l2norm:
                qq, kk = _gdn.l2norm(qq), _gdn.l2norm(kk)
            qq = (qq * float(scale)).repeat_interleave(rep, dim=2)        # [G, L, HV, K]
            kk = kk.repeat_interleave(rep, dim=2)
            g = -A_log.to(cdt).exp() * F.softplus(aa + dt_bias.to(cdt))   # [G, L, HV]
            beta = bb.sigmoid()
            S = torch.stack([ssm_state[sidx[i]].to(cdt) if hinit[i] else torch.zeros_like(ssm_state[sidx[i]]).to(cdt)
                             for i in grp])                               # [G, HV, V, K]
            o = vv.new_empty(vv.shape)
            for t in range(L):
                S = S * g[:, t].exp()[..., None, None]
                kt = kk[:, t]                                             # [G, HV, K]
                d = (vv[:, t] - (S @ kt.unsqueeze(-1)).squeeze(-1)) * beta[:, t, :, None]
                S = S + d.unsqueeze(-1) * kt.unsqueeze(-2)
                o[:, t] = (S @ qq[:, t].unsqueeze(-1)).squeeze(-1)
            for j, i in enumerate(grp):
                out[cu[i]:cu[i + 1]] = o[j].to(v.dtype)
                ssm_state[sidx[i]] = S[j].to(ssm_state.dtype)
    return out


# ------------------------------------------------------------------------------------------------------------
# the fork (read-only): paths, native op, reference backward
# ------------------------------------------------------------------------------------------------------------

def fork_paths(fork_root: str | os.PathLike | None = None) -> tuple[Path, Path]:
    """(R/python, R/tools/vhip_patches/gdn/torch-ext) of the minisgl fork (``$NGM_MINISGL_FORK`` or the default)."""
    R = Path(fork_root or os.environ.get(FORK_ROOT_ENV, _DEFAULT_FORK_ROOT))
    return R / "python", R / "tools" / "vhip_patches" / "gdn" / "torch-ext"


def _add_paths(*ps: Path) -> None:
    for p in ps:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def resolve_native_op(kind: str = "recurrent", fork_root=None) -> Callable | None:
    """The fork's native prefill op (``kind`` 'recurrent' = exact ``gdn_prefill``, 'wmma' = fp16-operand
    ``gdn_prefill_wmma``), importing ``gdn_hip`` from the fork's torch-ext dir if needed; None when the built
    extension is absent (always on this CPU box)."""
    op = _gdn.native_gdn_op(kind)
    if op is None:
        _add_paths(fork_paths(fork_root)[1])
        op = _gdn.native_gdn_op(kind)
    return op


_FORK_REF: dict[str, Callable] = {}


def load_fork_reference(fork_root=None) -> Callable:
    """The fork's ``ref_gdn_prefill_core`` (``gdn_hip/autograd.py``), loaded read-only without its .so (the module
    prefix before ``_native_bwd_ok`` is pure torch), as ``scripts/gdn_bench.load_fork_reference``."""
    path = fork_paths(fork_root)[1] / "gdn_hip" / "autograd.py"
    key = str(path)
    if key not in _FORK_REF:
        src = path.read_text().replace("from ._ops import add_op_namespace_prefix, ops", "ops = None")
        ns: dict = {"__name__": "gdn_fork_ref"}
        exec(compile(src[: src.index("def _native_bwd_ok")], key, "exec"), ns)  # noqa: S102 - read-only fork code
        _FORK_REF[key] = ns["ref_gdn_prefill_core"]
    return _FORK_REF[key]


# ------------------------------------------------------------------------------------------------------------
# autograd.Functions: native forward, recompute backward
# ------------------------------------------------------------------------------------------------------------

def _native_forward(op: Callable, q, k, v, a, b, A_log, dt_bias) -> torch.Tensor:
    """One varlen call in the layout of the fork's ``_GDNPrefillBatchFn.forward`` (as ``gdn._native_call``): zero
    initial state per sequence, fresh state buffer [B+1, HV, V, K], A_log / dt_bias fp32 (kept fp64 for fp64
    inputs, which only the CPU stand-ins accept, for gradchecks)."""
    B, T, H, Kk = q.shape
    HV, Vv = v.shape[2], v.shape[3]
    dev = q.device
    pdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    ssm_state = torch.zeros(B + 1, HV, Vv, Kk, dtype=torch.float32, device=dev)
    cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    state_idx = torch.arange(1, B + 1, dtype=torch.long, device=dev)
    has_init = torch.zeros(B, dtype=torch.uint8, device=dev)
    out = op(q.reshape(B * T, H, Kk).contiguous(), k.reshape(B * T, H, Kk).contiguous(),
             v.reshape(B * T, HV, Vv).contiguous(), a.reshape(B * T, HV).contiguous(),
             b.reshape(B * T, HV).contiguous(), A_log.detach().to(pdt).contiguous(),
             dt_bias.detach().to(pdt).contiguous(), cu, state_idx, has_init, ssm_state, float(Kk ** -0.5), 1)
    return out.reshape(B, T, HV, Vv).to(v.dtype)


def _cast(g, ref):
    return None if g is None else g.to(ref.dtype)


class NativeFwdHFBwd(torch.autograd.Function):
    """core = op(raw) forward; backward = autograd through HF's prep + ``torch_chunk_gated_delta_rule`` recomputed
    from the saved raw tensors. Inputs: q, k [B,T,H,K], v [B,T,HV,V], a, b [B,T,HV], A_log, dt_bias [HV]. Output
    [B,T,HV,V] in v's dtype. ``bwd_group``: sequences recomputed per pass (None = all B at once). ``beta_fp32``:
    recompute with sigmoid(b) in fp32 (the native kernel's math) instead of HF's activation-dtype sigmoid."""

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, op, chunk_size: int, bwd_group, beta_fp32: bool = False):
        out = _native_forward(op, q, k, v, a, b, A_log, dt_bias)
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.cfg = (int(chunk_size), bwd_group, bool(beta_fp32))
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        need = ctx.needs_input_grad[:7]
        chunk_size, bwd_group, beta_fp32 = ctx.cfg
        B = saved[0].shape[0]
        step = B if not bwd_group else max(1, int(bwd_group))
        per_seq: list[list[torch.Tensor]] = [[] for _ in range(5)]
        shared: list[torch.Tensor | None] = [None, None]
        for s in range(0, B, step):
            sl = slice(s, min(B, s + step))
            raw = [(t[sl] if i < 5 else t).detach().requires_grad_(need[i]) for i, t in enumerate(saved)]
            ins = [t for t in raw if t.requires_grad]
            if not ins:
                break
            with torch.enable_grad():
                o = hf_core(*raw, chunk_size=chunk_size, beta_fp32=beta_fp32)
            gs = iter(torch.autograd.grad(o, ins, dout[sl].to(o.dtype), allow_unused=True))
            del o
            for i, t in enumerate(raw):
                gi = next(gs) if t.requires_grad else None
                if i < 5:
                    if gi is not None:
                        per_seq[i].append(gi)
                elif gi is not None:
                    shared[i - 5] = gi if shared[i - 5] is None else shared[i - 5] + gi
        out = [torch.cat(per_seq[i]) if per_seq[i] else None for i in range(5)] + shared
        return (*[_cast(g, t) for g, t in zip(out, saved)], None, None, None, None)


class NativeFwdForkRefBwd(torch.autograd.Function):
    """The fork's batched training pairing (``_GDNPrefillBatchFn``): op(raw) forward; backward = autograd through
    the fork's ``ref_gdn_prefill_core`` with B folded into the heads and chunk ``max(4, 64 // B)``."""

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, op, ref_fn):
        out = _native_forward(op, q, k, v, a, b, A_log, dt_bias)
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.ref_fn = ref_fn
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, a, b, A_log, dt_bias = ctx.saved_tensors
        B, T, H, Kk = q.shape
        HV = v.shape[2]

        def fold(t):                       # [B,T,X,...] -> [T, B*X, ...]
            p = t.permute(1, 0, *range(2, t.dim()))
            return p.reshape(T, B * t.shape[2], *t.shape[3:])
        with torch.enable_grad():
            ins = [fold(t).detach().float().requires_grad_(True) for t in (q, k, v, a, b)]
            Ad = A_log.detach().float().repeat(B).requires_grad_(True)
            dd = dt_bias.detach().float().repeat(B).requires_grad_(True)
            ref = ctx.ref_fn(*ins, Ad, dd, Kk ** -0.5, True, chunk=max(4, 64 // max(1, B)))
            gs = torch.autograd.grad(ref, ins + [Ad, dd], fold(dout).float())

        def unfold(g, X):
            g = g.reshape(T, B, X, *g.shape[2:])
            return g.permute(1, 0, *range(2, g.dim()))
        res = [unfold(gs[0], H), unfold(gs[1], H), unfold(gs[2], HV), unfold(gs[3], HV), unfold(gs[4], HV),
               gs[5].reshape(B, HV).sum(0), gs[6].reshape(B, HV).sum(0)]
        res = [r if n else None for r, n in zip(res, ctx.needs_input_grad[:7])]
        return (*[_cast(g, t) for g, t in zip(res, (q, k, v, a, b, A_log, dt_bias))], None, None)


def _capture_state(op: Callable, sink: list) -> Callable:
    """Wrap a native-signature op so the final state it writes (``ssm_state[state_indices]``, native [HV, V, K]
    layout) is appended to ``sink`` as fp32 [B, HV, K, V] (HF's layout)."""
    def run(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices, has_initial_state, ssm_state, scale, l2):
        out = op(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices, has_initial_state, ssm_state, scale, l2)
        sink.append(ssm_state.index_select(0, state_indices.long()).transpose(-1, -2).float().contiguous())
        return out
    return run


def gdn_core(q, k, v, a, b, A_log, dt_bias, *, fwd: str = "native", bwd: str = "hf",
             native_op: Callable | None = None, native_kind: str = "recurrent", chunk_size: int = 64,
             tri: str = "solve", max_bh: int | None = None, save_states: bool = True, bwd_group: int | None = None,
             fork_ref: Callable | None = None, beta_fp32: bool | None = None,
             state_sink: list | None = None) -> torch.Tensor:
    """The GDN core [B,T,HV,V] (v's dtype) from the raw layer tensors, by arm:

    ======  ========  ==================================================================================
    fwd     bwd       what runs
    ======  ========  ==================================================================================
    hf      hf        plain autograd through HF's prep + torch chunk delta rule (the reference)
    native  hf        ``NativeFwdHFBwd``: native forward, HF torch-chunk recompute backward (the fix)
    native  ours      ``gdn.RawGDN``: native forward, our hand-written chunked WY backward
    native  fork_ref  ``NativeFwdForkRefBwd``: the fork's reference recompute backward
    ours    ours      ``gdn.RawGDN`` with our torch forward
    hf      ours/hf.. as native, with ``hf_standin_op`` in place of the native op (CPU)
    ======  ========  ==================================================================================

    ``native_op`` overrides the op for fwd='native' (else the fork's ``gdn_prefill`` / ``gdn_prefill_wmma`` by
    ``native_kind``, imported from the fork; RuntimeError if absent). ``beta_fp32`` applies to the HF math
    (bwd='hf' recompute, and the fwd='hf' / bwd='hf' reference); None (default) = True exactly when the forward is
    a native-math op (fwd='native' with any op but ``hf_standin_op``), because the native kernel computes
    sigmoid(b) in fp32 and the recompute should differentiate the function the forward computed.
    ``state_sink``: a list; the fp32 final recurrent state [B, HV, K, V] (HF's cache layout, detached) is
    appended to it (used to fill a fresh HF cache)."""
    if fwd not in FWD_ARMS or bwd not in BWD_ARMS:
        raise ValueError(f"unknown arm fwd={fwd!r} bwd={bwd!r}")
    if fwd == "hf" and bwd == "hf":
        if state_sink is None:
            return hf_core(q, k, v, a, b, A_log, dt_bias, chunk_size=chunk_size, beta_fp32=bool(beta_fp32)).to(v.dtype)
        o, sf = hf_core(q, k, v, a, b, A_log, dt_bias, chunk_size=chunk_size, beta_fp32=bool(beta_fp32),
                        output_final_state=True)
        state_sink.append(sf.detach().float())
        return o.to(v.dtype)
    if fwd == "ours":
        if bwd != "ours":
            raise ValueError("fwd='ours' pairs only with bwd='ours' (our forward has no native-op form)")
        if state_sink is None:
            return _gdn.gated_delta_rule_from_raw(q, k, v, a, b, A_log, dt_bias, impl="ours", chunk_size=chunk_size,
                                                  tri=tri, max_bh=max_bh, save_states=save_states)
        o, sf = _gdn.gated_delta_rule_from_raw(q, k, v, a, b, A_log, dt_bias, impl="ours", chunk_size=chunk_size,
                                               tri=tri, max_bh=max_bh, save_states=save_states,
                                               return_final_state=True)
        state_sink.append(sf.detach().float())
        return o
    if fwd == "hf":
        op = hf_standin_op
    else:
        op = native_op or resolve_native_op(native_kind)
        if op is None:
            raise RuntimeError("fwd='native' needs gdn_hip (GPU build of the fork) or an injected native_op")
    if beta_fp32 is None:
        beta_fp32 = op is not hf_standin_op
    if state_sink is not None:
        op = _capture_state(op, state_sink)
    if bwd == "hf":
        return NativeFwdHFBwd.apply(q, k, v, a, b, A_log, dt_bias, op, chunk_size, bwd_group, beta_fp32)
    if bwd == "ours":
        return _gdn.gated_delta_rule_from_raw(q, k, v, a, b, A_log, dt_bias, impl="native", native_op=op,
                                              chunk_size=chunk_size, tri=tri, max_bh=max_bh)
    return NativeFwdForkRefBwd.apply(q, k, v, a, b, A_log, dt_bias, op, fork_ref or load_fork_reference())


# ------------------------------------------------------------------------------------------------------------
# HF layer patching over gdn_core
# ------------------------------------------------------------------------------------------------------------

@dataclass
class NativePatchConfig:
    fwd: str = "native"
    bwd: str = "hf"
    native_op: Callable | None = None
    native_kind: str = "recurrent"
    chunk_size: int = 64
    tri: str = "solve"
    max_bh: int | None = None
    save_states: bool = True
    bwd_group: int | None = None
    fork_ref: Callable | None = None
    beta_fp32: bool | None = None      # None: True for a native-math forward (see gdn_core)
    strict: str | bool = "grad"        # raise instead of falling back to HF: True / 'grad' (when grad flows) / False
    calls: int = 0                     # prefill calls served by gdn_core (fresh-cache prefills included)
    fallbacks: int = 0                 # calls handed to HF's own forward (cache with state, varlen, T=1 decode)
    fresh_cache_calls: int = 0         # of ``calls``: prefills that also filled a fresh HF cache
    layers: list[str] = field(default_factory=list)


class GDNFallbackError(RuntimeError):
    """A patched GDN layer would have run HF's own forward (fla / slow torch) while gradients flow."""


def _fresh_cache(self, cache_params, T: int) -> bool:
    """A cache that holds no state for this layer yet (a training step or a plain prefill with use_cache=True,
    HF's default): the delta rule starts from a zero state, so gdn_core serves it and fills the cache as HF
    would (same as ``gdn._patched_forward``)."""
    if T <= 1:
        return False
    try:
        return (cache_params.get_conv_state(self.layer_idx, state_idx=0) is None
                and cache_params.get_recurrent_state(self.layer_idx, state_idx=0) is None)
    except Exception:  # noqa: BLE001 - an unknown cache type: let HF handle it
        return False


def _patched_forward_native(self, hidden_states: torch.Tensor, cache_params=None, attention_mask=None, **kwargs):
    """HF's Qwen3_5(Moe)GatedDeltaNet.forward (transformers 5.22.2) for a zero-state prefill (no cache, or a cache
    with no state for this layer yet, which is then filled), with the delta rule replaced by ``gdn_core`` and the
    conv by HF's torch ``causal_conv1d_fn`` body. A cache that already holds state (cached prefill / decode) or
    varlen metadata goes to HF's own forward; while gradients flow that raises ``GDNFallbackError`` unless
    ``strict=False``, because HF's path would silently skip the patched kernel (and call fla when installed)."""
    cfg: NativePatchConfig = self._ngm_gdn_native
    T_in = hidden_states.shape[1]
    fresh = cache_params is not None and _fresh_cache(self, cache_params, T_in)
    if (cache_params is not None and not fresh) or kwargs.get("cu_seq_lens_q") is not None \
            or kwargs.get("cu_seqlens") is not None:
        grad = torch.is_grad_enabled() and (hidden_states.requires_grad
                                            or any(p.requires_grad for p in self.parameters()))
        if cfg.strict is True or (cfg.strict == "grad" and grad):
            raise GDNFallbackError(
                f"{type(self).__name__} layer {getattr(self, 'layer_idx', '?')}: a cache with state or varlen "
                "metadata would run HF's own GDN forward, not the patched one; pass use_cache=False for training "
                "(or patch with strict=False)")
        cfg.fallbacks += 1
        return type(self).forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask,
                                  **kwargs)
    cfg.calls += 1
    cfg.fresh_cache_calls += int(fresh)
    hf = sys.modules[type(self).__module__]
    hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if fresh:
        mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
    conv = hf_torch_fn("causal_conv1d_fn", hf)
    mixed_qkv = conv(mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)
    if fresh:
        mixed_qkv = mixed_qkv[:, :, -T:]
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(B, T, -1, self.head_k_dim)
    key = key.reshape(B, T, -1, self.head_k_dim)
    value = value.reshape(B, T, -1, self.head_v_dim)
    sink: list | None = [] if fresh else None
    core = gdn_core(query, key, value, a, b, self.A_log, self.dt_bias, fwd=cfg.fwd, bwd=cfg.bwd,
                    native_op=cfg.native_op, native_kind=cfg.native_kind, chunk_size=cfg.chunk_size, tri=cfg.tri,
                    max_bh=cfg.max_bh, save_states=cfg.save_states, bwd_group=cfg.bwd_group, fork_ref=cfg.fork_ref,
                    beta_fp32=cfg.beta_fp32, state_sink=sink)
    if fresh:
        cache_params.update_recurrent_state(sink[-1], self.layer_idx)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(B, T, -1))


def patch_gdn_native(model: nn.Module, *, fwd: str = "auto", bwd: str = "hf", native_op: Callable | None = None,
                     native_kind: str = "recurrent", chunk_size: int = 64, tri: str = "solve",
                     max_bh: int | None = None, save_states: bool = True, bwd_group: int | None = None,
                     fork_ref: Callable | None = None, beta_fp32: bool | None = None,
                     strict: str | bool = "grad") -> NativePatchConfig:
    """Override the forward of every Qwen3_5GatedDeltaNet / Qwen3_5MoeGatedDeltaNet in ``model`` (per instance;
    parameters and state-dict keys unchanged) with ``gdn_core(fwd, bwd)``. ``fwd='auto'``: 'native' when the
    model's parameters are on a GPU and gdn_hip (or ``native_op``) is available, else 'hf' (HF's torch forward;
    with bwd='hf' that is plain HF autograd, with bwd='ours' the stand-in plus our backward). The native op is
    resolved now. ``beta_fp32`` None: fp32 sigmoid(b) in the recompute exactly when the forward is native math.
    ``strict``: what a call that would go to HF's own forward does (see ``_patched_forward_native``); 'grad'
    raises only while gradients flow. Raises if no GDN layer is found. Undo with ``unpatch_gdn_native``. After a
    model-level step, ``cfg.calls == len(cfg.layers)`` and ``cfg.fallbacks == 0`` confirm every layer ran
    the patched kernel."""
    mods = gdn_modules(model)
    if not mods:
        raise ValueError(f"no {' / '.join(GDN_CLASS_NAMES)} found")
    if strict not in (True, False, "grad"):
        raise ValueError(f"strict must be True, False or 'grad', not {strict!r}")
    if fwd == "auto":
        p = next(model.parameters(), None)
        op = native_op or (resolve_native_op(native_kind) if p is not None and p.is_cuda else None)
        fwd, native_op = ("native", op) if (op is not None and p is not None and p.is_cuda) else ("hf", None)
    if fwd not in FWD_ARMS or bwd not in BWD_ARMS or (fwd == "ours" and bwd != "ours"):
        raise ValueError(f"unsupported arm fwd={fwd!r} bwd={bwd!r}")
    if fwd == "native":
        native_op = native_op or resolve_native_op(native_kind)
        if native_op is None:
            raise RuntimeError("fwd='native' but gdn_hip is not importable and no native_op was given")
    if beta_fp32 is None and fwd == "native":
        beta_fp32 = native_op is not hf_standin_op
    if bwd == "fork_ref" and fork_ref is None:
        fork_ref = load_fork_reference()
    cfg = NativePatchConfig(fwd=fwd, bwd=bwd, native_op=native_op, native_kind=native_kind, chunk_size=chunk_size,
                            tri=tri, max_bh=max_bh, save_states=save_states, bwd_group=bwd_group, fork_ref=fork_ref,
                            beta_fp32=beta_fp32, strict=strict, layers=[n for n, _ in mods])
    for _, m in mods:
        if "forward" in m.__dict__:
            raise RuntimeError(f"{type(m).__name__} already has a patched forward (gdn.patch_gdn?); unpatch first")
        m._ngm_gdn_native = cfg
        m.forward = types.MethodType(_patched_forward_native, m)
    return cfg


def unpatch_gdn_native(model: nn.Module) -> int:
    """Undo patch_gdn_native. Returns the number of layers restored."""
    n = 0
    for _, m in gdn_modules(model):
        if "_ngm_gdn_native" in m.__dict__:
            m.__dict__.pop("forward", None)
            del m.__dict__["_ngm_gdn_native"]
            n += 1
    return n


# ------------------------------------------------------------------------------------------------------------
# the fork's NativeGDNShim, widened to both class names (N9)
# ------------------------------------------------------------------------------------------------------------

@dataclass
class ForkShimPatch:
    """Record of ``patch_fork_shims``: (parent, attribute, original HF module or None) per replaced layer."""
    replaced: list[tuple[nn.Module, str, nn.Module | None]]
    class_names: dict[str, int]

    def __len__(self) -> int:
        return len(self.replaced)

    def restore(self) -> int:
        """Put the original HF modules back (only when patched with ``keep_originals=True``)."""
        n = 0
        for parent, name, orig in self.replaced:
            if orig is None:
                raise RuntimeError("patched with keep_originals=False; originals were not kept")
            setattr(parent, name, orig)
            n += 1
        self.replaced = []
        return n


def _import_fork_shim(fork_root=None):
    _add_paths(*fork_paths(fork_root))
    from minisgl.gdn.hf_patch import NativeGDNShim  # type: ignore[import-not-found]
    return NativeGDNShim


def patch_fork_shims(model: nn.Module, *, shim_cls: Callable[[nn.Module], nn.Module] | None = None,
                     fork_root=None, keep_originals: bool = False,
                     class_names: tuple[str, ...] = GDN_CLASS_NAMES) -> ForkShimPatch:
    """The fork's ``patch_qwen3_5_gdn`` (``R/gdn/hf_patch.py:98-107``) with the class-name test widened from
    ``== "Qwen3_5GatedDeltaNet"`` to ``in class_names`` (both HF GDN classes): every match is replaced in place by
    ``shim_cls(child)`` (default: the fork's ``NativeGDNShim``, imported lazily from ``R/python`` with the
    ``gdn_hip`` torch-ext dir on ``sys.path``; it needs the built extension at call time, i.e. the GPU box).
    Matches are collected before any replacement. Raises if nothing matches. ``keep_originals`` keeps the HF
    modules for ``restore()`` (costs their memory: the shim holds its own dense copy of the weights)."""
    shim_cls = shim_cls or _import_fork_shim(fork_root)
    todo = [(parent, name, child) for parent in model.modules() for name, child in parent.named_children()
            if type(child).__name__ in class_names]
    if not todo:
        raise ValueError(f"patch_fork_shims: no {' / '.join(class_names)} in model (0 layers patched)")
    counts: dict[str, int] = {}
    rec = ForkShimPatch([], counts)
    for parent, name, child in todo:
        counts[type(child).__name__] = counts.get(type(child).__name__, 0) + 1
        setattr(parent, name, shim_cls(child))
        rec.replaced.append((parent, name, child if keep_originals else None))
    return rec
