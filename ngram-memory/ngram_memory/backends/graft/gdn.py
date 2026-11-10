"""Gated-delta-rule (GDN) training kernel for frozen Qwen3.5 / Qwen3.5-MoE hosts (DESIGN §4.6, review F1).

Why this exists. The fork's ``minisgl.gdn.hf_patch.patch_qwen3_5_gdn`` runs a native ``gdn_hip`` forward, but
its backward recomputes a pure-torch reference with a Python forward-substitution loop per chunk
(``R/../../tools/vhip_patches/gdn/torch-ext/gdn_hip/autograd.py:108-191, 305-347, 502-535``): 24.1 aten ops
per token per layer at B=4 (``S1/rt-feas/gdn_ref_ops.out``). It also matches layers only by the exact class
name ``Qwen3_5GatedDeltaNet`` and therefore misses ``Qwen3_5MoeGatedDeltaNet`` (``S1/designrev/gdn_class_check.out``).

What this module provides.

* ``chunk_gated_delta_rule``: a drop-in for HF's ``torch_chunk_gated_delta_rule`` (same arguments, same
  [B, T, H, d] layout, same fp32 math) implemented as one ``torch.autograd.Function`` with a hand-written
  backward. Forward and backward use the chunked WY / UT-transform form (DeltaNet, Yang et al. 2024;
  flash-linear-attention ``chunk_gated_delta_rule``). Everything inside a chunk is batched over all chunks
  at once; the only Python loop is the unavoidable sequential recurrence over chunk-boundary states (one
  ``baddbmm`` per chunk forward and one per chunk backward). The forward saves only its inputs and the
  chunk-boundary states (``save_states=True``) or only its inputs (``save_states=False``, the backward then
  re-runs the boundary scan). Gradients flow to q, k, v, g, beta and the initial state.
* ``RawGDN`` (used by the layer patch): the same kernel on the layer's raw tensors, saving only those (bf16,
  before GQA expansion) and recomputing the cheap elementwise prep in the backward. Its forward is either our
  torch kernel or the fork's native ``gdn_hip.gdn_prefill`` (GPU only; the op is injected, so the plumbing is
  testable on CPU with a stand-in).
* ``patch_gdn(model)`` / ``unpatch_gdn(model)``: per-instance forward override of every
  ``Qwen3_5GatedDeltaNet`` and ``Qwen3_5MoeGatedDeltaNet``. State dict keys and parameters are untouched, so
  the projections may be ``host_quant.QuantLinear`` modules. Calls with a cache (decode / cached prefill) or
  packed varlen metadata fall back to HF's own forward (its torch delta rule).

Chunk-local algebra (per batch*head, chunk of C positions, G = cumsum(g) inside the chunk, Gam_ij =
exp(G_i - G_j) for i >= j else 0, q already l2-normed and scaled, k l2-normed)::

    A   = strict_lower(beta_i (k_i . k_j) Gam_ij)       Tm = (I + A)^-1
    U   = Tm (beta * v)                                 W  = Tm (beta * exp(G) * k)
    P   = (q k^T) * Gam                                 Qt = q * exp(G) - P W
    Kd  = k * exp(G_C - G)                              gam = exp(G_C)
    O_n = Qt S_n + P U                                  S_{n+1} = (gam I - Kd^T W) S_n + Kd^T U

This is algebraically the per-token recurrence S <- exp(g) S; S <- S + k beta (v - S^T k)^T; o = S^T q that
HF's ``torch_recurrent_gated_delta_rule`` implements, with HF's state layout S: [dk, dv].
"""
from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

GDN_CLASS_NAMES: tuple[str, ...] = ("Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet")

__all__ = [
    "GDN_CLASS_NAMES", "chunk_gated_delta_rule", "gated_delta_rule_from_raw", "RawGDN",
    "native_gdn_op", "patch_gdn", "unpatch_gdn", "gdn_modules", "GDNPatchConfig", "l2norm",
    "recurrent_gated_delta_rule_reference",
]


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """HF / fla convention: x * rsqrt(sum(x^2) + eps) (eps inside the rsqrt)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ------------------------------------------------------------------------------------------------------------
# chunk-local quantities (all batched over [BH, N] chunks)
# ------------------------------------------------------------------------------------------------------------

def _unit_lower_inverse(A: torch.Tensor, method: str) -> torch.Tensor:
    """(I + A)^-1 for strictly lower-triangular A [..., C, C], batched (no per-row loop).

    'solve' uses one batched triangular solve; 'doubling' uses the nilpotent product
    prod_j (I + (-A)^(2^j)), ceil(log2 C) batched matmul pairs (no LAPACK/rocSOLVER dependency)."""
    C = A.shape[-1]
    eye = torch.eye(C, dtype=A.dtype, device=A.device)
    if method == "solve":
        return torch.linalg.solve_triangular(A + eye, eye.expand_as(A), upper=False, unitriangular=True)
    if method == "doubling":
        X = -A
        T = eye + X
        P = X
        for _ in range(max(0, math.ceil(math.log2(C)) - 1)):
            P = P @ P
            T = T + T @ P
        return T
    raise ValueError(f"unknown triangular-inverse method {method!r}")


@dataclass
class _Intra:
    Gc: torch.Tensor      # [BH,N,C]   in-chunk cumulative log decay
    Gam: torch.Tensor     # [BH,N,C,C] exp(G_i - G_j), lower incl. diagonal, else 0
    KK: torch.Tensor      # [BH,N,C,C] k k^T
    QK: torch.Tensor      # [BH,N,C,C] q k^T
    Tm: torch.Tensor      # [BH,N,C,C] (I + A)^-1
    eG: torch.Tensor      # [BH,N,C]   exp(G)
    eR: torch.Tensor      # [BH,N,C]   exp(G_C - G)
    gam: torch.Tensor     # [BH,N]     exp(G_C)
    U: torch.Tensor       # [BH,N,C,dv]
    W: torch.Tensor       # [BH,N,C,dk]
    P: torch.Tensor       # [BH,N,C,C]
    Qt: torch.Tensor      # [BH,N,C,dk]
    Kd: torch.Tensor      # [BH,N,C,dk]


def _intra(q, k, v, g, beta, tri: str) -> _Intra:
    C = q.shape[-2]
    Gc = g.cumsum(-1)
    lower = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    Gam = (Gc.unsqueeze(-1) - Gc.unsqueeze(-2)).masked_fill(~lower, float("-inf")).exp()
    KK = k @ k.transpose(-1, -2)
    Tm = _unit_lower_inverse((KK * Gam * beta.unsqueeze(-1)).tril(-1), tri)
    eG = Gc.exp()
    U = Tm @ (v * beta.unsqueeze(-1))
    W = Tm @ (k * (beta * eG).unsqueeze(-1))
    QK = q @ k.transpose(-1, -2)
    P = QK * Gam
    Qt = q * eG.unsqueeze(-1) - P @ W
    Glast = Gc[..., -1:]
    eR = (Glast - Gc).exp()
    Kd = k * eR.unsqueeze(-1)
    gam = Glast.squeeze(-1).exp()
    return _Intra(Gc, Gam, KK, QK, Tm, eG, eR, gam, U, W, P, Qt, Kd)


def _transition(I: _Intra) -> torch.Tensor:
    """Mf_n = gam_n I - Kd_n^T W_n  [BH,N,dk,dk] (the chunk's state transition)."""
    dk = I.W.shape[-1]
    eye = torch.eye(dk, dtype=I.W.dtype, device=I.W.device)
    return I.gam[..., None, None] * eye - I.Kd.transpose(-1, -2) @ I.W


def _scan(Mf: torch.Tensor, cv: torch.Tensor, S0: torch.Tensor | None) -> torch.Tensor:
    """states[n] = S at the start of chunk n, states[N] = final. Returns [N+1, BH, dk, dv].
    The one sequential loop of the forward: S_{n+1} = Mf_n S_n + cv_n."""
    BH, N, dk, dv = cv.shape
    M = Mf.transpose(0, 1).contiguous()     # [N,BH,dk,dk]
    c = cv.transpose(0, 1).contiguous()     # [N,BH,dk,dv]
    out = cv.new_empty(N + 1, BH, dk, dv)
    if S0 is None:
        out[0].zero_()
    else:
        out[0].copy_(S0)
    for n in range(N):
        torch.baddbmm(c[n], M[n], out[n], out=out[n + 1])
    return out


def _scan_back(Mf: torch.Tensor, a: torch.Tensor, dSN: torch.Tensor | None) -> torch.Tensor:
    """dS[n] = dL/dS_n for n = 0..N. dS_n = a_n + Mf_n^T dS_{n+1}. Returns [N+1, BH, dk, dv]."""
    BH, N, dk, dv = a.shape
    MT = Mf.transpose(-1, -2).transpose(0, 1).contiguous()
    at = a.transpose(0, 1).contiguous()
    out = a.new_empty(N + 1, BH, dk, dv)
    if dSN is None:
        out[N].zero_()
    else:
        out[N].copy_(dSN)
    for n in range(N - 1, -1, -1):
        torch.baddbmm(at[n], MT[n], out[n + 1], out=out[n])
    return out


def _core_forward(q, k, v, g, beta, S0, tri: str, keep_states: bool):
    """q,k [BH,N,C,dk]; v [BH,N,C,dv]; g,beta [BH,N,C]; S0 [BH,dk,dv] | None.
    Returns (O [BH,N,C,dv], S_final [BH,dk,dv], states [N+1,BH,dk,dv] | None)."""
    I = _intra(q, k, v, g, beta, tri)
    Mf = _transition(I)
    cv = I.Kd.transpose(-1, -2) @ I.U
    states = _scan(Mf, cv, S0)
    del Mf, cv
    Sn = states[:-1].transpose(0, 1)        # [BH,N,dk,dv] (view)
    O = I.Qt @ Sn + I.P @ I.U
    S_final = states[-1].clone()
    return O, S_final, (states if keep_states else None)


def _core_backward(q, k, v, g, beta, S0, states, dO, dSN, tri: str):
    """Hand-written backward of _core_forward. Returns (dq, dk, dv, dg, dbeta, dS0) in the chunked layout."""
    I = _intra(q, k, v, g, beta, tri)
    Mf = _transition(I)
    if states is None:
        states = _scan(Mf, I.Kd.transpose(-1, -2) @ I.U, S0)
    a = I.Qt.transpose(-1, -2) @ dO                        # [BH,N,dk,dv]
    dS = _scan_back(Mf, a, dSN)
    del Mf, a
    Sn = states[:-1].transpose(0, 1)                         # S_n          [BH,N,dk,dv]
    dSn1 = dS[1:].transpose(0, 1)                            # dL/dS_{n+1}  [BH,N,dk,dv]
    dS0 = dS[0].clone()

    vnew = I.U - I.W @ Sn                                    # [BH,N,C,dv]
    dU = I.P.transpose(-1, -2) @ dO + I.Kd @ dSn1
    dW = -(dU @ Sn.transpose(-1, -2))
    dP = dO @ vnew.transpose(-1, -2)
    dQg = dO @ Sn.transpose(-1, -2)
    dKd = vnew @ dSn1.transpose(-1, -2)
    dgam = (dSn1 * Sn).sum((-1, -2))                         # [BH,N]
    del vnew, dSn1, dS

    TmT = I.Tm.transpose(-1, -2)
    dVb = TmT @ dU
    dKbg = TmT @ dW
    dA = -(dVb @ I.U.transpose(-1, -2) + dKbg @ I.W.transpose(-1, -2))
    dA = dA.tril(-1)
    del TmT, dU, dW

    bcol = beta.unsqueeze(-1)
    dKK = dA * I.Gam * bcol
    dbeta = (dA * I.KK * I.Gam).sum(-1)
    dGam = dA * I.KK * bcol
    dQK = dP * I.Gam
    dGam = dGam + dP * I.QK
    del dA, dP

    eGc = I.eG.unsqueeze(-1)
    dq = dQK @ k + dQg * eGc
    dk_ = (dKK + dKK.transpose(-1, -2)) @ k + dQK.transpose(-1, -2) @ q + dKd * I.eR.unsqueeze(-1) \
        + dKbg * (beta * I.eG).unsqueeze(-1)
    dv_ = dVb * bcol
    dbeta = dbeta + (dKbg * k).sum(-1) * I.eG + (dVb * v).sum(-1)

    rKd = (dKd * I.Kd).sum(-1)                               # [BH,N,C]
    dGc = (dQg * q).sum(-1) * I.eG + (dKbg * k).sum(-1) * beta * I.eG - rKd
    X = dGam * I.Gam
    dGc = dGc + X.sum(-1) - X.sum(-2)
    dGlast = rKd.sum(-1) + dgam * I.gam
    dGc[..., -1] += dGlast
    dg = dGc.flip(-1).cumsum(-1).flip(-1)
    return dq, dk_, dv_, dg, dbeta, dS0


def _to_chunks(x: torch.Tensor, C: int) -> torch.Tensor:
    """[BH, T, ...] -> [BH, N, C, ...], zero-padded at the end."""
    T = x.shape[1]
    pad = (-T) % C
    if pad:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, pad))
    return x.reshape(x.shape[0], -1, C, *x.shape[2:])


def _groups(BH: int, max_bh: int | None) -> Iterator[slice]:
    step = BH if not max_bh else max(1, int(max_bh))
    for s in range(0, BH, step):
        yield slice(s, min(BH, s + step))


def _chunk_forward(q, k, v, g, beta, S0, C: int, tri: str, save_states: bool, max_bh):
    """q,k [BH,T,dk] (l2-normed, scaled), v [BH,T,dv], g,beta [BH,T], S0 [BH,dk,dv] | None.
    Returns (O [BH,T,dv], S_final [BH,dk,dv], states [N+1,BH,dk,dv] | None)."""
    BH, T, dk = q.shape
    dv = v.shape[-1]
    qc, kc, vc = _to_chunks(q, C), _to_chunks(k, C), _to_chunks(v, C)
    gc, bc = _to_chunks(g, C), _to_chunks(beta, C)
    N = qc.shape[1]
    O = vc.new_empty(BH, N, C, dv)
    Sf = vc.new_empty(BH, dk, dv)
    states = vc.new_empty(N + 1, BH, dk, dv) if save_states else None
    for sl in _groups(BH, max_bh):
        o, sf, st = _core_forward(qc[sl], kc[sl], vc[sl], gc[sl], bc[sl], None if S0 is None else S0[sl], tri,
                                  save_states)
        O[sl] = o
        Sf[sl] = sf
        if save_states:
            states[:, sl] = st
    return O.reshape(BH, N * C, dv)[:, :T], Sf, states


def _chunk_backward(q, k, v, g, beta, S0, states, dO, dSf, C: int, tri: str, max_bh):
    """Gradients (dq, dk, dv, dg, dbeta [BH,T,...], dS0 | None) of _chunk_forward. dO / dSf may be None."""
    BH, T, dk = q.shape
    dv = v.shape[-1]
    qc, kc, vc = _to_chunks(q, C), _to_chunks(k, C), _to_chunks(v, C)
    gc, bc = _to_chunks(g, C), _to_chunks(beta, C)
    N = qc.shape[1]
    dOc = _to_chunks(dO.contiguous(), C) if dO is not None else vc.new_zeros(BH, N, C, dv)
    dq, dk_, dv_ = torch.empty_like(qc), torch.empty_like(kc), torch.empty_like(vc)
    dg, db = torch.empty_like(gc), torch.empty_like(bc)
    dS0 = None if S0 is None else torch.empty_like(S0)
    for sl in _groups(BH, max_bh):
        r = _core_backward(qc[sl], kc[sl], vc[sl], gc[sl], bc[sl], None if S0 is None else S0[sl],
                           None if states is None else states[:, sl], dOc[sl], None if dSf is None else dSf[sl], tri)
        dq[sl], dk_[sl], dv_[sl], dg[sl], db[sl] = r[:5]
        if dS0 is not None:
            dS0[sl] = r[5]

    def un(x):
        return x.reshape(BH, N * C, *x.shape[3:])[:, :T]
    return un(dq), un(dk_), un(dv_), un(dg), un(db), dS0


class _ChunkGDR(torch.autograd.Function):
    """Tensor-level: q,k [BH,T,dk] (already l2-normed and scaled), v [BH,T,dv], g,beta [BH,T], S0 | None."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, S0, C: int, tri: str, save_states: bool, max_bh):
        ctx.set_materialize_grads(False)
        O, Sf, states = _chunk_forward(q, k, v, g, beta, S0, C, tri, save_states, max_bh)
        ctx.save_for_backward(q, k, v, g, beta, S0, states)
        ctx.cfg = (C, tri, max_bh)
        return O, Sf

    @staticmethod
    def backward(ctx, dO, dSf):
        q, k, v, g, beta, S0, states = ctx.saved_tensors
        return (*_chunk_backward(q, k, v, g, beta, S0, states, dO, dSf, *ctx.cfg), None, None, None, None)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    save_states: bool = True,
    tri: str = "solve",
    max_bh: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for HF ``torch_chunk_gated_delta_rule``: query/key [B,T,H,dk], value [B,T,H,dv], g/beta [B,T,H],
    initial_state [B,H,dk,dv]. Returns (out [B,T,H,dv] in query's dtype, final state [B,H,dk,dv] fp32 (fp64 for fp64 input) or None).

    Differentiable w.r.t. every tensor input. ``max_bh`` processes the batch*heads in groups of that size
    (bounds peak memory; more launches). ``kwargs`` (e.g. ``cu_seqlens``) are accepted and ignored, as HF does."""
    dtype = query.dtype
    B, T, H, dk = key.shape
    dv = value.shape[-1]
    cdt = torch.float64 if query.dtype == torch.float64 else torch.float32   # fp64 only for gradchecks
    q, k, v, b_, g_ = (x.transpose(1, 2).to(cdt) for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q = q * (q.shape[-1] ** -0.5)
    S0 = None if initial_state is None else initial_state.to(cdt).reshape(B * H, dk, dv)
    o, Sf = _ChunkGDR.apply(q.reshape(B * H, T, dk), k.reshape(B * H, T, dk), v.reshape(B * H, T, dv),
                            g_.reshape(B * H, T), b_.reshape(B * H, T), S0, int(chunk_size), tri,
                            bool(save_states), max_bh)
    o = o.reshape(B, H, T, dv).transpose(1, 2).to(dtype)
    return o, (Sf.reshape(B, H, dk, dv) if output_final_state else None)


def recurrent_gated_delta_rule_reference(q, k, v, g, beta, initial_state=None, use_qk_l2norm_in_kernel=True):
    """Per-token fp64-capable reference (HF ``torch_recurrent_gated_delta_rule`` math), for tests."""
    q, k, v, beta, g = (x.transpose(1, 2) for x in (q, k, v, beta, g))
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    q = q * q.shape[-1] ** -0.5
    B, H, T, dk = k.shape
    S = q.new_zeros(B, H, dk, v.shape[-1]) if initial_state is None else initial_state.to(q.dtype)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kv = (S * k[:, :, t, :, None]).sum(-2)
        d = (v[:, :, t] - kv) * beta[:, :, t, None]
        S = S + k[:, :, t, :, None] * d[:, :, None, :]
        outs.append((S * q[:, :, t, :, None]).sum(-2))
    return torch.stack(outs, 2).transpose(1, 2), S


# ------------------------------------------------------------------------------------------------------------
# native gdn_hip forward + our backward
# ------------------------------------------------------------------------------------------------------------

def native_gdn_op(kind: str = "recurrent") -> Callable | None:
    """The fork's native prefill op (``gdn_hip`` ``gdn_prefill`` = exact recurrent, ``gdn_prefill_wmma`` =
    fp16-operand chunked), or None when the extension is not importable (always the case on CPU here)."""
    try:
        from gdn_hip._ops import ops  # type: ignore[import-not-found]
    except Exception:
        return None
    name = "gdn_prefill" if kind == "recurrent" else "gdn_prefill_wmma"
    return getattr(ops, name, None)


def _native_call(op: Callable, q, k, v, a, b, A_log, dt_bias, scale: float) -> torch.Tensor:
    """Batched [B,T] call in the layout of the fork's _GDNPrefillBatchFn.forward (autograd.py:480-500):
    one varlen launch, zero initial state per sequence, fresh state buffer [B+1, HV, V, K]."""
    B, T, H, Kk = q.shape
    HV, Vv = v.shape[2], v.shape[3]
    dev = q.device
    ssm_state = torch.zeros(B + 1, HV, Vv, Kk, dtype=torch.float32, device=dev)
    cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    state_idx = torch.arange(1, B + 1, dtype=torch.long, device=dev)
    has_init = torch.zeros(B, dtype=torch.uint8, device=dev)
    out = op(q.reshape(B * T, H, Kk).contiguous(), k.reshape(B * T, H, Kk).contiguous(),
             v.reshape(B * T, HV, Vv).contiguous(), a.reshape(B * T, HV).contiguous(),
             b.reshape(B * T, HV).contiguous(), A_log.float().contiguous(), dt_bias.float().contiguous(), cu,
             state_idx, has_init, ssm_state,
             float(scale), 1)
    return out.reshape(B, T, HV, Vv)


def _prep_raw(q, k, v, a, b, A_log, dt_bias):
    """Raw layer tensors -> (qn [BH,T,dk] l2-normed+scaled+GQA-expanded, kn, v, g, beta) fp32, HF math."""
    B, T, H, dk = q.shape
    HV, dv = v.shape[2], v.shape[3]
    rep = HV // H
    beta = b.sigmoid().float()                 # HF order: sigmoid in the activation dtype, then fp32
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    qf, kf = q.float(), k.float()
    if rep > 1:
        qf = qf.repeat_interleave(rep, dim=2)
        kf = kf.repeat_interleave(rep, dim=2)
    qn = l2norm(qf) * dk ** -0.5
    kn = l2norm(kf)
    tr = lambda x: x.transpose(1, 2).reshape(B * HV, T, *x.shape[3:])  # noqa: E731
    return tr(qn), tr(kn), tr(v.float()), tr(g), tr(beta)


class RawGDN(torch.autograd.Function):
    """The GDN core from the layer's raw tensors (q,k [B,T,H,dk] pre-norm, v [B,T,HV,dv], a,b [B,T,HV], A_log,
    dt_bias [HV]) -> core [B,T,HV,dv] (fp32). Saves only the raw tensors (plus, for our forward with
    ``save_states``, the fp32 chunk-boundary states); the backward recomputes the cheap elementwise prep
    (l2norm, scale, GQA expansion, softplus/sigmoid gates) with autograd, runs the chunked core backward, and
    pulls the gradients through the prep. Forward: our chunked torch kernel (``op=None``) or an injected native
    op with gdn_hip.gdn_prefill's signature (boundary states then recomputed in the backward)."""

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, op, C: int, tri: str, max_bh, save_states: bool):
        B, T, HV, dv = v.shape
        states = None
        if op is not None:
            out = _native_call(op, q, k, v, a, b, A_log, dt_bias, q.shape[-1] ** -0.5).float()
            Sf = out.new_empty(0)   # final state not returned on the native path
        else:
            qn, kn, vv, gg, bb = _prep_raw(q, k, v, a, b, A_log, dt_bias)
            O, Sf, states = _chunk_forward(qn, kn, vv, gg, bb, None, C, tri, save_states, max_bh)
            out = O.reshape(B, HV, T, dv).transpose(1, 2)
            Sf = Sf.reshape(B, HV, *Sf.shape[1:])
            del qn, kn, vv, gg, bb
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias, states)
        ctx.cfg = (C, tri, max_bh)
        ctx.mark_non_differentiable(Sf)
        return out, Sf

    @staticmethod
    def backward(ctx, dout, dSf=None):
        q, k, v, a, b, A_log, dt_bias, states = ctx.saved_tensors
        C, tri, max_bh = ctx.cfg
        B, T, HV, dv = v.shape
        raw = [t.detach().requires_grad_(ctx.needs_input_grad[i])
               for i, t in enumerate((q, k, v, a, b, A_log, dt_bias))]
        with torch.enable_grad():
            prep = _prep_raw(*raw)
        qn, kn, vv, gg, bb = (t.detach() for t in prep)
        dOr = dout.float().transpose(1, 2).reshape(B * HV, T, dv)
        grads = _chunk_backward(qn, kn, vv, gg, bb, None, states, dOr, None, C, tri, max_bh)
        del qn, kn, vv, gg, bb, dOr
        want = [(t, gr) for t, gr in zip(prep, grads[:5]) if t.requires_grad]
        ins = [t for t in raw if t.requires_grad]
        res = torch.autograd.grad([t for t, _ in want], ins, [gr for _, gr in want], allow_unused=True) \
            if ins and want else []
        it = iter(res)
        out = [next(it) if t.requires_grad else None for t in raw]
        out = [None if gr is None else gr.to(t.dtype) for gr, t in zip(out, (q, k, v, a, b, A_log, dt_bias))]
        return (*out, None, None, None, None, None)


def gated_delta_rule_from_raw(q, k, v, a, b, A_log, dt_bias, *, impl: str = "ours", native_op: Callable | None = None,
                              chunk_size: int = 64, tri: str = "solve", max_bh: int | None = None,
                              save_states: bool = True, return_final_state: bool = False):
    """The GDN core from the layer's raw tensors (q,k [B,T,H,dk] pre-norm, v [B,T,HV,dv], a,b [B,T,HV]).
    ``return_final_state``: also return the fp32 final recurrent state [B,HV,dk,dv] (impl 'ours'; None for
    'native'), non-differentiable, in HF's cache layout.
    impl 'ours' = our torch forward + our backward; 'native' = ``native_op`` forward + our backward; 'hf_math' =
    the same prep as HF followed by ``chunk_gated_delta_rule`` with autograd through the prep (more saved
    memory; kept for cross-checks). Output dtype = v's."""
    if impl == "native":
        if native_op is None:
            raise RuntimeError("impl='native' needs native_op (gdn_hip not importable here)")
        o, sf = RawGDN.apply(q, k, v, a, b, A_log, dt_bias, native_op, chunk_size, tri, max_bh, False)
        return (o.to(v.dtype), None) if return_final_state else o.to(v.dtype)
    if impl == "ours":
        o, sf = RawGDN.apply(q, k, v, a, b, A_log, dt_bias, None, chunk_size, tri, max_bh, save_states)
        return (o.to(v.dtype), sf) if return_final_state else o.to(v.dtype)
    if impl != "hf_math":
        raise ValueError(f"unknown impl {impl!r}")
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    rep = v.shape[2] // q.shape[2]
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    o, sf = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=chunk_size, use_qk_l2norm_in_kernel=True,
                                   tri=tri, max_bh=max_bh, save_states=save_states)
    return (o, sf) if return_final_state else o


# ------------------------------------------------------------------------------------------------------------
# HF layer patching
# ------------------------------------------------------------------------------------------------------------

@dataclass
class GDNPatchConfig:
    impl: str = "ours"                 # 'ours' | 'native'
    native_op: Callable | None = None
    chunk_size: int = 64
    tri: str = "solve"
    max_bh: int | None = None
    save_states: bool = True
    calls: int = 0                     # prefill calls served by our kernel
    fallbacks: int = 0                 # cached / varlen calls handed to HF's own forward


def gdn_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every HF GDN mixer in ``model`` (both class names)."""
    return [(n, m) for n, m in model.named_modules() if type(m).__name__ in GDN_CLASS_NAMES]


def _patched_forward(self, hidden_states: torch.Tensor, cache_params=None, attention_mask=None, **kwargs):
    cfg: GDNPatchConfig = self._ngm_gdn
    T_in = hidden_states.shape[1]
    # A cache that holds no state for this layer yet (a plain prefill / training step with use_cache=True) has a
    # zero initial state: our kernel serves it and fills the cache exactly as HF would (VGK-1). Cached decode,
    # cached prefill with a previous state, varlen packing, or the native path with a cache go to HF.
    fresh = False
    if cache_params is not None:
        fresh = (cfg.impl == "ours" and T_in > 1
                 and cache_params.get_conv_state(self.layer_idx, state_idx=0) is None
                 and cache_params.get_recurrent_state(self.layer_idx, state_idx=0) is None)
    if (cache_params is not None and not fresh) or kwargs.get("cu_seq_lens_q") is not None \
            or kwargs.get("cu_seqlens") is not None:
        cfg.fallbacks += 1
        return type(self).forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask,
                                  **kwargs)
    cfg.calls += 1
    hf = sys.modules[type(self).__module__]
    hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if cache_params is not None:
        mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
    mixed_qkv = hf.causal_conv1d_fn(mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                    activation=self.activation)
    if cache_params is not None:
        mixed_qkv = mixed_qkv[:, :, -T:]
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(B, T, -1, self.head_k_dim)
    key = key.reshape(B, T, -1, self.head_k_dim)
    value = value.reshape(B, T, -1, self.head_v_dim)
    core, sf = gated_delta_rule_from_raw(query, key, value, a, b, self.A_log, self.dt_bias, impl=cfg.impl,
                                         native_op=cfg.native_op, chunk_size=cfg.chunk_size, tri=cfg.tri,
                                         max_bh=cfg.max_bh, save_states=cfg.save_states, return_final_state=True)
    if cache_params is not None:
        cache_params.update_recurrent_state(sf, self.layer_idx)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(B, T, -1))


def patch_gdn(model: nn.Module, *, impl: str = "auto", native_op: Callable | None = None, chunk_size: int = 64,
              tri: str = "solve", max_bh: int | None = None, save_states: bool = True) -> GDNPatchConfig:
    """Override the forward of every Qwen3_5GatedDeltaNet / Qwen3_5MoeGatedDeltaNet in ``model`` (in place,
    per instance; parameters and state-dict keys unchanged). impl: 'ours', 'native' (needs ``native_op`` or an
    importable gdn_hip), or 'auto' (native on CUDA/ROCm tensors when importable, else ours; resolved now).
    Returns the shared config; ``len(gdn_modules(model))`` layers are patched (0 raises)."""
    mods = gdn_modules(model)
    if not mods:
        raise ValueError("no Qwen3_5GatedDeltaNet / Qwen3_5MoeGatedDeltaNet found")
    if impl == "auto":
        op = native_op or native_gdn_op()
        on_gpu = next(model.parameters()).is_cuda if any(True for _ in model.parameters()) else False
        impl, native_op = ("native", op) if (op is not None and on_gpu) else ("ours", None)
    elif impl == "native":
        native_op = native_op or native_gdn_op()
        if native_op is None:
            raise RuntimeError("impl='native' but gdn_hip is not importable")
    elif impl != "ours":
        raise ValueError(f"unknown impl {impl!r}")
    cfg = GDNPatchConfig(impl=impl, native_op=native_op, chunk_size=chunk_size, tri=tri, max_bh=max_bh,
                         save_states=save_states)
    for _, m in mods:
        m._ngm_gdn = cfg
        m.forward = types.MethodType(_patched_forward, m)
    return cfg


def unpatch_gdn(model: nn.Module) -> int:
    """Undo patch_gdn. Returns the number of layers restored."""
    n = 0
    for _, m in gdn_modules(model):
        if "forward" in m.__dict__:
            del m.forward
            n += 1
        m.__dict__.pop("_ngm_gdn", None)
    return n
