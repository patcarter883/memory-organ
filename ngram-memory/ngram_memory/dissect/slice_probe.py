"""§D experiment S: a 2-layer Flash-Next slice (embedding + decoder layer 0 + the layer-1 PLE module) on CPU.

* Weights come from raw .bin files written by `scripts/fetch_slice.py` (bf16 as stored), read through np.memmap;
  dense layer-0 modules are transformers' own q4 classes built from a SimpleNamespace config, the 512 routed
  experts are streamed one at a time from the memmap (never all in fp32), so peak RSS stays ~2 GB.
* The PLE module is re-expressed op-for-op (`PLEProbe.parts`) so every intermediate (gate logit, gated value,
  normalised conv input, per-tap conv contributions) is visible; `PLEProbe.check_against_q4` compares its output
  with `q4.Qwen4ExpTextPLELayer` (built with a SparseRowEmbedding over the fetched rows) on the same input.
* Everything runs in fp32 on the bf16 weights (the served model computes in bf16; statistics only).

Notation (q4:1229-1249): e_t = concat of 16 rows (2560); v = W_v e; k_s = N_key(W_k e)_s; q_s = N_query(X_t)_s;
g_s = ssqrt(<k_s, q_s>/sqrt(d)); u_s = sigmoid(g_s) v; c = silu(sum_j w_j * N_conv(u)_{t-3(3-j)}); out_s = u_s + c_s.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4

L0 = "model.language_model.layers.0."
PLE = "model.language_model.layers.1.ple."


# ----------------------------------------------------------------------------------------------- weights
class SliceWeights:
    """Name -> tensor access over raw bf16 .bin files (manifest.json from fetch_slice.py) or an in-memory dict."""

    def __init__(self, root: str | Path | None = None, tensors: Mapping[str, torch.Tensor] | None = None):
        self.root = Path(root) if root is not None else None
        self.tensors = dict(tensors or {})
        self.man: dict = json.loads((self.root / "manifest.json").read_text()) if self.root is not None else {}
        self._mm: dict[str, np.memmap] = {}

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor]) -> "SliceWeights":
        return cls(None, tensors)

    def names(self) -> list[str]:
        return sorted(set(self.man) | set(self.tensors))

    def memmap(self, name: str) -> np.memmap:
        if name not in self._mm:
            m = self.man[name]
            if m["dtype"] != "BF16":
                raise TypeError(f"{name}: {m['dtype']}")
            self._mm[name] = np.memmap(self.root / m["path"], dtype=np.uint16, mode="r", shape=tuple(m["shape"]))
        return self._mm[name]

    def _pread(self, name: str, start: int, count: int) -> np.ndarray:
        """`count` bf16 words from element `start` of a raw file via os.pread (no mapping, so the bytes do not stay
        resident in this process: a memmap of the 5 GB experts counts toward RSS once touched)."""
        m = self.man[name]
        if m["dtype"] != "BF16":
            raise TypeError(f"{name}: {m['dtype']}")
        fd = os.open(self.root / m["path"], os.O_RDONLY)
        try:
            buf = os.pread(fd, 2 * count, 2 * start)
        finally:
            os.close(fd)
        if len(buf) != 2 * count:
            raise IOError(f"short read {name}")
        return np.frombuffer(buf, dtype=np.uint16)

    def t(self, name: str) -> torch.Tensor:
        """fp32 copy of a whole tensor."""
        if name in self.tensors:
            return self.tensors[name].detach().float().clone()
        shape = tuple(self.man[name]["shape"])
        return bf16_np_to_f32(self._pread(name, 0, int(np.prod(shape))).reshape(shape))

    def slab(self, name: str, i: int) -> torch.Tensor:
        """fp32 copy of tensor[i] (one expert)."""
        if name in self.tensors:
            return self.tensors[name][i].detach().float().clone()
        shape = tuple(self.man[name]["shape"])
        n = int(np.prod(shape[1:]))
        return bf16_np_to_f32(self._pread(name, i * n, n).reshape(shape[1:]))


def bf16_np_to_f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).float()


def text_namespace(text_cfg: Mapping) -> SimpleNamespace:
    """Attributes the q4 layer-0 classes read (no Qwen4ExpTextConfig validation)."""
    c = dict(text_cfg)
    return SimpleNamespace(
        hidden_size=c["hidden_size"], hc_count=c["hc_count"], hc_lowrank=c["hc_lowrank"],
        rms_norm_eps=c.get("rms_norm_eps", 1e-6), linear_num_value_heads=c["linear_num_value_heads"],
        linear_num_key_heads=c["linear_num_key_heads"], linear_key_head_dim=c["linear_key_head_dim"],
        linear_value_head_dim=c["linear_value_head_dim"], linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
        hidden_act=c.get("hidden_act", "silu"), output_gate_type=c.get("output_gate_type"),
        layer_types=list(c["layer_types"]), num_experts=c["num_experts"],
        num_experts_per_tok=c["num_experts_per_tok"], norm_topk_prob=c.get("norm_topk_prob", True),
        moe_intermediate_size=c["moe_intermediate_size"],
        shared_expert_intermediate_size=c["shared_expert_intermediate_size"],
        intermediate_size=c.get("intermediate_size", c["shared_expert_intermediate_size"]))


def _load(mod: torch.nn.Module, w: SliceWeights, prefix: str) -> None:
    sd = {k: w.t(prefix + k) for k in mod.state_dict()}
    mod.load_state_dict(sd, strict=True)


# --------------------------------------------------------------------------------------------- layer 0
class Layer0:
    """Flash-Next decoder layer 0 (GDN + MoE with hyper-connections), forward only, experts streamed."""

    def __init__(self, w: SliceWeights, text_cfg: Mapping):
        ns = self.ns = text_namespace(text_cfg)
        self.w = w
        self.d, self.hc = ns.hidden_size, ns.hc_count
        with torch.no_grad():
            self.attn_hc = q4.Qwen4ExpTextGatedResidual(ns).eval()
            _load(self.attn_hc, w, L0 + "attn_hyper_connection.")
            self.gdn = q4.Qwen4ExpTextGatedDeltaNet(ns, 0).eval()
            _load(self.gdn, w, L0 + "linear_attn.")
            self.mlp_hc = q4.Qwen4ExpTextGatedResidual(ns).eval()
            _load(self.mlp_hc, w, L0 + "mlp_hyper_connection.")
            self.shared = q4.Qwen4ExpTextMLP(ns, intermediate_size=ns.shared_expert_intermediate_size).eval()
            _load(self.shared, w, L0 + "mlp.shared_expert.")
        self.shared_gate = w.t(L0 + "mlp.shared_expert_gate.weight")
        self.router = w.t(L0 + "mlp.gate.weight")
        self.expert_hits = np.zeros(ns.num_experts, dtype=np.int64)

    @torch.no_grad()
    def attn_part(self, x0: torch.Tensor) -> torch.Tensor:
        """x0 [B, T, hc*d] -> state after the attention block's injection."""
        h, hyper, inj = self.attn_hc(x0)
        h = self.gdn(h)
        return hyper + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

    @torch.no_grad()
    def mlp_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mlp_hc(x)

    @torch.no_grad()
    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q4 TopKRouter: softmax over experts (fp32), top-k, renormalise. x [N, d] -> (weights [N,k], idx [N,k])."""
        logits = F.linear(x, self.router)
        p = torch.softmax(logits, dim=-1, dtype=torch.float)
        top, idx = torch.topk(p, self.ns.num_experts_per_tok, dim=-1)
        if self.ns.norm_topk_prob:
            top = top / top.sum(-1, keepdim=True)
        return top, idx

    @torch.no_grad()
    def moe(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse MoE block on x [N, d] (routed experts streamed from the memmap + gated shared expert)."""
        top, idx = self.route(x)
        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.ns.num_experts)
        self.expert_hits += counts.numpy()
        tok_of = torch.arange(x.shape[0]).repeat_interleave(idx.shape[1])
        wts = top.reshape(-1)
        start = 0
        for e in range(self.ns.num_experts):
            n = int(counts[e])
            if n == 0:
                continue
            sel = order[start:start + n]
            start += n
            toks = tok_of[sel]
            gu = self.w.slab(L0 + "mlp.experts.gate_up_proj", e)
            dn = self.w.slab(L0 + "mlp.experts.down_proj", e)
            g, u = F.linear(x[toks], gu).chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, dn) * wts[sel, None]
            out.index_add_(0, toks, y)
        sh = torch.sigmoid(F.linear(x, self.shared_gate)) * self.shared(x)
        return out + sh

    @torch.no_grad()
    def forward_docs(self, embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        """Per-document embeddings [T_i, d] -> layer-1 input X1 [T_i, hc*d] (docs are independent sequences; the
        MoE runs once over all their tokens so each expert is read once per call)."""
        mids, pre = [], []
        for e in embeds:
            x0 = e[None].repeat(1, 1, self.hc)
            mid = self.attn_part(x0)
            mixed, hyper, inj = self.mlp_pre(mid)
            pre.append((mixed[0], hyper[0], inj[0]))
        allx = torch.cat([p[0] for p in pre])
        mo = self.moe(allx)
        out, s = [], 0
        for mixed, hyper, inj in pre:
            T = mixed.shape[0]
            m = mo[s:s + T]
            s += T
            out.append(hyper + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2))
        return out


# --------------------------------------------------------------------------------------------- the PLE
def rms_group(x: torch.Tensor, weight: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm with group_size (scale 1 + w), fp32."""
    sh = x.shape
    y = x.float().reshape(*sh[:-1], -1, group)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.reshape(sh) * (1.0 + weight.float())


class PLEProbe:
    """Op-for-op PLE forward exposing every intermediate (hc streams, one shared value)."""

    def __init__(self, w: SliceWeights, hidden: int = 2560, hc: int = 4, n_heads: int = 16, eps: float = 1e-6,
                 dilation: int = 3, prefix: str = PLE):
        self.d, self.hc, self.H, self.eps, self.dil = hidden, hc, n_heads, eps, dilation
        self.Wk = w.t(prefix + "key_proj.weight")
        self.Wv = w.t(prefix + "value_proj.weight")
        self.nk = w.t(prefix + "norm_key.weight")
        self.nq = w.t(prefix + "norm_query.weight")
        self.nc = w.t(prefix + "norm_conv.weight")
        self.conv = w.t(prefix + "conv1d.weight")[:, 0, :]      # [hc*d, K]
        self.K = self.conv.shape[1]
        self.rw = self.Wv.shape[1] // n_heads

    def head_mask(self, heads) -> torch.Tensor:
        m = torch.zeros(self.Wv.shape[1])
        for h in heads:
            m[h * self.rw:(h + 1) * self.rw] = 1.0
        return m

    @torch.no_grad()
    def gate(self, x1: torch.Tensor, e: torch.Tensor, q: torch.Tensor | None = None):
        """(g [T, hc], v [T, d]) for embeddings e [T, E] and layer-1 input x1 [T, hc*d]."""
        k = rms_group(F.linear(e, self.Wk), self.nk, self.d, self.eps).unflatten(-1, (self.hc, self.d))
        if q is None:
            q = rms_group(x1, self.nq, self.d, self.eps).unflatten(-1, (self.hc, self.d))
        raw = (k * q).sum(-1) / math.sqrt(self.d)
        g = raw.abs().clamp_min(1e-6).sqrt() * raw.sign()
        return g, F.linear(e, self.Wv), q

    @torch.no_grad()
    def parts(self, x1: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """One document (no cache): x1 [T, hc*d], e [T, E] -> intermediates."""
        T = x1.shape[0]
        g, v, q = self.gate(x1, e)
        s = torch.sigmoid(g)                                            # [T, hc]
        u = s.unsqueeze(-1) * v.unsqueeze(-2)                           # [T, hc, d]
        n = rms_group(u.flatten(-2), self.nc, self.d, self.eps)         # [T, hc*d]
        pad = (self.K - 1) * self.dil
        npad = F.pad(n.T, (pad, 0))                                     # [C, pad+T]
        taps = torch.stack([self.conv[:, j:j + 1] * npad[:, j * self.dil:j * self.dil + T]
                            for j in range(self.K)])                    # [K, C, T]; tap K-1 = current position
        pre = taps.sum(0).T                                             # [T, C]
        c = F.silu(pre)
        out = u.flatten(-2) + c
        return dict(g=g, sig=s, v=v, u=u, n=n, taps=taps, pre=pre, conv=c, out=out, q=q)

    @torch.no_grad()
    def check_against_q4(self, x1: torch.Tensor, ids: torch.Tensor, rows_store, fmt) -> float:
        """max |out_probe - out_q4| for one document, where q4's PLE layer reads `rows_store` (SparseRowStore of
        the rows of `ids`) via the core package's build_ple_layer, with this probe's weights loaded."""
        from ..backends.graft.module import build_ple_layer
        from ..hashing import rows_for
        ple = build_ple_layer(fmt, table="sparse", rows=rows_store)
        with torch.no_grad():
            ple.key_proj.weight.copy_(self.Wk)
            ple.value_proj.weight.copy_(self.Wv)
            ple.norm_key.weight.copy_(self.nk)
            ple.norm_query.weight.copy_(self.nq)
            ple.norm_conv.weight.copy_(self.nc)
            ple.conv1d.weight.copy_(self.conv[:, None, :])
            ref = ple(x1[None], ids[None], None)[0]
        rid = torch.from_numpy(rows_for(fmt.key, ids.numpy()))
        e = rows_store.gather(rid).flatten(-2)
        mine = self.parts(x1, e)["out"]
        return float((ref - mine).abs().max())


def stream_rms(x: torch.Tensor, hc: int) -> torch.Tensor:
    """[T, hc*d] or [T, hc, d] -> per-(token, stream) RMS [T, hc]."""
    if x.dim() == 2:
        x = x.unflatten(-1, (hc, -1))
    return x.pow(2).mean(-1).sqrt()


@torch.no_grad()
def token_stats(probe: PLEProbe, x1: torch.Tensor, e: torch.Tensor) -> dict[str, np.ndarray]:
    """Per (token, stream) statistics for one document (all float32 numpy):

    g, sig [T,hc]; direct_rms (=rms(u_s)), conv_rms (=rms(c_s)), out_rms, x_rms [T,hc]; tap_rms [T,hc,K]
    (rms of each tap's contribution to the conv pre-activation); lin_share [T,hc] = rms(c - pre/2)/rms(c)
    (silu non-linearity share); leave-one-order-out: dsig_no2 / dsig_no3 [T,hc] = sig(full) - sig(without that
    order's heads), dv_no2 / dv_no3 [T] = |v - v_without|/|v| (= |v_n|/|v|), proj2 / proj3 [T] = <v, v_n>/|v|^2."""
    hc, d = probe.hc, probe.d
    p = probe.parts(x1, e)
    out = dict(g=p["g"], sig=p["sig"], direct_rms=stream_rms(p["u"], hc), conv_rms=stream_rms(p["conv"], hc),
               out_rms=stream_rms(p["out"], hc), x_rms=stream_rms(x1, hc), pre_rms=stream_rms(p["pre"], hc))
    out["tap_rms"] = p["taps"].permute(2, 1, 0).unflatten(1, (hc, d)).pow(2).mean(2).sqrt()   # [T, hc, K]
    c, pre = p["conv"], p["pre"]
    out["lin_share"] = stream_rms(c - pre / 2, hc) / stream_rms(c, hc).clamp_min(1e-30)
    v = p["v"]
    vv = (v * v).sum(-1).clamp_min(1e-30)
    for o, heads in (("2", range(0, 8)), ("3", range(8, 16))):
        keep = 1.0 - probe.head_mask(heads)
        g2, v2, _ = probe.gate(x1, e * keep, q=p["q"])
        vn = v - v2                                                     # this order's value
        out[f"dsig_no{o}"] = p["sig"] - torch.sigmoid(g2)
        out[f"dg_no{o}"] = p["g"] - g2
        out[f"dv_no{o}"] = vn.norm(dim=-1) / v.norm(dim=-1).clamp_min(1e-30)
        out[f"proj{o}"] = (v * vn).sum(-1) / vv
    out["v_norm"] = v.norm(dim=-1)
    out["e_norm"] = e.norm(dim=-1)
    return {k: t.float().numpy() for k, t in out.items()}


def _anova_within_share(N: float, G: float, sum_n2: float, ssw: float, sst: float) -> float:
    """One-way random-effects ANOVA (unbalanced): within share = s_w^2 / (s_w^2 + s_b^2) with s_w^2 = MSW and
    s_b^2 = max(0, (MSB - MSW) / n0), n0 = (N - sum n_i^2 / N) / (G - 1). Unbiased for the variance components,
    unlike SS_within / SS_total, which reads (N - G) / (N - 1) (~0.5 for pairs) when ALL variance is within
    groups (VD-1)."""
    if G < 2 or N - G < 1 or sst <= 0:
        return float("nan")
    ssb = sst - ssw
    msw = ssw / (N - G)
    msb = ssb / (G - 1)
    n0 = (N - sum_n2 / N) / (G - 1)
    sb2 = max(0.0, (msb - msw) / n0)
    tot = msw + sb2
    return msw / tot if tot > 0 else float("nan")


def variance_decomposition(values: np.ndarray, groups: np.ndarray, min_size: int = 2) -> dict[str, float]:
    """Within- vs between-group variance of `values` [N] over groups (int64 keys) with >= min_size members.
    within_share = the ANOVA variance-component share s_w^2 / (s_w^2 + s_b^2) (§D.3 rule: >= 0.5 within means
    context selection); within_share_raw = SS_within / SS_total (biased low for small groups, kept for reference)."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    uniq, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_size
    v, inv = v[keep], inv[keep]
    if v.size < 2:
        return dict(n=int(v.size), groups=0, within_share=float("nan"), within_share_raw=float("nan"))
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    ns = np.bincount(inv, minlength=uniq.size)
    means = np.where(ns > 0, sums / np.maximum(ns, 1), 0.0)
    within = float(((v - means[inv]) ** 2).sum())
    total = float(((v - v.mean()) ** 2).sum())
    G = int((ns > 0).sum())
    nn = ns[ns > 0].astype(np.float64)
    return dict(n=int(v.size), groups=G,
                within_share=_anova_within_share(float(v.size), float(G), float((nn ** 2).sum()), within, total),
                within_share_raw=within / total if total > 0 else float("nan"), total_var=total / v.size)


def within_share_ci(values: np.ndarray, groups: np.ndarray, min_size: int = 2, n_boot: int = 1000,
                    rng: np.random.Generator | None = None, alpha: float = 0.05) -> tuple[float, float, float]:
    """ANOVA within-group variance share (see variance_decomposition) with a bootstrap CI that resamples whole
    groups (members of groups with >= min_size occurrences only). Returns (estimate, lo, hi)."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, dtype=np.float64)
    uniq, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_size
    v, inv = v[keep], inv[keep]
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(inv, return_inverse=True)
    G = int(inv.max()) + 1
    n = np.bincount(inv, minlength=G).astype(np.float64)
    s1 = np.bincount(inv, weights=v, minlength=G)
    s2 = np.bincount(inv, weights=v * v, minlength=G)

    def share(w):  # w = multiplicity of each group in the resample
        N = (w * n).sum()
        S1, S2 = (w * s1).sum(), (w * s2).sum()
        within = (w * (s2 - s1 ** 2 / n)).sum()
        total = S2 - S1 ** 2 / N
        return _anova_within_share(N, w.sum(), (w * n * n).sum(), within, total)
    est = share(np.ones(G))
    reps = np.array([share(np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)) for _ in range(n_boot)])
    reps = reps[np.isfinite(reps)]
    return float(est), float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2))


def gdir_alpha_for_ratio(z: np.ndarray, target_rms: float, lo: float = 1e-4, hi: float = 1e3) -> float:
    """G-dir current-tap arm: c = silu(alpha * z) with z = N_conv(u) (unit-RMS per stream). Solve
    rms(silu(alpha z)) = target_rms for alpha by bisection in log space (rms is monotone in alpha > 0 for
    symmetric-ish z)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    def f(a):
        x = a * z
        return float(np.sqrt(np.mean((x / (1 + np.exp(-x))) ** 2)))
    a, b = math.log(lo), math.log(hi)
    for _ in range(80):
        m = 0.5 * (a + b)
        if f(math.exp(m)) < target_rms:
            a = m
        else:
            b = m
    return math.exp(0.5 * (a + b))


__all__ = ["SliceWeights", "Layer0", "PLEProbe", "rms_group", "stream_rms", "token_stats", "variance_decomposition",
           "gdir_alpha_for_ratio", "within_share_ci", "text_namespace", "bf16_np_to_f32", "L0", "PLE"]
