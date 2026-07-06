"""Learned, outcome-supervised gate router (Track 5, #99).

Replaces the hand-composed multigate product (store-conf x headroom x scope x sparse) with ONE small MLP
that maps a per-fact LABEL-FREE signal vector -> a scalar gain in [0,1] scaling the logit injection. Trained
by backprop through the logit-level injection: the frozen base's last-token logits `off` and the store's push
`raw` are pre-computed CONSTANTS (no base backward), so the only trainable path is router -> gain -> the added
term. The objective is outcome-supervised — raise log P(true) where it helps, penalised by KL(off||on) so the
push stays gentle — using the true label ONLY as a training target; the router's INPUTS are all label-free, so
it deploys without labels. Fit on a TRAIN split, evaluate on a HELD-OUT split to test whether a learned gate
GENERALISES (the ceiling claim).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SIG = 8  # store-conf, base-entropy, headroom, store-peak, agreement-margin, base-top-conf, base-margin, store-entropy


def signal_features(off, raw, conf):
    """off [B,V] frozen base last-token logits; raw [B,V] store push logits (out_proj(bank)@lmT); conf [B] or
    None store retrieval strength. Returns [B,N_SIG] label-free signal matrix (true label NOT used)."""
    B, V = off.shape
    p_off = torch.softmax(off, -1)
    ent = -(p_off * torch.log(p_off.clamp_min(1e-12))).sum(-1) / torch.log(torch.tensor(float(V), device=off.device))
    store_tok = raw.argmax(-1)
    base_top = off.argmax(-1)
    idx = torch.arange(B, device=off.device)
    p_tgt = p_off[idx, store_tok]                    # base mass on the store's delivered token
    p_top = p_off[idx, base_top]                     # base mass on its own top token
    store_prob = torch.softmax(raw, -1)
    store_peak = store_prob.max(-1).values
    store_ent = -(store_prob * torch.log(store_prob.clamp_min(1e-12))).sum(-1) / torch.log(torch.tensor(float(V), device=off.device))
    top2 = p_off.topk(2, -1).values
    base_margin = top2[:, 0] - top2[:, 1]            # base decisiveness (top1 - top2), distinct from entropy
    c = torch.zeros(B, device=off.device) if conf is None else torch.log1p(conf.float().to(off.device)) / 10.0
    return torch.stack([
        c,                       # 0 retrieval strength (locality)
        ent,                     # 1 base uncertainty (entropy)
        1.0 - p_tgt,             # 2 headroom on the target (dose)
        store_peak,              # 3 store decode confidence (scope / OOD)
        p_tgt - p_top,           # 4 agreement margin (<=0; how far target trails base's top)
        p_top,                   # 5 base top confidence
        base_margin,             # 6 base decisiveness (top1 - top2)
        store_ent,               # 7 store decode diffuseness (scope, complements peak)
    ], dim=-1)


class GateRouter(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_SIG, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def gain(self, sig):
        return torch.sigmoid(self.net(sig)).squeeze(-1)      # [B] in (0,1)


def _inj_base(raw, alpha_ref, topk):
    V = raw.shape[-1]
    inj = alpha_ref * raw
    if 0 < topk < V:
        keep = raw.topk(topk, -1).indices
        m = torch.zeros_like(raw); m.scatter_(1, keep, 1.0)
        inj = inj * m
    return inj


@torch.no_grad()
def oracle_gain(off, raw, true_tid, alpha_ref, *, kl_weight=0.3, topk=0, grid=41, device="cuda"):
    """Per-fact gain g* in [0,1] that MAXIMISES utility U(g)=logP_on(true) - kl_weight*KL(off||on), found by
    a grid line-search on the cached logits (no base forward). This is the OUTCOME target the regression
    router learns to predict from label-free signals — the truest 'outcome supervision' (train on the gain
    that actually worked). Returns g* [B] and the achieved best utility [B]."""
    off = off.detach().float(); raw = raw.detach().float()
    B = off.shape[0]
    idx = torch.arange(B, device=off.device)
    inj = _inj_base(raw, alpha_ref, topk)
    lp_off = torch.log_softmax(off, -1); p_off = lp_off.exp()
    gs = torch.linspace(0, 1, grid, device=off.device)
    best_u = torch.full((B,), -1e30, device=off.device)
    best_g = torch.zeros(B, device=off.device)
    for g in gs:
        lp_on = torch.log_softmax(off + g * inj, -1)
        u = lp_on[idx, true_tid.to(off.device)] - kl_weight * (p_off * (lp_off - lp_on)).sum(-1)
        upd = u > best_u
        best_g = torch.where(upd, g.expand(B), best_g)
        best_u = torch.where(upd, u, best_u)
    return best_g, best_u


def fit_router_regress(off, raw, true_tid, conf, alpha_ref, *, kl_weight=0.3, steps=400, lr=3e-3, topk=0,
                       grid=41, device="cuda"):
    """Train a GateRouter to REGRESS the oracle per-fact gain (MSE) from label-free signals. Decouples the
    router from the injection graph — it learns 'what dose actually worked here' as a function of the signals,
    which tends to generalise and calibrate better than the end-to-end differentiable objective."""
    off = off.detach().float(); raw = raw.detach().float()
    sig = signal_features(off, raw, conf).detach()
    g_star, _ = oracle_gain(off, raw, true_tid, alpha_ref, kl_weight=kl_weight, topk=topk, grid=grid, device=device)
    router = GateRouter().to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    for _ in range(steps):
        g = router.gain(sig)
        loss = F.mse_loss(g, g_star.detach())
        opt.zero_grad(); loss.backward(); opt.step()
    return router


def fit_router_hybrid(off, raw, true_tid, conf, alpha_ref, *, kl_weight=0.3, beta=0.5, steps=400, lr=3e-3,
                      topk=0, grid=41, device="cuda"):
    """Hybrid: the end-to-end differentiable objective (-logP_on(true) + kl_weight*KL) PLUS an oracle-dose
    MSE regulariser (beta * (gain - g*)^2). Chases the diff router's ΔP while inheriting the regress router's
    calibration/robustness — fixes the diff router's degeneration (dose-corr goes negative) at aggressive
    kl_weight by anchoring the gain to the per-fact utility-optimal dose."""
    off = off.detach().float(); raw = raw.detach().float()
    B = off.shape[0]
    idx = torch.arange(B, device=off.device)
    sig = signal_features(off, raw, conf).detach()
    inj = _inj_base(raw, alpha_ref, topk)
    lp_off = torch.log_softmax(off, -1); p_off = lp_off.exp()
    g_star, _ = oracle_gain(off, raw, true_tid, alpha_ref, kl_weight=kl_weight, topk=topk, grid=grid, device=device)
    g_star = g_star.detach()
    router = GateRouter().to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    tgt = true_tid.to(off.device)
    for _ in range(steps):
        g = router.gain(sig)
        on = off + g.unsqueeze(-1) * inj
        lp_on = torch.log_softmax(on, -1)
        push = -lp_on[idx, tgt]
        kl = (p_off * (lp_off - lp_on)).sum(-1)
        loss = (push + kl_weight * kl).mean() + beta * F.mse_loss(g, g_star)
        opt.zero_grad(); loss.backward(); opt.step()
    return router


def fit_router(off, raw, true_tid, conf, alpha_ref, *, kl_weight=0.3, steps=300, lr=3e-3, topk=0, device="cuda"):
    """Train a GateRouter on (off,raw,true_tid,conf). Returns the fitted router. off/raw are DETACHED constants;
    gradient flows only through the router. Loss = -logP_on(true) + kl_weight*KL(off||on), so the router learns
    to spend gain where the log-prob gain per unit KL is highest (i.e. where the base is unsure)."""
    off = off.detach().float(); raw = raw.detach().float()
    B, V = off.shape
    idx = torch.arange(B, device=off.device)
    sig = signal_features(off, raw, conf).detach()
    inj_base = alpha_ref * raw
    if 0 < topk < V:                                          # confine collateral to the store's top-k tokens
        keep = raw.topk(topk, -1).indices
        m = torch.zeros_like(raw); m.scatter_(1, keep, 1.0)
        inj_base = inj_base * m
    lp_off = torch.log_softmax(off, -1)
    p_off = lp_off.exp()
    router = GateRouter().to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    tgt = true_tid.to(off.device)
    for _ in range(steps):
        g = router.gain(sig)                                 # [B]
        on = off + g.unsqueeze(-1) * inj_base
        lp_on = torch.log_softmax(on, -1)
        push = -lp_on[idx, tgt]                              # raise P(true)
        kl = (p_off * (lp_off - lp_on)).sum(-1)             # stay gentle
        loss = (push + kl_weight * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return router
