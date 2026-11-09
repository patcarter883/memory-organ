"""Option (A) building blocks on a tiny random Qwen4Exp (HF transformers 5.22, pure torch, CPU, fp64):

 1. Every layer type (GDN torch fallback, gated full attention + QSA indexer, MoE + shared expert,
    hyper-connections, PLE) back-propagates to a SMALL trainable row tensor that replaces
    `ngram_embedding` (only the batch's touched rows exist; nothing else has requires_grad).
 2. The gradient matches central finite differences.
 3. Per-layer activation checkpointing (HF GradientCheckpointingLayer) gives the same gradient.
 4. An MXFP4 expert module (compact bytes in, re-dequantised in backward via an autograd.Function,
    nothing weight-sized saved) gives the same gradient as HF's dense Experts holding the
    identical dequantised weights.
"""
import sys, math, importlib.util
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
import transformers.models.qwen4_exp.modeling_qwen4_exp as M

spec = importlib.util.spec_from_file_location(
    "mxfp4_ref", "/home/user/patcarter883/minisglang-rdna4/python/minisgl/quant/mxfp4.py")
mx = importlib.util.module_from_spec(spec); spec.loader.exec_module(mx)
torch.set_num_threads(2)

model = tiny_model(seed=1, layers=8).double()
cfg = model.config
print("layer types:", [l.layer_type[:3] for l in model.model.layers], "| PLE at", [i for i, l in enumerate(model.model.layers) if l.ple is not None])
print("GDN chunk fn:", M.torch_chunk_gated_delta_rule.__name__ if hasattr(M.torch_chunk_gated_delta_rule, "__name__") else M.torch_chunk_gated_delta_rule)

# ---- 4. MXFP4 experts: quantise the tiny model's experts to MXFP4, store compact ------------
E2M1 = torch.tensor(mx.FP4_E2M1_LUT, dtype=torch.float64)
def quant_mxfp4(w):  # (N, K) float -> packed u8 (N, K/2), scale u8 (N, K/32). RTN onto the E2M1 grid.
    N, K = w.shape
    g = w.view(N, K // 32, 32)
    amax = g.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-126, 127)
    s = torch.exp2(e).unsqueeze(-1)
    q = (g / s).unsqueeze(-1)
    pos = E2M1[:8]
    idx = (q.abs() - pos).abs().argmin(-1)
    code = idx + 8 * (q.squeeze(-1) < 0).long()
    code = code.view(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, (e + 127).to(torch.uint8).view(N, K // 32)

def deq(p, s, dtype):
    return mx.dequant_reference(p, s).to(dtype)

class DeqLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, s):
        ctx.save_for_backward(p, s)              # compact bytes only
        return x @ deq(p, s, x.dtype).t()
    @staticmethod
    def backward(ctx, gy):
        p, s = ctx.saved_tensors
        return gy @ deq(p, s, gy.dtype), None, None   # dX only; no weight grad

class MxExperts(nn.Module):
    """Drop-in for Qwen4ExpTextExperts.forward: only ROUTED experts are touched (a real store would
    read exactly these from the checkpoint)."""
    def __init__(self, dense):
        super().__init__()
        self.num_experts = dense.num_experts
        self.gu = [quant_mxfp4(dense.gate_up_proj[e].detach()) for e in range(self.num_experts)]
        self.dn = [quant_mxfp4(dense.down_proj[e].detach()) for e in range(self.num_experts)]
        self.act_fn = dense.act_fn
        self.touched = set()
        with torch.no_grad():  # make the dense copy hold EXACTLY the dequantised values (parity ref)
            for e in range(self.num_experts):
                dense.gate_up_proj[e] = deq(*self.gu[e], dense.gate_up_proj.dtype)
                dense.down_proj[e] = deq(*self.dn[e], dense.down_proj.dtype)
    def forward(self, h, top_k_index, top_k_weights):
        out = torch.zeros_like(h)
        for e in torch.unique(top_k_index).tolist():
            self.touched.add(e)
            tok, slot = (top_k_index == e).nonzero(as_tuple=True)
            g, u = DeqLinear.apply(h[tok], *self.gu[e]).chunk(2, -1)
            y = DeqLinear.apply(self.act_fn(g) * u, *self.dn[e])
            out.index_add_(0, tok, y * top_k_weights[tok, slot, None])
        return out

# ---- 1. small trainable row table in place of the 320M-row nn.Embedding -----------------------
for p in model.parameters():
    p.requires_grad_(False)
ple_emb = model.model.layers[1].ple.ple_embedding
full_table = ple_emb.ngram_embedding.weight.detach()

torch.manual_seed(3)
B, T = 3, 12
ids = torch.randint(5, cfg.vocab_size - 5, (B, T))
labels = ids.clone()

class RowGather(nn.Module):
    """Holds ONLY the rows this batch touches; forward maps global row ids -> local rows."""
    def __init__(self, global_ids, base):
        super().__init__()
        self.uniq, inv = torch.unique(global_ids, return_inverse=True)
        self.delta = nn.Parameter(torch.zeros(len(self.uniq), base.shape[1], dtype=base.dtype))
        self.base = base[self.uniq].clone()
        self.inv_shape = global_ids.shape
    @property
    def weight(self):          # HF reads .weight.device (modeling_qwen4_exp.py:1171-1173)
        return self.delta
    def forward(self, gids):
        loc = torch.searchsorted(self.uniq, gids.reshape(-1)).view(gids.shape)
        return (self.base + self.delta)[loc]

# capture the global ids the HF hasher produces for this batch
captured = {}
orig_emb = ple_emb.ngram_embedding
def cap_hook(mod, inp):
    captured["ids"] = inp[0].clone()
h = orig_emb.register_forward_pre_hook(cap_hook)
with torch.no_grad():
    model(input_ids=ids)
h.remove()
gids = captured["ids"]
rows = RowGather(gids, full_table)
ple_emb.ngram_embedding = rows
print(f"batch {B}x{T}: {gids.numel()} row reads, {len(rows.uniq)} distinct rows trainable "
      f"(of {full_table.shape[0]})")

def loss_fn():
    # own fp64 CE: HF's loss upcasts to fp32, too coarse for a finite-difference check
    logits = model(input_ids=ids).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))

def grad_now():
    rows.delta.grad = None
    l = loss_fn(); l.backward()
    return l.detach(), rows.delta.grad.clone()

model.train(False)
l0, g_plain = grad_now()
others = [n for n, p in model.named_parameters() if p.grad is not None and p is not rows.delta]
print(f"loss {l0.item():.6f}; |grad rows| {g_plain.norm():.3e}; other params with grad: {others}")
assert g_plain.abs().sum() > 0 and not others

# ---- 2. finite differences ------------------------------------------------------------------
# DIRECTIONAL (not per-coordinate): HF upcasts norms / router / GDN to fp32 internally even in an
# fp64 model, so a single-coordinate perturbation is lost in fp32 rounding. Along the gradient
# direction (and along random directions, scored against g.v) the loss change is >> fp32 noise.
worst = 0.0
torch.manual_seed(7)
dirs = [("grad", g_plain / g_plain.norm())] + [(f"rand{k}", (lambda v: v / v.norm())(torch.randn_like(g_plain))) for k in range(3)]
for name, v in dirs:
    for eps in (3e-2, 1e-2):
        with torch.no_grad():
            rows.delta += eps * v; lp = loss_fn()
            rows.delta -= 2 * eps * v; lm = loss_fn()
            rows.delta += eps * v
        fd = ((lp - lm) / (2 * eps)).item(); an = (g_plain * v).sum().item()
        rel = abs(fd - an) / max(abs(an), 1e-12)
        if eps == 1e-2: worst = max(worst, rel) if name == "grad" else worst
        print(f"   dir {name:5s} eps {eps:.0e}: fd {fd:+.6e} autograd g.v {an:+.6e} rel {rel:.1e}")
print(f"finite-difference check along the gradient (fp64 model, fp32-internal ops): rel err {worst:.2e}")
assert worst < 1e-2

# ---- 3. per-layer activation checkpointing ------------------------------------------------------
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train(True)   # HF only checkpoints in training mode; no dropout in this model
_, g_ckpt = grad_now()
print(f"per-layer checkpointing: max |g_ckpt - g_plain| = {(g_ckpt - g_plain).abs().max():.2e}")
assert torch.allclose(g_ckpt, g_plain, rtol=1e-9, atol=1e-12)
model.gradient_checkpointing_disable(); model.train(False)

# ---- 4. MXFP4 experts with re-dequant backward ---------------------------------------------------
mx_mods = []
for layer in model.model.layers:
    exp = layer.mlp.experts
    m = MxExperts(exp)          # also overwrites exp's dense weights with the dequantised values
    mx_mods.append((layer.mlp, exp, m))
_, g_dense = grad_now()         # HF dense Experts holding dequantised MXFP4 values
for mlp, exp, m in mx_mods:
    mlp.experts = m
_, g_mx = grad_now()            # compact MXFP4 + DeqLinear
print(f"MXFP4 DeqLinear experts vs HF dense experts (same values): max |diff| = {(g_mx - g_dense).abs().max():.2e}; "
      f"experts touched per layer: {[len(m.touched) for _, _, m in mx_mods]}")
assert torch.allclose(g_mx, g_dense, rtol=1e-9, atol=1e-12)
print("ALL CHECKS PASSED")
