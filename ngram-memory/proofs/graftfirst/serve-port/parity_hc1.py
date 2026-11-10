"""Fork's Qwen4ExpPLE (minisgl, hc_count=1) vs transformers' Qwen4ExpTextPLELayer (the HF graft's module).

Checks, on CPU, small d (the code paths are width-agnostic):
  A. 'random' init (every path live: gate, norms, dilated conv): fork prefill-chunk + decode steps vs one HF
     full-sequence forward, fp32. Tolerance-level parity (different op order).
  B. 'editor' init (W_k = 0, conv = 0, norms 0): fork output == 0.5 * (e @ W_v^T) BIT-EXACTLY, in fp32 and bf16,
     and == HF bit-exactly; i.e. the linear fast path is not an approximation.
  C. empty cartridge (e = 0): output is exactly +0.0 everywhere (no -0.0), fp32 and bf16, for random W_v.
Run: MINISGL_TAIL_HIP=0 PYTHONPATH=<fork>/python:. python parity_hc1.py
"""
import math, sys, json
from types import SimpleNamespace
import torch
torch.set_num_threads(1)
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4  # before the stubs
import stubimport  # noqa: F401  (stubs msgpack/zmq/... the compute path never touches)
from minisgl.distributed.info import set_tp_info
set_tp_info(0, 1)
from minisgl.models.qwen4exp import Qwen4ExpPLE
from minisgl.ple.runtime import PLEBatch
from minisgl.ple.state import PLEStateCache

d, H = 256, 16
w = d // H
E = H * w
T, T0 = 40, 23  # total tokens; first T0 as one prefill chunk, rest decoded one by one
res = {}

def hf_layer(dtype):
    cfg = SimpleNamespace(hidden_size=d, hc_count=1, ple_embed_dim=E, ple_conv_kernel_size=4, rms_norm_eps=1e-6,
                          vocab_size=248320, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
                          make_ngram_vocab_size_divisible_by=128, seed=1234, eos_token_id=248044)
    with torch.device("meta"):
        L = q4.Qwen4ExpTextPLELayer(cfg, 1, 0)
    class Fixed(nn.Module):
        def forward(self, ids, pkv):
            return self.e
    L.ple_embedding = Fixed()
    L.to_empty(device="cpu")
    return L.to(dtype)

def init(L, kind, g):
    with torch.no_grad():
        if kind == "random":
            L.key_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            L.value_proj.weight.normal_(0, 1 / math.sqrt(E), generator=g)
            L.conv1d.weight.normal_(0, 0.5, generator=g)
            for n in (L.norm_key, L.norm_query, L.norm_conv):
                n.weight.normal_(0, 0.1, generator=g)
        else:  # editor init, as P/graft/graft_proto.py:209-226
            q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
            L.key_proj.weight.zero_(); L.conv1d.weight.zero_()
            L.value_proj.weight.copy_(q[:d, :E])
            for n in (L.norm_key, L.norm_query, L.norm_conv):
                n.weight.zero_()

def fork_block(L, dtype):
    cfg = SimpleNamespace(hidden_size=d, hc_hidden_size=d, hc_count=1, ple_embed_dim=E, ple_conv_kernel_size=4,
                          rms_norm_eps=1e-6, ngram_size=3)
    _old = torch.get_default_dtype(); torch.set_default_dtype(dtype)
    try:
        B = Qwen4ExpPLE(cfg)  # the engine builds under the model dtype
    finally:
        torch.set_default_dtype(_old)
    sd = {"conv1d_weight": L.conv1d.weight.detach().clone(), "key_proj.weight": L.key_proj.weight.detach().clone(),
          "value_proj.weight": L.value_proj.weight.detach().clone(), "norm_key.weight": L.norm_key.weight.detach().clone(),
          "norm_query.weight": L.norm_query.weight.detach().clone(), "norm_conv.weight": L.norm_conv.weight.detach().clone(),
          "ple_embedding.layer_multipliers": torch.zeros(3, dtype=torch.int64)}
    B.load_state_dict(sd)
    st = PLEStateCache(num_slots=3, wide=d, state_len=B.conv_state_len, context_len=2, eos_token_id=248044,
                       dtype=dtype, device=torch.device("cpu"))
    return B, st

def run_fork(B, st, hidden, emb):
    outs = []
    slot = 1
    idx = torch.tensor([slot])
    b = PLEBatch(embeddings=emb[:T0], state_indices=idx, slots=[slot], seq_lens=[T0], is_decode=False)
    outs.append(B.compute(hidden[:T0], emb[:T0], b, st.conv_state))
    for t in range(T0, T):
        b = PLEBatch(embeddings=emb[t:t+1], state_indices=idx, slots=[slot], seq_lens=[1], is_decode=True)
        outs.append(B.compute(hidden[t:t+1], emb[t:t+1], b, st.conv_state))
    return torch.cat(outs, 0)

def run_hf(L, hidden, emb):
    L.ple_embedding.e = emb.unsqueeze(0)
    return L(hidden.unsqueeze(0), torch.zeros(1, T, dtype=torch.long), None)[0]

for kind in ("random", "editor"):
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(7)
        L = hf_layer(dtype); init(L, kind, g)
        B, st = fork_block(L, dtype)
        hidden = torch.randn(T, d, generator=g).to(dtype)
        emb = (torch.randn(T, E, generator=g) * 0.05).to(dtype)
        with torch.no_grad():
            f = run_fork(B, st, hidden, emb).float(); h = run_hf(L, hidden, emb).float()
            lin = (0.5 * (emb @ L.value_proj.weight.t())).float()
            lin2 = (emb @ (0.5 * L.value_proj.weight).t()).float()  # 0.5 folded into W_v
            W = L.value_proj.weight
            chunks = [emb[:T0]] + [emb[t:t+1] for t in range(T0, T)]
            lin_c = torch.cat([0.5 * torch.nn.functional.linear(c, W) for c in chunks]).float()  # same GEMM shapes as the fork
        r = {"max_abs_fork_vs_hf": (f - h).abs().max().item(), "out_rms": h.pow(2).mean().sqrt().item(),
             "out_max_abs": h.abs().max().item()}
        if kind == "editor":
            r["fork_eq_linear_bitexact"] = bool(torch.equal(f, lin))
            r["fork_eq_linear_same_gemm_shapes_bitexact"] = bool(torch.equal(f, lin_c))
            r["hf_eq_linear_bitexact"] = bool(torch.equal(h, lin))
            r["fold_half_into_Wv_bitexact"] = bool(torch.equal(lin, lin2))
        res[f"{kind}/{str(dtype).split('.')[-1]}"] = r
    # C: empty cartridge
for dtype in (torch.float32, torch.bfloat16):
    g = torch.Generator().manual_seed(3)
    L = hf_layer(dtype); init(L, "random", g)
    B, st = fork_block(L, dtype)
    hidden = torch.randn(T, d, generator=g).to(dtype)
    emb = torch.zeros(T, E, dtype=dtype)
    with torch.no_grad():
        f = run_fork(B, st, hidden, emb)
    res[f"empty/{str(dtype).split('.')[-1]}"] = {
        "all_zero": bool((f == 0).all()), "any_negative_zero": bool(((f == 0) & torch.signbit(f)).any()),
        "note": "random W_k/conv/norms: e=0 still gives exactly 0 (value_proj has no bias)"}
print(json.dumps(res, indent=1))
