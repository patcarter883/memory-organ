"""Time forward + backward through HF's pure-torch qwen4_exp modules at the REAL Flash-Next widths
(hidden 2560, hyper-connection x4 = 10240, GDN 16k/48v heads x128, 24-head gated attention, 512 experts
top-10 + shared expert, PLE at decoder index 1), with 4 layers (3 GDN + 1 full attention) instead of
48, random weights, the n-gram table shrunk (ngram_vocab_size_base=1000; the table size does not change
the PLE module's compute). Routed experts are replaced by an MXFP4 module of OUR OWN that keeps only
compact bytes (the checkpoint layout: gate_up (E,1280,1280)+(E,1280,80), down (E,2560,320)+(E,2560,20))
and re-dequantises in backward; nothing weight-sized is saved for autograd. Only the PLE table rows
require grad. Reports total s/iter and the split routed-experts vs everything else, per layer type."""
import json, os, time, collections
import torch, torch.nn as nn, torch.nn.functional as F
T_ = int(os.environ.get("T", "4")); torch.set_num_threads(T_)
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fork_mxfp4", "/home/user/patcarter883/minisglang-rdna4/python/minisgl/quant/mxfp4.py")
fork = importlib.util.module_from_spec(spec); spec.loader.exec_module(fork)

tc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl/config.json")))["text_config"]
tc.pop("model_type", None); tc.pop("mtp", None)
tc.update(num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
          ngram_vocab_size_base=1000, split_ngram_parts=4, mtp_num_hidden_layers=0)
cfg = Qwen4ExpTextConfig(**tc); cfg._attn_implementation = "sdpa"; cfg._experts_implementation = "eager"
cfg.dtype = torch.bfloat16

LUT = torch.tensor(fork.FP4_E2M1_LUT, dtype=torch.float32)
PAIR = torch.stack([LUT[torch.arange(256) & 0xF], LUT[torch.arange(256) >> 4]], -1).to(torch.bfloat16).contiguous().view(torch.int32).view(256)
@torch.compile(dynamic=False)
def deq(p, s):
    N, K2 = p.shape
    w = PAIR[p.to(torch.int64)].view(torch.bfloat16).view(N, K2 * 2 // 32, 32)
    return (w * (s.to(torch.int32) << 23).view(torch.float32).to(torch.bfloat16).unsqueeze(-1)).view(N, K2 * 2)

STATS = collections.Counter()
class MxLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, s):
        t = time.perf_counter(); W = deq(p, s); y = x @ W.t()
        STATS["expert_s"] += time.perf_counter() - t; STATS["deq"] += 1
        ctx.save_for_backward(p, s); return y
    @staticmethod
    def backward(ctx, gy):
        t = time.perf_counter(); p, s = ctx.saved_tensors; gx = gy @ deq(p, s)
        STATS["expert_s"] += time.perf_counter() - t; STATS["deq"] += 1
        return gx, None, None

class MxExperts(nn.Module):
    def __init__(self, store, act):
        super().__init__(); self.store = store; self.act_fn = act
    def forward(self, h, top_k_index, top_k_weights):
        out = torch.zeros_like(h); gu_p, gu_s, dn_p, dn_s = self.store
        uniq = torch.unique(top_k_index).tolist(); STATS["experts_touched"] += len(uniq)
        for e in uniq:
            tok, slot = (top_k_index == e).nonzero(as_tuple=True)
            g_, u = MxLinear.apply(h[tok], gu_p[e], gu_s[e]).chunk(2, -1)
            y = MxLinear.apply(self.act_fn(g_) * u, dn_p[e], dn_s[e])
            out.index_add_(0, tok, (y * top_k_weights[tok, slot, None]).to(out.dtype))
        return out

gen = torch.Generator().manual_seed(0)
E = 512
store = (torch.randint(0, 256, (E, 1280, 1280), dtype=torch.uint8, generator=gen),
         torch.randint(118, 122, (E, 1280, 80), dtype=torch.uint8, generator=gen),
         torch.randint(0, 256, (E, 2560, 320), dtype=torch.uint8, generator=gen),
         torch.randint(118, 122, (E, 2560, 20), dtype=torch.uint8, generator=gen))

torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    model = M.Qwen4ExpForCausalLM(cfg)
torch.set_default_dtype(torch.float32)
for layer in model.model.layers:
    layer.mlp.experts = MxExperts(store, layer.mlp.experts.act_fn)
model = model.to_empty(device="cpu").to(torch.bfloat16)
torch.manual_seed(0)
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2: p.normal_(0, 0.02)
        elif "A_log" in n: p.uniform_(0, 2)
        else: p.fill_(1.0)
    for n, b in model.named_buffers():           # non-persistent buffers (rope inv_freq, PLE multipliers/heads) recomputed
        pass
# rebuild buffers that to_empty left uninitialised
model.model.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(cfg)
ple = model.model.layers[1].ple.ple_embedding
ref_ple = M.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, 1, 0)
for name, buf in ref_ple.named_buffers():
    getattr(ple, name).copy_(buf) if hasattr(ple, name) else None
for p in model.parameters(): p.requires_grad_(False)
table = ple.ngram_embedding.weight; table.requires_grad_(True)
model.train(False)

def run(B, T, n_tgt, reps=2):
    ids = torch.randint(10, cfg.vocab_size - 10, (B, T), generator=gen)
    res = []
    for r in range(reps + 1):
        STATS.clear(); table.grad = None
        t = time.perf_counter()
        # lm_head on a 2-D [targets, 2560] input: a 3-D input makes CPU autograd materialise a
        # broadcast copy of the 1.2 GiB weight PER SEQUENCE in backward (B=8 -> 10 GiB, found by OOM).
        hid = model.model(input_ids=ids).last_hidden_state[:, -(n_tgt + 1):-1].reshape(-1, cfg.hidden_size)
        logits = model.lm_head(hid).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, -n_tgt:].reshape(-1))
        tf = time.perf_counter() - t; ef = STATS["expert_s"]
        loss.backward()
        tot = time.perf_counter() - t
        assert table.grad is not None and table.grad.abs().sum() > 0
        if r: res.append((tot, tf, STATS["expert_s"], ef, STATS["experts_touched"]))
    tot, tf, es, ef, nt = min(res)
    print(f"B={B} T={T} ({B*T} tok, loss on last {n_tgt}/seq): fwd+bwd {tot:6.2f} s for 4 layers "
          f"[fwd {tf:5.2f} s] | routed experts {es:5.2f} s ({nt/4:.0f} distinct/layer, "
          f"{es/max(nt,1)*1e3:.2f} ms per touched expert incl. fwd+bwd) | rest {tot-es:5.2f} s", flush=True)
    return tot, es, nt

print(f"threads={T_} torch {torch.__version__}; real widths, 4 layers (GDN,GDN,GDN,attn), PLE at layer 1, bf16")
import resource
for spec_ in os.environ.get("SHAPES", "8x4,8x8,8x32").split(","):
    B, T = map(int, spec_.split("x"))
    run(B, T, min(3, T - 1))
    print(f"   peak RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GiB", flush=True)
