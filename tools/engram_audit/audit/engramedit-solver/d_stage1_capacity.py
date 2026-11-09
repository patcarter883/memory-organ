"""(c-post-check)+(d)+(f): run Gemini's REAL EngramEditSolver.apply_edit end-to-end on a tiny random Qwen2
(4 layers, d=64, graft at layer 2, r_mem=32), via Gemini's Layer2EngramGraft hook.
 - zero v_proj (as initialised; = what Stage 1 sees if warmup has not opened W_V)
 - random v_proj with the module's own short_conv init (nn.init.dirac_)
 - random v_proj with a corrected causal-identity conv
Measures: Stage-1 NLL, post-write NLL (which the code never checks), Jacobian rank of the residual write wrt delta,
relation-blindness and 2-gram-sharing leakage."""
import torch, copy
from common import make_engram, eg
from transformers import Qwen2Config, Qwen2ForCausalLM
torch.manual_seed(0)

cfg = Qwen2Config(vocab_size=5000, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False)
base_model = Qwen2ForCausalLM(cfg).eval()
SPINE, DOT, STEP, OTHER = 101, 13, 202, 303
M, T = 8, 12
g = torch.Generator().manual_seed(1)
TARGET = [777, 888, 999]
rows, anchors = [], []
for m in range(M):
    pre = torch.randint(1000, 5000, (T - 3 - 3,), generator=g).tolist()
    rows.append(pre + [SPINE, DOT, STEP] + TARGET); anchors.append(len(pre) + 2)
ids = torch.tensor(rows)
tmask = torch.zeros_like(ids); tmask[:, -3:] = 1


def target_nll(model, ids, tmask):
    with torch.no_grad():
        lg = model(input_ids=ids).logits[:, :-1].float()
    lab = ids[:, 1:]; mk = tmask[:, 1:].float()
    tl = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1), reduction="none").view(lab.shape)
    return ((tl * mk).sum() / mk.sum()).item()


def build(vproj, conv_fix):
    model = copy.deepcopy(base_model)
    engram = make_engram(d_model=64, r_mem=32, num_buckets=200_000)
    with torch.no_grad():
        if vproj == "random":
            torch.manual_seed(3); engram.v_proj.weight.normal_(0, 1.0)  # trained-like, opened W_V
            engram.q_proj.weight.normal_(0, 0.2); engram.k_proj.weight.normal_(0, 0.2)
        if conv_fix:
            engram.short_conv.weight.zero_(); engram.short_conv.weight[:, 0, -1] = 1.0  # causal identity
    eg.Layer2EngramGraft(model, model.model.layers[2], engram)
    return model, engram


def write_jacobian_rank(model, engram, ids, anchors):
    """Jacobian of layer-2 fused output (all positions) wrt delta at the anchor (r_mem=32 -> d_model=64 per pos)."""
    captured = {}
    layer = model.model.layers[2]
    h2 = {}
    hk = layer.register_forward_hook(lambda mod, i, o: h2.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()), prepend=True)
    with torch.no_grad(): model(input_ids=ids[:1])
    hk.remove()
    h = h2["h"]
    engram.prefetch_from_host(ids[:1])
    def f(delta):
        engram._active_perturbation = (delta, anchors[:1])
        out = engram.forward_fusion(h)
        engram._active_perturbation = None
        return out[0].reshape(-1)
    J = torch.autograd.functional.jacobian(f, torch.zeros(engram.r_mem))
    s = torch.linalg.svdvals(J)
    rank = int((s > s[0] * 1e-6).sum()) if s[0] > 0 else 0
    Jp = J.view(T, 64, engram.r_mem)
    per_pos = [round(Jp[t].norm().item(), 4) for t in range(T)]
    return rank, s[:4].tolist(), per_pos


for vproj, conv_fix in [("zero", False), ("random", False), ("random", True)]:
    model, engram = build(vproj, conv_fix)
    nll0 = target_nll(model, ids, tmask)
    rank, sv, per_pos = write_jacobian_rank(model, engram, ids, anchors)
    # how many channels of short_conv are non-zero
    nz_ch = int((engram.short_conv.weight.abs().sum(dim=(1, 2)) > 0).sum())
    solver = eg.EngramEditSolver(model, engram)
    # capture delta* by spying on the Stage-2 split
    spy = {}
    orig_split = torch.split
    def split_spy(t, *a, **k):
        if t.dim() == 1 and t.numel() == engram.r_mem: spy["d"] = t.clone()
        return orig_split(t, *a, **k)
    torch.split = split_spy
    stage1_nll = solver.apply_edit(ids, anchors, tmask, steps=20, lr=0.08)
    torch.split = orig_split
    nll_post = target_nll(model, ids, tmask)
    dn = spy["d"].float().norm().item()
    print(f"\n### v_proj={vproj:6s} conv={'fixed-identity' if conv_fix else 'dirac_ (as coded)'}: non-zero conv channels={nz_ch}/64")
    print(f"  Jacobian d(L2 out, all pos)/d delta: rank={rank} of max {engram.r_mem}; top sv={['%.3g'%x for x in sv]}")
    print(f"  per-position Jacobian norm (anchor at {anchors[0]}): {per_pos}")
    print(f"  target NLL pre-edit={nll0:.4f} | Stage-1 returned NLL (with live delta)={stage1_nll:.4f} | "
          f"post-write NLL (table only, never checked by code)={nll_post:.4f} | ||delta*||={dn:.4f}")

    if vproj == "random":
        # (f) relation-blindness and 2-gram leakage on the edited table
        def anchor_h3(model, seq, pos):
            out = {}
            hk = model.model.layers[3].register_forward_hook(lambda mod, i, o: out.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
            with torch.no_grad(): lg = model(input_ids=torch.tensor([seq])).logits[0, -1]
            hk.remove(); return out["h"][0], lg
        pre = [1500, 1600, 1700]
        tests = {"same subj, OTHER relation": pre + [SPINE, DOT, STEP] + [4321, 4322],
                 "unrelated subj sharing ('.','Step')": pre + [OTHER, DOT, STEP] + [4321, 4322],
                 "unrelated subj, no shared n-gram": pre + [OTHER, DOT, 404] + [4321, 4322]}
        clean_model = copy.deepcopy(base_model)
        cm, ce = build(vproj, conv_fix)  # same q/k/v weights, unedited tables
        for name, seq in tests.items():
            h_e, lg_e = anchor_h3(model, seq, 5)
            h_c, lg_c = anchor_h3(cm, seq, 5)
            dh = (h_e - h_c).norm(dim=-1)
            print(f"  [{name}] ||dh_L3|| per pos={['%.4f' % x for x in dh.tolist()]} | "
                  f"final-logit max|d|={(lg_e - lg_c).abs().max():.4f}, KL={torch.nn.functional.kl_div(lg_e.log_softmax(-1), lg_c.log_softmax(-1), log_target=True, reduction='sum'):.5f}")
