"""T2 proofs on a TINY random Gemma4 text model shrunk from gemma-4-E4B-it's text_config.
(a) (token, layer) -> table slice helper; context-independence of per_layer_input
(b) locality of a row edit (exact zeros where the token is absent / before it); KV-sharing reach per layer
(c) decode parity (prefill+cached decode == full forward; generate cache==nocache) with an edited table
(d) final-NLL gradient reaches the rows; write capacity PLE-row vs free residual delta; leak to other contexts
(e) saturation: post_per_layer_input_norm fixes the per-layer write magnitude
(f) alias-token mitigation: copy a token into a new vocab id (exact no-op), edit only the alias row
(g) graft: enable PLE on a 26B-like (MoE, k_eq_v, no kv-sharing) tiny config, zero-init -> exact no-op
"""
import json, math, copy, sys
import torch, torch.nn.functional as F
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
from transformers.cache_utils import DynamicCache

torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
E4B = json.load(open(sys.argv[1]))["text_config"]
OUT = {}

L, H, D, V = 12, 128, 16, 512
# keep E4B's 5:1 sliding/full pattern and its kv-shared fraction (18/42 -> 5/12)
layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(L)]
cfg = Gemma4TextConfig(
    vocab_size=V, vocab_size_per_layer_input=V, hidden_size=H, hidden_size_per_layer_input=D,
    intermediate_size=4 * H, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=1, head_dim=32,
    global_head_dim=64, num_kv_shared_layers=5, sliding_window=8, layer_types=layer_types,
    final_logit_softcapping=E4B["final_logit_softcapping"], rope_parameters=E4B["rope_parameters"],
    hidden_activation=E4B["hidden_activation"], rms_norm_eps=E4B["rms_norm_eps"], max_position_embeddings=512,
    tie_word_embeddings=True, pad_token_id=0, bos_token_id=2, eos_token_id=1, initializer_range=0.05,
    attention_k_eq_v=E4B["attention_k_eq_v"], enable_moe_block=False, use_double_wide_mlp=E4B["use_double_wide_mlp"],
)
cfg._attn_implementation = "eager"
model = Gemma4ForCausalLM(cfg).eval()
tm = model.model
# make the random model less degenerate: non-unit norm weights, PLE table rows at E4B-like relative scale
for n, p in model.named_parameters():
    if n.endswith("norm.weight") or "layernorm.weight" in n:
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
tm.embed_tokens_per_layer.weight.normal_(0, float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
base_state = copy.deepcopy(model.state_dict())
first_shared = L - cfg.num_kv_shared_layers
store_layers = [i for i, l in enumerate(tm.layers) if l.self_attn.store_full_length_kv]
OUT["config"] = dict(L=L, H=H, D=D, V=V, first_shared=first_shared, store_full_length_kv_layers=store_layers,
                     layer_types=layer_types)
print("store_full_length_kv layers:", store_layers, "first shared:", first_shared)

# ---------------- (a) helper ----------------
def ple_slice(model, tok: int, layer: int) -> torch.Tensor:
    """View into the editable table: raw (unscaled) row segment the decoder layer `layer` receives for token `tok`."""
    tm = model.model; D = tm.hidden_size_per_layer_input
    return tm.embed_tokens_per_layer.weight[tok, layer * D:(layer + 1) * D]

def ple_byte_range(tok, layer, D=256, L=42, dtype_bytes=2):
    """Byte range (relative to the tensor's data start) of that slice in a row-major [V, L*D] checkpoint tensor."""
    s = (tok * L * D + layer * D) * dtype_bytes
    return s, s + D * dtype_bytes

def effective_ple(model, tok: int) -> torch.Tensor:
    """[L, D] per_layer_input the decoder layers receive for `tok` (a pure function of the token id)."""
    tm = model.model; ids = torch.tensor([[tok]])
    return tm.project_per_layer_inputs(tm.embed_tokens(ids), tm.get_per_layer_inputs(ids))[0, 0]

captured = {}
def cap_hook(i):
    def f(mod, args, kwargs):
        captured[i] = (args[1] if len(args) > 1 else kwargs["per_layer_input"]).clone()
    return f
hs = [layer.register_forward_pre_hook(cap_hook(i), with_kwargs=True) for i, layer in enumerate(tm.layers)]
seq = torch.randint(3, V, (1, 20))
model(seq)
for h in hs: h.remove()
max_ctx_dev = 0.0; max_slice_dev = 0.0
for pos in range(seq.shape[1]):
    t = seq[0, pos].item(); eff = effective_ple(model, t)
    for i in range(L):
        max_ctx_dev = max(max_ctx_dev, (captured[i][0, pos] - eff[i]).abs().max().item())
    raw = tm.get_per_layer_inputs(seq[:, pos:pos + 1])[0, 0]
    for i in range(L):
        max_slice_dev = max(max_slice_dev, (raw[i] - ple_slice(model, t, i) * math.sqrt(D)).abs().max().item())
OUT["a"] = dict(max_dev_captured_vs_effective_ple=max_ctx_dev, max_dev_lookup_vs_slice_times_sqrtD=max_slice_dev,
                e4b_byte_range_tok94648_layer22=ple_byte_range(94648, 22))
print("(a)", OUT["a"])

# ---------------- (b) locality ----------------
X = 77
def logits(ids, m=model):
    return m(ids).logits.float()
noX = torch.randint(3, V, (64, 24)); noX[noX == X] = 78
withX = torch.randint(3, V, (64, 24)); withX[withX == X] = 78; P0 = 9; withX[:, P0] = X
lo_no, lo_with = logits(noX), logits(withX)
delta = torch.randn(L * D) * tm.embed_tokens_per_layer.weight[X].std() * 3
tm.embed_tokens_per_layer.weight[X] += delta
ln, lw = logits(noX), logits(withX)
d_no = (ln - lo_no).abs().max().item()
d_before = (lw[:, :P0] - lo_with[:, :P0]).abs().max().item()
d_at = (lw[:, P0] - lo_with[:, P0]).abs().max().item()
d_after = (lw[:, P0 + 1:] - lo_with[:, P0 + 1:]).abs().amax(dim=(0, 2))
kl_after = F.kl_div(lw[:, P0 + 1:].log_softmax(-1), lo_with[:, P0 + 1:].log_softmax(-1), log_target=True,
                    reduction="none").sum(-1).mean(0)
OUT["b"] = dict(max_abs_dlogit_seqs_without_X=d_no, max_abs_dlogit_positions_before_X=d_before,
                max_abs_dlogit_at_X=d_at, max_abs_dlogit_after_X_by_offset=[round(v, 4) for v in d_after.tolist()],
                mean_KL_after_X_by_offset=[round(v, 5) for v in kl_after.tolist()])
model.load_state_dict(base_state)
# per-layer reach: edit only the layer-l slice of X; does anything AFTER X's position change?
reach = []
for l in range(L):
    model.load_state_dict(base_state)
    ple_slice(model, X, l).add_(torch.randn(D) * 1.0)
    lw = logits(withX)
    reach.append(dict(layer=l, at_X=round((lw[:, P0] - lo_with[:, P0]).abs().max().item(), 5),
                      after_X=(lw[:, P0 + 1:] - lo_with[:, P0 + 1:]).abs().max().item()))
model.load_state_dict(base_state)
OUT["b_reach"] = reach
print("(b)", OUT["b"]); print("(b reach)", reach)

# ---------------- (c) decode parity ----------------
tm.embed_tokens_per_layer.weight[X] += delta
full = torch.randint(3, V, (1, 30)); full[full == X] = 78; full[0, 4] = X; full[0, 21] = X  # X in prompt AND in decode part
Pp = 12
ref = logits(full)[0]
cache = DynamicCache(config=cfg)
out = model(full[:, :Pp], past_key_values=cache, use_cache=True)
dec = [out.logits[0].float()]
for t in range(Pp, full.shape[1]):
    out = model(full[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
    dec.append(out.logits[0].float())
dec = torch.cat(dec)
g1 = model.generate(full[:, :Pp], max_new_tokens=25, do_sample=False, use_cache=True)
g2 = model.generate(full[:, :Pp], max_new_tokens=25, do_sample=False, use_cache=False)
model.load_state_dict(base_state)
ref_unedited = logits(full)[0]
OUT["c"] = dict(max_abs_diff_cached_decode_vs_full=(dec - ref).abs().max().item(),
                decode_step_at_X_pos21_diff=(dec[21] - ref[21]).abs().max().item(),
                edit_changes_pos21_by=(ref[21] - ref_unedited[21]).abs().max().item(),
                generate_cache_eq_nocache=bool(torch.equal(g1, g2)), seq_len=30, sliding_window=cfg.sliding_window)
print("(c)", OUT["c"])

# ---------------- (d) gradient + capacity ----------------
def nll_last(ids, target, m=model):
    return -m(ids).logits[:, -1].float().log_softmax(-1)[torch.arange(ids.shape[0]), target]

torch.set_grad_enabled(True)
prompt = torch.tensor([[2, 31, 45, 52, X, 101, 102]])   # "<bos> c c c SUBJ rel rel" ; X not last
target = torch.tensor([200])
model.zero_grad()
nll_last(prompt, target).sum().backward()
G = tm.embed_tokens_per_layer.weight.grad
nz_rows = sorted(set(torch.nonzero(G.abs().sum(1)).flatten().tolist()))
gX = G[X].view(L, D).norm(dim=1)
OUT["d_grad"] = dict(rows_with_nonzero_grad=nz_rows, prompt_tokens=sorted(set(prompt[0].tolist())),
                     X_grad_norm_by_layer=[float(f"{v:.3e}") for v in gX.tolist()])
print("(d grad)", OUT["d_grad"])
model.zero_grad()
for p in model.parameters(): p.requires_grad_(False)

def run_method(method, X, prompt, target, steps=200, lr=0.05, layer=2, layers=None):
    """Optimise only the method's parameters for final-position NLL. Returns trajectory + param."""
    model.load_state_dict(base_state)
    hooks = []
    posX = (prompt[0] == X).nonzero()[0].item()
    if method.startswith("ple"):
        ls = layers if layers is not None else list(range(L))
        p = torch.zeros(len(ls), D, requires_grad=True)
        W = tm.embed_tokens_per_layer.weight
        def pre(mod, args, kwargs):  # functional row write: W[X, l] += p
            return None
        # implement by temporarily replacing the embedding forward
        orig = tm.embed_tokens_per_layer.forward
        def fwd(ids):
            out = orig(ids)
            add = torch.zeros(L, D); add[ls] = p
            mask = (ids == X).unsqueeze(-1).float()
            return out + mask * (add.flatten() * tm.embed_tokens_per_layer.embed_scale)
        tm.embed_tokens_per_layer.forward = fwd
        undo = lambda: setattr(tm.embed_tokens_per_layer, "forward", orig)
    else:  # free residual delta added at every position where X occurs, after decoder layer `layer`
        p = torch.zeros(H, requires_grad=True)
        cur = {}
        def ids_hook(mod, args, kwargs):
            cur["ids"] = kwargs.get("input_ids", args[0] if args else None)
        hooks.append(model.register_forward_pre_hook(ids_hook, with_kwargs=True))
        def post(mod, args, out):
            mask = (cur["ids"] == X).unsqueeze(-1).float()
            return out + mask * p
        hooks.append(tm.layers[layer].register_forward_hook(post))
        undo = lambda: None
    opt = torch.optim.Adam([p], lr=lr)
    traj = []
    for s in range(steps):
        loss = nll_last(prompt, target).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if s in (0, 9, 29, 99, steps - 1): traj.append((s + 1, round(loss.item(), 4)))
    final = nll_last(prompt, target).item()
    return dict(p=p.detach().clone(), traj=traj, final_nll=final, hooks=hooks, undo=undo)

def cleanup(r):
    for h in r["hooks"]: h.remove()
    r["undo"]()

def eval_with(r, fn):
    with torch.no_grad():
        return fn()

methods = {
    "ple_all_layers": dict(method="ple"),
    "ple_readable_layers": dict(method="ple", layers=list(range(max(store_layers)))),
    "ple_single_layer2": dict(method="ple", layers=[2]),
    "ple_local_only_layers": dict(method="ple", layers=list(range(max(store_layers), L))),
    "free_resid_delta_after_layer2": dict(method="resid", layer=2),
}
torch.manual_seed(1)
facts = []
for k in range(int(sys.argv[5]) if len(sys.argv) > 5 else 8):
    subj = 300 + k
    ctx = torch.randint(3, 290, (3,)).tolist(); rel = torch.randint(3, 290, (2,)).tolist()
    facts.append(dict(X=subj, prompt=torch.tensor([[2] + ctx + [subj] + rel]), target=torch.tensor([400 + k])))
# other-context probes: same subject, other relations; subject embedded in unrelated text
probes_other_rel = [torch.randint(3, 290, (3,)).tolist() for _ in range(6)]
res = {m: [] for m in methods}
for f in facts:
    X_, pr, tg = f["X"], f["prompt"], f["target"]
    with torch.no_grad():
        model.load_state_dict(base_state)
        base_nll = nll_last(pr, tg).item()
        other = torch.stack([torch.tensor([2] + pr[0, 1:4].tolist() + [X_] + r) for r in probes_other_rel])
        para_ctx = torch.stack([torch.tensor([2] + torch.randint(3, 290, (3,)).tolist() + [X_] + pr[0, -2:].tolist())
                                for _ in range(6)])  # "paraphrase": new pre-subject context, same relation
        unrelated = torch.randint(3, 290, (16, 20)); unrelated[:, 6] = X_
        nox = torch.randint(3, 290, (16, 20))
        b_other = model(other).logits[:, -1].float().log_softmax(-1)
        b_para = nll_last(para_ctx, tg.expand(6))
        b_unrel = model(unrelated).logits.float().log_softmax(-1)
        b_nox = model(nox).logits.float().log_softmax(-1)
    for m, kw in methods.items():
        import time; _t = time.time(); r = run_method(X=X_, prompt=pr, target=tg, **kw); print("fact", X_, m, round(time.time() - _t, 1), "s nll", round(r["final_nll"], 4), flush=True)
        with torch.no_grad():
            a_other = model(other).logits[:, -1].float().log_softmax(-1)
            a_para = nll_last(para_ctx, tg.expand(6))
            a_unrel = model(unrelated).logits.float().log_softmax(-1)
            a_nox = model(nox).logits.float().log_softmax(-1)
            # size of the residual-stream change at the subject position at the last layer and at layer-2 output
            hs_a = model(pr, output_hidden_states=True).hidden_states
        cleanup(r)
        with torch.no_grad():
            model.load_state_dict(base_state)
            hs_b = model(pr, output_hidden_states=True).hidden_states
        posX = 4
        res[m].append(dict(
            base_nll=base_nll, final_nll=r["final_nll"], traj=r["traj"], n_params=r["p"].numel(),
            p_target_after=math.exp(-r["final_nll"]),
            para_p_target_before=torch.exp(-b_para).mean().item(), para_p_target_after=torch.exp(-a_para).mean().item(),
            other_rel_p_target_before=b_other[:, tg[0]].exp().mean().item(),
            other_rel_p_target_after=a_other[:, tg[0]].exp().mean().item(),
            other_rel_KL=F.kl_div(a_other, b_other, log_target=True, reduction="batchmean").item(),
            unrelated_KL_after_X=F.kl_div(a_unrel[:, 7:], b_unrel[:, 7:], log_target=True, reduction="none").sum(-1).mean().item(),
            unrelated_KL_before_X=F.kl_div(a_unrel[:, :6], b_unrel[:, :6], log_target=True, reduction="none").sum(-1).mean().item(),
            noX_max_abs_dlogprob=(a_nox - b_nox).abs().max().item(),
            resid_change_rel_at_X_final=((hs_a[-1][0, posX] - hs_b[-1][0, posX]).norm() / hs_b[-1][0, posX].norm()).item(),
            resid_change_rel_at_last_final=((hs_a[-1][0, -1] - hs_b[-1][0, -1]).norm() / hs_b[-1][0, -1].norm()).item(),
        ))
summary = {}
for m, rows in res.items():
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if isinstance(rows[0][k], float)}
    agg["n_params"] = rows[0]["n_params"]; agg["traj_fact0"] = rows[0]["traj"]
    agg["efficacy_p>0.5"] = sum(r["p_target_after"] > 0.5 for r in rows) / len(rows)
    summary[m] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg.items()}
OUT["d_capacity"] = summary
for m, s in summary.items(): print("(d)", m, s)
torch.set_grad_enabled(False)

# ---------------- (e) saturation of the per-layer write ----------------
model.load_state_dict(base_state)
l = 2; seqe = torch.tensor([[2, 31, 45, 52, X, 101, 102]])
branch = {}
def brk(mod, args, out):
    branch["h_in"] = args[0]
def post_norm_hook(mod, args, out):
    branch["ple_out"] = out
h1 = tm.layers[l].post_per_layer_input_norm.register_forward_hook(post_norm_hook)
sat = []
d_dir = torch.randn(D)
for scale in [0, 1e-2, 1e-1, 1, 10, 100, 1000]:
    model.load_state_dict(base_state)
    ple_slice(model, X, l).add_(d_dir * scale)
    model(seqe)
    o = branch["ple_out"][0, 4]
    sat.append(dict(row_scale=scale, ple_branch_out_norm=round(o.norm().item(), 4)))
h1.remove()
model.load_state_dict(base_state)
OUT["e_saturation"] = dict(layer=l, post_norm_weight_norm=tm.layers[l].post_per_layer_input_norm.weight.norm().item(),
                           sweep=sat)
print("(e)", OUT["e_saturation"])

# ---------------- (f) alias token: exact no-op install, then edit only the alias ----------------
model.load_state_dict(base_state)
m2 = copy.deepcopy(model)
old_V = m2.config.vocab_size
emb_before = type(m2.model.embed_tokens_per_layer).__name__
m2.resize_token_embeddings(old_V + 1, mean_resizing=False)
emb_after = type(m2.model.embed_tokens_per_layer).__name__
scale_after = getattr(m2.model.embed_tokens_per_layer, "embed_scale", None)
A = old_V
m2.model.embed_tokens.weight[A] = m2.model.embed_tokens.weight[X]
m2.model.embed_tokens_per_layer.weight[A] = m2.model.embed_tokens_per_layer.weight[X]
s1 = torch.tensor([[2, 31, 45, 52, X, 101, 102, 7, X, 9]])
s1a = s1.clone(); s1a[0, 4] = A           # tokenizer rule: X preceded by (52) -> alias
lo = model(s1).logits.float(); la = m2(s1a).logits.float()
noop = (la[..., :old_V] - lo).abs().max().item()
alias_logit_eq = (la[..., A] - lo[..., X]).abs().max().item()
m2.model.embed_tokens_per_layer.weight[A] += torch.randn(L * D)
la2 = m2(s1a).logits.float()
plain = torch.tensor([[2, 60, 61, X, 101, 102]]); lp_base = model(plain).logits.float(); lp_m2 = m2(plain).logits.float()
OUT["f_alias"] = dict(embed_type_before_resize=emb_before, embed_type_after_resize=emb_after,
                      embed_scale_after_resize=None if scale_after is None else float(scale_after),
                      vocab_size_per_layer_input_after=m2.config.vocab_size_per_layer_input,
                      install_max_abs_dlogit_orig_vocab=noop, alias_logit_minus_X_logit=alias_logit_eq,
                      after_alias_edit_change_at_alias_and_after=(la2[0, 4:, :old_V] - la[0, 4:, :old_V]).abs().max().item(),
                      after_alias_edit_change_before_alias=(la2[0, :4, :old_V] - la[0, :4, :old_V]).abs().max().item(),
                      plain_X_context_change=(lp_m2[..., :old_V] - lp_base).abs().max().item())
print("(f)", OUT["f_alias"])

# ---------------- (g) PLE graft on a 26B-like config (MoE, k_eq_v, no kv sharing, PLE disabled natively) -------
B26 = json.load(open(sys.argv[3]))["text_config"]
def cfg26(ple_dim):
    c = Gemma4TextConfig(
        vocab_size=V, vocab_size_per_layer_input=V, hidden_size=96, hidden_size_per_layer_input=ple_dim,
        intermediate_size=64, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=24,
        global_head_dim=48, num_global_key_value_heads=B26.get("num_global_key_value_heads"), num_kv_shared_layers=0,
        sliding_window=8, final_logit_softcapping=B26["final_logit_softcapping"], rope_parameters=B26["rope_parameters"],
        hidden_activation=B26["hidden_activation"], max_position_embeddings=512, tie_word_embeddings=True,
        pad_token_id=0, bos_token_id=2, eos_token_id=1, attention_k_eq_v=B26["attention_k_eq_v"],
        enable_moe_block=B26["enable_moe_block"], num_experts=8, top_k_experts=2, moe_intermediate_size=32)
    c._attn_implementation = "eager"; return c
torch.manual_seed(5)
host = Gemma4ForCausalLM(cfg26(0)).eval()
graft = Gemma4ForCausalLM(cfg26(8)).eval()
missing, unexpected = graft.load_state_dict(host.state_dict(), strict=False)
gm = graft.model
gm.embed_tokens_per_layer.weight.zero_()            # table starts empty
gm.per_layer_model_projection.weight.zero_()        # no token-identity/context term
gm.per_layer_projection_norm.weight.zero_()
for layer in gm.layers:                             # gate/proj stay random (needed for writes to have an effect)
    layer.post_per_layer_input_norm.weight.fill_(1.0)
ids = torch.randint(3, V, (4, 24))
noop_g = (graft(ids).logits - host(ids).logits).abs().max().item()
gc1 = graft.generate(ids[:1, :10], max_new_tokens=15, do_sample=False)
hc1 = host.generate(ids[:1, :10], max_new_tokens=15, do_sample=False)
Xg = ids[0, 5].item()
gm.embed_tokens_per_layer.weight[Xg] += torch.randn(6 * 8)
after = (graft(ids).logits - host(ids).logits).abs()
rows_have = (ids == Xg)
OUT["g_graft26"] = dict(missing_keys=sorted(missing), unexpected_keys=list(unexpected), install_max_abs_dlogit=noop_g,
                        generate_equal=bool(torch.equal(gc1, hc1)),
                        after_write_max_change_seqs_without_token=after[~rows_have.any(1)].max().item() if (~rows_have.any(1)).any() else None,
                        after_write_max_change_seq0_before_token=after[0, :5].max().item(),
                        after_write_max_change_seq0_from_token=after[0, 5:].max().item(),
                        moe=B26["enable_moe_block"], k_eq_v=B26["attention_k_eq_v"])
print("(g)", OUT["g_graft26"])
json.dump(OUT, open(sys.argv[4], "w"), indent=1, default=str)
