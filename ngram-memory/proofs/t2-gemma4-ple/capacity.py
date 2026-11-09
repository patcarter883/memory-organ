"""Write capacity: PLE-row edit (fixed-norm, gated, per-layer) vs free residual delta, longer optimisation.
Two settings for the tiny model's post_per_layer_input_norm: 'unit' (random ~1) and 'real' (rms ratio PLE/FF copied
from real gemma-4-E4B-it weights, real_stats.json). Usage: capacity.py E4B_config real_stats.json out.json"""
import json, math, copy, sys, time
import torch, torch.nn.functional as F
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
torch.set_num_threads(1)
E4B = json.load(open(sys.argv[1]))["text_config"]; RS = json.load(open(sys.argv[2]))
L, H, D, V = 12, 128, 16, 512
layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(L)]

def build(setting):
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(
        vocab_size=V, vocab_size_per_layer_input=V, hidden_size=H, hidden_size_per_layer_input=D, intermediate_size=4 * H,
        num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=1, head_dim=32, global_head_dim=64,
        num_kv_shared_layers=5, sliding_window=8, layer_types=layer_types,
        final_logit_softcapping=E4B["final_logit_softcapping"], rope_parameters=E4B["rope_parameters"],
        hidden_activation=E4B["hidden_activation"], max_position_embeddings=512, tie_word_embeddings=True,
        pad_token_id=0, bos_token_id=2, eos_token_id=1, initializer_range=0.05)
    cfg._attn_implementation = "eager"
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or "layernorm.weight" in n:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
        m.model.embed_tokens_per_layer.weight.normal_(0, 0.25)
        if setting == "real":
            ratio = RS["ratio_ple_to_ff"]
            # tiny 0..5 (readable) <-> real 0..22 ; tiny 6..11 (local) <-> real 23..41
            mp = [round(i * 22 / 5) for i in range(6)] + [round(23 + (i - 6) * 18 / 5) for i in range(6, 12)]
            for i, layer in enumerate(m.model.layers):
                wf = layer.post_feedforward_layernorm.weight; wp = layer.post_per_layer_input_norm.weight
                wp.mul_(ratio[mp[i]] * wf.pow(2).mean().sqrt() / wp.pow(2).mean().sqrt())
    for p in m.parameters(): p.requires_grad_(False)
    return m

def nll_last(m, ids, target):
    return -m(ids).logits[:, -1].float().log_softmax(-1)[torch.arange(ids.shape[0]), target]

def install(m, method, X, layers=None, layer=2, clamp=None):
    tm = m.model; undo = []
    if method == "ple":
        p = torch.zeros(len(layers), D, requires_grad=True)
        orig = tm.embed_tokens_per_layer.forward
        def fwd(ids):
            add = torch.zeros(L, D); add[layers] = p
            return orig(ids) + (ids == X).unsqueeze(-1).float() * (add.flatten() * tm.embed_tokens_per_layer.embed_scale)
        tm.embed_tokens_per_layer.forward = fwd
        undo.append(lambda: setattr(tm.embed_tokens_per_layer, "forward", orig))
        proj = lambda: None
    else:
        p = torch.zeros(H, requires_grad=True); cur = {}
        h1 = m.register_forward_pre_hook(lambda mod, a, k: cur.__setitem__("ids", a[0]), with_kwargs=True)
        h2 = tm.layers[layer].register_forward_hook(lambda mod, a, out: out + (cur["ids"] == X).unsqueeze(-1).float() * p)
        undo += [h1.remove, h2.remove]
        def proj():
            if clamp is not None and p.norm() > clamp:
                with torch.no_grad(): p.mul_(clamp / p.norm())
    return p, proj, undo

def branch_norm(m, ids, layer, pos):
    got = {}
    h = m.model.layers[layer].post_per_layer_input_norm.register_forward_hook(lambda mod, a, o: got.__setitem__("o", o))
    with torch.no_grad(): m(ids)
    h.remove(); return got["o"][0, pos].norm().item()

out = {}
for setting in ["unit", "real"]:
    m = build(setting)
    torch.manual_seed(1)
    facts = []
    for k in range(4):
        subj = 300 + k
        ctx = torch.randint(3, 290, (3,)).tolist(); rel = torch.randint(3, 290, (2,)).tolist()
        facts.append((subj, torch.tensor([[2] + ctx + [subj] + rel]), torch.tensor([400 + k])))
    rels = [torch.randint(3, 290, (2,)).tolist() for _ in range(6)]
    with torch.no_grad():
        hs = m(facts[0][1], output_hidden_states=True).hidden_states
    bn2 = branch_norm(m, facts[0][1], 2, 4)
    out[setting] = dict(ple_branch_norm_layer2_at_X=bn2, resid_norm_after_layer2_at_X=hs[3][0, 4].norm().item(),
                        ple_branch_norm_by_layer=[round(branch_norm(m, facts[0][1], l, 4), 3) for l in range(L)],
                        resid_norm_by_layer=[round(h[0, 4].norm().item(), 2) for h in hs[1:]])
    methods = {
        "ple_readable_layers0-5": dict(method="ple", layers=list(range(6))),
        "ple_single_layer2": dict(method="ple", layers=[2]),
        "free_resid_after_layer2": dict(method="resid", layer=2),
        "free_resid_after_layer2_clamped_2xPLEbranch": dict(method="resid", layer=2, clamp=2 * bn2),
    }
    for name, kw in methods.items():
        rows = []
        for X, pr, tg in facts:
            with torch.no_grad():
                other = torch.stack([torch.tensor(pr[0, :5].tolist() + r) for r in rels])
                para = torch.stack([torch.tensor([2] + torch.randint(3, 290, (3,)).tolist() + [X] + pr[0, -2:].tolist()) for _ in range(6)])
                unrel = torch.randint(3, 290, (16, 20)); unrel[:, 6] = X
                b_other = m(other).logits[:, -1].float().log_softmax(-1); b_para = nll_last(m, para, tg.expand(6))
                b_unrel = m(unrel).logits.float().log_softmax(-1); base = nll_last(m, pr, tg).item()
            p, proj, undo = install(m, X=X, **kw)
            opt = torch.optim.Adam([p], lr=0.1); t = time.time()
            for s in range(1000):
                loss = nll_last(m, pr, tg).sum(); opt.zero_grad(); loss.backward(); opt.step(); proj()
            with torch.no_grad():
                fin = nll_last(m, pr, tg).item()
                a_other = m(other).logits[:, -1].float().log_softmax(-1); a_para = nll_last(m, para, tg.expand(6))
                a_unrel = m(unrel).logits.float().log_softmax(-1)
            for u in undo: u()
            rows.append(dict(base=base, final=fin, p=math.exp(-fin), param_norm=p.norm().item(),
                             para_p_before=torch.exp(-b_para).mean().item(), para_p_after=torch.exp(-a_para).mean().item(),
                             other_rel_p_before=b_other[:, tg[0]].exp().mean().item(), other_rel_p_after=a_other[:, tg[0]].exp().mean().item(),
                             other_rel_KL=F.kl_div(a_other, b_other, log_target=True, reduction="batchmean").item(),
                             unrel_KL_after_X=F.kl_div(a_unrel[:, 7:], b_unrel[:, 7:], log_target=True, reduction="none").sum(-1).mean().item()))
            print(setting, name, X, f"{time.time()-t:.1f}s", {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
        agg = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}
        agg["efficacy_p>0.5"] = sum(r["p"] > 0.5 for r in rows) / len(rows)
        out[setting][name] = agg
        print("AGG", setting, name, agg, flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
