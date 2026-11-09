# End-to-end "write a fact" on the tiny Qwen4Exp: optimise ONLY the anchor's table rows (subject-last-token n-grams)
# for final-position NLL of a target token; measure efficacy, conv-spill, and collateral on other contexts.
import sys, torch, random
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
from ngram_format import NGramTableFormat
torch.set_num_threads(2); random.seed(1); torch.manual_seed(1)
m = tiny_model(vocab=5000, conv_scale=0.3); cfg = m.config
for p in m.parameters(): p.requires_grad_(False)
W = m.model.layers[1].ple.ple_embedding.ngram_embedding.weight; W.requires_grad_(True)
fmt = NGramTableFormat(vocab_size=cfg.vocab_size, eos_token_id=cfg.eos_token_id, ngram_vocab_size_base=cfg.ngram_vocab_size_base,
                       embed_dim=cfg.ple_embed_dim, split_parts=cfg.split_ngram_parts)
R = lambda T: [random.randrange(4999) for _ in range(T)]
subj = R(3); rel = R(4); target = 777
prompt = R(2) + subj + rel; t = 2 + len(subj) - 1           # anchor = last subject token
para = R(5) + subj + R(2) + rel                              # same subject, other prefix, extra tokens before relation
other_rel = R(2) + subj + R(4)                               # same subject, different relation
short_ctx = R(4) + subj[-2:] + rel                           # only the last 2 subject tokens (3-gram broken)
rand_txt = [R(32) for _ in range(30)]
def heads(ids, p, edit):
    return [h for h, r in fmt.rows_for(ids, p) if r in edit]
for name, hs in [("all16", range(16)), ("3gram8", range(8, 16)), ("2gram8", range(8))]:
    with torch.no_grad(): W0 = W.detach().clone()
    edit = sorted({r for h, r in fmt.rows_for(prompt, t) if h in hs})
    mask = torch.zeros(W.shape[0], 1); mask[edit] = 1
    def nll(ids):
        lo = m(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        return torch.nn.functional.cross_entropy(lo[None], torch.tensor([target])), lo
    with torch.no_grad(): base = {k: nll(v)[0].item() for k, v in dict(prompt=prompt, para=para, other_rel=other_rel, short=short_ctx).items()}
    with torch.no_grad(): base_rand = [m(input_ids=torch.tensor([x])).logits[0] for x in rand_txt]
    opt = torch.optim.Adam([W], lr=0.05)
    for step in range(150):
        opt.zero_grad(); l, _ = nll(prompt); l.backward(); W.grad *= mask; opt.step()
    with torch.no_grad():
        after = {k: nll(v)[0].item() for k, v in dict(prompt=prompt, para=para, other_rel=other_rel, short=short_ctx).items()}
        top1 = {k: int(nll(v)[1].argmax() == target) for k, v in dict(prompt=prompt, para=para, other_rel=other_rel, short=short_ctx).items()}
        dr = max((m(input_ids=torch.tensor([x])).logits[0] - b).abs().max().item() for x, b in zip(rand_txt, base_rand))
        nhit = sum(1 for x in rand_txt for p in range(len(x)) if heads(x, p, set(edit)))
        print(f"{name}: edited rows {len(edit)} | NLL base->after " + ", ".join(f"{k} {base[k]:.2f}->{after[k]:.2f}(top1={top1[k]})" for k in base)
              + f" | heads hit at anchor: para {len(heads(para, 5 + len(subj) - 1, set(edit)))}, short {len(heads(short_ctx, 5, set(edit)))}"
              + f" | random text: positions hitting edited rows {nhit}/{30*32}, max|dlogit| {dr:.2e} | row change/row norm {((W-W0)[edit].norm(dim=1).mean()/W0[edit].norm(dim=1).mean()).item():.2f}")
        W.copy_(W0)
