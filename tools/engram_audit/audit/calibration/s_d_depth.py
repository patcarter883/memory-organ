# (d) C4/C8 premise: by which layer does an in-context counterfactual fact reach the window/last positions?
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
tok, m = load_qwen()
NL = m.config.num_hidden_layers
FACTS = [  # (prompt, subject, new object)
 ("The Eiffel Tower is located in", "The Eiffel Tower", "Rome"), ("The Colosseum is located in", "The Colosseum", "London"),
 ("The Statue of Liberty is located in", "The Statue of Liberty", "Tokyo"), ("Big Ben is located in", "Big Ben", "Berlin"),
 ("The Taj Mahal is located in", "The Taj Mahal", "Canada"), ("Mount Fuji is located in", "Mount Fuji", "Egypt"),
 ("The Kremlin is located in", "The Kremlin", "Madrid"), ("The Louvre is located in", "The Louvre", "Sydney"),
 ("The Sydney Opera House is located in", "The Sydney Opera House", "Chicago"), ("The Acropolis is located in", "The Acropolis", "Dublin"),
 ("The capital of France is", "The capital of France", "Berlin"), ("The capital of Japan is", "The capital of Japan", "Rome"),
 ("The capital of Italy is", "The capital of Italy", "Paris"), ("The capital of Germany is", "The capital of Germany", "Madrid"),
 ("The capital of Spain is", "The capital of Spain", "Moscow"), ("Barack Obama was born in", "Barack Obama", "Paris"),
 ("Albert Einstein was born in", "Albert Einstein", "France"), ("The official language of Brazil is", "The official language of Brazil", "German"),
 ("The official language of Mexico is", "The official language of Mexico", "Japanese"), ("Toyota is a car company from", "Toyota", "Sweden"),
]
CTRL = "It was a quiet morning in the small town.\n"
def ids(s): return tok(s)["input_ids"]
def run(x, patch=None):
    """returns list of 25 states [T,d] (0=embeddings, b+1=after block b) and final logits. patch=(b, positions, states[len(pos),d]) replaces after block b."""
    caps, hooks = [None]*(NL+1), []
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if patch is not None and patch[0] + 1 == i:
                o = o.clone(); o[0, patch[1]] = patch[2]
            caps[i] = o[0].detach()
            return ((o,) + tuple(out[1:])) if isinstance(out, tuple) else o
        return h
    hooks.append(m.model.embed_tokens.register_forward_hook(mk(0)))
    for b in range(NL): hooks.append(m.model.layers[b].register_forward_hook(mk(b+1)))
    try:
        with torch.no_grad(): lg = m(input_ids=torch.tensor([x])).logits[0]
    finally:
        for h in hooks: h.remove()
    return caps, lg
def lens(hvec, tid):
    with torch.no_grad(): return F.softmax(m.lm_head(m.model.norm(hvec)), -1)[tid].item()
LN = lambda v: F.layer_norm(v, (v.shape[-1],))
R = {k: torch.zeros(NL+1) for k in ["d_TS", "d_CS", "d_TC", "dA_TC", "dA_CS", "lensT", "lensC", "lensS", "patch_last", "patch_win", "patch_win_ctrl", "patch_anchor"]}
finT = finS = 0.0; n = 0
for prompt, subj, new in FACTS:
    s = ids(prompt); tid = ids(" " + new)[0]
    pre = ids(f"{prompt} {new}.\n"); ctrl = ids(CTRL)
    anchor = len(ids(subj)) - 1; assert s[:anchor+1] == ids(subj)
    T, lgT = run(pre + s); C, _ = run(ctrl + s); S, lgS = run(s)
    P, Pc = len(pre), len(ctrl)
    finT += F.softmax(lgT[-1], -1)[tid].item(); finS += F.softmax(lgS[-1], -1)[tid].item(); n += 1
    win = list(range(anchor, len(s)))
    for l in range(NL+1):
        R["d_TS"][l] += (LN(T[l][-1]) - LN(S[l][-1])).norm().item(); R["d_CS"][l] += (LN(C[l][-1]) - LN(S[l][-1])).norm().item()
        R["d_TC"][l] += (LN(T[l][-1]) - LN(C[l][-1])).norm().item()
        R["dA_TC"][l] += (LN(T[l][P+anchor]) - LN(C[l][Pc+anchor])).norm().item(); R["dA_CS"][l] += (LN(C[l][Pc+anchor]) - LN(S[l][anchor])).norm().item()
        R["lensT"][l] += lens(T[l][-1], tid); R["lensC"][l] += lens(C[l][-1], tid); R["lensS"][l] += lens(S[l][-1], tid)
        b = l - 1  # patch after block b (b=-1 means embeddings: skip)
        if b >= 0:
            _, lg = run(s, (b, [len(s)-1], T[l][P+len(s)-1:P+len(s)])); R["patch_last"][l] += F.softmax(lg[-1], -1)[tid].item()
            _, lg = run(s, (b, win, T[l][[P+w for w in win]])); R["patch_win"][l] += F.softmax(lg[-1], -1)[tid].item()
            _, lg = run(s, (b, win, C[l][[Pc+w for w in win]])); R["patch_win_ctrl"][l] += F.softmax(lg[-1], -1)[tid].item()
            _, lg = run(s, (b, [anchor], T[l][P+anchor:P+anchor+1])); R["patch_anchor"][l] += F.softmax(lg[-1], -1)[tid].item()
for k in R: R[k] /= n
print(f"N facts={n}. Final P(new): teacher(fact in context)={finT/n:.3f}  student(bare prompt)={finS/n:.4f}")
print("state | depth | ||LN T-LN S||  ||LN C-LN S||  ||LN T-LN C|| @last | ||T-C||@anchor ||C-S||@anchor | lens P(new) T / C / S @last | P(new) after patching teacher state into student: last-pos / C8-window[anchor..end] / window-from-CONTROL / anchor-only")
for l in range(NL+1):
    tag = "emb" if l == 0 else f"L{l-1:02d}"
    print(f"{tag:4s} | {l/NL:4.2f} | {R['d_TS'][l]:7.2f} {R['d_CS'][l]:7.2f} {R['d_TC'][l]:7.2f} | {R['dA_TC'][l]:6.2f} {R['dA_CS'][l]:6.2f} | {R['lensT'][l]:.3f} / {R['lensC'][l]:.3f} / {R['lensS'][l]:.3f} | "
          f"{R['patch_last'][l]:.3f} / {R['patch_win'][l]:.3f} / {R['patch_win_ctrl'][l]:.3f} / {R['patch_anchor'][l]:.3f}")
