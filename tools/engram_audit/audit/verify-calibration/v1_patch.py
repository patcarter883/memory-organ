# Independent re-implementation of the F1 test (C8 Stage-1 optimum = teacher states at student positions after layer b).
# Adds: student baseline rank, an "all but position 0" variant (avoid destroying the student's attention sink),
# and a control (unrelated prefix) teacher, to separate "fact info" from "generic prefix" effects.
import torch, torch.nn.functional as F
torch.set_num_threads(2)
from transformers import AutoTokenizer, AutoModelForCausalLM
Q = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
tok = AutoTokenizer.from_pretrained(Q)
m = AutoModelForCausalLM.from_pretrained(Q, dtype=torch.float32, attn_implementation="sdpa").eval()
NL = m.config.num_hidden_layers
FACTS = [("The Eiffel Tower is located in", "The Eiffel Tower", "Rome"), ("The Colosseum is located in", "The Colosseum", "London"),
 ("The Statue of Liberty is located in", "The Statue of Liberty", "Tokyo"), ("Big Ben is located in", "Big Ben", "Berlin"),
 ("The Taj Mahal is located in", "The Taj Mahal", "Canada"), ("Mount Fuji is located in", "Mount Fuji", "Egypt"),
 ("The Kremlin is located in", "The Kremlin", "Madrid"), ("The Louvre is located in", "The Louvre", "Sydney"),
 ("The capital of France is", "The capital of France", "Berlin"), ("The capital of Japan is", "The capital of Japan", "Rome"),
 ("The capital of Italy is", "The capital of Italy", "Paris"), ("Barack Obama was born in", "Barack Obama", "Paris"),
 ("The official language of Brazil is", "The official language of Brazil", "German"), ("Toyota is a car company from", "Toyota", "Sweden")]
# a stronger in-context teacher format too (explicit instruction-like spec), to check the weak-teacher caveat
FMT = {"plain": "{p} {n}.\n", "spec": "Fact (updated): {p} {n}. Remember: {p} {n}.\n"}
CTRL = "It was a quiet morning in the small town, and nobody was awake yet.\n"
enc = lambda s: tok(s)["input_ids"]

def states(x):
    with torch.no_grad():
        return m(input_ids=torch.tensor([x]), output_hidden_states=True).hidden_states  # [emb, after L0, ..., after L23(normed? no: last is post-norm)]

def run_patch(x, b, pos, vecs):
    def hook(mod, inp, out):
        o = (out[0] if isinstance(out, tuple) else out).clone(); o[0, pos] = vecs
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = m.model.layers[b].register_forward_hook(hook)
    try:
        with torch.no_grad(): lg = m(input_ids=torch.tensor([x])).logits[0, -1]
    finally: h.remove()
    return F.softmax(lg.float(), -1)

def cap(x):  # per-layer outputs via hooks (avoid the final-norm ambiguity of hidden_states[-1])
    outs = [None]*NL; hs = []
    for b in range(NL):
        hs.append(m.model.layers[b].register_forward_hook(lambda mod, i, o, b=b: outs.__setitem__(b, (o[0] if isinstance(o, tuple) else o)[0].detach())))
    try:
        with torch.no_grad(): lg = m(input_ids=torch.tensor([x])).logits[0, -1]
    finally:
        for h in hs: h.remove()
    return outs, F.softmax(lg.float(), -1)

LAY = [2, 6, 14, 20, 21, 23]
for fmt, tmpl in FMT.items():
    acc = {}; base_p, base_r, teach_p = [], [], []
    for p, subj, new in FACTS:
        s = enc(p); tid = enc(" " + new)[0]; anchor = len(enc(subj)) - 1
        pre = enc(tmpl.format(p=p, n=new)); P = len(pre); ctrl = enc(CTRL); Pc = len(ctrl)
        T, pT = cap(pre + s); C, _ = cap(ctrl + s); _, pS = cap(s)
        teach_p.append(pT[tid].item()); base_p.append(pS[tid].item()); base_r.append(int((pS > pS[tid]).sum()))
        for b in LAY:
            for name, pos in [("all", list(range(len(s)))), ("all-but-0", list(range(1, len(s)))), ("window", list(range(anchor, len(s))))]:
                pr = run_patch(s, b, pos, T[b][[P + q for q in pos]])
                acc.setdefault((b, name), []).append((pr[tid].item(), int((pr > pr[tid]).sum())))
            pos = list(range(1, len(s)))
            pr = run_patch(s, b, pos, C[b][[Pc + q for q in pos]])
            acc.setdefault((b, "ctrl all-but-0"), []).append((pr[tid].item(), int((pr > pr[tid]).sum())))
    n = len(FACTS); med = lambda v: sorted(v)[len(v)//2]
    print(f"\n[{fmt} teacher] N={n}  teacher P(new)={sum(teach_p)/n:.3f}  student P(new)={sum(base_p)/n:.4f} (median rank {med(base_r)})")
    for b in LAY:
        print(f"  after L{b:02d} (depth {(b+1)/NL:.2f}): " + " | ".join(
            f"{k}: P={sum(a for a,_ in acc[(b,k)])/n:.3f} r~{med([r for _,r in acc[(b,k)]])}" for k in ["all", "all-but-0", "window", "ctrl all-but-0"]))
