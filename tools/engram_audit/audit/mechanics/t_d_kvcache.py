"""(d) KV-cache incremental decode vs full re-forward with NON-ZERO v_proj (tests claim C3)."""
from common import *
import torch.nn as nn

def run(conv_mode, seed=1):
    m = tiny_model()
    d = m.config.hidden_size
    eng = make_engram(d)
    torch.manual_seed(seed)
    with torch.no_grad():
        nn.init.normal_(eng.v_proj.weight, std=0.5)          # "trained" non-zero v_proj
        if conv_mode == "random":
            nn.init.normal_(eng.short_conv.weight, std=0.5)  # "trained" conv
        elif conv_mode == "dirac_grouped":
            nn.init.dirac_(eng.short_conv.weight, groups=d)
        # else: verbatim init (dirac_ without groups -> only ch0, k-index 2)
    calls = []
    orig = eng.prefetch_from_host
    def spy(ids):
        calls.append(ids.clone()); return orig(ids)
    eng.prefetch_from_host = spy
    g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
    ids = torch.randint(1, 1000, (1, 12))
    P = 6
    with torch.no_grad():
        full = m(input_ids=ids).logits  # full re-forward of the whole sequence
        out = m(input_ids=ids[:, :P], use_cache=True)
        past = out.past_key_values
        diffs = [(out.logits[0, -1] - full[0, P - 1]).abs().max().item()]
        for t in range(P, ids.size(1)):
            out = m(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            diffs.append((out.logits[0, -1] - full[0, t]).abs().max().item())
        # baseline: same comparison with graft effectively removed (v_proj=0)
        eng.v_proj.weight.zero_()
        full0 = m(input_ids=ids).logits
        out = m(input_ids=ids[:, :P], use_cache=True); past = out.past_key_values
        base = [(out.logits[0, -1] - full0[0, P - 1]).abs().max().item()]
        for t in range(P, ids.size(1)):
            out = m(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True); past = out.past_key_values
            base.append((out.logits[0, -1] - full0[0, t]).abs().max().item())
    print(f"[{conv_mode}] max|incremental - full| per step (prefill-last, then decode steps):")
    print("   graft v_proj!=0 :", [f"{x:.3g}" for x in diffs])
    print("   graft v_proj==0 :", [f"{x:.3g}" for x in base])
    print("   input_ids seen by prefetch during decode:", [c.tolist() for c in calls[2:5]])
    return eng

eng = run("verbatim")
run("dirac_grouped")
run("random")

print("\nWhat n-grams get hashed for a single decode token t=777?")
h = eng.hasher
one = h.hash_sequence(torch.tensor([[777]]))
pair = h.hash_sequence(torch.tensor([[0, 777]]))
tri = h.hash_sequence(torch.tensor([[0, 0, 777]]))
ctx = h.hash_sequence(torch.tensor([[5, 9, 777]]))
print(" decode 2-gram buckets       :", one[2][0, -1].tolist(), " == hash((0,777)):", torch.equal(one[2][0, -1], pair[2][0, -1]))
print(" decode 3-gram buckets       :", one[3][0, -1].tolist(), " == hash((0,0,777)):", torch.equal(one[3][0, -1], tri[3][0, -1]))
print(" true-context (5,9,777) 2g/3g:", ctx[2][0, -1].tolist(), ctx[3][0, -1].tolist())

# HF generate: greedy with cache vs greedy by full re-forward each step
print("\nHF generate (use_cache=True) vs manual full re-forward greedy, random conv, v_proj!=0")
m = tiny_model(); d = m.config.hidden_size; eng = make_engram(d)
torch.manual_seed(3)
with torch.no_grad():
    nn.init.normal_(eng.v_proj.weight, std=2.0); nn.init.normal_(eng.short_conv.weight, std=0.5)
g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
prompt = torch.randint(1, 1000, (1, 8))
with torch.no_grad():
    gen = m.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=True, pad_token_id=0)
    seq = prompt.clone()
    for _ in range(12):
        nxt = m(input_ids=seq).logits[0, -1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], 1)
print(" generate (cached):", gen[0, 8:].tolist())
print(" full re-forward  :", seq[0, 8:].tolist())
print(" identical:", torch.equal(gen, seq))
