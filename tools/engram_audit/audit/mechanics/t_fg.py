"""(d') is an edited table row read during cached decode? (f) bf16 table update loss, (g) left-padding/attention_mask."""
from common import *
import torch.nn as nn, torch.nn.functional as F

print("=== (d') edit written to rows of the true n-grams ending at symbol X; is it read when X is decoded with KV cache? ===")
m = tiny_model(); d = m.config.hidden_size; eng = make_engram(d)
torch.manual_seed(5)
with torch.no_grad():
    nn.init.normal_(eng.v_proj.weight, std=0.5); nn.init.dirac_(eng.short_conv.weight, groups=d)  # best-case: all channels alive
    eng.short_conv.weight[:, 0, 3] = 1.0  # also give the current-token tap weight, so decode CAN see current e_t
g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
seq = torch.randint(1, 1000, (1, 10)); X_pos = 7  # symbol X at position 7, then Y at 8, Z at 9
idx = eng.hasher.hash_sequence(seq)
def logits_full():
    with torch.no_grad(): return m(input_ids=seq).logits[0]
def logits_cached():
    with torch.no_grad():
        out = m(input_ids=seq[:, :X_pos], use_cache=True); past = out.past_key_values; L = [None] * X_pos
        for t in range(X_pos, seq.size(1)):
            out = m(input_ids=seq[:, t:t + 1], past_key_values=past, use_cache=True); past = out.past_key_values
            L.append(out.logits[0, -1])
        return torch.stack(L[X_pos:])
f0, c0 = logits_full()[X_pos:], logits_cached()
saved = {n: eng.cpu_tables[n].clone() for n in eng.orders}
for n in eng.orders:  # the "edit": big write into the K rows of the n-gram ending at X
    eng.cpu_tables[n][idx[n][0, X_pos]] += 3.0
f1, c1 = logits_full()[X_pos:], logits_cached()
print(" full re-forward: |logit change| at X, X+1, X+2:", [f"{v:.3f}" for v in (f1 - f0).abs().amax(-1).tolist()])
print(" KV-cache decode: |logit change| at X, X+1, X+2:", [f"{v:.3g}" for v in (c1 - c0).abs().amax(-1).tolist()])
dec_idx = eng.hasher.hash_sequence(seq[:, X_pos:X_pos + 1])
print(" buckets gathered at decode of X (2g):", dec_idx[2][0, 0].tolist(), " edited rows (2g):", idx[2][0, X_pos].tolist())
for n in eng.orders: eng.cpu_tables[n].copy_(saved[n])

print("\n=== (f) bf16 host table: are small updates lost? ===")
torch.manual_seed(0)
r, K, M = 256, 4, 8
tab = (torch.randn(4096, r) * 0.02).to(torch.bfloat16)
e_q_norm = (torch.randn(K, 2 * r) * 0.02).mean(0).norm()
maxn = max(0.4 * e_q_norm.item(), 0.1)
print(f" ||e_q|| ~ {e_q_norm:.3f} -> max ||delta*|| = {maxn:.3f}, per-coord ~ {maxn/(2*r)**0.5:.2e}")
for scale in [1e-2, 1e-3, 1e-4, 1e-5]:
    upd = torch.randn(4096, r) * scale
    new = tab + upd.to(torch.bfloat16)           # verbatim pattern: delta_E.to(dtype) then += in bf16
    realized = new.float() - tab.float()
    lost = (realized == 0).float().mean().item()
    relerr = ((realized - upd).norm() / upd.norm()).item()
    print(f"  update std {scale:.0e}: frac coords unchanged {lost:.3f}, rel. error of realized update {relerr:.3f}")
# replicate verbatim Stage-2 ridge numbers for a typical edit: same anchor n-gram across M paraphrases
for n, pen in [(2, 10.0), (3, 1.0)]:
    U = K; A = torch.full((M, U), 1.0 / K)
    gamma = pen * (1.0 + torch.log1p(torch.ones(U))) * 1e-2
    target = torch.randn(r); target = target / target.norm() * (maxn / 2 ** 0.5)
    dE = torch.linalg.solve(A.T @ A + torch.diag(gamma), A.sum(0, keepdim=True).T @ target.unsqueeze(0))
    realized_e = dE.mean(0)  # e = mean over K heads of the rows
    print(f"  order {n}: ||realized delta e|| / ||target slice|| = {realized_e.norm()/target.norm():.3f}; "
          f"per-coord |dE| ~ {dE.abs().mean():.1e}")
    rows = tab[:U].clone(); rows += dE.to(torch.bfloat16)
    rr = (rows.float() - tab[:U].float()).mean(0)
    print(f"           after bf16 write: realized/target cos={F.cosine_similarity(rr, target, 0):.4f}, "
          f"norm ratio {rr.norm()/target.norm():.3f}")
# sequential tiny edits on the same bucket (e.g. sparse-SGD warmup lr*grad ~ 1e-5): accumulate 1000 times
row = torch.full((r,), 0.02).to(torch.bfloat16); acc32 = torch.full((r,), 0.02)
for _ in range(1000):
    step = torch.full((r,), 1e-5); row += step.to(torch.bfloat16); acc32 += step
print(f"  1000 sequential +1e-5 updates on a 0.02 entry: bf16 moved {row[0].item()-0.02:+.2e} (fp32 {acc32[0].item()-0.02:+.2e})")

print("\n=== (f') is the gathered tensor pinned? ===")
print(" cannot test is_pinned()/async H2D: no CUDA here (pin_memory raises). Code path: cpu_tables[n][idx] is advanced indexing ->",
      "new tensor; .mean(); torch.cat -> fresh pageable allocations.")

print("\n=== (g) left padding + attention_mask, batch vs single ===")
m = tiny_model(); d = m.config.hidden_size; eng = make_engram(d)
torch.manual_seed(7)
with torch.no_grad():
    nn.init.normal_(eng.v_proj.weight, std=0.5); nn.init.dirac_(eng.short_conv.weight, groups=d)
g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
a = torch.randint(1, 1000, (1, 6)); PAD = 999
padded = torch.cat([torch.full((1, 4), PAD), a], 1); mask = torch.cat([torch.zeros(1, 4), torch.ones(1, 6)], 1).long()
for vz in [False, True]:
    if vz:
        with torch.no_grad(): eng.v_proj.weight.zero_()
    with torch.no_grad():
        single = m(input_ids=a).logits[0]
        lp = m(input_ids=padded, attention_mask=mask).logits[0, 4:]
    print(f" v_proj {'==0' if vz else '!=0'}: max|single - leftpadded| at real positions:",
          [f"{v:.3g}" for v in (single - lp).abs().amax(-1).tolist()])
print(" hasher signature:", "hash_sequence(self, input_ids)  -- no attention_mask argument")
