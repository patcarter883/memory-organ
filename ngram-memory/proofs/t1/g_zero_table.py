# Graft no-op check: a zero-initialised table makes the native PLE block output exactly 0, and what the first edit does.
import sys, torch
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
torch.manual_seed(0)
m = tiny_model(vocab=5000); ple = m.model.layers[1].ple; W = ple.ple_embedding.ngram_embedding.weight
with torch.no_grad(): W.zero_()
ids = torch.randint(0, 4999, (1, 20)); X = torch.randn(1, 20, 4 * 128)
o = ple(X, ids, None); print("zero table: max|PLE out| =", o.abs().max().item())
for s in (1e-6, 1e-4, 1e-2, 1.0):   # put a tiny value into the 16 rows of position 10 only
    r = ple.ple_embedding  # get rows at position 10
    cap = []; h = r.ngram_embedding.register_forward_hook(lambda mm, i, oo: cap.append(i[0])); ple(X, ids, None); h.remove()
    rows = cap[0][0, 10]
    with torch.no_grad(): W.zero_(); W[rows] = s * torch.randn(len(rows), W.shape[1])
    with torch.no_grad(): o = ple(X, ids, None)[0]
    u = o.norm(dim=-1)
    print(f"row scale {s:g}: |out| at t=10 {u[10]:.3e}, t=13 {u[13]:.3e}, t=16 {u[16]:.3e}, t=19 {u[19]:.3e}, others max {u[[p for p in range(20) if p not in (10,13,16,19)]].max():.1e}")
