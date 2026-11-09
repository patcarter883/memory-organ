# V5: row-diff ledger on a bf16 native table: quantisation loss of small diffs, and exact rollback.
import torch
torch.manual_seed(0)
rows, w = 64, 160
orig = (torch.randn(rows, w) * 0.09 / 160 ** 0.5).bfloat16()          # real per-head row L2 ~0.09 (t1/real_stats.out)
for s in (0.003, 0.03, 0.3, 1.0, 5.0):
    u = torch.randn(rows, w); u = u / u.norm(dim=1, keepdim=True) * s * orig.float().norm(dim=1, keepdim=True)
    got = (orig.float() + u).bfloat16().float() - orig.float()
    print(f"diff scale {s:5.3f}x row: realised/intended = {(got*u).sum().item()/u.pow(2).sum().item():.4f}, rel.err {((got-u).norm()/u.norm()).item():.4f}")
# three overlapping edits, then roll back edit #2
ids = [torch.arange(0, 40), torch.arange(20, 60), torch.arange(10, 30)]
U = [torch.randn(len(i), w) * 0.02 for i in ids]
naive = orig.clone()
for i, u in zip(ids, U): naive[i] = (naive[i].float() + u).bfloat16()
naive[ids[1]] = (naive[ids[1]].float() - U[1]).bfloat16()               # in-place subtract rollback
def materialise(active):
    acc = orig.float().clone()
    for k in active: acc.index_add_(0, ids[k], U[k])
    return acc.bfloat16()
ref = materialise([0, 2])
print(f"naive in-place rollback: max|W - ref| = {(naive.float()-ref.float()).abs().max().item():.2e}, rows differing {int((naive != ref).any(1).sum())}/{rows}")
led = materialise([0, 1, 2]); led = materialise([0, 2])
print(f"ledger (orig bits + fp32 diffs, re-materialise touched rows): max|W - ref| = {(led.float()-ref.float()).abs().max().item():.2e}; order-independent: {bool((materialise([2,0])==ref).all())}")
