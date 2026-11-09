"""Per-row update magnitude (what a hash co-tenant receives) and sequential-edit interference via a shared 2-gram,
using Gemini's VERBATIM Stage 2."""
import torch
from common import make_engram, run_stage2, read_e, eg
torch.manual_seed(0)
DOT, STEP = 13, 202
def para(subj, M=8, seed=1):
    g = torch.Generator().manual_seed(seed); rows, anc = [], []
    for m in range(M):
        pre = torch.randint(1000, 5000, (6,), generator=g).tolist(); rows.append(pre + subj + [7, 8, 9]); anc.append(6 + len(subj) - 1)
    return torch.tensor(rows), anc
engram = make_engram(); solver = eg.EngramEditSolver(None, engram); r = engram.r_per_order
idsA, aA = para([101, DOT, STEP]); idsB, aB = para([303, DOT, STEP], seed=2)
torch.manual_seed(5); dA = torch.randn(2 * r); dA = dA / dA.norm() * 0.1
torch.manual_seed(6); dB = torch.randn(2 * r); dB = dB / dB.norm() * 0.1
e0A = read_e(engram, idsA)[0, aA[0]]
ns = run_stage2(solver, idsA, aA, dA)
dE = ns["delta_E"]  # last order processed = 3-gram
print("3-gram slice: per-row delta_E / t3 (each of K=4 rows):", [round((row @ dA[r:] / (dA[r:] @ dA[r:])).item(), 3) for row in dE])
print("-> a hash co-tenant sharing ONE of the 4 head-buckets reads mean over heads: receives", round(0.25 * (dE[0] @ dA[r:] / (dA[r:] @ dA[r:])).item(), 3), "x the 3-gram slice")
e1A = read_e(engram, idsA)[0, aA[0]]
print(f"after edit A: realised at A anchor = {((e1A-e0A)@dA/(dA@dA)):.3f} of dA, cos={torch.cosine_similarity(e1A-e0A, dA, 0):.4f}")
run_stage2(solver, idsB, aB, dB)   # edit B = OtherGraph.Step, shares 2-gram ('.','Step') rows with A
e2A = read_e(engram, idsA)[0, aA[0]]
d = e2A - e0A
print(f"after edit B: change at A anchor: proj on dA={(d@dA/(dA@dA)):.3f}, cos(d, dA)={torch.cosine_similarity(d, dA, 0):.4f}, "
      f"||d - realisedA||/||dA|| = {((e2A-e1A).norm()/dA.norm()):.3f}; 2g slice now cos to dA_2g={torch.cosine_similarity(d[:r], dA[:r], 0):.3f}, "
      f"cos to dB_2g={torch.cosine_similarity(d[:r], dB[:r], 0):.3f}")
