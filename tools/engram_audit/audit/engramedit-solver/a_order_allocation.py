"""(a)+(b)+(c): run Gemini's VERBATIM Stage 2 with a controlled delta*, measure what is realised per order slice,
whether order_penalties can move norm between orders, ridge shrinkage vs Gamma, bf16 effect, and contrast with
EngramEdit's SUM-fusion allocation (Eq. 10 + App. A.3.3 weights)."""
import torch
from common import make_engram, run_stage2, read_e, eg
torch.manual_seed(0)

VOCAB = 5000
SPINE, DOT, STEP, OTHER = 101, 13, 202, 303          # "SpineGraph" "." "Step" ; "OtherGraph"
M = 8


def paraphrases(subject, M=M, T=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    rows, anchors = [], []
    for m in range(M):
        pre = torch.randint(1000, VOCAB, (T - len(subject) - 3,), generator=g).tolist()  # varied wording before subject
        post = torch.randint(1000, VOCAB, (3,), generator=g).tolist()                      # target tokens
        rows.append(pre + subject + post)
        anchors.append(len(pre) + len(subject) - 1)                                        # last subject token
    return torch.tensor(rows), anchors


def realised(engram, solver, ids, anchors, dstar, pen):
    e0 = read_e(engram, ids)
    run_stage2(solver, ids, anchors, dstar.to(engram.dtype), pen)
    e1 = read_e(engram, ids)
    d = torch.stack([e1[m, a] - e0[m, a] for m, a in enumerate(anchors)]).mean(0)
    r = engram.r_per_order
    t2, t3 = dstar[:r], dstar[r:]
    d2, d3 = d[:r], d[r:]
    return dict(ratio2=(d2 @ t2 / (t2 @ t2)).item(), ratio3=(d3 @ t3 / (t3 @ t3)).item(),
                share3_of_realised_sq=(d3.pow(2).sum() / d.pow(2).sum()).item(),
                share3_of_target_sq=(t3.pow(2).sum() / dstar.pow(2).sum()).item(),
                total_ratio=(d @ dstar / (dstar @ dstar)).item(), cos=torch.cosine_similarity(d, dstar, 0).item())


print("=== (a) Gemini CONCAT allocation: per-order realised fraction, verbatim Stage 2 ===")
ids, anchors = paraphrases([SPINE, DOT, STEP])
for pen in [{2: 10.0, 3: 1.0}, {2: 1.0, 3: 1.0}, {2: 1000.0, 3: 1.0}, {2: 1e5, 3: 1.0}]:
    for name, mk in [("isotropic", lambda r: torch.randn(2 * r)),
                     ("all-in-2gram-half", lambda r: torch.cat([torch.randn(r), torch.zeros(r)]))]:
        engram = make_engram(); solver = eg.EngramEditSolver(None, engram)
        torch.manual_seed(5); ds = mk(engram.r_per_order); ds = ds / ds.norm() * 0.1
        out = realised(engram, solver, ids, anchors, ds, pen)
        print(f"pen={pen} delta*={name:18s} realised/target 2g={out['ratio2']:.3f} 3g={out['ratio3']:.3f} | "
              f"3g share of ||delta*||^2={out['share3_of_target_sq']:.3f} -> of realised={out['share3_of_realised_sq']:.3f} "
              f"| total realised={out['total_ratio']:.3f} cos={out['cos']:.4f}")

print("\n=== unrelated prompt sharing only the 2-gram ('.','Step') (OtherGraph.Step) ===")
engram = make_engram(); solver = eg.EngramEditSolver(None, engram)
torch.manual_seed(5); ds = torch.randn(2 * engram.r_per_order); ds = ds / ds.norm() * 0.1
uid, uanc = paraphrases([OTHER, DOT, STEP], M=1, seed=99)
eu0 = read_e(engram, uid)[0, uanc[0]]
run_stage2(solver, ids, anchors, ds, {2: 10.0, 3: 1.0})
eu1 = read_e(engram, uid)[0, uanc[0]]
du = eu1 - eu0; r = engram.r_per_order
print(f"unrelated anchor change / ||delta*|| = {du.norm()/ds.norm():.3f}; 2g-slice change/||t2|| = "
      f"{(du[:r]@ds[:r]/(ds[:r]@ds[:r])):.3f}; 3g-slice change norm = {du[r:].norm():.2e}")

print("\n=== (c) ridge shrinkage vs Gamma (M identical anchor rows, K=4): realised = M/(M+g*K) ===")
for scale in [1, 10, 100, 1000]:
    pen = {2: 10.0 * scale, 3: 1.0 * scale}
    engram = make_engram(); solver = eg.EngramEditSolver(None, engram)
    out = realised(engram, solver, ids, anchors, ds, pen)
    g2 = pen[2] * (1 + torch.log1p(torch.tensor(1.0))).item() * 1e-2
    g3 = pen[3] * (1 + torch.log1p(torch.tensor(1.0))).item() * 1e-2
    print(f"penalty x{scale:5d}: g2={g2:.4f} g3={g3:.4f} realised 2g={out['ratio2']:.3f} (pred {M/(M+g2*4):.3f}) "
          f"3g={out['ratio3']:.3f} (pred {M/(M+g3*4):.3f})")
for Mx in [1, 2, 4, 8, 32]:
    engram = make_engram(); solver = eg.EngramEditSolver(None, engram)
    i2, a2 = paraphrases([SPINE, DOT, STEP], M=Mx)
    out = realised(engram, solver, i2, a2, ds, {2: 10.0, 3: 1.0})
    print(f"M={Mx:2d}: realised 2g={out['ratio2']:.3f} 3g={out['ratio3']:.3f}")

print("\n=== (b) bucket_counts is an EDIT counter: re-editing the same subject 6 times ===")
engram = make_engram(); solver = eg.EngramEditSolver(None, engram)
for k in range(6):
    hm = engram.hasher.hash_sequence(ids)
    b2 = hm[2][0, anchors[0]]; b3 = hm[3][0, anchors[0]]
    c2 = engram.bucket_counts[2][b2].tolist(); c3 = engram.bucket_counts[3][b3].tolist()
    out = realised(engram, solver, ids, anchors, ds, {2: 10.0, 3: 1.0})
    print(f"edit #{k+1}: counts 2g={c2} 3g={c3} -> realised 2g={out['ratio2']:.3f} 3g={out['ratio3']:.3f}")
print("(counts for a never-edited bucket, e.g. ('.','Step') as used across a corpus:",
      engram.bucket_counts[2][:3].tolist(), ")")

print("\n=== bf16 tables (the code's default dtype) ===")
engram = make_engram(dtype=torch.bfloat16); solver = eg.EngramEditSolver(None, engram)
out = realised(engram, solver, ids, anchors, ds, {2: 10.0, 3: 1.0})
print(f"bf16: realised 2g={out['ratio2']:.3f} 3g={out['ratio3']:.3f} cos={out['cos']:.4f}")
e = read_e(engram, ids); print(f"baseline ||e_q|| (r_mem=32) = {e[0, anchors[0]].norm():.4f}")
engram512 = make_engram(r_mem=512, num_buckets=200_000)
e = read_e(engram512, ids)
print(f"baseline ||e_q|| at r_mem=512 = {e[0, anchors[0]].norm():.4f}; 0.4*||e_q|| = {0.4*e[0, anchors[0]].norm():.4f} "
      f"-> max_delta_norm = clamp(min=0.1) = {max(0.1, 0.4*e[0, anchors[0]].norm().item()):.4f}")

print("\n=== EngramEdit SUM fusion (Eq. 9-10, App. A.3.3) for the same request ===")
def w_engramedit(n, pi, eta={2: 8, 3: 2, 4: 1}, gamma=9, beta=1, wfmax=10, wmax=64):
    wfreq = min(1 + gamma * pi ** beta, wfmax)
    return min(eta[n] + wfreq - 1, wmax)
def lam(w): return 0.05 * w + 0.01
for pi2, pi3 in [(0.99, 0.05), (0.99, 0.5), (0.5, 0.5)]:
    l2, l3 = lam(w_engramedit(2, pi2)), lam(w_engramedit(3, pi3))
    for P in [1, 4, 8]:
        A = torch.ones(P, 2)                      # each expression activates the one 2-gram and one 3-gram (binary)
        B = ds.unsqueeze(0).repeat(P, 1)          # b_p = delta*
        U = torch.linalg.solve(A.T @ A + torch.diag(torch.tensor([l2, l3])), A.T @ B)
        real = U.sum(0)                           # sum fusion: h = e(g2) + e(g3)
        print(f"pi2={pi2} pi3={pi3} lam2={l2:.3f} lam3={l3:.3f} P={P}: realised={real@ds/(ds@ds):.3f}, "
              f"3-gram share of realised = {U[1].norm()/(U[0].norm()+U[1].norm()):.3f}; "
              f"unrelated X.Step gets {U[0]@ds/(ds@ds):.3f} of delta*")
# Gemini's own gamma values plugged into a SUM model
g2 = 10 * 1.6931 * 1e-2; g3 = 1 * 1.6931 * 1e-2
A = torch.ones(8, 2); U = torch.linalg.solve(A.T @ A + torch.diag(torch.tensor([g2, g3])), A.T @ ds.unsqueeze(0).repeat(8, 1))
print(f"Gemini gammas in a SUM model (P=8): 3-gram share = {U[1].norm()/(U[0].norm()+U[1].norm()):.3f}, realised={U.sum(0)@ds/(ds@ds):.3f}")
