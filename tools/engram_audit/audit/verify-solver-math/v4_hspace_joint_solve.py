"""EngramEdit A.3.2 (gated/concat host) matches targets in memory-OUTPUT space h via a Jacobian, solved jointly.
Does a joint h-space ridge solve reallocate between the concatenated 2-gram and 3-gram slices?
Model: h = alpha * W_V z, z=[e2; e3] (gate held fixed = first-order Jacobian J = alpha*W_V), one 2-gram and one
3-gram embedding at the anchor, P identical expressions. Solve min P||J2 u2 + J3 u3 - b||^2 + l2||u2||^2 + l3||u3||^2."""
import torch
torch.manual_seed(0); torch.set_num_threads(2)
d, r = 64, 16   # d_model, r_per_order
P, alpha = 8, 0.5
def solve(J2, J3, b, l2, l3):
    J = torch.cat([J2, J3], 1)
    L = torch.diag(torch.cat([torch.full((r,), l2), torch.full((r,), l3)]))
    u = torch.linalg.solve(P * J.T @ J + L, P * J.T @ b)
    c2, c3 = J2 @ u[:r], J3 @ u[r:]
    tot = c2 + c3
    return (c3 @ tot / (tot @ tot)).item(), (tot @ b / (b @ b)).item()
b = torch.randn(d)
W2 = torch.randn(d, r) / d**0.5
W3_indep = torch.randn(d, r) / d**0.5
Q, _ = torch.linalg.qr(torch.randn(r, r)); W3_same = W2 @ Q   # same column space as the 2-gram block
for name, W3 in [("random independent blocks (generic)", W3_indep), ("blocks share column space (e.g. both aligned to one FFN subspace)", W3_same)]:
    for l2, l3 in [(0.1693, 0.1693), (0.1693, 0.01693), (1.693, 0.01693)]:
        bb = b if name.startswith("blocks") is False else b
        s, real = solve(alpha*W2, alpha*W3, bb, l2, l3)
        print(f"{name:65} l2={l2:<7} l3={l3:<7}: 3g share of realised (projection)={s:.3f}, realised/target={real:.3f}")
    # target restricted to span of the blocks (what a z-space delta would produce)
    dz = torch.randn(2*r); bz = alpha*torch.cat([W2, W3], 1) @ dz
    s, real = solve(alpha*W2, alpha*W3, bz, 0.1693, 0.01693)
    print(f"{name:65} target=J*delta_z (in span), l2=0.1693 l3=0.01693: 3g share={s:.3f}, realised/target={real:.3f}")
