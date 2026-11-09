# light-weight summary of (partial) results.json, no model needed
import json, sys
r = json.load(open(sys.argv[1]))
pf = r["per_fact"]; print("n facts", len(pf))
arms = []
for f in pf:
    for a in f["arms"]:
        if a not in arms: arms.append(a)
def m(a, g, k):
    v = [f["arms"][a][g][k] for f in pf if a in f["arms"] and g in f["arms"][a]]
    return sum(v)/len(v) if v else float("nan"), len(v)
print(f"{'arm':24s} n  rw.eff rw.top1 rw.Pnew opt.eff held.eff hCF.eff held.P leak.KL leak.t1")
for a in arms:
    n = m(a,"rewrite","eff")[1]
    print(f"{a:24s} {n:2d} " + " ".join(f"{m(a,g,k)[0]:6.3f}" for g,k in [("rewrite","eff"),("rewrite","top1"),("rewrite","pnew"),("opt_para","eff"),("held","eff"),("held_cf","eff"),("held","pnew"),("leak","kl"),("leak","top1")]))
for f in pf:
    print(f["subject"], f["old"], "->", f["new"], "A2k6", f["arms"]["A2_k6_pos0_rho4.0"]["rewrite"]["eff"], "A2k18", f["arms"]["A2_k18_pos0_rho4.0"]["rewrite"]["eff"], "A2k18 pnew %.3f" % f["arms"]["A2_k18_pos0_rho4.0"]["rewrite"]["pnew"], "base pnew %.3f" % f["arms"]["A0"]["rewrite"]["pnew"], "oracle", {k: round(v["window"]["pnew"], 3) for k, v in f["oracle"].items()})
