# Recompute the "held-out (CF paraphrases)" column and CF-paraphrase hash hit from results.json, fixing the
# held_cf index bug: run.py uses held_cf=range(no, no+len(f["held"])-1), which assumes an extra template is always
# present. For facts whose rewrite prompt IS one of the 4 templates (facts 1,8,10), there is no extra template,
# so held == the 2 CF paraphrases and held_cf silently keeps only paraphrase 1. Correct value there = held.
import json
EXP = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/experiment"
r = json.load(open(EXP + "/results.json"))
pf = r["per_fact"]
noextra = {"Maurice Bucaille", "Steve Swell", "Antonio Vivaldi"}  # from check_data.log (n_held 2)
for f in pf:
    assert (len(f["hash_hit"]["held_2"]) == 2) == (f["subject"] in noextra)
arms = ["A0", "A1_L2_rho0.5", "A1_L2_rho1.0", "A1_L2_rho4.0", "A1_L12_rho1.0", "A2_k6_pos0_rho4.0", "A2_k12_pos0_rho4.0",
        "A2_k18_pos0_rho4.0", "A3_ICL"]
print(f"{'arm':22s} {'reported heldCF.eff':>19s} {'corrected':>9s} | {'rep heldCF.pnew':>15s} {'corr':>6s} | {'rep heldCF.top1':>15s} {'corr':>6s}")
for a in arms:
    rows = [f for f in pf if a in f["arms"]]
    rep = {m: sum(f["arms"][a]["held_cf"][m] for f in rows) / len(rows) for m in ("eff", "pnew", "top1")}
    cor = {m: sum((f["arms"][a]["held"] if f["subject"] in noextra else f["arms"][a]["held_cf"])[m] for f in rows) / len(rows)
           for m in ("eff", "pnew", "top1")}
    print(f"{a:22s} {rep['eff']:19.3f} {cor['eff']:9.3f} | {rep['pnew']:15.3f} {cor['pnew']:6.3f} | {rep['top1']:15.3f} {cor['top1']:6.3f}")
for n in (2, 3):
    rep = sum(x for f in pf for x in f["hash_hit"][f"held_{n}"][:-1]) / sum(len(f["hash_hit"][f"held_{n}"][:-1]) for f in pf)
    cfl = [f["hash_hit"][f"held_{n}"][:2] for f in pf]
    cor = sum(sum(x) for x in cfl) / sum(len(x) for x in cfl)
    print(f"CF-paraphrase {n}-gram hash hit: reported {rep:.3f} corrected {cor:.3f}")
# per-fact A2 'efficacy' flips: P(new) and base P(new)
print("\nA2 rewrite flips (eff=1) with P(new):")
for f in pf:
    for a in ("A2_k6_pos0_rho4.0", "A2_k12_pos0_rho4.0", "A2_k18_pos0_rho4.0"):
        m = f["arms"][a]["rewrite"]
        if m["eff"] > 0:
            print(f"  {f['subject']:22s} {a:20s} pnew={m['pnew']:.4f} top1={m['top1']} base pnew={f['arms']['A0']['rewrite']['pnew']:.4f}")
print("\nA2 rho grid rows present for facts:", [i + 1 for i, f in enumerate(pf) if "A2_k6_pos0_rho0.5" in f["arms"]])
print("posoff rows present for facts:", [i + 1 for i, f in enumerate(pf) if "A2_k6_posoff_rho1.0" in f["arms"]])
print("A2 delta norms (k6 pos0) per fact:", [[round(x, 2) for x in f["info"]["A2_k6_pos0"]["delta_norms"]] for f in pf])
print("A1_L2 delta norms per fact (rho .5,1,4):", [[round(x, 2) for x in f["info"]["A1_L2"]["delta_norms"]] for f in pf])
print("A1_L2 anchor norm per fact:", [round(f["info"]["A1_L2"]["anchor_norm"], 2) for f in pf])
