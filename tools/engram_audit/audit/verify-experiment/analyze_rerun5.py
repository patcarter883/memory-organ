import json
V = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/verify-experiment"
R = json.load(open(V + "/rerun5.json"))
E = json.load(open("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/experiment/results.json"))["per_fact"]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
arms = ["A0", "A1", "A2_k6", "A2_k18", "A2_k6_long", "A3"]
runmap = {"A0": "A0", "A1": "A1_L2_rho1.0", "A2_k6": "A2_k6_pos0_rho4.0", "A2_k18": "A2_k18_pos0_rho4.0", "A3": "A3_ICL"}
print("Cross-check vs run.py results.json (rewrite P(new), held eff):")
for r in R:
    e = E[r["fact"] - 1]
    for a in runmap:
        rw = r[a][0]; held = [x for x in r[a] if x["grp"] == "held"]
        print(f"  f{r['fact']:2d} {a:6s} rw.pnew mine {rw['pnew']:.4f} run.py {e['arms'][runmap[a]]['rewrite']['pnew']:.4f} | "
              f"held.eff mine {mean([float(x['pnew']>x['pold']) for x in held]):.3f} run.py {e['arms'][runmap[a]]['held']['eff']:.3f}")
print("\nSummary over 5 facts (mine):")
print(f"{'arm':11s} rw.pnew rw.top1 | heldNOVEL eff/top1/pnew (n) | heldOVERLAP eff/top1/pnew (n) | leak top1 kl")
for a in arms:
    rw = [r[a][0] for r in R]
    nov, ovl = [], []
    for r in R:
        held = [x for x in r[a] if x["grp"] == "held"]
        for x, o in zip(held, r["held_overlap"]):
            (ovl if o else nov).append(x)
    lk = [x for r in R for x in r[a] if x["grp"] == "leak"]
    s = lambda xs: f"{mean([float(x['pnew']>x['pold']) for x in xs]):.2f}/{mean([x['top1'] for x in xs]):.2f}/{mean([x['pnew'] for x in xs]):.3f} ({len(xs)})"
    print(f"{a:11s} {mean([x['pnew'] for x in rw]):.3f}   {mean([x['top1'] for x in rw]):.2f}   | {s(nov):28s} | {s(ovl):30s} | {mean([x['top1'] for x in lk]):.2f} {mean([x['kl'] for x in lk]):.2f}")
print("\nheld-out prompts that are verbatim optimisation prompts behind a junk prefix:")
for r in R:
    held = [x for x in r["A0"] if x["grp"] == "held"]
    for x, o in zip(held, r["held_overlap"]):
        if o: print("  f%d" % r["fact"], repr(x["prompt"]))
print("\nper-prompt leakage under A1 (top-1 token; * = new object):")
for r in R:
    for x in r["A1"]:
        if x["grp"] == "leak":
            print(f"  f{r['fact']:2d} {x['prompt']!r:55s} top1={x['top1tok']!r}{'*' if x['top1'] else ''} pnew={x['pnew']:.3f} kl={x['kl']:.2f}")
print("\nA2 losses / norms / oracle:")
for r in R:
    print(f"  f{r['fact']} k6 {r['A2_k6_hist']} |d|={r['A2_k6_norm']:.2f}; k18 {r['A2_k18_hist']} |d|={r['A2_k18_norm']:.2f}; "
          f"k6 long {r['A2_k6_long_hist']} |d|={r['A2_k6_long_norm']:.2f} cos(A1)={r['cos_A2long_vs_A1']:.3f}; "
          f"oracle k6 (pnew,eff,top1)={[round(v,3) for v in r['oracle_k6']]} k18={[round(v,3) for v in r['oracle_k18']]}; "
          f"run.py oracle k6 pnew={E[r['fact']-1]['oracle']['k6']['window']['pnew']:.3f} k18={E[r['fact']-1]['oracle']['k18']['window']['pnew']:.3f}")
print("\nteacher logit lens (index 0 = embeddings, 24 = final logits), mean over 5 facts:")
L = [mean([r["teacher_lens"][i] for r in R]) for i in range(25)]
print("  " + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(L)))
print("  HF hidden_states[-1] already final-normed? max|HEAD(hs[-1]) - logits| =", [round(r["hs_last_is_normed"], 6) for r in R])
