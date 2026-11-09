# Robustness of the A2 negative result: Gemini's subgraph loss with NO L2 reg (0.005*|d|^2 removed), no clamp, 150 steps,
# k=6 and k=18, on 3 facts (Faty, Clickteam, Shinoda: those where the k18 oracle carries the most fact). Reuses rerun5.py helpers.
import sys, json, torch, torch.nn.functional as F
V = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/verify-experiment"
src = open(V + "/rerun5.py").read()
src = src[: src.index("inj2 = Inject(LAYERS[2])")]
src = src.replace("(lossw + 0.005 * d.pow(2).sum()).backward()", "(lossw + REG * d.pow(2).sum()).backward()")
g = {"__name__": "x", "REG": 0.0}
exec(compile(src, "rerun5_head", "exec"), g)
facts, opt_A2, evalp, Inject, LAYERS = g["facts"], g["opt_A2"], g["evalp"], g["Inject"], g["LAYERS"]
inj2 = Inject(LAYERS[2]); out = []
for fi in (4, 8, 3):
    f = facts[fi]
    for k in (6, 18):
        d, hist = opt_A2(f, k, inj2, 1e9, 150)
        rows = evalp(f, inj2, d)
        rw = rows[0]; held = [r for r in rows if r["grp"] == "held"]
        rec = dict(fact=fi + 1, subject=f["subject"], k=k, loss=hist, dnorm=float(d.norm()), rw_pnew=rw["pnew"], rw_pold=rw["pold"],
                   rw_top1=rw["top1"], rw_top1tok=rw["top1tok"], held_pnew=sum(r["pnew"] for r in held) / len(held),
                   held_top1=sum(r["top1"] for r in held) / len(held))
        print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open(V + "/a2_noreg.json", "w"), indent=1)
