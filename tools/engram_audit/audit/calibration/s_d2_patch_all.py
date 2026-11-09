# (d2) Global optimum of C8 Stage 1 = exact teacher L_b states at the student's positions. Patch ALL student positions (and the window) at block b.
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
exec(open(__file__.rsplit('/',1)[0] + "/s_d_depth.py").read().split("R = {k:")[0])   # reuse FACTS / run() / helpers
LAY = [2, 6, 10, 14, 17, 19, 20, 21, 23]
res = {b: [0.0, 0.0, 0.0] for b in LAY}; base = 0.0; teach = 0.0; ranks = {b: [] for b in LAY}
for prompt, subj, new in FACTS:
    s = ids(prompt); tid = ids(" " + new)[0]; pre = ids(f"{prompt} {new}.\n"); P = len(pre)
    anchor = len(ids(subj)) - 1
    T, lgT = run(pre + s); _, lgS = run(s)
    teach += F.softmax(lgT[-1], -1)[tid].item(); base += F.softmax(lgS[-1], -1)[tid].item()
    for b in LAY:
        allp = list(range(len(s)))
        _, lg = run(s, (b, allp, T[b+1][[P+w for w in allp]]))
        pr = F.softmax(lg[-1], -1); res[b][0] += pr[tid].item(); ranks[b].append(int((pr > pr[tid]).sum()))
        win = list(range(anchor, len(s)))
        _, lg = run(s, (b, win, T[b+1][[P+w for w in win]])); res[b][1] += F.softmax(lg[-1], -1)[tid].item()
n = len(FACTS)
print(f"teacher P(new)={teach/n:.3f}  student P(new)={base/n:.4f}  (24 layers; 'after Lb' state; depth=(b+1)/24)")
for b in LAY:
    print(f"patch after L{b:02d} (depth {(b+1)/24:.2f}): ALL student positions -> P(new)={res[b][0]/n:.3f} (median rank of new obj {sorted(ranks[b])[n//2]}) | C8 window [anchor..end] -> P(new)={res[b][1]/n:.3f}")
