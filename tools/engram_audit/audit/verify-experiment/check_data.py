# Re-executes run.py's own data/gate code (lines 1-206, exact source) with --n 12, then audits:
# anchor-token alignment, held-out composition, held_cf indexing, teacher concatenation, prompt endings, leak/relation overlap.
import sys, json
EXP = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/experiment"
src = open(EXP + "/run.py").read().splitlines()
cut = next(i for i, l in enumerate(src) if l.startswith("def add_deltas"))
sys.argv = ["run.py", "--n", "12", "--skip_decode"]
code = "\n".join(src[:cut]).replace("torch.set_num_threads(3)", "torch.set_num_threads(2)")
g = {"__name__": "x"}
exec(compile(code, "run_head", "exec"), g)
facts, tok, prefix_ids = g["facts"], g["tok"], g["prefix_ids"]
res = json.load(open(EXP + "/results.json"))
assert [f["case_id"] for f in facts] == [r["case_id"] for r in res["per_fact"]], "fact set mismatch"
print("fact set identical to results.json per_fact:", True)
d = lambda i: repr(tok.decode([i]))
issues = []
for fi, f in enumerate(facts):
    print(f"\n=== fact {fi+1} [{f['rel']}] subj={f['subject']!r} old={f['old']!r}({d(f['tt'])}) new={f['new']!r}({d(f['tn'])})")
    print("   rewrite in templates?", f["opt"][0] in [t.format(f["subject"]) for t in g["TEMPL"][f["rel"]]],
          "| n_opt", len(f["opt"]), "| n_held", len(f["held"]), "(2 CF paraphrases + extra:", len(f["held"]) - 2, ")")
    if len(f["held"]) == 2:
        issues.append(f"fact {fi+1}: no extra template -> held_cf=range(no,no+1) drops paraphrase 2; hash held_cf [:-1] also drops it")
    for grp in ("opt", "held", "leak"):
        for p, (ids, a) in zip(f[grp], f[grp + "_enc"]):
            subj_toks = tok(" " + f["subject"])["input_ids"]
            last_ok = ids[a] in (subj_toks[-1], tok(f["subject"])["input_ids"][-1])
            nxt = d(ids[a + 1]) if a + 1 < len(ids) else "<END>"
            flag = "" if last_ok else "  <-- anchor token != subject's last BPE token"
            if a == 0:
                flag += "  <-- ANCHOR AT POSITION 0 (attention-sink position)"
            if p.endswith(" "):
                flag += "  <-- prompt ends with space"
            if flag:
                issues.append(f"fact {fi+1} {grp}: {p!r}:{flag}")
            print(f"   {grp:4s} a={a:2d}/{len(ids):2d} tok={d(ids[a])} next={nxt} last={d(ids[-1])} | {p!r}{flag}")
    pre = prefix_ids(f)
    print("   teacher prefix:", repr(tok.decode(pre)), "len", len(pre), "last2", [d(x) for x in pre[-2:]])
    # natural tokenisation of the concatenated string vs token concatenation
    for p, (ids, a) in zip(f["opt"], f["opt_enc"]):
        nat = tok(f["fact"] + ".\n" + p)["input_ids"]
        if nat != pre + ids:
            issues.append(f"fact {fi+1}: teacher token concat != natural tokenisation for {p!r}")
print("\nISSUES:")
for i in issues:
    print(" -", i)
# held_cf / per-fact leak top1
print("\nper-fact leak.top1 / leak.kl for A1_L2_rho1.0 and held_cf eff:")
for r in res["per_fact"]:
    a = r["arms"]["A1_L2_rho1.0"]
    print(f"  {r['subject']:32s} {r['rel']:6s} leak.top1={a['leak']['top1']:.2f} leak.kl={a['leak']['kl']:.2f} held_cf.eff={a['held_cf']['eff']:.2f} held.eff={a['held']['eff']:.2f}")
