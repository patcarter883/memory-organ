import json, sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                   "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/experiment/results.json"))
arms = [a for a in "ABC" if f"arm_{a}" in R and not R[f"arm_{a}"].get("skipped")]
rows = [("heldout", "mean"), ("heldout", "unseen_2gram"), ("heldout", "seen_2gram_any"), ("heldout", "seen_2gram_only"),
        ("heldout", "seen_3gram_new_next"), ("heldout", "seen_3gram_seen_next"), ("train", "mean"),
        ("stdlib_py", "mean"), ("stdlib_py", "unseen_2gram"), ("stdlib_py", "seen_2gram_any"),
        ("general_text", "mean"), ("general_text", "unseen_2gram"), ("general_text", "seen_2gram_any")]
fr = R["category_fractions"]
print("| set / token class | frac | base | " + " | ".join(f"{a} | d{a}" for a in arms) + " |")
for s, k in rows:
    b = R["base"]["summary"][s][k]
    f = fr[s].get(k) if k != "seen_2gram_any" else 1 - fr[s]["unseen_2gram"]
    f = "" if k == "mean" else f"{f:.3f}"
    cells = []
    for a in arms:
        v = R[f"arm_{a}"]["eval"][s][k]
        cells += [f"{v:.4f}", f"{v - b:+.4f}"]
    print(f"| {s} / {k} | {f} | {b:.4f} | " + " | ".join(cells) + " |")
for a in arms:
    t = R[f"arm_{a}"]["train"]
    e = R[f"arm_{a}"]["eval"]
    print(a, json.dumps({k: v for k, v in t.items() if k != "train_curve_nll_per16"}))
    print(a, "curve", [round(x, 3) for x in t["train_curve_nll_per16"]])
    for s in e:
        print(a, s, {k: v for k, v in e[s].items() if k.startswith(("inj", "rows_hit", "collision"))})
print({k: R[k] for k in ("noop_zero_table", "host_decode_parity_no_graft", "host_layer1_input") if k in R})
print({k: v for k, v in R["data"].items() if not k.endswith("_files")})
print("total min", R.get("total_wall_min"))
