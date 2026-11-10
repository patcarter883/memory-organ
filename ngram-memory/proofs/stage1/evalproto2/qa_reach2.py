"""Code-ORDER QA kinds (cue precedes answer, as in source): import completion and constant value. Same reach stats as
qa_reach.py (imports its tables by exec of the shared prefix)."""
import ast, json, re
exec(open(__file__.replace("qa_reach2.py", "qa_reach.py")).read().split("items = [json.loads")[0])
items = []
for f in files:
    m = f[len("python/"):-3].replace("/", ".").removesuffix(".__init__")
    try: tree = ast.parse(src[f])
    except SyntaxError: continue
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
    if names: items.append(dict(kind="import_completion", held=f in held, p=f"from {m} import", a=" " + names[0] + "\n"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) \
           and isinstance(n.value.value, (int, float, str)) and len(n.targets[0].id) >= 6 and len(str(n.value.value)) <= 40:
            items.append(dict(kind="constant_value", held=f in held, p=f"# {f[len('python/'):]}\n{n.targets[0].id} =", a=" " + repr(n.value.value) + "\n"))
R = {}
for kind in ("import_completion", "constant_value"):
    for split in (False, True):
        sel = [it for it in items if it["kind"] == kind and it["held"] == split]
        agg = Counter(); first = Counter(); allok = 0; n = 0
        for it in sel:
            r = reach(it["p"], it["a"])
            if not r: continue
            n += 1; allok += all(x["tri_argmax"] for x in r)
            for x in r:
                agg["tok"] += 1
                for k, v in x.items(): agg[k] += v if k != "k3_branching" else 0
            for k, v in r[0].items(): first[k] += v if k != "k3_branching" else 0
        key = f"{kind}/{'heldout_file' if split else 'train_file'}"
        R[key] = dict(items=n, first_tok={k: round(first[k] / n, 3) for k in ("k3_in_train", "tri_seen", "tri_argmax", "k3_proj_specific")},
                      all_tok={k: round(agg[k] / agg["tok"], 3) for k in ("tri_seen", "tri_argmax", "k3_proj_specific")},
                      whole_answer_by_3gram_argmax_chain=round(allok / n, 3))
        print(key, json.dumps(R[key]))
json.dump(R, open(f"{OUT}/qa_reach2.json", "w"), indent=1)
