"""Which repo-QA / completion items can a 2/3-gram table *directly* reach?
For each item's answer tokens (tokenised in the prompt context), check whether the 2-gram key (x_{t-1}) and 3-gram key
(x_{t-2},x_{t-1}) occur in the TRAIN files (by-file split identical to evalproto.py: sha256(path)<0.15 & len>400 held out;
files ingested with a '# file: <path>' header) and whether the key's argmax continuation in train == x_t.
'key project-specific' = the 3-gram key never occurs in a general-Python reference (stdlib+torch+transformers sample).
This is an n-gram-statistics *upper bound on direct lookup*, not a model result: the rows add a vector at layer 1;
whether the frozen model turns it into the answer is what the eval measures."""
import hashlib, json, os, random, re, subprocess, sys
from collections import Counter, defaultdict
from transformers import AutoTokenizer
REPO = "/home/user/patcarter883/minisglang-rdna4"; SUB = "python/minisgl"
OUT = os.path.dirname(os.path.abspath(__file__))
QA = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/evalproto/qa_seed.jsonl"
MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
SITE = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/venv/lib/python3.13/site-packages"
tok = AutoTokenizer.from_pretrained(MODEL)
files = sorted(f for f in subprocess.run(["git", "-C", REPO, "ls-files", SUB], capture_output=True, text=True).stdout.split() if f.endswith(".py"))
src = {f: open(f"{REPO}/{f}", encoding="utf-8", errors="replace").read() for f in files}
h = lambda f: int(hashlib.sha256(f.encode()).hexdigest()[:8], 16) / 2**32
held = {f for f in files if h(f) < 0.15 and len(src[f]) > 400}
tri = defaultdict(Counter); bi = defaultdict(Counter)
for f in files:
    if f in held: continue
    ids = tok(f"# file: {f[len('python/'):]}\n" + src[f], add_special_tokens=False)["input_ids"]
    for i in range(len(ids)):
        if i >= 1: bi[ids[i-1]][ids[i]] += 1
        if i >= 2: tri[(ids[i-2], ids[i-1])][ids[i]] += 1
# general reference 3-gram keys (sample of ~2M tokens)
ref = set(); rng = random.Random(0); pool = []
for d in ["/usr/lib/python3.12", f"{SITE}/torch", f"{SITE}/transformers"]:
    for r, _, fs in os.walk(d):
        pool += [os.path.join(r, x) for x in fs if x.endswith(".py")]
rng.shuffle(pool); n = 0
for p in pool:
    try: ids = tok(open(p, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"]
    except Exception: continue
    ref.update(zip(ids, ids[1:])); n += len(ids)
    if n > 2_000_000: break
print("train 3-gram keys", len(tri), "ref tokens", n, file=sys.stderr)

def reach(prompt, ans):
    full = prompt + ans; enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    a_idx = [i for i, (a, b) in enumerate(offs) if b > len(prompt)]
    out = []
    for i in a_idx:
        if i < 2: continue
        k3 = (ids[i-2], ids[i-1]); c3 = tri.get(k3); c2 = bi.get(ids[i-1])
        out.append(dict(k3_in_train=c3 is not None, tri_seen=bool(c3 and c3[ids[i]] > 0),
                        tri_argmax=bool(c3 and c3.most_common(1)[0][0] == ids[i]),
                        bi_argmax=bool(c2 and c2.most_common(1)[0][0] == ids[i]),
                        k3_proj_specific=k3 not in ref, k3_branching=len(c3) if c3 else 0))
    return out

items = [json.loads(l) for l in open(QA)]
FORMS = {
  "defines_module": [("nl", lambda it: (it["nl"], it["answer"] + "`")),
                     ("code_dotted", lambda it: (f"# {it['name']} lives in ", it["answer"] + "\n"))],
  "signature": [("code_def", lambda it: (it["cloze_prefix"] + it["answer"].split("(")[0] + "(", it["answer"].split("(", 1)[1] + ")"))],
  "env_var_reader": [("nl", lambda it: (it["nl"], it["answer"] + "`"))],
}
R = {}
for kind, forms in FORMS.items():
    for split in ("train_file", "heldout_file"):
        sel = [it for it in items if it["kind"] == kind and it["heldout_file"] == (split == "heldout_file")]
        for fname, fn in forms:
            agg = Counter(); first = Counter(); allok = 0; n = 0
            for it in sel:
                r = reach(*fn(it))
                if not r: continue
                n += 1
                for x in r:
                    for k, v in x.items():
                        if k != "k3_branching": agg[k] += v
                    agg["tok"] += 1
                for k, v in r[0].items():
                    if k != "k3_branching": first[k] += v
                allok += all(x["tri_argmax"] for x in r)
            if not n: continue
            key = f"{kind}/{fname}/{split}"
            R[key] = dict(items=n, answer_tokens=agg["tok"],
                          all_tok={k: round(agg[k] / agg["tok"], 3) for k in ("k3_in_train", "tri_seen", "tri_argmax", "bi_argmax", "k3_proj_specific")},
                          first_tok={k: round(first[k] / n, 3) for k in ("k3_in_train", "tri_seen", "tri_argmax", "bi_argmax", "k3_proj_specific")},
                          whole_answer_by_3gram_argmax_chain=round(allok / n, 3))
            print(key, json.dumps(R[key]))
json.dump(R, open(f"{OUT}/qa_reach.json", "w"), indent=1)
