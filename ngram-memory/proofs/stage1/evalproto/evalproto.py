"""Stage-1 evaluation-protocol measurements on minisglang-rdna4 (python/minisgl @ HEAD df29117).

1. split statistics (by-file hash split, near-duplicate leakage; by-time split at a4a5543 = last commit before 2026-10-28)
2. token-class shares on held-out code (identifier / project-specific identifier / first-in-file / cross-file ...)
3. base Qwen2.5-0.5B NLL and teacher-forced exact match per class (proxy host; CPU)
4. repo-QA seed generation (AST) + base cloze accuracy
Outputs: evalproto.out (log), evalproto.json, qa_seed.jsonl
"""
import ast, hashlib, io, json, keyword, os, re, subprocess, sys, time, tokenize, builtins, random
from collections import Counter, defaultdict

REPO = "/home/user/patcarter883/minisglang-rdna4"
SUB = "python/minisgl"
OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
SITE = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/venv/lib/python3.13/site-packages"
REF_DIRS = ["/usr/lib/python3.12", f"{SITE}/torch", f"{SITE}/numpy", f"{SITE}/transformers"]
T_CUT = "a4a55431aeeec6c39079dcd943dfc2d6129beb01"  # last commit before 2026-10-28
N_SCORE_TOK = int(os.environ.get("N_SCORE_TOK", "24576"))
CHUNK = 1024
T0 = time.time()
LOG = open(f"{OUT}/evalproto.out", "w")
R = {}


def L(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout


files = sorted(f for f in git("ls-files", SUB).split() if f.endswith(".py"))
src = {f: open(f"{REPO}/{f}", encoding="utf-8", errors="replace").read() for f in files}
rel = {f: f[len("python/"):] for f in files}           # minisgl/...
modname = {f: rel[f][:-3].replace("/", ".").removesuffix(".__init__") for f in files}
L("files", len(files), "bytes", sum(len(s) for s in src.values()))

# ---------------------------------------------------------------- 1. splits
def h(f):
    return int(hashlib.sha256(f.encode()).hexdigest()[:8], 16) / 2**32
held = [f for f in files if h(f) < 0.15 and len(src[f]) > 400]
train = [f for f in files if f not in held]
train_lines = set()
for f in train:
    for ln in src[f].splitlines():
        s = ln.strip()
        if len(s) >= 20: train_lines.add(s)
dup = {}
for f in held:
    ls = [ln.strip() for ln in src[f].splitlines() if len(ln.strip()) >= 20]
    dup[f] = sum(s in train_lines for s in ls) / max(1, len(ls))
dvals = sorted(dup.values())
R["split_file"] = dict(n_train=len(train), n_held=len(held),
                       held_bytes=sum(len(src[f]) for f in held),
                       verbatim_line_dup_frac_median=dvals[len(dvals) // 2],
                       files_dup_gt_0p5=[f for f in held if dup[f] > 0.5],
                       dup_frac_by_file={rel[f]: round(dup[f], 3) for f in held})
L("by-file split", json.dumps({k: v for k, v in R["split_file"].items() if k != "dup_frac_by_file"}))

# by-time: lines added after T_CUT in HEAD files
diff = git("diff", "-U0", T_CUT, "HEAD", "--", SUB)
added = defaultdict(set); cur = None
for ln in diff.splitlines():
    if ln.startswith("+++ "):
        cur = ln[6:] if ln.startswith("+++ b/") else None
    elif ln.startswith("@@") and cur:
        m = re.search(r"\+(\d+)(?:,(\d+))?", ln); a = int(m.group(1)); n = int(m.group(2) or 1)
        added[cur].update(range(a, a + n))
new_files = [f for f in git("diff", "--diff-filter=A", "--name-only", T_CUT, "HEAD", "--", SUB).split() if f.endswith(".py")]
R["split_time"] = dict(cut=T_CUT, commits_after=int(git("rev-list", "--count", f"{T_CUT}..HEAD")),
                       files_with_added_lines=len([f for f in added if f.endswith(".py")]),
                       added_lines=sum(len(v) for f, v in added.items() if f.endswith(".py")),
                       new_files=new_files)
L("by-time split", json.dumps(R["split_time"]))

# ---------------------------------------------------------------- 2. identifier vocabularies
t = time.time()
ref_vocab = Counter()
IDRE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
nref = 0
for d in REF_DIRS:
    for root, _, fs in os.walk(d):
        for fn in fs:
            if fn.endswith(".py"):
                try: s = open(os.path.join(root, fn), encoding="utf-8", errors="ignore").read()
                except OSError: continue
                nref += 1
                ref_vocab.update(set(IDRE.findall(s)))
L(f"reference identifier vocab from {nref} files: {len(ref_vocab)} names ({time.time() - t:.0f}s)")
BUILTIN = set(dir(builtins))
STRTYPES = {tokenize.STRING} | {getattr(tokenize, k) for k in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END", "TSTRING_START", "TSTRING_MIDDLE", "TSTRING_END") if hasattr(tokenize, k)}

# project definitions
defs = defaultdict(set)          # name -> files defining it (top-level def/class, methods, module-level assigns)
sigs = {}                        # (file, qualname) -> signature string
toplevel = defaultdict(set)
for f in files:
    try: tree = ast.parse(src[f])
    except SyntaxError: continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            toplevel[node.name].add(f)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for tg in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(tg, ast.Name): toplevel[tg.id].add(f)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name].add(f)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in tree.body:
            try: sigs[(f, node.name)] = f"def {node.name}({ast.unparse(node.args)})"
            except Exception: pass
proj_only = {n for n in defs if ref_vocab[n] == 0}
R["defs"] = dict(defined_names=len(defs), defined_not_in_reference=len(proj_only), toplevel_names=len(toplevel))
L("project defs", json.dumps(R["defs"]))

# ---------------------------------------------------------------- 3. token classes on held-out code
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)


def char_classes(s, f):
    """per-char class labels from python tokenize."""
    lab = ["ws"] * len(s)
    lines = s.splitlines(keepends=True); off = [0]
    for ln in lines: off.append(off[-1] + len(ln))
    seen = set(); local = set()
    try:
        tree = ast.parse(s)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): local.add(node.name)
    except SyntaxError: pass
    spans = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(s).readline):
            (r0, c0), (r1, c1) = tk.start, tk.end
            a, b = off[r0 - 1] + c0, off[r1 - 1] + c1
            if tk.type == tokenize.NAME:
                n = tk.string
                if keyword.iskeyword(n) or keyword.issoftkeyword(n): c = "keyword"
                elif n in BUILTIN: c = "id_builtin"
                elif ref_vocab[n] == 0: c = "id_proj"          # never seen in stdlib/torch/numpy/transformers
                else: c = "id_general"
                first = n not in seen; seen.add(n)
                xfile = (n in defs) and (f not in defs[n])     # defined elsewhere in the project, not in this file
                spans.append((a, b, n, c, first, xfile))
            elif tk.type in STRTYPES:
                c = "string"
            elif tk.type == tokenize.COMMENT: c = "comment"
            elif tk.type == tokenize.NUMBER: c = "number"
            elif tk.type == tokenize.OP: c = "op"
            else: continue
            for i in range(a, min(b, len(s))): lab[i] = c
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return lab, spans


docs = []  # per held-out file: ids, per-token class, identifier spans in token idx, line spans
for f in held:
    s = f"# file: {rel[f]}\n" + src[f]   # path header (recommended ingestion format)
    lab, spans = char_classes(s, f)
    enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    tcls = []
    for (a, b) in offs:
        cs = Counter(lab[a:b]) if b > a else Counter(["ws"])
        # identifier wins if token overlaps an identifier
        pick = next((c for c in ("id_proj", "id_general", "id_builtin", "keyword") if cs.get(c)), cs.most_common(1)[0][0])
        tcls.append(pick)
    # identifier spans -> token index ranges
    starts = [a for a, _ in offs]; import bisect
    idsp = []
    for (a, b, n, c, first, xf) in spans:
        i0 = bisect.bisect_right(starts, a) - 1; i1 = bisect.bisect_left(starts, b)
        idsp.append((max(i0, 0), max(i1, i0 + 1), n, c, first, xf))
    # lines
    lsp = []; pos = 0
    for ln in s.splitlines(keepends=True):
        body = ln.rstrip("\n")
        if len(body.strip()) >= 8 and not body.strip().startswith("#"):
            a, b = pos + (len(body) - len(body.lstrip())), pos + len(body)
            i0 = bisect.bisect_right(starts, a) - 1; i1 = bisect.bisect_left(starts, b)
            lsp.append((max(i0, 0), max(i1, i0 + 1)))
        pos += len(ln)
    docs.append(dict(f=f, ids=ids, cls=tcls, idsp=idsp, lines=lsp))

ntok = sum(len(d["ids"]) for d in docs)
cls_share = Counter(c for d in docs for c in d["cls"])
R["heldout_token_classes"] = {k: round(v / ntok, 4) for k, v in cls_share.most_common()}
occ = Counter();
for d in docs:
    for (_, _, n, c, first, xf) in d["idsp"]:
        occ[c] += 1; occ[c + ("_first" if first else "_repeat")] += 1
        if xf: occ["xfile"] += 1; occ["xfile_first" if first else "xfile_repeat"] += 1
R["heldout_identifier_occurrences"] = dict(occ)
R["heldout_tokens"] = ntok
L("held-out tokens", ntok, "classes", json.dumps(R["heldout_token_classes"]))
L("identifier occurrences", json.dumps(dict(occ)))

# ---------------------------------------------------------------- 4. base NLL + teacher-forced EM (proxy host)
import torch
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

agg = defaultdict(lambda: [0.0, 0]); em = defaultdict(lambda: [0, 0])
scored = 0; t = time.time()
rng = random.Random(0)
order = list(range(len(docs))); rng.shuffle(order)
for di in order:
    if scored >= N_SCORE_TOK: break
    d = docs[di]; ids = d["ids"]
    n = min(len(ids), CHUNK * 2)            # first <=2 chunks of each file, in-file prefix context only
    nll = torch.empty(n); hit = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for c0 in range(0, n, CHUNK):
            ctx0 = max(0, c0 - CHUNK)       # 1 chunk of left context
            x = torch.tensor([ids[ctx0:min(n, c0 + CHUNK)]])
            lg = model(x).logits[0].float()
            lp = torch.log_softmax(lg[:-1], -1); y = x[0, 1:]
            tl = -lp.gather(1, y[:, None])[:, 0]; am = lg[:-1].argmax(-1) == y
            k = c0 - ctx0                    # target index i (predicting ids[i]) lives at row i-1-ctx0
            for i in range(max(c0, 1), min(n, c0 + CHUNK)):
                nll[i] = tl[i - 1 - ctx0]; hit[i] = am[i - 1 - ctx0]
    for i in range(1, n):
        c = d["cls"][i]; agg[c][0] += float(nll[i]); agg[c][1] += 1
        agg["all"][0] += float(nll[i]); agg["all"][1] += 1
        if c.startswith("id_"): agg["id_any"][0] += float(nll[i]); agg["id_any"][1] += 1
        else: agg["non_id"][0] += float(nll[i]); agg["non_id"][1] += 1
    for (i0, i1, nm, c, first, xf) in d["idsp"]:
        if i1 > n or i0 < 1: continue
        ok = bool(hit[i0:i1].all()); s_nll = float(nll[i0:i1].sum())
        for key in (c, f"{c}_{'first' if first else 'repeat'}") + (("xfile_" + ("first" if first else "repeat")),) * bool(xf):
            em[key][0] += ok; em[key][1] += 1
            agg["span_" + key][0] += s_nll; agg["span_" + key][1] += 1
    for (i0, i1) in d["lines"]:
        if i1 > n or i0 < 1: continue
        em["line"][0] += bool(hit[i0:i1].all()); em["line"][1] += 1
    scored += n
    L(f"scored {scored} tok ({time.time() - t:.0f}s) file {rel[d['f']]}")
R["base_nll_per_token"] = {k: dict(nll=round(v[0] / v[1], 4), n=v[1]) for k, v in sorted(agg.items()) if not k.startswith("span_")}
R["base_nll_per_identifier_span"] = {k[5:]: dict(nll=round(v[0] / v[1], 3), n=v[1]) for k, v in sorted(agg.items()) if k.startswith("span_")}
R["base_tf_exact_match"] = {k: dict(em=round(v[0] / v[1], 4), n=v[1]) for k, v in sorted(em.items())}
R["score_sec_per_1k_tok"] = round((time.time() - t) / scored * 1000, 2)
L("base NLL/token", json.dumps(R["base_nll_per_token"]))
L("base NLL/identifier span", json.dumps(R["base_nll_per_identifier_span"]))
L("base TF exact match", json.dumps(R["base_tf_exact_match"]))

# ---------------------------------------------------------------- 5. repo QA seed
corpus_all = "\n".join(src.values())
items = []
for n, fs in toplevel.items():
    if len(fs) != 1 or len(n) < 6 or n.startswith("_") or ref_vocab[n] > 0: continue
    f = next(iter(fs)); m = "minisgl." + modname[f] if not modname[f].startswith("minisgl") else modname[f]
    m = modname[f]
    imp = f"from {m} import"
    attested = bool(re.search(rf"from {re.escape(m)} import[^\n]*\b{re.escape(n)}\b", corpus_all)) or \
               bool(re.search(rf"from {re.escape(m)} import \([^)]*\b{re.escape(n)}\b", corpus_all))
    items.append(dict(kind="defines_module", name=n, answer=m, file=rel[f], heldout_file=f in held,
                      attested_import=attested,
                      nl=f"Q: In the minisgl codebase, which module defines `{n}`?\nA: `",
                      cloze_prefix="from minisgl.", cloze_suffix=f" import {n}"))
    if (f, n) in sigs and sigs[(f, n)].count(",") >= 1:
        items.append(dict(kind="signature", name=n, answer=sigs[(f, n)], file=rel[f], heldout_file=f in held,
                          nl=f"Q: What is the signature of `{n}` in {m}?\nA: `",
                          cloze_prefix=f"# {rel[f]}\n", cloze_target=sigs[(f, n)]))
envs = Counter(re.findall(r"""["'](MINISGL_[A-Z0-9_]+)["']""", corpus_all))
env_files = defaultdict(set)
for f in files:
    for e in set(re.findall(r"""["'](MINISGL_[A-Z0-9_]+)["']""", src[f])): env_files[e].add(rel[f])
for e, fs in env_files.items():
    if len(fs) == 1:
        items.append(dict(kind="env_var_reader", name=e, answer=next(iter(fs)), heldout_file=next(iter(fs)) in {rel[x] for x in held},
                          nl=f"Q: Which file in minisgl reads the environment variable {e}?\nA: `"))
kc = Counter(i["kind"] for i in items)
R["qa_seed"] = dict(counts=dict(kc), env_vars_total=len(envs),
                    defines_attested_import=sum(1 for i in items if i["kind"] == "defines_module" and i["attested_import"]),
                    defines_heldout_file=sum(1 for i in items if i["kind"] == "defines_module" and i["heldout_file"]))
L("qa seed", json.dumps(R["qa_seed"]))
with open(f"{OUT}/qa_seed.jsonl", "w") as fo:
    for it in items: fo.write(json.dumps(it) + "\n")

# base cloze accuracy for "from minisgl.<module> import X"  ->  score the module path given "...import X" can't be causal;
# use the causal form:  "import X  # defined in minisgl."  is unnatural, so use NL form with teacher-forced answer EM.
qa = [i for i in items if i["kind"] in ("defines_module", "env_var_reader")]
rng.shuffle(qa); qa = qa[:160]
mods = sorted(set(modname.values())); relfiles = sorted(rel.values())


def seq_lp(prompt, ans):
    p = tok(prompt, add_special_tokens=False)["input_ids"]; a = tok(ans, add_special_tokens=False)["input_ids"]
    x = torch.tensor([p + a]); lg = model(x).logits[0, len(p) - 1:-1]
    lp = torch.log_softmax(lg, -1).gather(1, torch.tensor(a)[:, None]).sum()
    return float(lp), bool((lg.argmax(-1) == torch.tensor(a)).all())


def distractors(ans, pool, k=3):
    pkg = ans.rsplit(".", 1)[0] if "." in ans else ans.rsplit("/", 1)[0]
    same = [m for m in pool if m != ans and m.startswith(pkg)]
    other = [m for m in pool if m != ans and m not in same]
    r = random.Random(ans); r.shuffle(same); r.shuffle(other)
    return (same + other)[:k]

res = defaultdict(lambda: defaultdict(float))
with torch.no_grad():
    for it in qa:
        pool = mods if it["kind"] == "defines_module" else relfiles
        cands = [it["answer"]] + distractors(it["answer"], pool)
        nl = [seq_lp(it["nl"], c + "`")[0] for c in cands]
        lp0, ok = seq_lp(it["nl"], it["answer"] + "`")
        keys = [it["kind"]] + ([it["kind"] + "_attested"] if it.get("attested_import") else [])
        if it["kind"] == "defines_module":
            code = [seq_lp(f"# file: {it['file'][:0]}", f"from {c} import {it['name']}\n")[0] for c in cands]
        for k in keys:
            r = res[k]; r["n"] += 1; r["nl_tf_em"] += ok; r["nl_mc4"] += int(max(range(4), key=lambda j: nl[j]) == 0)
            r["answer_nll"] += -lp0
            if it["kind"] == "defines_module": r["code_mc4"] += int(max(range(4), key=lambda j: code[j]) == 0)
R["qa_base"] = {k: {m: round(v / r["n"], 3) if m != "n" else int(v) for m, v in r.items()} for k, r in res.items()}
L("qa base (chance mc4=0.25)", json.dumps(R["qa_base"]))
json.dump(R, open(f"{OUT}/evalproto.json", "w"), indent=1)
L("done")
