"""Per-token-class breakdown of the VERIFIED rows-only graft run (stage1/verify-experiment, arm A, Qwen2.5-0.5B):
re-derives the 32 held-out 512-token windows exactly as verify.py does (same file split/seed/chunk sampling; tokenizer only,
no model), labels every target token with a Python-lexical class, and joins with base_verify.npz (OFF) and
nll_full_heldout.npy (ON). Classes: identifier FIRST sub-token vs CONTINUATION sub-token, project-specific
(name never in stdlib/torch/numpy/transformers) vs general, first occurrence in file vs repeat."""
import glob, io, json, keyword, os, random, re, tokenize, builtins, bisect
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
VE = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment"
SITE = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/venv/lib/python3.13/site-packages"
OUT = os.path.dirname(os.path.abspath(__file__))
EOS, T = 151643, 512
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]; random.Random(0).shuffle(shuf)
held = sorted(shuf[:round(0.15 * len(files))])
ref = Counter(); IDRE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
for d in ["/usr/lib/python3.12", f"{SITE}/torch", f"{SITE}/numpy", f"{SITE}/transformers"]:
    for r, _, fs in os.walk(d):
        for fn in fs:
            if fn.endswith(".py"):
                try: ref.update(set(IDRE.findall(open(os.path.join(r, fn), encoding="utf-8", errors="ignore").read())))
                except OSError: pass
BI = set(dir(builtins))
STR = {tokenize.STRING} | {getattr(tokenize, k) for k in dir(tokenize) if k.startswith(("FSTRING_", "TSTRING_"))}

def labels(s):
    lab = ["other"] * len(s); idstart = {}; lines = s.splitlines(keepends=True); off = [0]
    for ln in lines: off.append(off[-1] + len(ln))
    seen = set()
    try:
        for tk in tokenize.generate_tokens(io.StringIO(s).readline):
            (r0, c0), (r1, c1) = tk.start, tk.end; a, b = off[r0-1] + c0, off[r1-1] + c1
            if tk.type == tokenize.NAME:
                n = tk.string
                if keyword.iskeyword(n) or keyword.issoftkeyword(n): c = "keyword"
                else:
                    c = ("proj" if ref[n] == 0 else ("builtin" if n in BI else "gen")) + ("_first" if n not in seen else "_rep")
                    seen.add(n)
                for i in range(a, b): lab[i] = "ID:" + c
                idstart[a] = True
            elif tk.type in STR: c = "string"
            elif tk.type == tokenize.COMMENT: c = "comment"
            else: continue
            if not c.startswith(("proj", "gen", "builtin")) and c != "keyword":
                for i in range(a, b): lab[i] = c
            elif c == "keyword":
                for i in range(a, b): lab[i] = "keyword"
    except (tokenize.TokenError, SyntaxError, IndentationError): pass
    return lab, idstart

ids_all, cls_all = [], []
for p in held:
    s = Path(REPO, p).read_text(errors="replace")
    enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == tok(s, add_special_tokens=False).input_ids
    lab, idstart = labels(s)
    for (a, b) in enc["offset_mapping"]:
        seg = [lab[i] for i in range(a, b)] or ["other"]
        idl = [x for x in seg if x.startswith("ID:")]
        if idl:
            # first sub-token of an identifier if the identifier starts inside this token's span
            first = any(i in idstart for i in range(a, b))
            c = idl[0][3:] + ("|sub0" if first else "|cont")
        else:
            c = Counter(seg).most_common(1)[0][0]
        cls_all.append(c)
    ids_all += enc["input_ids"] + [EOS]; cls_all.append("eos")
a = np.asarray(ids_all); c = np.asarray(cls_all, dtype=object)
n = len(a) // T; ho = a[:n*T].reshape(n, T); hc = c[:n*T].reshape(n, T)
idx = np.sort(np.random.default_rng(1).choice(n, size=min(32, n), replace=False))
held_x, held_c = ho[idx], hc[idx][:, 1:]          # class of the TARGET token
base = np.load(f"{VE}/base_verify.npz")["heldout"].astype(np.float64)
on = np.load(f"{VE}/nll_full_heldout.npy").astype(np.float64)
assert base.shape == held_c.shape
d = on - base; R = {}
def grp(name, m):
    if m.sum() == 0: return
    R[name] = dict(n=int(m.sum()), frac=round(float(m.mean()), 4), base=round(float(base[m].mean()), 3),
                   delta=round(float(d[m].mean()), 4), rel_pct=round(100*float(d[m].mean()/base[m].mean()), 2),
                   share_of_gain=round(float(d[m].sum() / d.sum()), 3),
                   base_top1_proxy_p=round(float(np.exp(-base[m]).mean()), 3), on_p=round(float(np.exp(-on[m]).mean()), 3))
grp("all", np.ones_like(d, bool))
for k in sorted(set(held_c.ravel().tolist())): grp(k, held_c == k)
isid = np.vectorize(lambda s: "|" in s)(held_c)
grp("ID|sub0 (all)", np.vectorize(lambda s: s.endswith("|sub0"))(held_c))
grp("ID|cont (all)", np.vectorize(lambda s: s.endswith("|cont"))(held_c))
grp("ID proj (all)", np.vectorize(lambda s: s.startswith("proj"))(held_c))
grp("ID gen+builtin (all)", np.vectorize(lambda s: s.startswith(("gen", "builtin")))(held_c))
grp("non-ID", ~isid)
for k, v in R.items(): print(f"{k:24s} {json.dumps(v)}")
json.dump(R, open(f"{OUT}/tokdelta.json", "w"), indent=1)
