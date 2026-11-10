"""Class each held-out eval target token as comment / string(docstring) / identifier / other-code via Python tokenize."""
import glob, io, json, keyword, os, random, tokenize
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"; EOS, T = 151643, 512
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]; random.Random(0).shuffle(shuf); held = sorted(shuf[:round(0.15 * len(files))])
ids, cls = [], []
for f in held:
    txt = Path(REPO, f).read_text(errors="replace")
    lines = txt.splitlines(keepends=True); starts = np.cumsum([0] + [len(l) for l in lines])
    lab = np.full(len(txt), 3, np.int8)  # 3 other code
    try:
        for t in tokenize.generate_tokens(io.StringIO(txt).readline):
            a = starts[t.start[0] - 1] + t.start[1]; b = starts[t.end[0] - 1] + t.end[1]
            if t.type == tokenize.COMMENT: lab[a:b] = 0
            elif t.type == tokenize.STRING or t.type in (getattr(tokenize, "FSTRING_MIDDLE", -1),): lab[a:b] = 1
            elif t.type == tokenize.NAME and not keyword.iskeyword(t.string): lab[a:b] = 2
    except tokenize.TokenError:
        pass
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids += enc.input_ids + [EOS]
    for (a, b) in enc.offset_mapping:
        seg = lab[a:b]; seg = seg[seg != 3] if (seg != 3).any() else seg  # token touching a comment/string/name
        cls.append(int(np.bincount(seg, minlength=4).argmax()) if len(seg) else 3)
    cls.append(3)
ids = np.asarray(ids); cls = np.asarray(cls)
n = len(ids) // T
hidx = np.sort(np.random.default_rng(1).choice(n, size=32, replace=False))
hx = ids[:n * T].reshape(n, T)[hidx]; hc = cls[:n * T].reshape(n, T)[hidx][:, 1:]
b = np.load("base_verify.npz")["heldout"]; full = np.load("nll_full_heldout.npy"); ex = np.load("nll_exact_heldout.npy")
d = full - b
names = ["comment", "string/docstring", "identifier", "other code"]
out = {}
for i, nm in enumerate(names):
    m = hc == i
    out[nm] = dict(frac=float(m.mean()), base=float(b[m].mean()), delta=float(d[m].mean()), share_of_gain=float(d[m].sum() / d.sum()))
print(json.dumps(out, indent=1))
json.dump(out, open("tokclass.json", "w"), indent=1)
plain = []
for f in held: plain += tok(Path(REPO, f).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
plain = np.asarray(plain); assert (plain == ids).all(); print("ids reconstruction identical to ingest.py stream")
