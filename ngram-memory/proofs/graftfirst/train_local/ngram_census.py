# Distinct 2/3-gram census of a real codebase under the Flash-Next tokenizer -> rows a cartridge trains.
import os, sys, json, numpy as np
from tokenizers import Tokenizer
TOK = sys.argv[1]; ROOT = sys.argv[2]
tok = Tokenizer.from_file(TOK)
EXT = (".py",".md",".sh",".yml",".yaml",".toml",".txt",".h",".hip",".cpp",".c",".cu",".json")
files = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if not d.startswith(".") and d not in ("__pycache__","node_modules","out")]
    for f in fn:
        p = os.path.join(dp, f)
        if f.endswith(EXT) and os.path.getsize(p) < 2_000_000: files.append(p)
EOS = 248044
ids = []
for p in sorted(files):
    try: t = open(p, encoding="utf-8").read()
    except Exception: continue
    ids.extend(tok.encode(t).ids); ids.append(EOS)   # one EOS segment per file
a = np.array(ids, dtype=np.int64); N = len(a)
# history with eos reset (x_k := eos across file boundary), as in KeySpec 2.1
x0 = a; x1 = np.r_[EOS, a[:-1]]; x2 = np.r_[EOS, EOS, a[:-2]]
seg = np.cumsum(a == EOS); seg = np.r_[0, seg[:-1]]
x1 = np.where(np.r_[False, seg[1:] == seg[:-1]], x1, EOS)
x2 = np.where(np.r_[False, False, seg[2:] == seg[:-2]], x2, EOS)
big = x0 * 248320 + x1
tri = (x0 * 248320 + x1) * 248320 + x2   # fits int64? 248320^3=1.53e16 < 9.2e18 yes
nb = len(np.unique(big)); nt = len(np.unique(tri))
# growth curve: distinct per prefix
curve = []
for frac in (0.125, 0.25, 0.5, 1.0):
    k = int(N*frac); curve.append((k, len(np.unique(big[:k])), len(np.unique(tri[:k]))))
rows = 8*nb + 8*nt   # upper bound (hash collisions merge a few)
res = dict(root=ROOT, files=len(files), tokens=N, distinct_bigrams=nb, distinct_trigrams=nt,
           rows_16head_upper=rows, curve=curve)
for host, w in (("Qwen3.6-35B-A3B", 128), ("Qwen3.8-27B", 320), ("Flash-Next(native, ref)", 160)):
    P = rows * w
    res[host] = dict(row_width=w, params_M=round(P/1e6,1),
                     bf16_GB=round(P*2/1e9,2), fp32_master_plus_adam_GB=round(P*12/1e9,2),
                     bf16_rows_plus_rowwise_adagrad_GB=round((P*2 + rows*4)/1e9,2),
                     per_Mtok_rows=round(rows/N*1e6/1e6,2))
print(json.dumps(res, indent=1))
