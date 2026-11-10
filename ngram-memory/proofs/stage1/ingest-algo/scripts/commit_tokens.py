"""Tokens in the .py files changed per first-parent commit (b80007f..df29117): size of re-train windows.
Usage: python -I commit_tokens.py <out.json>"""
import json, subprocess, sys
import numpy as np
from tokenizers import Tokenizer
R = "/home/user/patcarter883/minisglang-rdna4"
tok = Tokenizer.from_file("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/ingest-algo/tok/tokenizer.json")
g = lambda *a: subprocess.run(["git", "-C", R, *a], capture_output=True, text=True).stdout
cs = g("rev-list", "--first-parent", "--reverse", "b80007f..df29117").split(); prev = "b80007f"; per = []
for c in cs:
    n, nf, add_lines = 0, 0, 0
    for line in g("diff", "--numstat", "--no-renames", prev, c, "--", "python").splitlines():
        a, d, p = line.split("\t")
        if not p.endswith(".py"): continue
        nf += 1; add_lines += int(a) if a != "-" else 0
        t = subprocess.run(["git", "-C", R, "show", f"{c}:{p}"], capture_output=True).stdout.decode("utf-8", "replace")
        n += len(tok.encode(t, add_special_tokens=False).ids)
    if nf: per.append((n, nf, add_lines))
    prev = c
a = np.array(per)
res = {"commits_touching_py": len(per), "changed_file_tokens": {q: float(np.percentile(a[:, 0], p)) for q, p in (("median", 50), ("p90", 90), ("max", 100))},
       "files": {"median": float(np.median(a[:, 1])), "p90": float(np.percentile(a[:, 1], 90))},
       "added_lines": {"median": float(np.median(a[:, 2])), "p90": float(np.percentile(a[:, 2], 90))}}
print(json.dumps(res)); open(sys.argv[1], "w").write(json.dumps(res, indent=1))
