# Count tokens and 2/3-gram types of a codebase (minisgl fork *.py) vs a general-code
# background (CPython stdlib + transformers/torch python sources), using the Qwen2.5 tokenizer
# as a proxy for the Qwen3.x tokenizers. Answers: how big is "a project codebase" in tokens,
# how many n-gram rows would Stage 1 touch, and what share of those rows are repo-specific
# (unseen in background) vs shared with general code (retention risk if written).
import os, sys, glob, collections, json
from transformers import AutoTokenizer

TOK = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
tok = AutoTokenizer.from_pretrained(TOK)

def files(root, exclude=()):
    out = []
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if any(e in p for e in exclude):
            continue
        try:
            if os.path.getsize(p) < 2_000_000:
                out.append(p)
        except OSError:
            pass
    return sorted(out)

def ngrams(paths, cap_tokens=None):
    c2, c3 = collections.Counter(), collections.Counter()
    n = 0
    for p in paths:
        try:
            s = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        ids = tok(s, add_special_tokens=False)["input_ids"]
        n += len(ids)
        c2.update(zip(ids, ids[1:]))
        c3.update(zip(ids, ids[1:], ids[2:]))
        if cap_tokens and n >= cap_tokens:
            break
    return n, c2, c3

repo = files("/home/user/patcarter883/minisglang-rdna4", exclude=("/.git/", "/venv", "/node_modules/"))
sp = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/venv/lib"
bg = files("/usr/lib/python3.12", exclude=("/test/", "/tests/")) + \
     files(glob.glob(sp + "/python3*/site-packages/transformers")[0]) + \
     files(glob.glob(sp + "/python3*/site-packages/torch")[0], exclude=("/testing/",))
print("repo files", len(repo), "background files", len(bg), flush=True)
nr, r2, r3 = ngrams(repo)
nb, b2, b3 = ngrams(bg, cap_tokens=60_000_000)
res = {"repo_tokens": nr, "background_tokens": nb}
for name, r, b in (("2gram", r2, b2), ("3gram", r3, b3)):
    types = len(r); occ = sum(r.values())
    shared_types = sum(1 for k in r if k in b)
    shared_occ = sum(v for k, v in r.items() if k in b)
    singletons = sum(1 for v in r.values() if v == 1)
    # repo-specific types that recur (>=2 occurrences, unseen in background): the "learnable" rows
    spec_rec = sum(1 for k, v in r.items() if k not in b and v >= 2)
    res[name] = dict(types=types, occurrences=occ, shared_types_frac=round(shared_types / types, 4),
                     shared_occ_frac=round(shared_occ / occ, 4), singleton_types_frac=round(singletons / types, 4),
                     repo_specific_recurring_types=spec_rec, background_types=len(b))
print(json.dumps(res, indent=1))
