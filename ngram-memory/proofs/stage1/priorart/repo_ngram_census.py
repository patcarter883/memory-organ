# Census of a codebase under a Qwen tokenizer: tokens, distinct 2/3-grams, expected
# hash-sharing at Flash-Next head size (~20M rows/head) and at Engram-Adapter size (90,007).
import os, sys, math, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
roots = sys.argv[2:]
exts = {'.py','.md','.rs','.c','.h','.cpp','.hip','.cu','.ts','.js','.toml','.yaml','.yml','.sh','.txt','.json'}
ids_all = []; nfiles = 0; nbytes = 0
for root in roots:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith('.') and d not in ('node_modules','venv','__pycache__','build','dist')]
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.splitext(f)[1] not in exts: continue
            try:
                s = open(p, encoding='utf-8').read()
            except Exception: continue
            if len(s) > 2_000_000: continue
            nfiles += 1; nbytes += len(s)
            ids_all.append(tok(s, add_special_tokens=False)['input_ids'])
T = sum(map(len, ids_all))
bi = collections.Counter(); tri = collections.Counter()
for ids in ids_all:
    for i in range(1, len(ids)):
        bi[(ids[i-1], ids[i])] += 1
        if i >= 2: tri[(ids[i-2], ids[i-1], ids[i])] += 1
print(f"files={nfiles} bytes={nbytes} tokens={T}")
for name, c in (("2-gram", bi), ("3-gram", tri)):
    n = len(c); once = sum(1 for v in c.values() if v == 1)
    print(f"{name}: distinct={n} singletons={once} ({once/n:.1%})")
    for m in (20_000_003, 90_007):
        # prob a given key shares its bucket with >=1 other repo key (uniform hash), per head
        share = 1 - math.exp(-(n-1)/m)
        print(f"   head size {m}: P(key shares bucket with another repo key, one head) = {share:.4f}")
# rows touched by ingestion on Flash-Next layout: 8 heads per order
print(f"Flash-Next rows touched (upper bound, 8 heads x distinct keys): {8*(len(bi)+len(tri))}")
print(f"fp32 overlay at width 160: {8*(len(bi)+len(tri))*(160*4+8)/1e9:.2f} GB; bf16 delta: {8*(len(bi)+len(tri))*(160*2+8)/1e9:.2f} GB")
