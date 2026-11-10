"""Vectorised DECODE-step hash for a whole batch (one numpy pass), checked bit-for-bit against the fork's
per-sequence Qwen4ExpNGramHasher.row_ids on EOS-dense histories. numpy only (torch blocked)."""
import sys, time, json, builtins, importlib.util
_ri = builtins.__import__
def _nt(n, *a, **k):
    if n == "torch" or n.startswith("torch."): raise ImportError("torch blocked")
    return _ri(n, *a, **k)
builtins.__import__ = _nt
import numpy as np
R = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m
hashing = _load("minisgl.ple.hashing", f"{R}/ple/hashing.py"); row_table = _load("minisgl.weights.row_table", f"{R}/weights/row_table.py")
V, EOS = 248320, 248044
P = np.array([20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000081, 20000099,
              20000101, 20000107, 20000111, 20000113, 20000117, 20000123, 20000159, 20000171], dtype=np.int64)
heads = row_table.NgramHeads(offsets=np.concatenate([[0], np.cumsum(P)[:-1]]).astype(np.int64), vocab_sizes=P)
hasher = hashing.Qwen4ExpNGramHasher(ngram_size=3, heads_per_ngram=8, layer_multipliers=hashing.build_layer_multipliers(V, 3, 0, 1234), eos_token_id=EOS)
M = hasher.layer_multipliers

def decode_row_ids(hist: np.ndarray, tok: np.ndarray) -> np.ndarray:
    """hist (bs, 2) = PLEStateCache.token_history rows [t-2, t-1]; tok (bs,) = this step's token. -> (bs, 16).
    An EOS ends ITS segment (it is still part of it), so: h1==EOS -> x1=x2=EOS; else h0==EOS -> x2=EOS."""
    h0, h1 = hist[:, 0], hist[:, 1]
    x1 = np.where(h1 == EOS, EOS, h1)
    x2 = np.where((h1 == EOS) | (h0 == EOS), EOS, h0)
    m2 = (tok * M[0]) ^ (x1 * M[1]); m3 = m2 ^ (x2 * M[2])
    return heads.row_ids(np.repeat(np.stack([m2, m3], 1), 8, axis=1))

rng = np.random.default_rng(1)
bad = 0; n = 0
for trial in range(2000):
    bs = 32
    pool = np.array([EOS, 5, 7, 248046, 0, 248319], dtype=np.int64)
    hist = np.where(rng.random((bs, 2)) < 0.4, EOS, rng.integers(0, V, (bs, 2)))
    hist[rng.random((bs, 2)) < 0.1] = pool[rng.integers(0, pool.size, size=1)]
    tok = np.where(rng.random(bs) < 0.3, EOS, rng.integers(0, V, bs))
    ref = np.stack([hasher.row_ids(heads, np.concatenate([hist[i], tok[i:i+1]]), 1)[0] for i in range(bs)])
    got = decode_row_ids(hist, tok)
    bad += int((ref != got).any(1).sum()); n += bs
def med(f, k=300):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1e6, 1)
res = {"checked_rows": n, "mismatched_rows": bad, "torch_imported": "torch" in sys.modules}
for bs in (1, 4, 16, 64):
    hist = rng.integers(0, V, (bs, 2)); tok = rng.integers(0, V, bs)
    res[f"bs{bs}"] = {"per_seq_loop_us": med(lambda: [hasher.row_ids(heads, np.concatenate([hist[i], tok[i:i+1]]), 1) for i in range(bs)]),
                      "vectorised_us": med(lambda: decode_row_ids(hist, tok))}
print(json.dumps(res, indent=1))
