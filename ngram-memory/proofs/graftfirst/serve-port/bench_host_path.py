"""Host-side cost of the GRAFT row path, torch BLOCKED (proves the path is numpy-only).

Pipeline per forward (what PLEEmbeddingSource would do with a SparseRowTable instead of the NVMe table):
  1. hash        : fork's Qwen4ExpNGramHasher.row_ids per sequence (unchanged code, Flash-Next KeySpec)
  2. lookup      : composite key (ns<<32 | row_id) -> np.searchsorted into ONE sorted array of all loaded cartridges
  3. stage       : (a) DENSE  -> zero a (T, 16*w) uint16 (bf16-bit) staging slice, scatter hit rows   [decode path]
                   (b) COMPACT-> only tokens with >=1 hit: (T_hit, 16*w) rows + (T_hit,) token index    [prefill path]
Single thread (another job owns 3 of 4 cores). Times are medians, microseconds. CPU here != owner's CPU: est. only.
"""
import sys, time, json, builtins
_real_import = builtins.__import__
def _no_torch(name, *a, **k):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch blocked")
    return _real_import(name, *a, **k)
builtins.__import__ = _no_torch
sys.path.insert(0, "/home/user/patcarter883/minisglang-rdna4/python")
import importlib.util, numpy as np
def _load(name, path):  # load single files so the package __init__ (which is lazy anyway) cannot pull torch
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
R = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
hashing = _load("minisgl.ple.hashing", f"{R}/ple/hashing.py")
row_table = _load("minisgl.weights.row_table", f"{R}/weights/row_table.py")

V, EOS, NS, HPN = 248320, 248044, 3, 8
PRIMES = np.array([20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000081, 20000099,
                   20000101, 20000107, 20000111, 20000113, 20000117, 20000123, 20000159, 20000171], dtype=np.int64)
heads = row_table.NgramHeads(offsets=np.concatenate([[0], np.cumsum(PRIMES)[:-1]]).astype(np.int64), vocab_sizes=PRIMES)
hasher = hashing.Qwen4ExpNGramHasher(ngram_size=NS, heads_per_ngram=HPN,
                                     layer_multipliers=hashing.build_layer_multipliers(V, NS, 0, 1234), eos_token_id=EOS)
assert hasher.layer_multipliers.tolist() == [23703573157769, 20109073645365, 8052911324071], hasher.layer_multipliers
rng = np.random.default_rng(0)
NH = 16

class SparseCartridgeTable:
    """All loaded cartridges in one sorted composite-key array. ns 0 == 'no cartridge' and is never stored."""
    def __init__(self, keys, rows):
        o = np.argsort(keys); self.keys = keys[o]; self.rows = rows[o]
    def lookup(self, ns, ids):  # ns (N,), ids (N,) -> pos (N,), hit (N,) bool
        k = (ns.astype(np.int64) << 32) | ids
        pos = np.searchsorted(self.keys, k)
        pos[pos == self.keys.size] = 0
        return pos, self.keys[pos] == k

def med(f, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6)

out = {"note": "median us, 1 thread, this sandbox CPU (est. for owner box)"}
for w in (128, 320):            # 35B-A3B, 27B
    for anchors in (10_000, 100_000):
        U = anchors * NH
        ids = rng.integers(0, int(PRIMES.sum()), size=U, dtype=np.int64)
        tab = SparseCartridgeTable((np.int64(1) << 32) | ids, rng.integers(0, 65535, size=(U, w), dtype=np.uint16))
        dense = np.zeros((8192 * NH, w), dtype=np.uint16)
        for label, T, bs in (("decode_bs1", 1, 1), ("decode_bs4", 1, 4), ("decode_bs16", 1, 16),
                             ("prefill_2048", 2048, 1), ("prefill_8192", 8192, 1)):
            toks = [rng.integers(0, V, size=T, dtype=np.int64) for _ in range(bs)]
            hist = np.full(2, EOS, dtype=np.int64)
            def h():
                return np.concatenate([hasher.row_ids(heads, np.concatenate([hist, t]), T) for t in toks]).reshape(-1)
            flat = h(); N = flat.size; ns = np.ones(N, dtype=np.int64)
            n = 300 if T == 1 else 7
            t_hash = med(h, n)
            def look(): return tab.lookup(ns, flat)
            t_look = med(look, n)
            pos, hit = look()
            def stage_dense():
                dense[:N] = 0
                dense[:N][hit] = tab.rows[pos[hit]]
            t_dense = med(stage_dense, n)
            # 100%-hit worst case (every row of every token is a cartridge row)
            flat_hit = ids[rng.integers(0, U, size=N)]
            def look_hit(): return tab.lookup(ns, flat_hit)
            t_look_hit = med(look_hit, n)
            pos_h, hit_h = look_hit()
            def stage_dense_hit():
                dense[:N] = 0
                dense[:N][hit_h] = tab.rows[pos_h[hit_h]]
            t_dense_hit = med(stage_dense_hit, n)
            def stage_compact(pos=pos, hit=hit):
                th = hit.reshape(-1, NH).any(1); tok = np.flatnonzero(th)
                rows = np.zeros((tok.size * NH, w), dtype=np.uint16)
                sel = hit.reshape(-1, NH)[tok].reshape(-1)
                rows[sel] = tab.rows[pos.reshape(-1, NH)[tok].reshape(-1)[sel]]
                return tok, rows
            t_compact = med(stage_compact, n)
            t_compact_hit = med(lambda: stage_compact(pos_h, hit_h), n)
            out[f"w{w}/anchors{anchors}/{label}"] = {
                "tokens": T * bs, "rows": int(N), "hash": round(t_hash, 1), "lookup_0pct_hit": round(t_look, 1),
                "stage_dense_0pct": round(t_dense, 1), "stage_compact_0pct": round(t_compact, 1),
                "lookup_100pct_hit": round(t_look_hit, 1), "stage_dense_100pct": round(t_dense_hit, 1),
                "stage_compact_100pct": round(t_compact_hit, 1),
                "h2d_bytes_dense_bf16": int(N * w * 2),
                "cartridge_ram_MiB": round(U * (8 + 2 * w) / 2**20, 1)}
# vectorised decode hash (one numpy pass for the whole decode batch instead of a per-sequence loop)
# NOTE: the EOS rule in vec_decode below is WRONG for EOS-bearing histories (an EOS belongs to the segment it ends);
# it is kept only as a timing probe. The correct, EOS-checked version is vec_decode_hash.py (64,000 rows, 0 mismatches).
M = hasher.layer_multipliers
def vec_decode(hist3):  # hist3 (bs, 3) = [t-2, t-1, t]; EOS rule: a key reaching back across an EOS uses EOS
    x0, x1, x2 = hist3[:, 2], hist3[:, 1].copy(), hist3[:, 0].copy()
    x1[x0 == EOS] = EOS  # (simplified boundary handling; real port must reuse shift_right_ignore_eos semantics)
    m2 = (x0 * M[0]) ^ (x1 * M[1]); m3 = m2 ^ (x2 * M[2])
    hs = np.repeat(np.stack([m2, m3], 1), HPN, axis=1)
    return heads.row_ids(hs)
for bs in (1, 16, 64):
    hist3 = rng.integers(0, V - 10, size=(bs, 3), dtype=np.int64)
    ref = np.stack([hasher.row_ids(heads, hist3[i], 1)[0] for i in range(bs)])
    assert np.array_equal(vec_decode(hist3), ref)
    out[f"decode_hash_vectorised_bs{bs}"] = {"loop_us": round(med(lambda: [hasher.row_ids(heads, hist3[i], 1) for i in range(bs)], 300), 1),
                                            "vectorised_us": round(med(lambda: vec_decode(hist3), 300), 1)}
out["torch_imported"] = "torch" in sys.modules
print(json.dumps(out, indent=1))
