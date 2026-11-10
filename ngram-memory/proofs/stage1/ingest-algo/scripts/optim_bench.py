"""Sparse optimiser cost for a cartridge (CPU, 4 threads): row-wise Adagrad vs SparseAdam on a compact row store.

(a) distinct (head,row) pairs touched per microbatch of S_MB tokens of the real project (Flash-Next hash),
(b) step time of the optimiser update alone for that many touched rows, compact store of R_STORE rows x 160 fp32,
(c) bytes per touched row for each optimiser (fp32 master delta + state + int64 id).
Usage: python -I optim_bench.py <out.json>
"""
import importlib.util, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from tokenizers import Tokenizer

torch.set_num_threads(4); torch.manual_seed(0)
SCR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad")
REPO = Path("/home/user/patcarter883/minisglang-rdna4")
spec = importlib.util.spec_from_file_location("hsh", REPO / "python/minisgl/ple/hashing.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
V, EOS = 248320, 248044
mult = H.build_layer_multipliers(V, 3, 0); sizes = H.build_head_vocab_sizes(20_000_000, 16, 0)
offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
hs = H.Qwen4ExpNGramHasher(ngram_size=3, heads_per_ngram=8, layer_multipliers=mult, eos_token_id=EOS)
tok = Tokenizer.from_file(str(SCR / "nm/stage1/ingest-algo/tok/tokenizer.json"))
files = [p for p in subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "df29117", "python"],
                                   capture_output=True, text=True).stdout.split() if p.endswith(".py")]
docs = []
for p in files:
    t = subprocess.run(["git", "-C", str(REPO), "show", f"df29117:{p}"], capture_output=True).stdout.decode("utf-8", "replace")
    d = np.asarray(tok.encode(t, add_special_tokens=False).ids, np.int64)
    if d.size:
        docs.append(offs[None, :] + np.mod(np.repeat(hs.mixed_ids(d), 8, axis=1), sizes[None, :]))
R = np.concatenate(docs)
res = {"corpus_tokens": int(R.shape[0])}
rng = np.random.default_rng(0)
for S_MB in (2048, 8192, 32768):
    u = []
    for _ in range(20):          # microbatch = random set of 512-token windows (shuffled training order)
        starts = rng.integers(0, R.shape[0] - 512, S_MB // 512)
        u.append(np.unique(np.concatenate([R[s:s + 512] for s in starts]).reshape(-1)).size)
    res[f"touched_rows_per_mb_{S_MB}"] = float(np.mean(u)); res[f"reads_per_mb_{S_MB}"] = S_MB * 16
print(json.dumps(res), flush=True)

W, R_STORE = 160, 2_000_000
vals = torch.zeros(R_STORE, W)
acc = torch.zeros(R_STORE)
m = torch.zeros(R_STORE, W); v = torch.zeros(R_STORE, W)
for S_MB in (8192, 32768):
    reads = 16 * S_MB
    idx_reads = torch.randint(0, R_STORE, (reads,))
    g_reads = torch.randn(reads, W)
    def coalesce():
        uq, inv = torch.unique(idx_reads, return_inverse=True)
        g = torch.zeros(uq.numel(), W).index_add_(0, inv, g_reads)
        return uq, g
    def adagrad(uq, g, lr=1e-2, eps=1e-10):
        a = acc[uq] + (g * g).mean(1); acc[uq] = a
        vals.index_add_(0, uq, -lr * g / (a.sqrt() + eps)[:, None])
    def adam(uq, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, t=10):
        mm = m[uq].mul_(b1).add_(g, alpha=1 - b1); vv = v[uq].mul_(b2).addcmul_(g, g, value=1 - b2)
        m[uq] = mm; v[uq] = vv
        vals.index_add_(0, uq, -lr * (mm / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps))
    for name, fn in (("coalesce", None), ("rowwise_adagrad", adagrad), ("lazy_adam", adam)):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); uq, g = coalesce()
            if fn: fn(uq, g)
            ts.append(time.perf_counter() - t0)
        res[f"{name}_s_reads{reads}_rows{int(uq.numel())}"] = float(np.median(ts))
res["bytes_per_row"] = {"rowwise_adagrad": W * 4 + 4 + 8, "lazy_adam(SparseAdam)": 3 * W * 4 + 8,
                        "adam_bf16_moments": W * 4 + 2 * W * 2 + 8, "served_bf16": W * 2 + 8, "served_int6g32": 130 + 8}
print(json.dumps(res, indent=1))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
