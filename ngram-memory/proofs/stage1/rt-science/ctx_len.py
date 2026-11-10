# Does the 0.5B proxy's held-out gain survive the §6.A.1 fixed scoring context (C >= 1024 in-file tokens)?
# Re-uses the skeptic's trained arm-A rows (rows_armA.pt) and the exact data/key pipeline of verify.py.
import glob, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/user/memory-organ/ngram-memory/proofs/graft")
from graft_proto import NGramGraft
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
V_DIR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment")
OUT = Path(__file__).resolve().parent
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
STDLIB = "/usr/local/lib/python3.13"
STDLIB_FILES = ["argparse.py", "dataclasses.py", "functools.py", "typing.py", "json/decoder.py", "json/encoder.py",
                "asyncio/tasks.py", "asyncio/queues.py", "collections/__init__.py", "logging/__init__.py",
                "subprocess.py", "threading.py", "inspect.py", "enum.py", "contextlib.py", "shutil.py"]
EOS, V, T = 151643, 151936, 512
KEY = dict(vocab_size=V, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=200_000,
           make_ngram_vocab_size_divisible_by=128, seed=1234, eos_token_id=EOS, ple_layer_index=0)
NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else 16
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]; random.Random(0).shuffle(shuf)
n_held = round(0.15 * len(files)); held_files, train_files = sorted(shuf[:n_held]), sorted(shuf[n_held:])
def stream(paths, root):
    ids = []
    for p in paths:
        ids += tok(Path(root, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
    return np.asarray(ids, dtype=np.int64)
def chunk(s):
    n = len(s) // T; return s[: n * T].reshape(n, T)
def sample_idx(n, k, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))
tr_all = chunk(stream(train_files, REPO))
train_x = torch.from_numpy(tr_all[sample_idx(len(tr_all), 96, 0)])
model = AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.float32).eval()
for p in model.parameters(): p.requires_grad_(False)
D = model.config.hidden_size
_ref = NGramGraft(D, KEY, layer_idx=1, table="sparse", init="editor")
TRAIN_ROW_IDS = torch.unique(_ref.row_ids(train_x)); N_TOTAL = _ref.ple.ple_embedding.ngram_embedding.num_embeddings
class CompactRows(nn.Module):
    def __init__(self, n_total, w, row_ids):
        super().__init__(); self.num_embeddings, self.embedding_dim = n_total, w
        self.register_buffer("row_ids", row_ids); self.emb = nn.Embedding(len(row_ids), w, sparse=True)
        with torch.no_grad(): self.emb.weight.zero_()
    @property
    def weight(self): return self.emb.weight
    def forward(self, ids):
        n = self.row_ids.numel(); pos = torch.searchsorted(self.row_ids, ids.contiguous()).clamp_max(n - 1)
        hit = (self.row_ids[pos] == ids).unsqueeze(-1).to(self.emb.weight.dtype)
        return self.emb(pos) * hit
g = NGramGraft(D, KEY, layer_idx=1, table="sparse", init="editor", value_seed=0)
emb = g.ple.ple_embedding
emb.ngram_embedding = CompactRows(N_TOTAL, emb.ngram_embedding.embedding_dim, TRAIN_ROW_IDS)
for p in g.parameters(): p.requires_grad_(False)
g.install(model)
W = emb.ngram_embedding.weight
ROWS = torch.load(V_DIR / "rows_armA.pt")
assert ROWS.shape == W.shape, (ROWS.shape, W.shape)

def blocks(paths, root, nblk, seed):
    out = []
    for fi, p in enumerate(paths):
        ids = tok(Path(root, p).read_text(errors="replace"), add_special_tokens=False).input_ids
        for b in range(1024, len(ids) - 1024 + 1, 1024):
            out.append((fi, np.asarray(ids[b - 1024: b + 1024], dtype=np.int64)))
    rng = np.random.default_rng(seed); sel = sorted(rng.choice(len(out), size=min(nblk, len(out)), replace=False))
    return [out[i] for i in sel], len(out)

@torch.no_grad()
def nll(x):  # x: 1D tensor, returns per-target nll for x[1:]
    lg = model(input_ids=x[None], use_cache=False).logits[0, :-1]
    return F.cross_entropy(lg, x[1:], reduction="none")

def score(blks):
    # targets = positions 1024+[1..511] and 1024+512+[1..511]; short: cold 512-token windows (verify.py protocol);
    # long: context >= 1024 in-file tokens (window [b-1024, b+1024)).
    res = {"short": [], "long": []}
    for fi, a in blks:
        x = torch.from_numpy(a)
        s = torch.cat([nll(x[1024:1536]), nll(x[1536:2048])])  # 511 + 511 targets
        l = nll(x)  # 2047 targets; target index j predicts x[j+1]
        keep = torch.tensor([j for j in range(1024, 2047) if (j + 1 - 1024) % 512 != 0])  # targets x[1025..1535], x[1537..2047]
        res["short"].append(s.numpy()); res["long"].append(l[keep].numpy())
    return {k: np.stack(v) for k, v in res.items()}

R = {}
for name, paths, root, nb, seed in (("heldout", held_files, REPO, NBLK, 7), ("stdlib", [f for f in STDLIB_FILES if Path(STDLIB, f).exists()], STDLIB, NBLK // 2, 8)):
    blks, avail = blocks(paths, root, nb, seed)
    t = time.time()
    with torch.no_grad(): W.zero_()
    base = score(blks)
    with torch.no_grad(): W.copy_(ROWS)
    on = score(blks)
    fid = np.array([fi for fi, _ in blks])
    r = {"blocks": len(blks), "available_blocks": avail, "files": int(len(np.unique(fid))), "sec": time.time() - t}
    for c in ("short", "long"):
        d = on[c] - base[c]
        bm = base[c].mean(); dm = d.mean()
        # file-cluster bootstrap
        rng = np.random.default_rng(0); U = np.unique(fid); boots = []
        for _ in range(5000):
            k = rng.choice(U, len(U)); m = np.concatenate([np.where(fid == u)[0] for u in k])
            boots.append(d[m].mean() / base[c][m].mean())
        r[c] = dict(base=float(bm), delta=float(dm), rel_pct=float(100 * dm / bm), ci95_rel_pct=[float(100 * q) for q in np.percentile(boots, [2.5, 97.5])])
    R[name] = r; print(name, json.dumps(r), flush=True)
(OUT / "ctx_len.json").write_text(json.dumps(R, indent=1))
