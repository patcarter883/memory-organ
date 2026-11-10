# Position-level "none-hit" check: are ON and OFF equal at held-out positions whose 16 rows are all untrained?
import glob, os, random, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/user/memory-organ/ngram-memory/proofs/graft")
from graft_proto import NGramGraft
from transformers import AutoTokenizer
torch.set_num_threads(1)
V_DIR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment")
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
EOS, V, T = 151643, 151936, 512
KEY = dict(vocab_size=V, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=200_000,
           make_ngram_vocab_size_divisible_by=128, seed=1234, eos_token_id=EOS, ple_layer_index=0)
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]; random.Random(0).shuffle(shuf)
n_held = round(0.15 * len(files)); held, train = sorted(shuf[:n_held]), sorted(shuf[n_held:])
def stream(paths):
    ids = []
    for p in paths: ids += tok(Path(REPO, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
    return np.asarray(ids, dtype=np.int64)
def chunk(s): n = len(s) // T; return s[: n * T].reshape(n, T)
def sidx(n, k, seed): return np.sort(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))
tr = chunk(stream(train)); ho = chunk(stream(held))
train_x = torch.from_numpy(tr[sidx(len(tr), 96, 0)]); held_x = torch.from_numpy(ho[sidx(len(ho), 32, 1)])
g = NGramGraft(896, KEY, layer_idx=1, table="sparse", init="editor")
trained = torch.unique(g.row_ids(train_x)).numpy()
ids = g.row_ids(held_x).numpy()  # [32, 512, 16]
W = torch.load(V_DIR / "rows_armA.pt").numpy()
nz = np.linalg.norm(W, axis=1) > 0
trained_nz = trained[nz] if len(trained) == len(W) else trained
hit = np.isin(ids, trained_nz).any(-1)[:, :-1]   # position t reads >= 1 nonzero trained row -> affects target t+1
base = np.load(V_DIR / "base_verify.npz")["heldout"]; on = np.load(V_DIR / "nll_full_heldout.npy")
d = on - base
nohit = ~hit
prefix_hit = np.maximum.accumulate(hit, axis=1)
clean = ~prefix_hit
R = dict(pos=int(d.size), frac_nohit=float(nohit.mean()), nohit_mean_abs_delta=float(np.abs(d[nohit]).mean()),
         nohit_frac_exact_equal=float((d[nohit] == 0).mean()), nohit_mean_delta=float(d[nohit].mean()),
         nohit_contrib_share=float(d[nohit].sum() / d.sum()),
         frac_prefix_clean=float(clean.mean()), clean_mean_abs_delta=float(np.abs(d[clean]).mean()) if clean.any() else None)
print(json.dumps(R))
