# File-clustered bootstrap of the 0.5B proxy held-out and stdlib deltas (re-uses the skeptic's saved NLL arrays).
import glob, os, random, json, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
V = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment")
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
EOS, T = 151643, 512
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]; random.Random(0).shuffle(shuf)
n_held = round(0.15 * len(files)); held = sorted(shuf[:n_held])
src = []
for i, p in enumerate(held):
    t = tok(Path(REPO, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
    src += [i] * len(t)
src = np.asarray(src); n = len(src) // T
srcc = src[: n * T].reshape(n, T)
idx = np.sort(np.random.default_rng(1).choice(n, size=32, replace=False))
hs = srcc[idx][:, 1:]
base = dict(np.load(V / "base_verify.npz"))
d = np.load(V / "nll_full_heldout.npy") - base["heldout"]
b = base["heldout"]
print("held files", len(held), "pool chunks", n, "files in eval", len(np.unique(hs)))
print("mean delta", d.mean(), "rel", d.mean() / b.mean())
fids = np.unique(hs)
S = np.array([d[hs == f].sum() for f in fids]); N = np.array([(hs == f).sum() for f in fids]); B = np.array([b[hs == f].sum() for f in fids])
rng = np.random.default_rng(0); res = []; rel = []
for _ in range(10000):
    k = rng.integers(0, len(fids), len(fids))
    res.append(S[k].sum() / N[k].sum()); rel.append(S[k].sum() / B[k].sum())
print("file-cluster 95% CI delta", np.percentile(res, [2.5, 97.5]), "rel", np.percentile(rel, [2.5, 97.5]))
# chunk bootstrap for comparison
cs = d.sum(1); cn = np.full(32, d.shape[1])
r2 = [cs[k].sum() / cn[k].sum() for k in (rng.integers(0, 32, 32) for _ in range(10000))]
print("chunk 95% CI", np.percentile(r2, [2.5, 97.5]))
per = S / N; print("per-file mean deltas sd", per.std(ddof=1), "token shares", np.round(N / N.sum(), 3))
# unweighted file-mean MDE at 31 files (80% power, alpha .05 two-sided) using observed between-file SD of delta
sd = per.std(ddof=1); print("MDE nats @31 files", 2.8 * sd / np.sqrt(31), "rel to base 2.661/3.249", 2.8 * sd / np.sqrt(31) / 2.661, 2.8 * sd / np.sqrt(31) / 3.249)
# stdlib
ds = np.load(V / "nll_full_stdlib_py.npy") - base["stdlib_py"]
print("stdlib tokens", ds.size, "delta", ds.mean(), "rel", ds.mean() / base["stdlib_py"].mean())
