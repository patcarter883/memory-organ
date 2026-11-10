"""Stage-1 ingestion extras (complements count_rows.py).

1. transformers qwen4_exp hashing parity: rows from HF Qwen4ExpTextNGramEmbedding.forward (table replaced by an
   index-returning stub, so the 320M x 160 table is never allocated) == minisgl/ple/hashing.py, real config.json.
2. Corpus scope: python/**/*.py vs every text file under python/ (adds .cu/.cuh/.h/.cpp/.md).
3. Presentation framing: the same files as (a) bare EOS segments, (b) cat -n style line-numbered tool output,
   (c) inside a chat turn in a ```python fence (no EOS reset across the turn). How many rows does each framing touch,
   how many are shared with (a), and what fraction of positions get a different 2/3-gram key.
4. Git history: tokens/rows across every distinct .py blob reachable on first-parent history vs HEAD.
5. Row-load skew inside the project: share of position-reads served by the top-k rows (penalty / budget design).

Usage: python -I extras.py <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

T0 = time.time()
SCR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad")
REPO = Path("/home/user/patcarter883/minisglang-rdna4")
HEAD = "df29117"
TOK = SCR / "nm/stage1/ingest-algo/tok/tokenizer.json"
CFG = SCR / "models/Qwen/Qwen3.8-Flash-Next/config.json"
OUT = Path(sys.argv[1])
RES: dict = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


spec = importlib.util.spec_from_file_location("hsh", REPO / "python/minisgl/ple/hashing.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
tc = json.loads(CFG.read_text())["text_config"]
V, EOS, NS, HPN, BASE = tc["vocab_size"], tc["eos_token_id"], tc["ngram_size"], tc["heads_per_ngram"], tc["ngram_vocab_size_base"]
NH = (NS - 1) * HPN
mult = H.build_layer_multipliers(V, NS, 0)
sizes = H.build_head_vocab_sizes(BASE, NH, 0)
offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
hasher = H.Qwen4ExpNGramHasher(ngram_size=NS, heads_per_ngram=HPN, layer_multipliers=mult, eos_token_id=EOS)
tok = Tokenizer.from_file(str(TOK))


def rows_np(ids):
    mixed = hasher.mixed_ids(ids)
    return offs[None, :] + np.mod(np.repeat(mixed, HPN, axis=1), sizes[None, :])


def keys(ids):
    s1 = H.shift_right_ignore_eos(ids, 1, EOS); s2 = H.shift_right_ignore_eos(ids, 2, EOS)
    return s1 * V + ids, (s2 * V + s1) * V + ids


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True).stdout


def blob(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True).stdout.decode("utf-8", "replace")


def enc(t):
    return np.asarray(tok.encode(t, add_special_tokens=False).ids, dtype=np.int64)


# ------------------------------------------------------------------ 1. HF parity
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4  # noqa: E402

_real_emb = torch.nn.Embedding
torch.nn.Embedding = lambda n, d, *a, **k: _real_emb(1, 1)    # never allocate the 320M-row table
try:
    cfgns = types.SimpleNamespace(ngram_size=NS, heads_per_ngram=HPN, vocab_size=V, ngram_vocab_size_base=BASE,
                                  seed=tc.get("seed", 1234), eos_token_id=EOS,
                                  make_ngram_vocab_size_divisible_by=tc.get("make_ngram_vocab_size_divisible_by", 128))
    hf = q4.Qwen4ExpTextNGramEmbedding(cfgns, embedding_dim=NH * 160, layer_idx=1, ple_layer_index=0)
finally:
    torch.nn.Embedding = _real_emb


class IdxStub(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        return ids.unsqueeze(-1)


hf.ngram_embedding = IdxStub()
files = [p for p in git("ls-tree", "-r", "--name-only", HEAD, "python").split("\n") if p.endswith(".py")]
npos, nbad = 0, 0
for p in files[::12]:
    ids = enc(blob(HEAD, p))
    if ids.size == 0:
        continue
    ids = np.concatenate([ids[:300], [EOS], ids[300:600]])     # include a mid-sequence EOS reset
    with torch.no_grad():
        r_hf = hf(torch.from_numpy(ids)[None], None)[0].numpy()
    r_np = rows_np(ids)
    npos += ids.size; nbad += int((r_hf != r_np).any(axis=1).sum())
RES["hf_parity"] = dict(files=len(files[::12]), positions=npos, mismatched_positions=nbad,
                        hf_multipliers=hf.layer_multipliers.tolist(), hf_total_rows=int(hf.total_vocab_size))
log("HF qwen4_exp vs minisgl hashing:", RES["hf_parity"])


# ------------------------------------------------------------------ helpers
def census(docs):
    R = np.concatenate([rows_np(d) for d in docs]); K = [keys(d) for d in docs]
    k2 = np.concatenate([k[0] for k in K]); k3 = np.concatenate([k[1] for k in K])
    per_head = [int(np.unique(R[:, h]).size) for h in range(NH)]
    return dict(tokens=int(R.shape[0]), distinct_2gram=int(np.unique(k2).size), distinct_3gram=int(np.unique(k3).size),
                rows_per_head_2g_mean=float(np.mean(per_head[:8])), rows_per_head_3g_mean=float(np.mean(per_head[8:])),
                rows_total=int(sum(per_head)), fp32_GiB=sum(per_head) * 640 / 2**30, bf16_GiB=sum(per_head) * 320 / 2**30), R


# ------------------------------------------------------------------ 2. scope
texts_py = {p: blob(HEAD, p) for p in files}
allf = [p for p in git("ls-tree", "-r", "--name-only", HEAD, "python").split("\n") if p and not p.endswith((".gz",))]
texts_all = {p: blob(HEAD, p) for p in allf}
c_py, R_py = census([enc(t) for t in texts_py.values()])
c_all, _ = census([enc(t) for t in texts_all.values()])
RES["scope_py"], RES["scope_all_text"] = c_py, c_all
RES["scope_all_text"]["files"] = len(allf)
log("scope .py:", c_py); log("scope all text files:", c_all)

# ------------------------------------------------------------------ 3. presentation framing
def numbered(t):
    return "".join(f"{i:>6}\t{line}\n" for i, line in enumerate(t.split("\n"), 1))


def chat(p, t):
    return f"<|im_start|>user\nHere is `{p}`:\n```python\n{t}\n```\nWhat does it do?<|im_end|>\n<|im_start|>assistant\n"


def rowsets(R):
    return [np.unique(R[:, h]) for h in range(NH)]


base_sets = rowsets(R_py)
for name, fn in (("cat_n_numbered", lambda p, t: numbered(t)), ("chat_fenced", chat)):
    docs = [enc(fn(p, t)) for p, t in texts_py.items()]
    c, R = census(docs)
    sets = rowsets(R)
    shared = sum(int(np.isin(sets[h], base_sets[h]).sum()) for h in range(NH))
    union = sum(int(np.union1d(sets[h], base_sets[h]).size) for h in range(NH))
    # positions of the framed corpus whose 16 rows are all already in the bare-file row set
    cov = np.ones(R.shape[0], bool)
    for h in range(NH):
        cov &= np.isin(R[:, h], base_sets[h])
    c.update(rows_shared_with_bare=shared, rows_new_vs_bare=c["rows_total"] - shared, rows_union_with_bare=union,
             positions_fully_covered_by_bare_rows=float(cov.mean()))
    RES[f"framing_{name}"] = c
    log(f"framing {name}:", c)

# ------------------------------------------------------------------ 4. git history (first parent)
commits = git("rev-list", "--first-parent", HEAD).split()
blobs = set()
for c in commits:
    for line in git("ls-tree", "-r", c, "python").split("\n"):
        if line.endswith(".py"):
            meta, path = line.split("\t", 1)
            blobs.add(meta.split()[2])
hist_tok, hist_sets = 0, [set() for _ in range(NH)]
hist_rows_arr = [[] for _ in range(NH)]
for b in blobs:
    t = subprocess.run(["git", "-C", str(REPO), "cat-file", "-p", b], capture_output=True).stdout.decode("utf-8", "replace")
    ids = enc(t)
    if ids.size == 0:
        continue
    hist_tok += ids.size
    R = rows_np(ids)
    for h in range(NH):
        hist_rows_arr[h].append(np.unique(R[:, h]))
hist_union = [np.unique(np.concatenate(x)) for x in hist_rows_arr]
hist_rows = sum(int(u.size) for u in hist_union)
live = sum(int(np.isin(hist_union[h], base_sets[h]).sum()) for h in range(NH))
RES["git_history"] = dict(first_parent_commits=len(commits), distinct_py_blobs=len(blobs), tokens_all_blobs=hist_tok,
                          rows_all_blobs=hist_rows, rows_live_at_head=live, rows_only_in_history=hist_rows - live,
                          fp32_GiB_all_blobs=hist_rows * 640 / 2**30)
log("git history:", RES["git_history"])

# ------------------------------------------------------------------ 5. row-load skew in project
skew = {}
for h in (0, 8):
    _, cnt = np.unique(R_py[:, h], return_counts=True)
    cs = np.sort(cnt)[::-1]; cum = np.cumsum(cs) / cs.sum()
    skew[f"head{h}"] = {f"top{k}": float(cum[min(k, cs.size) - 1]) for k in (1000, 10000, 50000, 100000)}
    skew[f"head{h}"]["rows_for_50pct_reads"] = int(np.searchsorted(cum, 0.5) + 1)
    skew[f"head{h}"]["rows_for_90pct_reads"] = int(np.searchsorted(cum, 0.9) + 1)
    skew[f"head{h}"]["singleton_rows_frac"] = float((cs == 1).mean())
RES["row_load_skew"] = skew
log("row-load skew:", skew)
OUT.write_text(json.dumps(RES, indent=1))
log("wrote", OUT)
