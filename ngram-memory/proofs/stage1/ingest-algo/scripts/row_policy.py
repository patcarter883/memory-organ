"""Trainable-row selection policies for a project cartridge: size vs reach vs collateral.

Train = by-file 85% split of minisgl python/*.py at df29117 (same seed as count_rows.py).
Reference ("what general traffic looks like", used to DEFINE specificity): Python 3.13 stdlib sample
(2M tok) + wikitext-2 train (2.5M tok).
Evaluation on corpora NOT used to define specificity: held-out project files (reach), transformers
source sample (near collateral), wikitext-2 test (general collateral).

Usage: python -I row_policy.py <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

T0 = time.time()
SCR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad")
REPO = Path("/home/user/patcarter883/minisglang-rdna4")
HEAD = "df29117"
TOK = SCR / "nm/stage1/ingest-algo/tok/tokenizer.json"
WK = Path("/root/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
          "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/")
STDLIB = Path("/usr/lib/python3.13")
TFM = SCR / "venv/lib/python3.13/site-packages/transformers"
OUT = Path(sys.argv[1])


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


spec = importlib.util.spec_from_file_location("h", REPO / "python/minisgl/ple/hashing.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
V, EOS = 248320, 248044
mult = H.build_layer_multipliers(V, 3, 0)
sizes = H.build_head_vocab_sizes(20_000_000, 16, 0)
offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
hasher = H.Qwen4ExpNGramHasher(ngram_size=3, heads_per_ngram=8, layer_multipliers=mult, eos_token_id=EOS)
tok = Tokenizer.from_file(str(TOK))


def enc(texts):
    out = []
    for i in range(0, len(texts), 256):
        out += [np.asarray(e.ids, np.int64) for e in tok.encode_batch(texts[i:i + 256], add_special_tokens=False)]
    return [d for d in out if d.size]


def keys(docs):
    K2, K3, R = [], [], []
    for d in docs:
        mixed = hasher.mixed_ids(d)
        R.append(offs[None, :] + np.mod(np.repeat(mixed, 8, axis=1), sizes[None, :]))
        s1 = H.shift_right_ignore_eos(d, 1, EOS); s2 = H.shift_right_ignore_eos(d, 2, EOS)
        K2.append(s1 * V + d); K3.append((s2 * V + s1) * V + d)
    return np.concatenate(K2), np.concatenate(K3), np.concatenate(R)


def git_blob(p):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{HEAD}:{p}"], capture_output=True).stdout.decode("utf-8", "replace")


files = [p for p in subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", HEAD, "python"],
                                   capture_output=True, text=True).stdout.split("\n") if p.endswith(".py")]
docs = enc([git_blob(p) for p in files])
idx = list(range(len(docs))); random.Random(0).shuffle(idx)
nh = int(round(0.15 * len(idx)))
tr = keys([docs[i] for i in sorted(idx[nh:])]); ho = keys([docs[i] for i in sorted(idx[:nh])])


def pyfiles(root, cap, seed, exclude):
    fs = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in exclude))
    random.Random(seed).shuffle(fs)
    out, n = [], 0
    for p in fs:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append(t); n += len(t) / 3.8
        if n >= cap:
            break
    return enc(out)


ref_std = keys(pyfiles(STDLIB, 2_000_000, 0, ("test", "idlelib", "tkinter", "lib2to3", "site-packages")))
wtr = pq.read_table(WK / "train-00000-of-00001.parquet").column("text").to_pylist()
ref_gen = keys(enc(["".join(wtr[i:i + 200]) for i in range(0, len(wtr), 200)]))
near = keys(pyfiles(TFM, 2_000_000, 0, ("test",)))
wte = pq.read_table(WK / "test-00000-of-00001.parquet").column("text").to_pylist()
gen = keys(enc(["".join(wte[i:i + 200]) for i in range(0, len(wte), 200)]))
log(f"tokens train={tr[0].size} heldout={ho[0].size} ref_stdlib={ref_std[0].size} ref_wiki={ref_gen[0].size} "
    f"eval_near_transformers={near[0].size} eval_wikitext_test={gen[0].size}")

N_tr, N_ref = tr[0].size, ref_std[0].size + ref_gen[0].size


def counts(k):
    u, c = np.unique(k, return_counts=True)
    return u, c


def lookup(u_ref, c_ref, q):
    pos = np.clip(np.searchsorted(u_ref, q), 0, max(u_ref.size - 1, 0))
    return np.where(u_ref[pos] == q, c_ref[pos], 0)


res = {}
sel_by_policy = {}
for order, ktr, kref in ((2, tr[0], np.concatenate([ref_std[0], ref_gen[0]])), (3, tr[1], np.concatenate([ref_std[1], ref_gen[1]]))):
    u, c = counts(ktr)
    ur, cr = counts(kref)
    cref = lookup(ur, cr, u)
    rate_p = c / N_tr; rate_r = cref / N_ref
    enrich = rate_p / (rate_r + 1.0 / N_ref)        # add-one smoothing on the reference
    pol = {
        "all": np.ones(u.size, bool),
        "count>=2": c >= 2,
        "count>=5": c >= 5,
        "specific(ref=0)": cref == 0,
        "specific & count>=2": (cref == 0) & (c >= 2),
        "enrich>=4": enrich >= 4,
        "enrich>=4 & count>=2": (enrich >= 4) & (c >= 2),
        "enrich>=16 & count>=2": (enrich >= 16) & (c >= 2),
    }
    for name, m in pol.items():
        sel_by_policy.setdefault(name, {})[order] = u[m]

# rows per policy (via the hash of the selected n-grams -> rows of each head), reach and collateral
for name, sel in sel_by_policy.items():
    d = {}
    # rows: take train positions whose key is selected, collect their rows per head
    rows = 0
    rowsets = []
    for h in range(16):
        order = 2 if h < 8 else 3
        m = np.isin(tr[0] if order == 2 else tr[1], sel[order])
        rs = np.unique(tr[2][m, h]); rowsets.append(rs); rows += rs.size
    d["selected_2grams"], d["selected_3grams"] = int(sel[2].size), int(sel[3].size)
    d["rows"] = rows
    d["overlay_MiB_bf16+ids"] = rows * (160 * 2 + 8) / 2**20
    d["overlay_MiB_fp32+ids"] = rows * (160 * 4 + 8) / 2**20
    d["collision_rate_per_head_2g_3g"] = [rowsets[0].size / sizes[0], rowsets[8].size / sizes[8]]
    for tag, (k2, k3, R) in (("heldout_project", ho), ("near_transformers", near), ("general_wikitest", gen)):
        e2 = np.isin(k2, sel[2]); e3 = np.isin(k3, sel[3])
        anyrow = np.zeros(k2.size, bool)
        for h in range(16):
            anyrow |= np.isin(R[:, h], rowsets[h])
        d[f"{tag}: exact2|exact3|either|any_row(incl. collisions)"] = [round(float(e2.mean()), 4), round(float(e3.mean()), 4),
                                                                        round(float((e2 | e3).mean()), 4), round(float(anyrow.mean()), 4)]
    res[name] = d
    log(name, json.dumps(d))

OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
