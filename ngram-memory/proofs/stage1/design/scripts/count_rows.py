"""Stage-1 ingestion census: how many Flash-Next n-gram rows does a real codebase touch?

Corpus  : `git ls-files 'python/*.py'` of the minisgl fork at a pinned commit (blobs via `git show`,
          so the working tree is irrelevant), one file = one EOS segment.
Tokenizer: Qwen/Qwen3.8-Flash-Next tokenizer.json (downloaded with hf_hub_download, tokenizer files only).
Hash    : the fork's numpy-only `minisgl/ple/hashing.py` (bit-identical to HF qwen4_exp; see
          ngram-memory/proofs/serve/ple_path/hash_parity.py), real config.json values.
Outputs : JSON + human-readable lines on stdout.

Usage: python -I count_rows.py <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

T0 = time.time()
SCR = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad")
REPO = Path("/home/user/patcarter883/minisglang-rdna4")
HEAD = "df29117"
C_OLD = "b80007f"  # first-parent commit 80 before HEAD (2026-10-28), for the time split
TOK = SCR / "nm/stage1/design/tok/tokenizer.json"
CFG = SCR / "models/Qwen/Qwen3.8-Flash-Next/config.json"
WIKI = Path("/root/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet")
STDLIB = Path("/usr/lib/python3.13")
TFM = SCR / "venv/lib/python3.13/site-packages/transformers"
OUT = Path(sys.argv[1])
ROW_W = 160
RES: dict = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- hash
spec = importlib.util.spec_from_file_location("minisgl_ple_hashing", REPO / "python/minisgl/ple/hashing.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
tc = json.loads(CFG.read_text())["text_config"]
V, EOS, NS, HPN, BASE = (tc["vocab_size"], tc["eos_token_id"], tc["ngram_size"], tc["heads_per_ngram"],
                         tc["ngram_vocab_size_base"])
NH = (NS - 1) * HPN
mult = H.build_layer_multipliers(V, NS, 0)
assert mult.tolist() == [23703573157769, 20109073645365, 8052911324071], mult
sizes = H.build_head_vocab_sizes(BASE, NH, 0)
offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
TOTAL_ROWS = int(sizes.sum())
hasher = H.Qwen4ExpNGramHasher(ngram_size=NS, heads_per_ngram=HPN, layer_multipliers=mult, eos_token_id=EOS)
RES["keyspec"] = dict(V=V, eos=EOS, ngram_size=NS, heads_per_ngram=HPN, base=BASE, heads=NH,
                      multipliers=mult.tolist(), head_sizes_first_last=[int(sizes[0]), int(sizes[-1])],
                      logical_rows=TOTAL_ROWS)
log("keyspec ok", RES["keyspec"])

tok = Tokenizer.from_file(str(TOK))


def rows_of(ids: np.ndarray):
    """(L,) ids of one EOS segment -> rows [L,16] global, k2 [L], k3 [L] exact n-gram keys."""
    mixed = hasher.mixed_ids(ids)                              # [L,2] (order 2, order 3)
    hh = np.repeat(mixed, HPN, axis=1)                         # [L,16]
    rows = offs[None, :] + np.mod(hh, sizes[None, :])
    s1 = H.shift_right_ignore_eos(ids, 1, EOS)
    s2 = H.shift_right_ignore_eos(ids, 2, EOS)
    k2 = s1 * V + ids                                          # exact 2-gram (x_{t-1}, x_t)
    k3 = (s2 * V + s1) * V + ids                               # exact 3-gram, < 2^63
    return rows, k2, k3


def encode_docs(texts):
    out = []
    for i in range(0, len(texts), 256):
        for e in tok.encode_batch(texts[i:i + 256], add_special_tokens=False):
            out.append(np.asarray(e.ids, dtype=np.int64))
    return out


class Corpus:
    def __init__(self, name, docs):
        self.name = name
        self.docs = [d for d in docs if d.size]
        R, K2, K3, IDS, NXT = [], [], [], [], []
        for d in self.docs:
            r, k2, k3 = rows_of(d)
            R.append(r); K2.append(k2); K3.append(k3); IDS.append(d)
            nx = np.empty_like(d); nx[:-1] = d[1:]; nx[-1] = -1   # next token (-1 = none)
            NXT.append(nx)
        self.rows = np.concatenate(R); self.k2 = np.concatenate(K2); self.k3 = np.concatenate(K3)
        self.ids = np.concatenate(IDS); self.nxt = np.concatenate(NXT)
        self.n = int(self.ids.size)


def git(*a, text=True):
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=text).stdout


def tree_files(commit):
    return [p for p in git("ls-tree", "-r", "--name-only", commit, "python").split("\n") if p.endswith(".py")]


def blob(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True).stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------- 1. project corpus at HEAD
files = tree_files(HEAD)
texts = [blob(HEAD, f) for f in files]
docs = encode_docs(texts)
proj = Corpus("project", docs)
nbytes = sum(len(t.encode()) for t in texts)
log(f"project: files={len(files)} bytes={nbytes} tokens={proj.n} bytes/token={nbytes / proj.n:.2f}")

# sanity: tokenizers == transformers AutoTokenizer on 5 files
from transformers import AutoTokenizer  # noqa: E402
atok = AutoTokenizer.from_pretrained(str(TOK.parent))
same = all(atok(texts[i], add_special_tokens=False)["input_ids"] == docs[i].tolist() for i in range(0, 250, 50))
log("AutoTokenizer parity on 5 files:", same)


def census(c: Corpus):
    d = {}
    d["tokens"] = c.n
    d["distinct_2gram"] = int(np.unique(c.k2).size)
    d["distinct_3gram"] = int(np.unique(c.k3).size)
    u2, n2 = np.unique(c.k2, return_counts=True)
    u3, n3 = np.unique(c.k3, return_counts=True)
    d["2gram_singleton_frac"] = float((n2 == 1).mean())
    d["3gram_singleton_frac"] = float((n3 == 1).mean())
    for k in (2, 5, 20):
        d[f"distinct_2gram_count>={k}"] = int((n2 >= k).sum())
        d[f"distinct_3gram_count>={k}"] = int((n3 >= k).sum())
    per_head, per_head_ge2, per_head_ge5 = [], [], []
    for h in range(NH):
        u, n = np.unique(c.rows[:, h], return_counts=True)
        per_head.append(int(u.size)); per_head_ge2.append(int((n >= 2).sum())); per_head_ge5.append(int((n >= 5).sum()))
    d["distinct_rows_per_head"] = per_head
    d["distinct_rows_per_head_count>=2"] = per_head_ge2
    d["distinct_rows_per_head_count>=5"] = per_head_ge5
    R = sum(per_head)
    d["distinct_rows_total"] = R
    d["rows_frac_of_table"] = R / TOTAL_ROWS
    for tag, Rk in (("all", R), ("count>=2", sum(per_head_ge2)), ("count>=5", sum(per_head_ge5))):
        d[f"overlay_bytes[{tag}]"] = dict(
            rows=Rk,
            fp32_values=Rk * ROW_W * 4, fp32_plus_int64_ids=Rk * (ROW_W * 4 + 8),
            bf16_values=Rk * ROW_W * 2, bf16_plus_int64_ids=Rk * (ROW_W * 2 + 8),
            int6g32_plus_ids=Rk * (130 + 8),
            sparseadam_fp32_state=Rk * (ROW_W * 4 * 3 + 8),       # delta + m + v (+id)
            rowwise_adagrad_fp32_state=Rk * (ROW_W * 4 + 4 + 8),  # delta + 1 accumulator (+id)
        )
    return d


RES["project_head"] = census(proj)
log("project census:", json.dumps({k: v for k, v in RES["project_head"].items() if not k.startswith("overlay")}))
for k, v in RES["project_head"].items():
    if k.startswith("overlay"):
        log(k, {a: (f"{b / 2**20:.1f} MiB" if a != "rows" else b) for a, b in v.items()})

# ---------------------------------------------------------------- 2. token classes on code
uniq_ids = np.unique(proj.ids)
dec = {int(i): tok.decode([int(i)]) for i in uniq_ids}


def tclass(s):
    if s.strip() == "":
        return "ws" if "\n" not in s else "nl"      # nl = newline(+indent) token
    if re.search(r"[A-Za-z0-9]", s):
        return "word" if s[:1] in (" ", "\t") else ("word_cont" if s[:1].isalnum() or s[:1] == "_" else "punct+word")
    return "punct"


cls_of = {i: tclass(s) for i, s in dec.items()}
cls_arr = np.array([cls_of[int(i)] for i in proj.ids])
RES["token_class_position_share"] = {k: v / proj.n for k, v in Counter(cls_arr.tolist()).most_common()}
log("token class shares (positions):", {k: round(v, 3) for k, v in RES["token_class_position_share"].items()})

# ---------------------------------------------------------------- 3. near/general corpora
def py_files(root: Path, cap_tokens: int, seed: int, exclude=("test", "idlelib", "tkinter", "lib2to3", "site-packages")):
    fs = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in exclude))
    random.Random(seed).shuffle(fs)
    out, n = [], 0
    for p in fs:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        e = np.asarray(tok.encode(t, add_special_tokens=False).ids, dtype=np.int64)
        out.append(e); n += e.size
        if n >= cap_tokens:
            break
    return out


near_std = Corpus("near_stdlib", py_files(STDLIB, 2_000_000, 0))
near_ml = Corpus("near_ml_transformers", py_files(TFM, 2_000_000, 0, exclude=("test",)))
import pyarrow.parquet as pq  # noqa: E402
wl = pq.read_table(WIKI).column("text").to_pylist()
wdocs = ["".join(wl[i:i + 200]) for i in range(0, len(wl), 200)]
gen = Corpus("general_wikitext2", encode_docs(wdocs))
log(f"near_stdlib tokens={near_std.n} near_ml tokens={near_ml.n} general tokens={gen.n}")


def overlap(P: Corpus, O: Corpus):
    """How much of project P's n-gram/row set also fires in corpus O, and how often O hits P's rows."""
    d = {}
    for order, kP, kO in ((2, P.k2, O.k2), (3, P.k3, O.k3)):
        uP, nP = np.unique(kP, return_counts=True)
        inO = np.isin(uP, np.unique(kO))
        d[f"proj_distinct_{order}gram_shared_frac"] = float(inO.mean())
        d[f"proj_{order}gram_occurrence_shared_frac"] = float(nP[inO].sum() / nP.sum())
        d[f"other_positions_exact_hit_{order}gram"] = float(np.isin(kO, uP).mean())
    hit_any, coll = [], []
    for h in range(NH):
        order = 2 if h < HPN else 3
        rP = np.unique(P.rows[:, h])
        hit = np.isin(O.rows[:, h], rP)
        exact = np.isin(O.k2 if order == 2 else O.k3, P.k2 if order == 2 else P.k3)
        hit_any.append(float(hit.mean())); coll.append(float((hit & ~exact).mean()))
    d["other_positions_row_hit_per_head"] = hit_any
    d["other_positions_collision_only_hit_per_head"] = coll
    allrows = np.zeros(O.n, bool)
    for h in range(NH):
        allrows |= np.isin(O.rows[:, h], np.unique(P.rows[:, h]))
    d["other_positions_hit_any_of_16_heads"] = float(allrows.mean())
    # rows of P that are also read by O (shared-row collateral surface)
    shared = sum(int(np.isin(np.unique(P.rows[:, h]), np.unique(O.rows[:, h])).sum()) for h in range(NH))
    d["proj_rows_also_read_by_other_frac"] = shared / RES["project_head"]["distinct_rows_total"]
    return d


for O in (near_std, near_ml, gen):
    RES[f"overlap_{O.name}"] = overlap(proj, O)
    log(f"overlap with {O.name}:", json.dumps({k: (v if not isinstance(v, list) else [round(x, 5) for x in v[::8]])
                                                 for k, v in RES[f'overlap_{O.name}'].items()}))

# project-specific n-grams (absent from all three other corpora)
other2 = np.unique(np.concatenate([near_std.k2, near_ml.k2, gen.k2]))
other3 = np.unique(np.concatenate([near_std.k3, near_ml.k3, gen.k3]))
spec2 = ~np.isin(proj.k2, other2)
spec3 = ~np.isin(proj.k3, other3)
RES["project_specific"] = dict(
    positions_with_project_specific_2gram=float(spec2.mean()),
    positions_with_project_specific_3gram=float(spec3.mean()),
    distinct_project_specific_2gram=int(np.unique(proj.k2[spec2]).size),
    distinct_project_specific_3gram=int(np.unique(proj.k3[spec3]).size),
)
# class of the 2-gram (x_{t-1}, x_t) at project-specific vs shared positions
prev_cls = np.concatenate([["bos"], cls_arr[:-1]])
pair = np.char.add(np.char.add(prev_cls.astype(str), ">"), cls_arr.astype(str))
for tag, m in (("all", np.ones_like(spec2)), ("project_specific_2gram", spec2), ("shared_2gram", ~spec2)):
    RES[f"2gram_class_pairs[{tag}]"] = {k: round(v / m.sum(), 4) for k, v in Counter(pair[m].tolist()).most_common(8)}
log("project-specific:", RES["project_specific"])
for tag in ("all", "project_specific_2gram", "shared_2gram"):
    log(f"2-gram class pairs [{tag}]:", RES[f"2gram_class_pairs[{tag}]"])


def show(k, order):
    ids = []
    for j in range(order):
        ids.append(int(k % V)); k //= V
    ids = ids[::-1]
    return "|".join("<eos>" if i == EOS else dec.get(i, tok.decode([i])) for i in ids)


u3, n3 = np.unique(proj.k3[spec3], return_counts=True)
top = np.argsort(-n3)[:25]
RES["top_project_specific_3grams"] = [(show(int(u3[i]), 3), int(n3[i])) for i in top]
u2, n2 = np.unique(proj.k2[spec2], return_counts=True)
top = np.argsort(-n2)[:25]
RES["top_project_specific_2grams"] = [(show(int(u2[i]), 2), int(n2[i])) for i in top]
log("top project-specific 3-grams:", RES["top_project_specific_3grams"][:15])
log("top project-specific 2-grams:", RES["top_project_specific_2grams"][:15])

# ---------------------------------------------------------------- 4. held-out splits and n-gram predictability
def split_by_file(seed=0, frac=0.15):
    idx = list(range(len(proj.docs)))
    random.Random(seed).shuffle(idx)
    nh = int(round(frac * len(idx)))
    return sorted(idx[nh:]), sorted(idx[:nh])


def sub(c: Corpus, sel, name):
    return Corpus(name, [c.docs[i] for i in sel])


def coverage(Tr: Corpus, Ho: Corpus, Ho_mask=None):
    m = np.ones(Ho.n, bool) if Ho_mask is None else Ho_mask
    d = {"heldout_positions": int(m.sum())}
    for order, kT, kH in ((2, Tr.k2, Ho.k2), (3, Tr.k3, Ho.k3)):
        uT, nT = np.unique(kT, return_counts=True)
        for kmin in (1, 2, 5):
            d[f"exact_{order}gram_seen_in_train(count>={kmin})"] = float(np.isin(kH[m], uT[nT >= kmin]).mean())
    rowhit = np.zeros(Ho.n, bool)
    for h in range(NH):
        rowhit |= np.isin(Ho.rows[:, h], np.unique(Tr.rows[:, h]))
    d["any_trained_row_read"] = float(rowhit[m].mean())
    coll = []
    for h in (0, 8):
        exact = np.isin(Ho.k2 if h < 8 else Ho.k3, Tr.k2 if h < 8 else Tr.k3)
        coll.append(float((np.isin(Ho.rows[:, h], np.unique(Tr.rows[:, h])) & ~exact)[m].mean()))
    d["collision_only_row_hit_head0_head8"] = coll
    # n-gram LM proxy: does the train-set argmax continuation of the context predict the next token?
    for order, kT, kH in ((2, Tr.k2, Ho.k2), (3, Tr.k3, Ho.k3)):
        valid = Tr.nxt >= 0
        ctx, nxt = kT[valid], Tr.nxt[valid]
        pairk = np.stack([ctx, nxt], 1)
        up, cnt = np.unique(pairk, axis=0, return_counts=True)
        # per context: argmax next, its count, total
        order_ = np.lexsort((-cnt, up[:, 0]))
        up, cnt = up[order_], cnt[order_]
        first = np.concatenate([[True], up[1:, 0] != up[:-1, 0]])
        ctx_u = up[first, 0]; arg = up[first, 1]
        tot = np.add.reduceat(cnt, np.flatnonzero(first)); top = cnt[first]
        hv = m & (Ho.nxt >= 0)
        pos = np.searchsorted(ctx_u, kH)
        pos = np.clip(pos, 0, ctx_u.size - 1)
        found = (ctx_u[pos] == kH) & hv
        conf = found & (tot[pos] >= 2) & (top[pos] / np.maximum(tot[pos], 1) >= 0.5)
        d[f"{order}gram_ctx_seen_frac"] = float(found[hv].mean())
        d[f"{order}gram_argmax_next_acc_when_seen"] = float((arg[pos] == Ho.nxt)[found].mean()) if found.any() else None
        d[f"{order}gram_confident_ctx_frac"] = float(conf[hv].mean())
        d[f"{order}gram_argmax_next_acc_when_confident"] = float((arg[pos] == Ho.nxt)[conf].mean()) if conf.any() else None
        # same, restricted to project-specific contexts (absent from near/general)
        if Ho is proj or True:
            spec = ~np.isin(kH, other2 if order == 2 else other3)
            f2 = found & spec
            d[f"{order}gram_projspecific_ctx_seen_frac"] = float(f2[hv].mean())
            d[f"{order}gram_projspecific_argmax_acc"] = float((arg[pos] == Ho.nxt)[f2].mean()) if f2.any() else None
    return d


tr_idx, ho_idx = split_by_file()
TR, HO = sub(proj, tr_idx, "train_byfile"), sub(proj, ho_idx, "heldout_byfile")
RES["split_by_file"] = dict(train_files=len(tr_idx), heldout_files=len(ho_idx), train_tokens=TR.n, heldout_tokens=HO.n,
                            train_census=census(TR), coverage=coverage(TR, HO))
log("by-file split:", json.dumps(RES["split_by_file"]["coverage"]))
RES["split_by_file"]["coverage_on_near_stdlib"] = coverage(TR, near_std)
RES["split_by_file"]["coverage_on_general"] = coverage(TR, gen)
log("by-file train rows on near_stdlib:", json.dumps(RES["split_by_file"]["coverage_on_near_stdlib"]))
log("by-file train rows on general:", json.dumps(RES["split_by_file"]["coverage_on_general"]))

# ---------------------------------------------------------------- 5. time split via git
old_files = tree_files(C_OLD)
old_texts = {f: blob(C_OLD, f) for f in old_files}
OLD = Corpus("old", encode_docs(list(old_texts.values())))
# added lines of HEAD relative to C_OLD, located inside HEAD files (real context)
added_docs, added_masks = [], []
for f, t in zip(files, texts):
    if f in old_texts and old_texts[f] == t:
        continue
    if f in old_texts:
        diff = git("diff", "-U0", C_OLD, HEAD, "--", f)
        add = set()
        for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
            s, n = int(m.group(1)), int(m.group(2) or 1)
            add.update(range(s, s + n))
    else:
        add = set(range(1, t.count("\n") + 2))
    if not add:
        continue
    enc = tok.encode(t, add_special_tokens=False)
    line_start = np.cumsum([0] + [len(x) + 1 for x in t.split("\n")])
    starts = np.array([o[0] for o in enc.offsets])
    line_of = np.searchsorted(line_start, starts, side="right")  # 1-based line number
    mask = np.isin(line_of, sorted(add))
    added_docs.append(np.asarray(enc.ids, dtype=np.int64)); added_masks.append(mask)
NEW = Corpus("head_changed_files", added_docs)
NEWMASK = np.concatenate([m for d, m in zip(added_docs, added_masks) if d.size])
RES["split_by_time"] = dict(c_old=C_OLD, head=HEAD, old_files=len(old_files), old_tokens=OLD.n,
                            changed_or_new_files=len(added_docs), added_positions=int(NEWMASK.sum()),
                            coverage=coverage(OLD, NEW, NEWMASK))
log("by-time split:", json.dumps({k: v for k, v in RES["split_by_time"].items() if k != "coverage"}))
log("by-time coverage:", json.dumps(RES["split_by_time"]["coverage"]))

# ---------------------------------------------------------------- 6. incremental: per-commit dirty rows
commits = git("rev-list", "--first-parent", "--reverse", f"{C_OLD}..{HEAD}").split()
cnt = np.zeros(TOTAL_ROWS, dtype=np.int32)
np.add.at(cnt, OLD.rows.reshape(-1), 1)
cache: dict = {}


def file_rows(commit, path):
    key = (commit, path)
    if key not in cache:
        t = blob(commit, path)
        e = np.asarray(tok.encode(t, add_special_tokens=False).ids, dtype=np.int64)
        cache[key] = rows_of(e)[0].reshape(-1) if e.size else np.zeros(0, np.int64)
    return cache[key]


per = []
union_dirty, union_new, union_dead = set(), set(), set()
prev = C_OLD
for c in commits:
    ns = git("diff", "--name-status", "--no-renames", prev, c, "--", "python").split("\n")
    dirty_parts, nfiles, tok_changed = [], 0, 0
    for line in ns:
        if not line.strip():
            continue
        st, path = line.split("\t")[0], line.split("\t")[-1]
        if not path.endswith(".py"):
            continue
        nfiles += 1
        ro = file_rows(prev, path) if st != "A" else np.zeros(0, np.int64)
        rn = file_rows(c, path) if st != "D" else np.zeros(0, np.int64)
        uo, co = np.unique(ro, return_counts=True)
        un, cn = np.unique(rn, return_counts=True)
        allu = np.union1d(uo, un)
        a = np.zeros(allu.size, np.int64); b = np.zeros(allu.size, np.int64)
        a[np.searchsorted(allu, uo)] = co; b[np.searchsorted(allu, un)] = cn
        ch = allu[a != b]
        before = cnt[allu].copy()
        cnt[allu] += (b - a).astype(np.int32)
        dirty_parts.append((ch, before[a != b], cnt[ch]))
        tok_changed += abs(int(rn.size - ro.size)) // NH
    if dirty_parts:
        ch = np.concatenate([p[0] for p in dirty_parts])
        bef = np.concatenate([p[1] for p in dirty_parts]); aft = np.concatenate([p[2] for p in dirty_parts])
        new = ch[(bef == 0) & (aft > 0)]; dead = ch[(bef > 0) & (aft == 0)]
        uch = np.unique(ch)
        per.append(dict(commit=c, files=nfiles, dirty_rows=int(uch.size), new_rows=int(np.unique(new).size),
                        dead_rows=int(np.unique(dead).size)))
        union_dirty.update(uch.tolist()); union_new.update(new.tolist()); union_dead.update(dead.tolist())
    prev = c
# check final counts equal HEAD census
head_cnt_rows = int((cnt > 0).sum())
dr = np.array([p["dirty_rows"] for p in per]); nr = np.array([p["new_rows"] for p in per]); dd = np.array([p["dead_rows"] for p in per])
RES["incremental"] = dict(
    commits=len(commits), commits_touching_py=len(per),
    dirty_rows_per_commit=dict(median=float(np.median(dr)), p90=float(np.percentile(dr, 90)), max=int(dr.max()), mean=float(dr.mean())),
    new_rows_per_commit=dict(median=float(np.median(nr)), p90=float(np.percentile(nr, 90)), max=int(nr.max())),
    dead_rows_per_commit=dict(median=float(np.median(dd)), p90=float(np.percentile(dd, 90)), max=int(dd.max())),
    files_per_commit_median=float(np.median([p["files"] for p in per])),
    union_dirty_rows_over_window=len(union_dirty), union_new_rows=len(union_new), union_dead_rows=len(union_dead),
    rows_live_at_head_from_replay=head_cnt_rows, rows_live_at_head_census=RES["project_head"]["distinct_rows_total"],
)
log("incremental:", json.dumps(RES["incremental"]))

OUT.write_text(json.dumps(RES, indent=1, default=str))
log("wrote", OUT)
