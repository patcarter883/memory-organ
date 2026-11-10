"""Skeptic re-run of experiment/ingest.py arm A (identical data, seeds, hyperparameters) + extra checks.

Stages: data (copied verbatim from ingest.py) -> leakage stats -> host hash -> train arm A (requires_grad audit)
-> host hash again -> save rows -> eval (graft ON, per-token) -> ablations -> decode parity w/ power check -> top rows.
"""
from __future__ import annotations

import glob, hashlib, json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/memory-organ/ngram-memory/proofs/graft")
from graft_proto import NGramGraft  # noqa: E402

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent
EXP = OUT.parent / "experiment"
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
STDLIB = "/usr/local/lib/python3.13"
STDLIB_FILES = ["argparse.py", "dataclasses.py", "functools.py", "typing.py", "json/decoder.py", "json/encoder.py",
                "asyncio/tasks.py", "asyncio/queues.py", "collections/__init__.py", "logging/__init__.py",
                "subprocess.py", "threading.py", "inspect.py", "enum.py", "contextlib.py", "shutil.py"]
EOS, V, T = 151643, 151936, 512
KEY = dict(vocab_size=V, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=200_000,
           make_ngram_vocab_size_divisible_by=128, seed=1234, eos_token_id=EOS, ple_layer_index=0)
TRAIN_CHUNKS, EPOCHS, HELD_CHUNKS, TE_CHUNKS, GEN_CHUNKS, LR_ROWS, LAM = 96, 2, 32, 12, 12, 1e-2, 1e-3
T0 = time.time()
LOG = open(OUT / "verify.log", "a")
R = json.loads((OUT / "verify.json").read_text()) if (OUT / "verify.json").exists() else {}


def L(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def save():
    (OUT / "verify.json").write_text(json.dumps(R, indent=1))


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402
hf_logging.set_verbosity_error()
tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]
random.Random(0).shuffle(shuf)
n_held = round(0.15 * len(files))
held_files, train_files = sorted(shuf[:n_held]), sorted(shuf[n_held:])
assert held_files == json.loads((EXP / "results.json").read_text())["data"]["heldout_files"]


def stream(paths, root, with_src=False):
    ids, src = [], []
    for i, p in enumerate(paths):
        t = tok(Path(root, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
        ids += t; src += [i] * len(t)
    a = np.asarray(ids, dtype=np.int64)
    return (a, np.asarray(src)) if with_src else a


def chunk(s):
    n = len(s) // T
    return s[: n * T].reshape(n, T)


def sample_idx(n, k, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))


tr_stream = stream(train_files, REPO)
ho_stream, ho_src = stream(held_files, REPO, with_src=True)
tr_all, ho_all = chunk(tr_stream), chunk(ho_stream)
tr_idx, ho_idx = sample_idx(len(tr_all), TRAIN_CHUNKS, 0), sample_idx(len(ho_all), HELD_CHUNKS, 1)
train_x, held_x = torch.from_numpy(tr_all[tr_idx]), torch.from_numpy(ho_all[ho_idx])
held_src = chunk(ho_src)[ho_idx]  # file index per held-out token
train_eval_x = train_x[torch.randperm(len(train_x), generator=torch.Generator().manual_seed(2))[:TE_CHUNKS]]
std_stream = stream([f for f in STDLIB_FILES if Path(STDLIB, f).exists()], STDLIB)
std_all = chunk(std_stream)
std_x = torch.from_numpy(std_all[sample_idx(len(std_all), GEN_CHUNKS, 3)])
from datasets import load_dataset  # noqa: E402
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wt_ids = np.asarray(tok("".join(wt["text"]), add_special_tokens=False).input_ids, dtype=np.int64)
wt_all = chunk(wt_ids)
wiki_x = torch.from_numpy(wt_all[sample_idx(len(wt_all), GEN_CHUNKS, 4)])
EVALSETS = dict(heldout=held_x, train=train_eval_x, stdlib_py=std_x, general_text=wiki_x)
L("data", {k: tuple(v.shape) for k, v in EVALSETS.items()})

# ------------------------------------------------------------------------------------------------ leakage: long spans
def span_set(x2d, n):
    s = set()
    for row in x2d:
        r = row.tolist()
        for i in range(len(r) - n + 1):
            s.add(tuple(r[i:i + n]))
    return s


if "leak" not in R:
    R["leak"] = {}
    for n in (8, 16, 32):
        S_used = span_set(train_x.numpy(), n)
        S_pool = span_set(tr_all, n)
        h = held_x.numpy()
        hit_used = np.zeros((h.shape[0], h.shape[1] - 1), bool); hit_pool = hit_used.copy()
        for b in range(h.shape[0]):
            r = h[b].tolist()
            for t in range(n - 1, len(r)):  # span r[t-n+1 .. t] ending at TARGET position t (target index t-1)
                sp = tuple(r[t - n + 1:t + 1])
                hit_used[b, t - 1] = sp in S_used; hit_pool[b, t - 1] = sp in S_pool
        R["leak"][f"{n}gram_target_span_in_train_used"] = float(hit_used.mean())
        R["leak"][f"{n}gram_target_span_in_train_pool"] = float(hit_pool.mean())
        np.save(OUT / f"leak_hit_used_{n}.npy", hit_used)
    # same stats for stdlib (control: unrelated code)
    S16 = span_set(train_x.numpy(), 16); s = std_x.numpy()
    R["leak"]["stdlib_16gram_target_span_in_train_used"] = float(np.mean(
        [tuple(s[b, t - 15:t + 1].tolist()) in S16 for b in range(len(s)) for t in range(15, T)]))
    L("leak", R["leak"]); save()

# --------------------------------------------------------------------------------------------- host + keys + cats
model = AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.float32)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
D = model.config.hidden_size
layers = model.model.layers


def host_hash():
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


_ref = NGramGraft(D, KEY, layer_idx=1, table="sparse", init="editor")
SHIFT = _ref.ple.ple_embedding._shift_right_ignore_eos


def keys(x):
    x0, x1, x2 = [SHIFT(x, k).numpy() for k in range(3)]
    return x1 * V + x0, (x2 * V + x1) * V + x0


tr_k2, tr_k3 = keys(train_x)
seen2, seen3 = set(tr_k2.ravel().tolist()), set(tr_k3.ravel().tolist())
seen4 = set(zip(tr_k3[:, :-1].ravel().tolist(), train_x[:, 1:].numpy().ravel().tolist()))
isin = lambda a, s: np.vectorize(s.__contains__, otypes=[bool])(a)


def categories(x):
    k2, k3 = keys(x)
    k2, k3, y = k2[:, :-1], k3[:, :-1], x[:, 1:].numpy()
    s2, s3 = isin(k2, seen2), isin(k3, seen3)
    s4 = np.array([(a, b) in seen4 for a, b in zip(k3.ravel().tolist(), y.ravel().tolist())]).reshape(k3.shape)
    c = np.zeros(k2.shape, dtype=np.int8); c[s2] = 1; c[s3] = 2; c[s3 & s4] = 3
    return c


CATS = {k: categories(v) for k, v in EVALSETS.items()}
TRAIN_ROW_IDS = torch.unique(_ref.row_ids(train_x))
N_TOTAL = _ref.ple.ple_embedding.ngram_embedding.num_embeddings

# key <-> row-id consistency: identical exact keys must give identical rows (the seen/unseen split is only valid if so)
if "key_consistency" not in R:
    kc = {}
    for name, x in EVALSETS.items():
        ids = _ref.row_ids(x).numpy()
        k2, k3 = keys(x)
        for order, kk, hs in ((2, k2, slice(0, 8)), (3, k3, slice(8, 16))):
            d = {}
            bad = 0
            for key, r in zip(kk.ravel().tolist(), map(tuple, ids[..., hs].reshape(-1, 8).tolist())):
                if d.setdefault(key, r) != r:
                    bad += 1
            kc[f"{name}_{order}gram_key_to_row_mismatch"] = bad
        # every seen-2gram position must hit only trainable rows
        hit = np.isin(ids, TRAIN_ROW_IDS.numpy())
        s2 = isin(k2, seen2); s3 = isin(k3, seen3)
        kc[f"{name}_seen2_positions_all_rows_trainable"] = bool(hit[..., :8][s2].all()) if s2.any() else None
        kc[f"{name}_seen3_positions_all_rows_trainable"] = bool(hit[..., 8:][s3].all()) if s3.any() else None
    R["key_consistency"] = kc; L("key consistency", kc); save()


@torch.no_grad()
def nll_of(x):
    out = []
    for i in range(len(x)):
        xb = x[i:i + 1]
        lg = model(input_ids=xb, use_cache=False).logits
        out.append(F.cross_entropy(lg[:, :-1].flatten(0, 1), xb[:, 1:].flatten(), reduction="none").view(1, -1))
    return torch.cat(out).numpy()


class CompactRows(nn.Module):  # verbatim semantics of ingest.py CompactRows + optional per-(pos,head) mask
    def __init__(self, n_total, w, row_ids):
        super().__init__()
        self.num_embeddings, self.embedding_dim = n_total, w
        self.register_buffer("row_ids", row_ids)
        self.emb = nn.Embedding(len(row_ids), w, sparse=True)
        with torch.no_grad():
            self.emb.weight.zero_()
        self.mask = None

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        n = self.row_ids.numel()
        pos = torch.searchsorted(self.row_ids, ids).clamp_max(n - 1)
        hit = (self.row_ids[pos] == ids).unsqueeze(-1).to(self.emb.weight.dtype)
        out = self.emb(pos) * hit
        if self.mask is not None:
            out = out * self.mask.unsqueeze(-1).to(out.dtype)
        return out


def make_graft():
    g = NGramGraft(D, KEY, layer_idx=1, table="sparse", init="editor", value_seed=0)
    emb = g.ple.ple_embedding
    emb.ngram_embedding = CompactRows(N_TOTAL, emb.ngram_embedding.embedding_dim, TRAIN_ROW_IDS)
    for p in g.parameters():
        p.requires_grad_(False)
    emb.ngram_embedding.emb.weight.requires_grad_(True)
    return g


# --------------------------------------------------------------------------------------------- base (sanity vs npz)
base_npz = dict(np.load(EXP / "base_raw_full.npz"))
if not (OUT / "base_verify.npz").exists():
    b = {k: nll_of(v) for k, v in EVALSETS.items()}
    np.savez(OUT / "base_verify.npz", **b)
base = dict(np.load(OUT / "base_verify.npz"))
R["base_matches_original_npz_maxabs"] = {k: float(np.abs(base[k] - base_npz[k]).max()) for k in base}
L("base vs original npz", R["base_matches_original_npz_maxabs"]); save()

# --------------------------------------------------------------------------------------------- train arm A (exact)
ROWS = OUT / "rows_armA.pt"
g = make_graft().install(model)
emb = g.ple.ple_embedding
W = emb.ngram_embedding.weight
if not ROWS.exists():
    h0 = host_hash()
    tr_params = [(n, p.numel()) for n, p in list(model.named_parameters()) + [("graft." + n, p) for n, p in g.named_parameters()]
                 if p.requires_grad]
    R["requires_grad_params"] = dict(names=[n for n, _ in tr_params], numel=sum(c for _, c in tr_params),
                                     host_params_requiring_grad=sum(p.requires_grad for p in model.parameters()))
    L("requires_grad", R["requires_grad_params"])
    graft_init = {n: p.detach().clone() for n, p in g.named_parameters() if n != "ple.ple_embedding.ngram_embedding.emb.weight"}
    stash = {}
    hh = emb.register_forward_hook(lambda m, i, o: stash.__setitem__("e", o))
    opt = torch.optim.SparseAdam([W], lr=LR_ROWS)
    rng = np.random.default_rng(10)
    st, run, curve, grad_seen = [], [], [], set()
    for ep in range(EPOCHS):
        for j in rng.permutation(len(train_x)):
            t = time.time()
            x = train_x[j:j + 1]
            lg = model(input_ids=x, use_cache=False).logits
            nll = F.cross_entropy(lg[0, :-1], x[0, 1:])
            pen = LAM * stash["e"].pow(2).sum(-1).mean()
            (nll + pen).backward()
            if not grad_seen:
                grad_seen = {n for n, p in list(model.named_parameters()) + list(g.named_parameters()) if p.grad is not None}
            opt.step(); opt.zero_grad(set_to_none=True)
            st.append(time.time() - t); run.append(float(nll))
            if len(run) == 16:
                curve.append(float(np.mean(run))); run = []
                L(f"step {len(st)} nll16 {curve[-1]:.4f} {np.mean(st[-16:]):.2f}s/step")
    hh.remove()
    W.requires_grad_(False)
    R["params_with_grad_after_step1"] = sorted(grad_seen)
    R["train_curve"] = curve
    R["step_time_s"] = dict(median=float(np.median(st)), mean=float(np.mean(st)))
    h1 = host_hash()
    R["host_sha256_before"], R["host_sha256_after"], R["host_bit_identical"] = h0, h1, h0 == h1
    R["graft_nonrow_params_unchanged"] = all(torch.equal(graft_init[n], p) for n, p in g.named_parameters() if n in graft_init)
    torch.save(W.detach().clone(), ROWS)
    L("host bit identical", h0 == h1, "graft non-row unchanged", R["graft_nonrow_params_unchanged"]); save()
with torch.no_grad():
    W.copy_(torch.load(ROWS)); W.requires_grad_(False)
W0 = W.detach().clone()
TAB = emb.ngram_embedding


def summ(nll, cats, b):
    d = dict(mean=float(nll.mean() - b.mean()))
    for i, n in enumerate(["unseen_2gram", "seen_2gram_only", "seen_3gram_new_next", "seen_3gram_seen_next"]):
        m = cats == i
        d[n] = float(nll[m].mean() - b[m].mean()) if m.any() else None
        d[n + "_contrib"] = float((nll[m] - b[m]).sum() / nll.size) if m.any() else None
    return d


# -------------------------------------------------------------------------------------------- eval: full graft ON
if "full" not in R:
    R["full"] = {}
    for k, x in EVALSETS.items():
        nl = nll_of(x)
        np.save(OUT / f"nll_full_{k}.npy", nl)
        R["full"][k] = summ(nl, CATS[k], base[k])
    L("full graft deltas", json.dumps(R["full"])); save()

# graft really ON during eval? remove and compare
g.remove()
R["graft_removed_heldout2_maxabs_vs_base"] = float(np.abs(nll_of(held_x[:2]) - base["heldout"][:2]).max())
g.install(model)
L("removed graft vs base", R["graft_removed_heldout2_maxabs_vs_base"])


# -------------------------------------------------------------------------------------------- ablations (heldout+stdlib)
def exact_masks(x):
    k2, k3 = keys(x)
    m = np.zeros(k2.shape + (16,), bool)
    m[..., :8] = isin(k2, seen2)[..., None]
    m[..., 8:] = isin(k3, seen3)[..., None]
    return torch.from_numpy(m)


def nll_masked(x, mask_fn):
    out = []
    for i in range(len(x)):
        TAB.mask = mask_fn(x[i:i + 1])
        out.append(nll_of(x[i:i + 1]))
    TAB.mask = None
    return np.concatenate(out)


if "ablate" not in R:
    R["ablate"] = {}
    for k in ("heldout", "stdlib_py"):
        x = EVALSETS[k]
        a = {}
        nl = nll_masked(x, exact_masks); a["exact_key_rows_only (no collisions)"] = summ(nl, CATS[k], base[k])
        np.save(OUT / f"nll_exact_{k}.npy", nl)
        nl = nll_masked(x, lambda xb: ~exact_masks(xb)); a["collision_rows_only"] = summ(nl, CATS[k], base[k])
        R["ablate"][k] = a
        L("ablate", k, json.dumps(a)); save()

# constant-bias control: mean injected graft output over the 96 training chunks, added at every position
if "const_bias" not in R:
    acc, n = torch.zeros(D), 0
    hk = layers[1].register_forward_pre_hook(lambda m, a, kw: None, with_kwargs=True)
    hk.remove()
    with torch.no_grad():
        for i in range(len(train_x)):
            model(input_ids=train_x[i:i + 1], use_cache=False)
            o = g.debug_last["out"][0]
            acc += o.sum(0); n += o.shape[0]
    mean_out = acc / n
    g.remove()
    hb = layers[1].register_forward_pre_hook(
        lambda m, a, kw: ((a[0] + mean_out,) + tuple(a[1:]), kw) if a else (a, {**kw, "hidden_states": kw["hidden_states"] + mean_out}),
        with_kwargs=True)
    R["const_bias"] = {"mean_out_norm": float(mean_out.norm())}
    for k in ("heldout", "stdlib_py", "general_text"):
        R["const_bias"][k] = summ(nll_of(EVALSETS[k]), CATS[k], base[k])
    hb.remove(); g.install(model)
    L("const bias", json.dumps(R["const_bias"])); save()

# frequent-row ablation: zero rows of the top-K most frequent train 2-grams and 3-grams (all 8 heads each)
ids_tr = _ref.row_ids(train_x)
if "freq_ablate" not in R:
    R["freq_ablate"] = {}
    c2, c3 = Counter(tr_k2.ravel().tolist()), Counter(tr_k3.ravel().tolist())
    k2f, k3f = tr_k2.ravel(), tr_k3.ravel()
    ids_flat = ids_tr.reshape(-1, 16)
    for K in (100, 1000):
        top2 = {k for k, _ in c2.most_common(K)}; top3 = {k for k, _ in c3.most_common(K)}
        rows = torch.cat([ids_flat[torch.from_numpy(isin(k2f, top2))][:, :8].reshape(-1),
                          ids_flat[torch.from_numpy(isin(k3f, top3))][:, 8:].reshape(-1)]).unique()
        cover2 = sum(c for k, c in c2.most_common(K)) / len(k2f)
        pos = torch.searchsorted(TRAIN_ROW_IDS, rows)
        with torch.no_grad():
            W.copy_(W0); W[pos] = 0
        r = dict(rows_zeroed=int(rows.numel()), train_2gram_token_coverage=cover2)
        for k in ("heldout", "stdlib_py"):
            r[k] = summ(nll_of(EVALSETS[k]), CATS[k], base[k])
        R["freq_ablate"][f"top{K}"] = r
        L("freq ablate", K, json.dumps(r)); save()
    with torch.no_grad():
        W.copy_(W0)

# -------------------------------------------------------------------------------------------- decode parity + power
def decode(ids, P, break_state=False):
    full = model(input_ids=ids, use_cache=False).logits
    out = model(input_ids=ids[:, :P], use_cache=True)
    d = [(out.logits - full[:, :P]).abs().max().item()]
    cache = out.past_key_values
    for t in range(P, ids.shape[1]):
        if break_state:
            cache.__dict__.pop(g._side_key, None)  # drop the n-gram side state -> history forgotten
        o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        d.append((o.logits[:, -1] - full[:, t]).abs().max().item())
    return max(d)


if "decode" not in R:
    with torch.no_grad():
        R["decode"] = dict(parity_heldout=decode(held_x[:1, :40], 20), parity_train=decode(train_x[:1, :40], 20),
                           broken_state_heldout=decode(held_x[:1, :40], 20, True),
                           broken_state_train=decode(train_x[:1, :40], 20, True))
    L("decode", R["decode"]); save()

# -------------------------------------------------------------------------------------------- top rows by norm
if "top_rows" not in R:
    rn = W0.norm(dim=-1)
    rowmap = defaultdict(Counter)
    flat = ids_tr.reshape(-1, 16).numpy()
    x0, x1, x2 = [SHIFT(train_x, k).numpy().ravel() for k in range(3)]
    for p in range(flat.shape[0]):
        for h in range(16):
            ng = (x1[p], x0[p]) if h < 8 else (x2[p], x1[p], x0[p])
            rowmap[int(flat[p, h])][ng] += 1
    top = torch.argsort(rn, descending=True)[:25]
    rows = []
    for i in top.tolist():
        rid = int(TRAIN_ROW_IDS[i]); c = rowmap[rid]
        rows.append(dict(row=rid, head=int(np.searchsorted(_ref.ple.ple_embedding.ngram_heads_offsets.numpy(), rid, "right") - 1),
                         norm=float(rn[i]), n_distinct_ngrams=len(c),
                         ngrams=[(repr(tok.decode([t for t in ng])), n) for ng, n in c.most_common(3)]))
    # row-norm vs n-gram train frequency; and common-direction test
    counts = np.array([sum(rowmap[int(r)].values()) for r in TRAIN_ROW_IDS.tolist()])
    nz = rn.numpy() > 0
    rho = float(np.corrcoef(np.log(counts[nz]), rn.numpy()[nz])[0, 1])
    # per head: ||usage-weighted mean row|| / mean ||row||  (1 = all rows the same vector = global bias)
    heads = np.searchsorted(_ref.ple.ple_embedding.ngram_heads_offsets.numpy(), TRAIN_ROW_IDS.numpy(), "right") - 1
    coh = []
    Wn = W0.numpy()
    for h in range(16):
        m = heads == h
        w = counts[m][:, None]
        coh.append(float(np.linalg.norm((Wn[m] * w).sum(0) / w.sum()) / (np.linalg.norm(Wn[m], axis=1) * w[:, 0]).sum() * w.sum()))
    R["top_rows"] = dict(rows=rows, corr_log_count_vs_norm=rho, head_mean_direction_coherence=coh,
                         norm_quantiles={q: float(np.quantile(rn.numpy()[nz], q)) for q in (0.5, 0.9, 0.99, 0.999)})
    L("top rows", json.dumps(R["top_rows"])[:3000]); save()

# per-file breakdown of heldout gain + gain on tokens whose 16-gram span was seen in train_used
if "per_file" not in R:
    nl = np.load(OUT / "nll_full_heldout.npy"); d = nl - base["heldout"]
    src = held_src[:, 1:]
    pf = {}
    for i in np.unique(src):
        m = src == i
        pf[held_files[i]] = dict(tokens=int(m.sum()), mean_delta=float(d[m].mean()), contrib=float(d[m].sum() / d.size))
    R["per_file"] = dict(sorted(pf.items(), key=lambda kv: kv[1]["contrib"]))
    for n in (8, 16, 32):
        hu = np.load(OUT / f"leak_hit_used_{n}.npy")
        R["leak"][f"delta_on_{n}gram_span_seen"] = float(d[hu].mean()) if hu.any() else None
        R["leak"][f"contrib_{n}gram_span_seen"] = float(d[hu].sum() / d.size)
    L("per file", json.dumps(R["per_file"])[:2500]); L("leak", R["leak"]); save()
L("done")
