"""Stage-1 ingestion through a GRAFT: first real test (CPU, frozen Qwen2.5-0.5B).

Host  : frozen Qwen2.5-0.5B (fp32, CPU).
Graft : Flash-Next-format n-gram module = transformers' Qwen4ExpTextPLELayer built by ngram-memory's
        proofs/graft/graft_proto.py::NGramGraft (hashed 2/3-grams, 8 heads per order, prime per-head sizes,
        hc_count=1, editor init per DESIGN.md §2.4: table=0, W_k=0 (gate==0.5), conv1d=0, norms 0, W_v seeded orthogonal),
        installed at the INPUT of decoder block 1 (as Flash-Next). KeySpec scaled down: V=151936 (host vocab),
        eos=151643, seed 1234, base 200_000 -> 16 primes 200003..., row width d/16 = 56.
Corpus: owner's minisgl python package (*.py), split BY FILE 85/15 with seed 0.
Arms  : A = rows only (nn.Embedding(sparse=True) + SparseAdam, L2-on-looked-up-embeddings penalty)
        B = rows + value_proj (Engram-Adapter-style)
        C = LoRA r=8 on q/v of all layers (reference), graft removed.
Usage : python ingest.py [--smoke]
"""
from __future__ import annotations

import argparse, glob, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/memory-organ/ngram-memory/proofs/graft")
from graft_proto import NGramGraft  # noqa: E402

torch.set_num_threads(3)
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

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--train_chunks", type=int, default=160)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--heldout_chunks", type=int, default=40)
ap.add_argument("--train_eval_chunks", type=int, default=16)
ap.add_argument("--gen_chunks", type=int, default=16)
ap.add_argument("--arms", default="A,B,C")
ap.add_argument("--lr_rows", type=float, default=1e-2)
ap.add_argument("--lr_v", type=float, default=5e-5)
ap.add_argument("--lr_lora", type=float, default=2e-4)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--budget_min", type=float, default=85.0, help="skip arm C if elapsed exceeds budget - est")
args = ap.parse_args()
if args.smoke:
    args.train_chunks, args.epochs, args.heldout_chunks, args.train_eval_chunks, args.gen_chunks = 4, 1, 2, 2, 2
tag = "smoke" if args.smoke else "full"
LOG = open(OUT / f"ingest_{tag}.log", "w")
RES_PATH = OUT / ("results_smoke.json" if args.smoke else "results.json")
T0 = time.time()


def L(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


R = dict(args=vars(args), key_spec=KEY, host=HOST, corpus=REPO, torch=torch.__version__)


def save():
    RES_PATH.write_text(json.dumps(R, indent=1))


# ------------------------------------------------------------------------------------------------------------ corpus
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402
hf_logging.set_verbosity_error()

tok = AutoTokenizer.from_pretrained(HOST)
files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
shuf = files[:]
random.Random(0).shuffle(shuf)
n_held = round(0.15 * len(files))
held_files, train_files = sorted(shuf[:n_held]), sorted(shuf[n_held:])


def stream(paths, root):
    ids = []
    for p in paths:
        ids += tok(Path(root, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
    return np.asarray(ids, dtype=np.int64)


def chunk(s):
    n = len(s) // T
    return s[: n * T].reshape(n, T)


def sample(ch, k, seed):
    idx = np.sort(np.random.default_rng(seed).choice(len(ch), size=min(k, len(ch)), replace=False))
    return torch.from_numpy(ch[idx])


tr_stream, ho_stream = stream(train_files, REPO), stream(held_files, REPO)
tr_all, ho_all = chunk(tr_stream), chunk(ho_stream)
train_x = sample(tr_all, args.train_chunks, 0)
held_x = sample(ho_all, args.heldout_chunks, 1)
train_eval_x = train_x[torch.randperm(len(train_x), generator=torch.Generator().manual_seed(2))[: args.train_eval_chunks]]
std_stream = stream([f for f in STDLIB_FILES if Path(STDLIB, f).exists()], STDLIB)
std_x = sample(chunk(std_stream), args.gen_chunks, 3)
try:
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_ids = np.asarray(tok("".join(wt["text"]), add_special_tokens=False).input_ids, dtype=np.int64)
    gen_src = "wikitext-2-raw-v1 test"
except Exception as e:  # fallback: repo markdown
    L("wikitext unavailable:", repr(e))
    wt_ids = stream(sorted(os.path.relpath(f, "/home/user/memory-organ/docs")
                           for f in glob.glob("/home/user/memory-organ/docs/**/*.md", recursive=True)),
                    "/home/user/memory-organ/docs")
    gen_src = "memory-organ/docs markdown"
wiki_x = sample(chunk(wt_ids), args.gen_chunks, 4)
R["data"] = dict(n_files=len(files), n_train_files=len(train_files), n_heldout_files=len(held_files),
                 heldout_files=held_files, train_files=train_files,
                 train_pool_tokens=int(len(tr_stream)), heldout_pool_tokens=int(len(ho_stream)),
                 train_tokens_used=int(train_x.numel()), heldout_eval_tokens=int(held_x.numel()),
                 train_eval_tokens=int(train_eval_x.numel()), stdlib_eval_tokens=int(std_x.numel()),
                 general_eval_tokens=int(wiki_x.numel()), general_source=gen_src,
                 stdlib_files=[f for f in STDLIB_FILES if Path(STDLIB, f).exists()], chunk_len=T)
L("files", len(files), "train", len(train_files), "held", len(held_files), "train pool tok", len(tr_stream),
  "held pool tok", len(ho_stream), "train used", train_x.numel())
save()

# ------------------------------------------------------------------------------------------------------------ host
model = AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.float32)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
D = model.config.hidden_size
layers = model.model.layers


# --------------------------------------------------------------------------------------------- exact n-gram key sets
def shifts(x):
    """x0,x1,x2 histories per q4 _shift_right_ignore_eos semantics (start of chunk == start of sequence)."""
    return [SHIFT(x, k).numpy() for k in range(3)]


_ref_graft = NGramGraft(D, KEY, layer_idx=1, table="sparse", init="editor")
SHIFT = _ref_graft.ple.ple_embedding._shift_right_ignore_eos


def keys(x):
    x0, x1, x2 = shifts(x)
    k2 = x1 * V + x0
    k3 = (x2 * V + x1) * V + x0
    return k2, k3


tr_k2, tr_k3 = keys(train_x)
seen2, seen3 = set(tr_k2.ravel().tolist()), set(tr_k3.ravel().tolist())
seen4 = set(zip(tr_k3[:, :-1].ravel().tolist(), train_x[:, 1:].numpy().ravel().tolist()))
R["data"]["train_unique_2grams"], R["data"]["train_unique_3grams"] = len(seen2), len(seen3)


def categories(x):
    """per predicted token (input pos t -> target t+1): 0 unseen-2gram, 1 seen2 only, 2 seen3 (not cont.), 3 seen3+cont."""
    k2, k3 = keys(x)
    k2, k3, y = k2[:, :-1], k3[:, :-1], x[:, 1:].numpy()
    s2 = np.vectorize(seen2.__contains__, otypes=[bool])(k2)
    s3 = np.vectorize(seen3.__contains__, otypes=[bool])(k3)
    s4 = np.array([(a, b) in seen4 for a, b in zip(k3.ravel().tolist(), y.ravel().tolist())]).reshape(k3.shape)
    c = np.zeros(k2.shape, dtype=np.int8)
    c[s2] = 1
    c[s3] = 2
    c[s3 & s4] = 3
    return c


CAT_NAMES = ["unseen_2gram", "seen_2gram_only", "seen_3gram_new_next", "seen_3gram_seen_next"]
EVALSETS = dict(heldout=held_x, train=train_eval_x, stdlib_py=std_x, general_text=wiki_x)
CATS = {k: categories(v) for k, v in EVALSETS.items()}
R["category_fractions"] = {k: {n: float((c == i).mean()) for i, n in enumerate(CAT_NAMES)} for k, c in CATS.items()}
L("category fractions", json.dumps(R["category_fractions"]))


# ------------------------------------------------------------------------------------------------------------ eval
class InjMeter:
    """pre-hook on layers[1] registered AFTER the graft: sees hidden+out; ratio ||out|| / ||hidden|| per token."""

    def __init__(self, graft):
        self.graft, self.r = graft, []
        self.h = layers[1].register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, mod, a, kw):
        hid = a[0] if a else kw["hidden_states"]
        out = self.graft.debug_last["out"]
        base = (hid - out).norm(dim=-1)
        self.r.append((out.norm(dim=-1) / base)[:, 1:].reshape(-1).detach())

    def done(self):
        self.h.remove()
        r = torch.cat(self.r) if self.r else torch.zeros(1)
        return dict(mean=float(r.mean()), median=float(r.median()), p99=float(r.quantile(0.99)), max=float(r.max()))


@torch.no_grad()
def nll_of(x, bs=2):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        lg = model(input_ids=xb, use_cache=False).logits
        out.append(F.cross_entropy(lg[:, :-1].flatten(0, 1).float(), xb[:, 1:].flatten(), reduction="none")
                   .view(len(xb), -1))
    return torch.cat(out).numpy()


def summarise(nll, cats, base=None):
    s = dict(mean=float(nll.mean()))
    for i, n in enumerate(CAT_NAMES):
        m = cats == i
        s[n] = float(nll[m].mean()) if m.any() else None
    m = cats >= 1
    s["seen_2gram_any"] = float(nll[m].mean()) if m.any() else None
    s["unseen_2gram"] = s["unseen_2gram"]
    if base is not None:
        s["delta"] = {k: (None if s[k] is None or base[k] is None else s[k] - base[k])
                      for k in ["mean", "seen_2gram_any"] + CAT_NAMES}
    return s


def eval_all(label, graft=None, touched=None, base=None):
    t = time.time()
    res, raw = {}, {}
    for k, x in EVALSETS.items():
        meter = InjMeter(graft) if graft is not None else None
        nll = nll_of(x)
        raw[k] = nll
        res[k] = summarise(nll, CATS[k], None if base is None else base["summary"][k])
        if meter is not None:
            res[k]["inj_ratio"] = meter.done()
        if touched is not None:  # collision view: fraction of the 16 hashed rows at each input position that were trained
            ids = graft.row_ids(x)[:, :-1]  # (B, T-1, 16)
            hit = touched[ids]
            c = CATS[k]
            res[k]["rows_hit_frac_2gram_heads"] = float(hit[..., :8].float().mean())
            res[k]["rows_hit_frac_3gram_heads"] = float(hit[..., 8:].float().mean())
            unseen = torch.from_numpy(c == 0)
            res[k]["collision_hit_frac_2gram_heads_at_unseen_2gram"] = float(hit[..., :8][unseen].float().mean()) \
                if unseen.any() else None
            new3 = torch.from_numpy(c <= 1)
            res[k]["collision_hit_frac_3gram_heads_at_unseen_3gram"] = float(hit[..., 8:][new3].float().mean()) \
                if new3.any() else None
    L(f"eval {label} ({time.time() - t:.0f}s):", json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                              for kk, vv in v.items() if kk != "delta"}
                                                          for k, v in res.items()}))
    return dict(summary=res), raw


def decode_parity(ids, P):
    full = model(input_ids=ids, use_cache=False).logits
    out = model(input_ids=ids[:, :P], use_cache=True)
    d = [(out.logits - full[:, :P]).abs().max().item()]
    cache = out.past_key_values
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        d.append((o.logits[:, -1] - full[:, t]).abs().max().item())
    return max(d)


# ------------------------------------------------------------------------------------------------------------ graft
def make_graft():
    g = NGramGraft(D, KEY, layer_idx=1, table="dense", init="editor", value_seed=0)
    emb = g.ple.ple_embedding
    n, w = emb.ngram_embedding.num_embeddings, emb.ngram_embedding.embedding_dim
    emb.ngram_embedding = nn.Embedding(n, w, sparse=True)
    with torch.no_grad():
        emb.ngram_embedding.weight.zero_()
    for p in g.parameters():
        p.requires_grad_(False)
    emb.ngram_embedding.weight.requires_grad_(True)
    return g


def overlay_bytes(n_rows, w, extra_params=0):
    return dict(rows=int(n_rows), fp32_rows_plus_int64_ids=int(n_rows * (4 * w + 8) + 4 * extra_params),
                bf16_rows_plus_int64_ids=int(n_rows * (2 * w + 8) + 2 * extra_params),
                int6_g32_est=int(n_rows * (math.ceil(6 * w / 8) + 2 * math.ceil(w / 32) + 8) + 2 * extra_params))


# ------------------------------------------------------------------------------------------------------------ base
base, base_raw = eval_all("base")
R["base"] = base
save()
hs = []
hk = layers[1].register_forward_pre_hook(lambda m, a, kw: hs.append((a[0] if a else kw["hidden_states"])[0, 1:].detach()),
                                         with_kwargs=True)
with torch.no_grad():
    model(input_ids=held_x[:1], use_cache=False)
hk.remove()
h = hs[0]
R["host_layer1_input"] = dict(token_norm_median=float(h.norm(dim=-1).median()), elem_std=float(h.std()),
                              elem_abs_median=float(h.abs().median()))
L("layer-1 input hidden:", R["host_layer1_input"])

# no-op check
g = make_graft().install(model)
with torch.no_grad():
    a = model(input_ids=held_x[:2], use_cache=False).logits
g.remove()
with torch.no_grad():
    b = model(input_ids=held_x[:2], use_cache=False).logits
R["noop_zero_table"] = dict(max_abs_dlogit=float((a - b).abs().max()), bitwise_equal=bool(torch.equal(a, b)))
L("no-op:", R["noop_zero_table"])
with torch.no_grad():
    R["host_decode_parity_no_graft"] = decode_parity(held_x[:1, :40], 20)
del g, a, b
save()


def train_graft(arm):
    g = make_graft().install(model)
    emb = g.ple.ple_embedding
    W = emb.ngram_embedding.weight
    stash = {}
    hh = emb.register_forward_hook(lambda m, i, o: stash.__setitem__("e", o))
    opts = [torch.optim.SparseAdam([W], lr=args.lr_rows)]
    if arm == "B":
        g.ple.value_proj.weight.requires_grad_(True)
        opts.append(torch.optim.Adam([g.ple.value_proj.weight], lr=args.lr_v))
    rng = np.random.default_rng(10)
    st, curve, run = [], [], []
    for ep in range(args.epochs):
        for j in rng.permutation(len(train_x)):
            t = time.time()
            x = train_x[j:j + 1]
            lg = model(input_ids=x, use_cache=False).logits
            nll = F.cross_entropy(lg[0, :-1], x[0, 1:])
            pen = args.lam * stash["e"].pow(2).sum(-1).mean()
            (nll + pen).backward()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
            st.append(time.time() - t)
            run.append(float(nll.detach()))
            if len(run) == 16:
                curve.append(float(np.mean(run))); run = []
                L(f"arm {arm} ep {ep} step {len(st)} nll16 {curve[-1]:.4f} pen {float(pen):.2e} step {np.mean(st[-16:]):.2f}s")
    hh.remove()
    for p in g.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        touched = (W != 0).any(-1)
    n_rows = int(touched.sum())
    rn = W[touched].norm(dim=-1)
    info = dict(train_curve_nll_per16=curve, step_time_s=dict(mean=float(np.mean(st)), median=float(np.median(st))),
                train_steps=len(st), train_tokens_seen=len(st) * T,
                rows_touched=n_rows, rows_touched_frac_of_table=n_rows / W.shape[0],
                row_norm=dict(mean=float(rn.mean()), median=float(rn.median()), max=float(rn.max())),
                overlay=overlay_bytes(n_rows, W.shape[1], extra_params=(D * D if arm == "B" else 0)))
    if arm == "B":
        Wv0 = make_graft().ple.value_proj.weight
        info["value_proj_rel_change"] = float((g.ple.value_proj.weight - Wv0).norm() / Wv0.norm())
    return g, touched, info


class LoRALinear(nn.Module):
    def __init__(self, base_lin, r=8, alpha=16):
        super().__init__()
        self.base = base_lin
        self.A = nn.Parameter(torch.empty(r, base_lin.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base_lin.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.s


def train_lora():
    orig = []
    for lyr in layers:
        a = lyr.self_attn
        orig.append((a.q_proj, a.v_proj))
        a.q_proj, a.v_proj = LoRALinear(a.q_proj), LoRALinear(a.v_proj)
    params = [p for lyr in layers for m in (lyr.self_attn.q_proj, lyr.self_attn.v_proj) for p in (m.A, m.B)]
    opt = torch.optim.Adam(params, lr=args.lr_lora)
    rng = np.random.default_rng(10)
    st, curve, run = [], [], []
    for ep in range(args.epochs):
        for j in rng.permutation(len(train_x)):
            t = time.time()
            x = train_x[j:j + 1]
            lg = model(input_ids=x, use_cache=False).logits
            nll = F.cross_entropy(lg[0, :-1], x[0, 1:])
            nll.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            st.append(time.time() - t)
            run.append(float(nll.detach()))
            if len(run) == 16:
                curve.append(float(np.mean(run))); run = []
                L(f"arm C ep {ep} step {len(st)} nll16 {curve[-1]:.4f} step {np.mean(st[-16:]):.2f}s")
    for p in params:
        p.requires_grad_(False)
    n_par = sum(p.numel() for p in params)
    info = dict(train_curve_nll_per16=curve, step_time_s=dict(mean=float(np.mean(st)), median=float(np.median(st))),
                train_steps=len(st), lora_params=n_par, overlay=dict(fp32=4 * n_par, bf16=2 * n_par))
    return orig, info


# ------------------------------------------------------------------------------------------------------------ arms
arm_time = {}
for arm in args.arms.split(","):
    ta = time.time()
    if arm == "C":
        est = np.mean(list(arm_time.values())) * 1.3 if arm_time else 0
        if not args.smoke and (time.time() - T0 + est) / 60 > args.budget_min:
            L(f"skip arm C: elapsed {(time.time() - T0) / 60:.1f} min + est {est / 60:.1f} min > budget")
            R["arm_C"] = dict(skipped=True)
            continue
        orig, info = train_lora()
        ev, _ = eval_all("C", base=base)
        for lyr, (q, v) in zip(layers, orig):
            lyr.self_attn.q_proj, lyr.self_attn.v_proj = q, v
    else:
        g, touched, info = train_graft(arm)
        ev, _ = eval_all(arm, graft=g, touched=touched, base=base)
        with torch.no_grad():
            info["decode_parity_trained_graft_max_abs_dlogit"] = decode_parity(held_x[:1, :40], 20)
            info["decode_parity_trained_graft_train_chunk"] = decode_parity(train_x[:1, :40], 20)
        g.remove()
        del g, touched
    with torch.no_grad():  # host restored exactly?
        restored = nll_of(held_x[:2])
        info["host_restored_max_abs_dnll"] = float(np.abs(restored - base_raw["heldout"][:2]).max())
    arm_time[arm] = time.time() - ta
    info["arm_wall_s"] = arm_time[arm]
    R[f"arm_{arm}"] = dict(train=info, eval=ev["summary"])
    L(f"arm {arm} done in {arm_time[arm] / 60:.1f} min:", json.dumps(info)[:2000])
    save()

R["total_wall_min"] = (time.time() - T0) / 60
save()
L("done; total", f"{R['total_wall_min']:.1f} min")
