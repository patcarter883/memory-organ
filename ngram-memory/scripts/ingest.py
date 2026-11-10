#!/usr/bin/env python
"""Stage-1 ingestion driver (§I): corpus -> splits -> windows -> row policy -> KL targets -> sparse-row training.

Subcommands
  splits   build Split A (frozen once written) and Split B for the pinned minisgl commit and write
           ingest_splits/split_a.json, split_b.json (--write), or check the stored files (default).
  train    train a graft cartridge on a frozen HF host (Qwen2.5-0.5B proxy by default) on the Split-A train files.

Example (the M0 smoke: 20 steps of the base arm on 4k windows with KL anchors):
  python scripts/ingest.py train --out OUT_DIR --steps 20 --threads 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory.ingest import corpus as C  # noqa: E402
from ngram_memory.ingest import rowpolicy as RP  # noqa: E402

QWEN25 = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
STDLIB = Path("/usr/lib/python3.13")
# the stdlib files the §I.8 proxy used for its R2 evaluation: never used as anchors
STDLIB_EVAL = {"argparse.py", "dataclasses.py", "functools.py", "typing.py", "json/decoder.py", "json/encoder.py",
               "asyncio/tasks.py", "asyncio/queues.py", "collections/__init__.py", "logging/__init__.py",
               "subprocess.py", "threading.py", "inspect.py", "enum.py", "contextlib.py", "shutil.py"}
WIKI = Path("/root/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet")
T0 = time.time()


def rss() -> str:
    import resource
    try:
        cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        cur = float("nan")
    return f"rss {cur:.0f} MB peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"


def trim() -> None:
    """Return freed heap pages to the OS (glibc), so RSS reflects live memory."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------- splits
def build_splits(repo: str, commit: str, t_cut: str, frozen: C.SplitA | None) -> tuple[C.Corpus, C.SplitA, C.SplitB]:
    corp = C.load_corpus(repo, commit)
    files = {f.path: f.text for f in corp.files}
    sa = C.assign_split_a(files, corp.commit, near_clusters=corp.dedup.near_clusters if corp.dedup else (),
                          frozen=frozen)
    sb = C.build_split_b(repo, t_cut, commit)
    return corp, sa, sb


def cmd_splits(a) -> int:
    frozen = None if a.rebuild else C.load_split_a()
    corp, sa, sb = build_splits(a.repo, a.commit, a.t_cut, frozen)
    summ = dict(corpus=corp.summary(), split_a=dict(train=len(sa.train), dev=len(sa.dev), test=len(sa.test),
                                                    neardup_diag=sa.neardup_diag, cluster_moves=sa.cluster_moves),
                split_b=dict(commits=len(sb.commits), first_parent=len(sb.first_parent_commits),
                             files_with_added_lines=len(sb.added_lines), added_lines=sb.n_added_lines,
                             new_files=sb.new_files, train_files_at_t_cut=len(sb.train_files)))
    print(json.dumps(summ, indent=1, default=str))
    if a.write:
        C.write_split_files(sa, sb)
        log("wrote", C.SPLITS_DIR)
        return 0
    old_a, old_b = C.load_split_a(), C.load_split_b()
    ok = old_a is not None and old_a.to_json() == sa.to_json() and old_b is not None and old_b.to_json() == sb.to_json()
    log("stored splits match a rebuild:", ok)
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------- anchors
def stdlib_texts(cap_tokens: int, seed: int, tok: C.TokenizerJSON) -> list[np.ndarray]:
    ex = ("test", "tests", "idlelib", "tkinter", "lib2to3", "site-packages", "__pycache__")
    fs = sorted(p for p in STDLIB.rglob("*.py") if not any(x in p.parts for x in ex)
                and str(p.relative_to(STDLIB)) not in STDLIB_EVAL)
    random.Random(seed).shuffle(fs)
    out, n = [], 0
    for p in fs:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        d = tok.encode(t)
        if d.size:
            out.append(d); n += d.size
        if n >= cap_tokens:
            break
    return out


def wiki_texts(cap_tokens: int, tok: C.TokenizerJSON) -> list[np.ndarray]:
    import pyarrow.parquet as pq
    wl = pq.read_table(WIKI).column("text").to_pylist()
    out, n = [], 0
    for i in range(0, len(wl), 200):
        d = tok.encode("".join(wl[i:i + 200]))
        if d.size:
            out.append(d); n += d.size
        if n >= cap_tokens:
            break
    return out


def specific_mask(windows: list[C.Window], key, specific: dict[int, np.ndarray]) -> tuple[list[C.Window], int]:
    """Drop KL positions whose exact 2/3-gram is project-specific (§I.1: anchors exclude project behaviour)."""
    out, n_masked = [], 0
    for w in windows:
        ix = RP.NGramIndex(key, [w.ids], contexts=[w.context], rows=False)
        bad = np.zeros(w.ids.size, bool)
        for o, ks in specific.items():
            if o in ix.keys:
                bad |= np.isin(ix.keys[o], ks)
        lab = w.labels.copy()
        n_masked += int(((lab != C.IGNORE) & bad).sum())
        lab[bad] = C.IGNORE
        out.append(C.Window(w.ids, w.context, lab, w.start))
    return out, n_masked


# ------------------------------------------------------------------------------------------------- train
def cmd_train(a) -> int:
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM
    from ngram_memory.ingest import kltargets as KL
    from ngram_memory.ingest import train as TR
    from ngram_memory.spec import graft_format, qwen25_proxy_key_spec
    from ngram_memory.tables import SparseRowStore

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = dict(args=vars(a))

    # ---- corpus and splits
    sa = C.load_split_a()
    if sa is None:
        raise SystemExit("ingest_splits/split_a.json missing: run `scripts/ingest.py splits --write` first")
    corp = C.load_corpus(a.repo, a.commit)
    by = corp.by_path()
    train_files = [by[p] for p in sa.train if p in by]
    dev_files = [by[p] for p in sa.dev if p in by]
    tok = C.TokenizerJSON(Path(a.host) / "tokenizer.json")
    host_cfg = json.loads((Path(a.host) / "config.json").read_text())
    key = qwen25_proxy_key_spec() if a.key == "qwen25-proxy" else None
    if key is None:
        raise SystemExit(f"unknown --key {a.key}")
    if key.vocab_size != host_cfg["vocab_size"]:
        raise SystemExit("KeySpec vocab != host vocab")
    eos = key.eos_token_id
    docs = C.tokenize_files(tok, train_files, a.framing)
    stream = C.pack_stream(docs, eos)
    windows = C.make_windows(stream, a.window, a.stride, key.context_len, eos)
    dev_stream = C.pack_stream(C.tokenize_files(tok, dev_files, a.framing), eos)
    dev_windows = C.make_windows(dev_stream, a.window, a.stride, key.context_len, eos)[: a.dev_windows]
    summary["data"] = dict(train_files=len(train_files), dev_files=len(dev_files), train_tokens=int(stream.size),
                           train_windows=len(windows), dev_tokens=int(dev_stream.size), framing=a.framing,
                           tokenizer_sha256=tok.sha256, window=a.window, stride=a.stride)
    log("data", summary["data"])

    # ---- row policy, reuse weights, loss-weight cap
    ix = RP.NGramIndex(key, [stream])
    near_docs = stdlib_texts(a.reference_tokens, a.seed, tok)
    chat_docs = wiki_texts(a.reference_tokens, tok)
    ref = RP.NGramIndex(key, near_docs + chat_docs)
    pol = RP.RowPolicy(a.policy, a.k, a.r)
    rows = RP.select_rows(ix, pol, ref)
    omega = RP.reuse_weights(rows, ix, ref, gamma=a.gamma, pi_other=a.pi_other)
    lw, cap = RP.ngram_loss_weights(ix, pct=a.weight_cap_pct)
    win_w = [lw[w.start:w.start + w.ids.size] for w in windows]
    specific = {o: np.setdiff1d(np.unique(ix.keys[o]), np.unique(ref.keys[o])) for o in ix.keys}
    summary["rows"] = dict(policy=pol.name(), rows=int(rows.size), per_head=RP.per_head_counts(key, rows),
                           all_rows_read=int(np.unique(ix.rows).size), omega_mean=float(omega.mean()),
                           omega_p90=float(np.percentile(omega, 90)), omega_max=float(omega.max()),
                           loss_weight_cap=cap, positions_capped=int((lw < 1).sum()),
                           reference_tokens=ref.n, project_specific_ngrams={o: int(v.size) for o, v in specific.items()})
    log("rows", summary["rows"])
    del ref
    trim()
    log("after row policy:", rss())

    # ---- anchors (separate streams, true history, project-specific positions masked)
    anchors_w = {}
    for name, ds in (("near", near_docs), ("chat", chat_docs)):
        aw = C.make_windows(C.pack_stream(ds, eos), a.anchor_window, a.anchor_window, key.context_len, eos)
        aw = aw[: a.anchor_windows]
        aw, nm = specific_mask(aw, key, specific)
        anchors_w[name] = aw
        summary.setdefault("anchors", {})[name] = dict(windows=len(aw), tokens=int(sum(w.ids.size for w in aw)),
                                                       kl_positions=int(sum(w.n_targets for w in aw)),
                                                       masked_project_specific=nm)
    log("anchors", summary["anchors"])

    # ---- host
    t = time.time()
    host = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32)
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)
    trim()
    fp0 = TR.host_fingerprint(host)
    trim()
    summary["host"] = dict(path=a.host, model_type=host.config.model_type, hidden=host.config.hidden_size,
                           load_s=time.time() - t, fingerprint=fp0, rss_after_load=TR.rss_mb())
    log("host", summary["host"])

    # ---- KL targets (graft off), cached by content
    anchors = {}
    for name, aw in anchors_w.items():
        h = hashlib.sha256()
        h.update(a.host.encode()); h.update(str(a.topk).encode())
        for w in aw:
            h.update(w.ids.tobytes())
        cache = out / f"kl_{name}_{h.hexdigest()[:16]}.safetensors"
        t = time.time()
        if cache.exists():
            tg = KL.KLTargets.load(cache)
        else:
            tg = KL.compute_kl_targets(host, aw, k=a.topk, meta=dict(set=name))
            tg.save(cache)
        anchors[name] = (aw, tg)
        summary["anchors"][name].update(kl_targets=str(cache), kl_bytes=tg.nbytes(), kl_s=time.time() - t)
        log(f"KL targets {name}: {len(tg)} windows in {time.time() - t:.1f}s ({tg.nbytes() / 2**20:.1f} MiB)")

    # ---- graft format and row scale
    fmt = graft_format(host.config.hidden_size, key, arm="G-dir" if a.arm == "G-dir" else "G-lin",
                       gdir_alpha=a.gdir_alpha)
    hn = TR.layer_input_norm(host, windows[:2], fmt.module.layer_idx, max_tokens=2048)
    s = TR.glin_row_scale(fmt, hn)
    summary["format"] = dict(fingerprint=fmt.fingerprint(), row_width=fmt.row_width, padded_rows=fmt.padded_rows,
                             layer1_input_norm_median=hn, row_scale=s)
    log("format", summary["format"])
    store = SparseRowStore.reserved(fmt.padded_rows, fmt.row_width, torch.from_numpy(rows))
    cfg = TR.TrainConfig(arm=a.arm, lr=a.lr, anchor_lambdas={"near": a.lambda_near, "chat": a.lambda_chat},
                         lambda_l2=a.lambda_l2, lambda_l1=a.lambda_l1, kappa=a.kappa, rho=a.rho,
                         lambda_rho=a.lambda_rho, loss_chunk=a.loss_chunk, seed=a.seed, max_steps=a.steps,
                         ckpt_every=a.ckpt_every)
    (out / "config.json").write_text(json.dumps(dict(train=cfg.to_dict(), fmt=json.loads(fmt.to_json())), indent=1))
    trainer = TR.RowTrainer(host, fmt, store, windows, cfg, anchors=anchors, omega=omega, row_scale=s,
                            loss_weights=win_w, log_path=out / "metrics.jsonl")
    ckpt = out / "checkpoint.pt"
    if a.resume and ckpt.exists():
        trainer.load_checkpoint(ckpt)
        log("resumed at step", trainer.step_idx)
    trim()
    summary["rss_before_train"] = TR.rss_mb()
    log("before training:", rss())
    if a.eval_dev:
        summary["dev_before"] = trainer.eval_nll(dev_windows)
        log("dev NLL before", summary["dev_before"])

    def cb(r):
        log(f"step {r['step']} nll {r['nll']:.4f} kl_near {r.get('kl_near', 0):.5f} kl_chat {r.get('kl_chat', 0):.5f} "
            f"rho_train_med {r['rho_train_median']:.4f} rho_near_med {r.get('rho_median_near', float('nan')):.4f} "
            f"rows_read {r['rows_read']} nonzero {r['rows_nonzero']} step {r['step_s']:.1f}s "
            f"rss {r['rss_mb']:.0f}/{r['peak_rss_mb']:.0f} MB")

    recs = trainer.train(None, ckpt, cb)
    if a.eval_dev:
        summary["dev_after"] = trainer.eval_nll(dev_windows)
        log("dev NLL after", summary["dev_after"])
    st = [r["step_s"] for r in recs]
    summary["train"] = dict(steps=len(recs), step_s_mean=float(np.mean(st)) if st else None,
                            step_s_median=float(np.median(st)) if st else None,
                            step_s_after_first=float(np.mean(st[1:])) if len(st) > 1 else None,
                            tokens_per_step=int(np.mean([windows[r["window"]].ids.size for r in recs])) if recs else 0,
                            nll_first=recs[0]["nll"] if recs else None, nll_last=recs[-1]["nll"] if recs else None,
                            rows_nonzero=recs[-1]["rows_nonzero"] if recs else 0)
    cart = trainer.cartridge()
    trainer.close()
    summary["cartridge"] = dict(rows=len(cart), bytes_bf16_plus_ids=len(cart) * (2 * fmt.row_width + 8))
    summary["host_unchanged"] = TR.host_fingerprint(host) == fp0
    summary["peak_rss_mb"] = TR.rss_mb()[1]
    summary["wall_s"] = time.time() - T0
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log("summary", json.dumps({k: summary[k] for k in ("train", "cartridge", "host_unchanged", "peak_rss_mb")}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("splits", "train"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=str(C.MINISGL_REPO))
        p.add_argument("--commit", default=C.MINISGL_HEAD)
    sp = sub.choices["splits"]
    sp.add_argument("--t-cut", default=C.MINISGL_T_CUT)
    sp.add_argument("--write", action="store_true")
    sp.add_argument("--rebuild", action="store_true", help="ignore the frozen Split A lists")
    tp = sub.choices["train"]
    tp.add_argument("--out", required=True)
    tp.add_argument("--host", default=str(QWEN25))
    tp.add_argument("--key", default="qwen25-proxy")
    tp.add_argument("--framing", default="bare", choices=C.FRAMINGS)
    tp.add_argument("--window", type=int, default=4096)
    tp.add_argument("--stride", type=int, default=3584)
    tp.add_argument("--anchor-window", type=int, default=1024)
    tp.add_argument("--anchor-windows", type=int, default=24)
    tp.add_argument("--reference-tokens", type=int, default=500_000)
    tp.add_argument("--policy", default="count", choices=("all", "count", "row_count", "enrich", "specific"))
    tp.add_argument("--k", type=int, default=2)
    tp.add_argument("--r", type=float, default=4.0)
    tp.add_argument("--gamma", type=float, default=4.0)
    tp.add_argument("--pi-other", type=float, default=0.45)
    tp.add_argument("--weight-cap-pct", type=float, default=99.9)
    tp.add_argument("--arm", default="G-lin", choices=("G-lin", "G-dir", "rows+Wv", "key-shuffle"))
    tp.add_argument("--gdir-alpha", type=float, default=1.0)
    tp.add_argument("--lr", type=float, default=1e-2)
    tp.add_argument("--lambda-near", type=float, default=0.3)
    tp.add_argument("--lambda-chat", type=float, default=0.3)
    tp.add_argument("--lambda-l2", type=float, default=1e-4)
    tp.add_argument("--lambda-l1", type=float, default=1e-5)
    tp.add_argument("--kappa", type=float, default=None)
    tp.add_argument("--rho", type=float, default=0.01)
    tp.add_argument("--lambda-rho", type=float, default=0.1)
    tp.add_argument("--topk", type=int, default=64)
    tp.add_argument("--loss-chunk", type=int, default=128)
    tp.add_argument("--steps", type=int, default=20)
    tp.add_argument("--ckpt-every", type=int, default=10)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--threads", type=int, default=1)
    tp.add_argument("--resume", action="store_true")
    tp.add_argument("--eval-dev", action="store_true")
    tp.add_argument("--dev-windows", type=int, default=2)
    a = ap.parse_args()
    return cmd_splits(a) if a.cmd == "splits" else cmd_train(a)


if __name__ == "__main__":
    sys.exit(main())
