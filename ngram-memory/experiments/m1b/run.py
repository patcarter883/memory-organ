#!/usr/bin/env python
"""M1b runner (pre-registered in experiments/m1b/PREREG.md, grid in experiments/m1b/grid.json).

    python experiments/m1b/run.py --out DIR [--smoke] [--threads 4] [--rss-limit-gb 11] [--stop-after STAGE]

Stages run sequentially in the pre-registered dependency order (grid.json "stages"). Every stage writes
DIR/<stage>/result.json when it finishes; a stage whose result.json exists is skipped (resumable). Training
stages checkpoint every `ckpt_every` steps and resume from DIR/<arm>/checkpoint.pt. Each stage runs in a child
process whose resident set is polled; above --rss-limit-gb the child is killed and the stage is recorded as
aborted (its dependants then use the remaining candidates). DIR/progress.json is rewritten after every event.

Data discipline: training uses only Split-A train files (or, for the foreign control, the foreign corpus; for the
Split-B stage, the tree at T_cut); every selection reads Split-A dev results only. The Split-A test files are read
exactly once, by the stage named "test", after the operating point is fixed; Split B is scored after that.
Worker modes (internal): --worker train --stage NAME.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
from ngram_memory import paths as _paths  # noqa: E402
QWEN25 = str(_paths.qwen25())          # HF cache at the pinned revision
FOREIGN = str(_paths.foreign_dir())     # NM_FOREIGN_DIR
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s]", *a, flush=True)


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


# ================================================================================================ grid
def load_grid(smoke: bool) -> dict:
    g = jload(HERE / "grid.json")
    g["budget"] = g["budgets"]["smoke" if smoke else "full"]
    g["eval_flags"] = g["eval"]["smoke" if smoke else "full"]
    return g


def resolve_arm(grid: dict, name: str, out: Path) -> dict:
    """Arm config = common recipe, then the recipe fields of a selected earlier stage ('from_selection'), then the
    arm's own overrides (its factor and, for controls, their corpus/arm)."""
    arm = dict(grid["common"])
    spec = grid["arms"][name]
    for field, sel in spec.get("from_selection", {}).items():
        s = jload(out / sel / "result.json")
        if field == "*":                       # copy the selected configuration's recipe fields
            for k in grid["recipe_fields"]:
                if k in s["config"]:
                    arm[k] = s["config"][k]
        else:
            arm[field] = s["config"][field]
    arm.update({k: v for k, v in spec.items() if k not in ("from_selection", "kind", "note")})
    arm["name"] = name
    return arm


# ======================================================================================= child processes
def rss_mb_of(pid: int) -> float:
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def tree_rss_mb(pid: int) -> float:
    tot = rss_mb_of(pid)
    try:
        kids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        kids = []
    return tot + sum(tree_rss_mb(int(k)) for k in kids)


def run_guarded(cmd: list[str], log_path: Path, rss_limit_mb: float, threads: int) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    peak = 0.0
    with open(log_path, "a") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                             start_new_session=True)
        aborted = None
        while p.poll() is None:
            r = tree_rss_mb(p.pid)
            peak = max(peak, r)
            if r > rss_limit_mb:
                aborted = f"rss {r:.0f} MB > limit {rss_limit_mb:.0f} MB"
                lf.write(f"\n=== RSS GUARD: {aborted}; killing\n")
                lf.flush()
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                break
            time.sleep(2.0)
    return dict(returncode=p.returncode, peak_rss_mb=peak, wall_s=time.time() - t, aborted=aborted)


# ============================================================================================ worker: train
def worker_train(a) -> int:
    """Train one arm (fresh process). Writes metrics.jsonl, checkpoint.pt, cartridge.pt, fmt.json, train.json."""
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from ngram_memory.ingest import corpus as C
    from ngram_memory.ingest import kltargets as KL
    from ngram_memory.ingest import rowpolicy as RP
    from ngram_memory.ingest import train as TR
    from ngram_memory.spec import graft_format, qwen25_proxy_key_spec
    from ngram_memory.tables import SparseRowStore
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest_cli", ROOT / "scripts/ingest.py")
    ING = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ING)

    out = Path(a.out)
    grid = load_grid(a.smoke)
    B = grid["budget"]
    cfgd = jload(out / a.stage / "arm.json")
    d = out / a.stage
    torch.manual_seed(cfgd["seed"])
    summ: dict = dict(config=cfgd, budget=B)

    key = qwen25_proxy_key_spec()
    eos = key.eos_token_id
    tok = C.TokenizerJSON(Path(QWEN25) / "tokenizer.json")
    sa = C.load_split_a()
    corpus = cfgd.get("corpus", "split_a")
    if corpus == "split_a":
        corp = C.load_corpus(C.MINISGL_REPO, C.MINISGL_HEAD)
        by = corp.by_path()
        train_files = [by[p] for p in sa.train if p in by]
        if len(train_files) != len(sa.train):
            raise SystemExit("Split-A train files missing from the corpus")
    elif corpus == "split_b":
        sb = C.load_split_b()
        corp = C.load_corpus(C.MINISGL_REPO, sb.t_cut)
        by = corp.by_path()
        train_files = [by[p] for p in sb.train_files if p in by]
    elif corpus == "foreign":
        root = Path(cfgd.get("foreign_dir", FOREIGN)) / "corpus"
        man = jload(root.parent / "manifest.json")
        fs = sorted(p for p in root.rglob("*.py"))
        train_files = [C.SourceFile(str(p.relative_to(root)), "", p.stat().st_size, p.read_text(encoding="utf-8"))
                       for p in fs]
        summ["foreign"] = dict(package=man["package"], version=man["version"], sdist_sha256=man["sdist_sha256"],
                               files=len(train_files))
    else:
        raise SystemExit(f"unknown corpus {corpus}")
    dev_corp = C.load_corpus(C.MINISGL_REPO, C.MINISGL_HEAD) if corpus != "split_a" else corp
    dby = dev_corp.by_path()
    dev_files = [dby[p] for p in sa.dev if p in dby]
    docs = C.tokenize_files(tok, train_files, "bare")
    stream = C.pack_stream(docs, eos)
    windows = C.make_windows(stream, B["window"], B["stride"], key.context_len, eos)
    dev_stream = C.pack_stream(C.tokenize_files(tok, dev_files, "bare"), eos)
    dev_windows = C.make_windows(dev_stream, B["window"], B["stride"], key.context_len, eos)[: B["monitor_dev_windows"]]
    summ["data"] = dict(corpus=corpus, train_files=len(train_files), train_tokens=int(stream.size),
                        train_windows=len(windows), tokenizer_sha256=tok.sha256)
    log("data", summ["data"])

    ix = RP.NGramIndex(key, [stream])
    near_docs = ING.stdlib_texts(B["reference_tokens"], 0, tok)
    chat_docs = ING.wiki_texts(B["reference_tokens"], tok)
    ref = RP.NGramIndex(key, near_docs + chat_docs)
    pol = RP.RowPolicy(cfgd["policy"], cfgd.get("k", 2), cfgd.get("r", 4.0))
    rows = RP.select_rows(ix, pol, ref)
    omega = RP.reuse_weights(rows, ix, ref, gamma=cfgd["gamma"], pi_other=cfgd["pi_other"])
    lw, cap = RP.ngram_loss_weights(ix, pct=cfgd["weight_cap_pct"])
    win_w = [lw[w.start:w.start + w.ids.size] for w in windows]
    specific = {o: np.setdiff1d(np.unique(ix.keys[o]), np.unique(ref.keys[o])) for o in ix.keys}
    summ["rows"] = dict(policy=pol.name(), rows=int(rows.size), omega_mean=float(omega.mean()),
                        loss_weight_cap=cap, reference_tokens=int(ref.n))
    log("rows", summ["rows"])
    del ref, ix
    ING.trim()

    anchors_w = {}
    for name, ds in (("near", near_docs), ("chat", chat_docs)):
        aw = C.make_windows(C.pack_stream(ds, eos), B["anchor_window"], B["anchor_window"], key.context_len, eos)
        aw = aw[: B["anchor_windows"]]
        aw, nm = ING.specific_mask(aw, key, specific)
        anchors_w[name] = aw
        summ.setdefault("anchors", {})[name] = dict(windows=len(aw), masked_project_specific=nm)

    host = AutoModelForCausalLM.from_pretrained(QWEN25, dtype=torch.float32)
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)
    fp0 = TR.host_fingerprint(host)
    ING.trim()
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for name, aw in anchors_w.items():
        h = hashlib.sha256()
        h.update(QWEN25.encode()); h.update(str(B["topk"]).encode())
        for w in aw:
            h.update(w.ids.tobytes()); h.update(w.labels.tobytes())
        cp = cache / f"kl_{name}_{h.hexdigest()[:16]}.safetensors"
        t = time.time()
        if cp.exists():
            tg = KL.KLTargets.load(cp)
        else:
            tg = KL.compute_kl_targets(host, aw, k=B["topk"], meta=dict(set=name))
            tg.save(str(cp) + ".part")
            os.replace(str(cp) + ".part", cp)
        anchors[name] = (aw, tg)
        log(f"KL targets {name}: {len(tg)} windows, {time.time() - t:.1f}s")

    gdir = cfgd["arm"] == "G-dir"
    fmt = graft_format(host.config.hidden_size, key, arm="G-dir" if gdir else "G-lin",
                       gdir_alpha=cfgd.get("gdir_alpha", 1.0))
    hn = TR.layer_input_norm(host, windows[:2], fmt.module.layer_idx, max_tokens=2048)
    s = TR.glin_row_scale(fmt, hn)
    (d / "fmt.json").write_text(fmt.to_json())
    store = SparseRowStore.reserved(fmt.padded_rows, fmt.row_width, torch.from_numpy(rows))
    steps = int(B["steps"]) if not a.max_steps else min(int(B["steps"]), a.max_steps)
    if a.max_steps:
        summ["max_steps_override"] = a.max_steps
    tc = TR.TrainConfig(arm=cfgd["arm"], lr=cfgd["lr"], anchor_lambdas={"near": cfgd["lambda_near"],
                                                                        "chat": cfgd["lambda_chat"]},
                        lambda_l2=cfgd["lambda_l2"], lambda_l1_rel=cfgd["lambda_l1_rel"], kappa=cfgd.get("kappa"),
                        rho=cfgd["rho"], lambda_rho=cfgd["lambda_rho"], loss_chunk=cfgd["loss_chunk"],
                        seed=cfgd["seed"], max_steps=steps, ckpt_every=B["ckpt_every"],
                        shuffle_seed=cfgd.get("shuffle_seed", 1))
    trainer = TR.RowTrainer(host, fmt, store, windows, tc, anchors=anchors, omega=omega, row_scale=s,
                            loss_weights=win_w, log_path=d / "metrics.jsonl")
    ckpt = d / "checkpoint.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
        log("resumed at step", trainer.step_idx)
        summ["resumed_at"] = trainer.step_idx
    if trainer.step_idx == 0:
        summ["dev_monitor_before"] = trainer.eval_nll(dev_windows)

    def cb(r):
        log(f"step {r['step']}/{steps} nll {r['nll']:.4f} kl_near {r.get('kl_near', 0):.5f} "
            f"kl_chat {r.get('kl_chat', 0):.5f} rho_med {r['rho_train_median']:.4f} rows_read {r['rows_read']} "
            f"nonzero {r['rows_nonzero']} step_s {r['step_s']:.1f} rss {r['rss_mb']:.0f}/{r['peak_rss_mb']:.0f} MB")
        jdump(d / "train_progress.json", dict(step=r["step"] + 1, steps=steps, step_s=r["step_s"],
                                              rss_mb=r["rss_mb"], peak_rss_mb=r["peak_rss_mb"], t=time.time()))

    trainer.train(None, ckpt, cb)
    summ["dev_monitor_after"] = trainer.eval_nll(dev_windows)
    allrec = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
    st = [r["step_s"] for r in allrec]
    summ["train"] = dict(steps=len(allrec), tokens=int(sum(windows[r["window"]].ids.size for r in allrec)),
                         step_s_mean=float(np.mean(st)) if st else None,
                         step_s_median=float(np.median(st)) if st else None,
                         rows_nonzero=allrec[-1]["rows_nonzero"] if allrec else 0,
                         peak_rss_mb=max(r["peak_rss_mb"] for r in allrec) if allrec else None)
    cart = trainer.cartridge()
    trainer.close()
    torch.save(cart.state_dict(), d / "cartridge.pt")
    summ["cartridge"] = dict(rows=len(cart), path=str(d / "cartridge.pt"))
    summ["host_unchanged"] = TR.host_fingerprint(host) == fp0
    if not summ["host_unchanged"]:
        raise SystemExit("host weights changed during training")
    jdump(d / "train.json", summ)
    log("train done", summ["train"], summ["cartridge"])
    return 0


# ========================================================================================= worker: frontier
def worker_frontier(a) -> int:
    """Pruned copies of a trained cartridge (§I.8 correction 7): zero the rows of the top-K most frequent 2-grams
    and the top-K 3-grams of the TRAIN corpus (all 8 heads of each order). Writes pruned_<K>.pt + fmt.json."""
    import numpy as np
    import torch
    from ngram_memory.ingest import corpus as C
    from ngram_memory.ingest import rowpolicy as RP
    from ngram_memory.spec import qwen25_proxy_key_spec
    from ngram_memory.tables import SparseRowStore
    out = Path(a.out)
    grid = load_grid(a.smoke)
    st = next(x for x in grid["stages"] if x["name"] == a.stage)
    src = out / st["source"]
    d = out / a.stage
    key = qwen25_proxy_key_spec()
    tok = C.TokenizerJSON(Path(QWEN25) / "tokenizer.json")
    sa = C.load_split_a()
    corp = C.load_corpus(C.MINISGL_REPO, C.MINISGL_HEAD)
    by = corp.by_path()
    stream = C.pack_stream(C.tokenize_files(tok, [by[p] for p in sa.train if p in by], "bare"), key.eos_token_id)
    ix = RP.NGramIndex(key, [stream])
    store = SparseRowStore.from_state_dict(torch.load(src / "cartridge.pt", map_location="cpu", weights_only=True))
    info = {}
    for K in st["top_k"]:
        kill = []
        for o in (2, 3):
            u, c = np.unique(ix.keys[o], return_counts=True)
            top = u[np.argsort(-c, kind="stable")[:K]]
            pos = np.isin(ix.keys[o], top)
            heads = list(key.heads_of_order(o))
            kill.append(np.unique(ix.rows[pos][:, heads]))
        kill = np.unique(np.concatenate(kill))
        pr = store.copy()
        present = kill[np.isin(kill, pr.row_ids.numpy())]
        pr.delete(torch.from_numpy(present))
        torch.save(pr.state_dict(), d / f"pruned_{K}.pt")
        info[str(K)] = dict(rows_zeroed=int(present.size), rows_left=len(pr))
    (d / "fmt.json").write_text((src / "fmt.json").read_text())
    jdump(d / "pruned.json", info)
    log("frontier", info)
    return 0


# ============================================================================================ orchestration
class Runner:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.grid = load_grid(a.smoke)
        self.ppath = self.out / "progress.json"
        self.progress = jload(self.ppath) if self.ppath.exists() else dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                            smoke=a.smoke, stages={})
        if self.progress.get("smoke") != a.smoke:
            raise SystemExit("output dir was created with a different --smoke setting")
        grid_sha = hashlib.sha256((HERE / "grid.json").read_bytes()).hexdigest()
        if self.progress.setdefault("grid_sha256", grid_sha) != grid_sha:
            raise SystemExit("grid.json changed since this run started (pre-registration): refusing to continue")
        self.save()

    def save(self) -> None:
        self.progress["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.progress["elapsed_s_this_invocation"] = time.time() - T0
        jdump(self.ppath, self.progress)

    def mark(self, stage: str, **kw) -> None:
        self.progress["stages"].setdefault(stage, {}).update(kw)
        self.save()

    def done(self, stage: str) -> bool:
        return (self.out / stage / "result.json").exists()

    # ---------------------------------------------------------------------------------- evaluate.py calls
    def splits_json(self) -> Path:
        p = self.out / "_cache" / "splits.json"
        if not p.exists():
            from ngram_memory.eval.stage1.splits import Splits
            from ngram_memory.ingest.corpus import SPLITS_DIR
            sp = Splits.from_ingest(SPLITS_DIR / "split_a.json", SPLITS_DIR / "split_b.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            sp.save(p)
        return p

    def evaluate(self, stage: str, sets: str, conditions: str, *, reuse_base: Path | None = None,
                 fmt_map: dict | None = None, extra: list[str] | None = None) -> dict:
        d = self.out / stage / "eval"
        E = self.grid["eval_flags"]
        cmd = [PY, str(ROOT / "scripts/evaluate.py"), "--splits", str(self.splits_json()), "--host", QWEN25,
               "--out", str(d), "--sets", sets, "--conditions", conditions, "--threads", str(self.a.threads),
               "--context", str(E["context"]), "--stride", str(E["stride"]), "--min-context", str(E["min_context"]),
               "--retention-tokens", str(E["retention_tokens"]), "--tune-files", str(E["tune_files"]),
               "--ref-vocab-cache", str(self.out / "_cache" / "ref_vocab.json")]
        if E.get("max_files") is not None:
            cmd += ["--max-files", str(E["max_files"])]
        if E.get("max_dev_files") is not None:
            cmd += ["--max-dev-files", str(E["max_dev_files"])]
        if E.get("max_windows") is not None:
            cmd += ["--max-windows", str(E["max_windows"])]
        if E.get("tune_max_windows") is not None:
            cmd += ["--tune-max-windows", str(E["tune_max_windows"])]
        if reuse_base is not None:
            cmd += ["--reuse-base", str(reuse_base)]
        if fmt_map:
            fm = self.out / stage / "fmt_map.json"
            jdump(fm, fmt_map)
            cmd += ["--fmt-map", str(fm)]
        cmd += extra or []
        r = run_guarded(cmd, self.out / stage / "eval.log", self.a.rss_limit_gb * 1024, self.a.threads)
        if r["aborted"] or r["returncode"] != 0:
            raise StageFailed(stage, r)
        s = jload(d / "summary.json")
        return dict(run=r, summary=s)

    # ---------------------------------------------------------------------------------- stage kinds
    def stage_base(self, st: dict) -> dict:
        ev = self.evaluate(st["name"], st["sets"], "base")
        return dict(sets=st["sets"], summary=_brief(ev["summary"]), eval_dir=str(self.out / st["name"] / "eval"),
                    wall_s=ev["run"]["wall_s"], peak_rss_mb=ev["run"]["peak_rss_mb"])

    def stage_train_eval(self, st: dict) -> dict:
        name = st["name"]
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        arm = resolve_arm(self.grid, name, self.out)
        if (d / "arm.json").exists() and jload(d / "arm.json") != arm:
            raise SystemExit(f"{name}: resolved config differs from the one this arm started with")
        jdump(d / "arm.json", arm)
        if not (d / "train.json").exists():
            self.mark(name, status="training", config=arm)
            cmd = [PY, str(Path(__file__).resolve()), "--worker", "train", "--stage", name, "--out", str(self.out),
                   "--threads", str(self.a.threads)] + (["--smoke"] if self.a.smoke else [])
            r = run_guarded(cmd, d / "train.log", self.a.rss_limit_gb * 1024, self.a.threads)
            if r["aborted"] or r["returncode"] != 0:
                raise StageFailed(name, r)
        tr = jload(d / "train.json")
        self.mark(name, status="evaluating", train=tr["train"])
        base_dir = self.out / self.grid["base_stage"] / "eval"
        ev = self.evaluate(name, self.grid["dev_sets"], f"rows:{d / 'cartridge.pt'}", reuse_base=base_dir,
                           fmt_map={"cartridge": str(d / "fmt.json")})
        res = dev_metrics(ev["summary"], "rows[cartridge]")
        return dict(config=arm, train=tr["train"], rows=tr["rows"], cartridge=tr["cartridge"],
                    dev=res, eval_dir=str(self.out / name / "eval"), eval_wall_s=ev["run"]["wall_s"],
                    eval_peak_rss_mb=ev["run"]["peak_rss_mb"])

    def stage_lora(self, st: dict) -> dict:
        B = self.grid["budget"]
        L = self.grid["lora"]
        tokens = int(B["steps"]) * int(B["window"]) if L["tokens"] == "match_row_arms" else int(L["tokens"])
        if self.a.smoke:
            tokens = min(tokens, B.get("lora_tokens_smoke", tokens))
        extra = ["--lora-tokens", str(tokens), "--lora-window", str(L["window"] if not self.a.smoke else B["lora_window"]),
                 "--lora-lr", str(L["lr"]), "--lora-r", str(L["r"]), "--lora-epochs", "1"]
        base_dir = self.out / self.grid["base_stage"] / "eval"
        ev = self.evaluate(st["name"], self.grid["dev_sets"], "lora", reuse_base=base_dir, extra=extra)
        return dict(config=dict(L, name="lora", tokens=tokens), dev=dev_metrics(ev["summary"], "lora"),
                    adapter=str(self.out / st["name"] / "eval" / "lora_adapter.pt"),
                    tuning=ev["summary"]["tuning"].get("lora"), eval_wall_s=ev["run"]["wall_s"])

    def stage_select(self, st: dict) -> dict:
        cands = []
        for c in st["candidates"]:
            p = self.out / c / "result.json"
            if p.exists():
                cands.append(jload(p))
            else:
                self.mark(st["name"], warning=f"candidate {c} has no result (aborted?)")
        if not cands:
            raise StageFailed(st["name"], dict(error="no candidate results"))
        R = self.grid["selection"]
        ok = [c for c in cands if c["dev"]["r2_rel_pct"] <= R["r2_max_pct"] and c["dev"]["r1_rel_pct"] <= R["r1_max_pct"]
              and c["dev"]["r1_mean_kl"] <= R["r1_max_kl"]]
        if ok:
            best = max(ok, key=lambda c: (c["dev"]["e1_gain_pct"], -c["config"]["rho"]))
            rule = "largest dev E1 gain among candidates with dev R2 <= r2_max_pct and R1 within PASS"
        else:
            best = min(cands, key=lambda c: (c["dev"]["r2_rel_pct"], c["config"]["rho"]))
            rule = "no candidate meets the retention bars: smallest dev R2 (recorded as the kill-zone case)"
        table = [dict(stage=c["config"]["name"], **{k: c["dev"][k] for k in ("e1_gain_pct", "e1_ci_pct", "r1_rel_pct",
                                                                               "r1_mean_kl", "r2_rel_pct")})
                 for c in cands]
        return dict(selected=best["config"]["name"], config=best["config"], dev=best["dev"], rule=rule,
                    any_meets_bars=bool(ok), candidates=table, cartridge=best.get("cartridge"))

    def stage_test(self, st: dict) -> dict:
        marker = self.out / st["name"] / "STARTED"
        if marker.exists():
            self.mark(st["name"], note="test stage restarted after an interruption (same pre-registered read)")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        op = jload(self.out / st["operating_point"] / "result.json")

        conds, fmap = [], {}

        def rows_of(stage: str, tag: str) -> str:
            sd = self.out / stage
            link = self.out / st["name"] / f"{tag}.pt"
            if not link.exists():
                link.symlink_to(sd / "cartridge.pt")
            fmap[tag] = str(sd / "fmt.json")
            return str(link)
        op_rows = rows_of(op["selected"], "op")
        for c in st["conditions"]:
            if c == "op":
                conds.append(f"rows:{op_rows}")
            elif c in ("perm_op", "random_op", "constbias_op"):
                conds.append(f"{c.split('_')[0]}:{op_rows}")
            elif c == "rag4k+op":
                conds.append(f"rag4k+rows:{op_rows}")
            elif c.startswith("control:"):
                stg = c.split(":", 1)[1]
                if (self.out / stg / "result.json").exists():
                    conds.append(f"rows:{rows_of(stg, stg)}")
            elif c == "lora":
                lr = self.out / st["lora_stage"] / "result.json"
                if lr.exists():
                    conds.append(f"lora:{jload(lr)['adapter']}")
            else:
                conds.append(c)
        extra = ["--knm-tokens", str(self.grid["eval_flags"]["knm_tokens"]), "--r3-items", "0"]
        ev = self.evaluate(st["name"], "test", ",".join(conds), fmt_map=fmap, extra=extra)
        return dict(operating_point=op["selected"], op_config=op["config"], conditions=conds,
                    summary=_brief(ev["summary"]), eval_dir=str(self.out / st["name"] / "eval"),
                    wall_s=ev["run"]["wall_s"])

    def stage_splitb(self, st: dict) -> dict:
        name = st["name"]
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        op = jload(self.out / st["operating_point"] / "result.json")
        arm = dict(op["config"], name=name, corpus="split_b")
        if arm["arm"] == "key-shuffle":
            raise SystemExit("operating point cannot be a control")
        jdump(d / "arm.json", arm)
        if not (d / "train.json").exists():
            self.mark(name, status="training", config=arm)
            cmd = [PY, str(Path(__file__).resolve()), "--worker", "train", "--stage", name, "--out", str(self.out),
                   "--threads", str(self.a.threads)] + (["--smoke"] if self.a.smoke else [])
            r = run_guarded(cmd, d / "train.log", self.a.rss_limit_gb * 1024, self.a.threads)
            if r["aborted"] or r["returncode"] != 0:
                raise StageFailed(name, r)
        tr = jload(d / "train.json")
        conds = ",".join(["base", f"rows:{d / 'cartridge.pt'}"] + st["conditions"])
        ev = self.evaluate(name, "splitb", conds, fmt_map={"cartridge": str(d / "fmt.json")})
        return dict(config=arm, train=tr["train"], summary=_brief(ev["summary"]), eval_dir=str(d / "eval"))

    def stage_frontier(self, st: dict) -> dict:
        name = st["name"]
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "pruned.json").exists():
            cmd = [PY, str(Path(__file__).resolve()), "--worker", "frontier", "--stage", name, "--out", str(self.out),
                   "--threads", str(self.a.threads)] + (["--smoke"] if self.a.smoke else [])
            r = run_guarded(cmd, d / "frontier.log", self.a.rss_limit_gb * 1024, self.a.threads)
            if r["aborted"] or r["returncode"] != 0:
                raise StageFailed(name, r)
        ks = st["top_k"]
        conds = ",".join(f"rows:{d / f'pruned_{K}.pt'}" for K in ks)
        ev = self.evaluate(name, self.grid["dev_sets"], conds, reuse_base=self.out / self.grid["base_stage"] / "eval",
                           fmt_map={f"pruned_{K}": str(d / "fmt.json") for K in ks})
        src = jload(self.out / st["source"] / "result.json")["dev"]
        pts = [dict(point="base", r2=0.0, e1=0.0)]
        for K in sorted(ks, reverse=True):
            m = dev_metrics(ev["summary"], f"rows[pruned_{K}]")
            pts.append(dict(point=f"{st['source']} minus top-{K}", r2=m["r2_rel_pct"], e1=m["e1_gain_pct"], dev=m))
        pts.append(dict(point=st["source"], r2=src["r2_rel_pct"], e1=src["e1_gain_pct"]))
        op = jload(self.out / "S3_op" / "result.json")
        front = sorted(pts, key=lambda x: x["r2"])

        def interp(r2):  # piecewise-linear E1 of the frontier at a given R2 (clamped at the ends)
            if r2 <= front[0]["r2"]:
                return front[0]["e1"]
            for p0, p1 in zip(front, front[1:]):
                if p0["r2"] <= r2 <= p1["r2"]:
                    t = 0.0 if p1["r2"] == p0["r2"] else (r2 - p0["r2"]) / (p1["r2"] - p0["r2"])
                    return p0["e1"] + t * (p1["e1"] - p0["e1"])
            return front[-1]["e1"]
        fe = interp(op["dev"]["r2_rel_pct"])
        return dict(points=pts, pruned=jload(d / "pruned.json"), operating_point=op["selected"],
                    op_dev_e1_gain_pct=op["dev"]["e1_gain_pct"], op_dev_r2_pct=op["dev"]["r2_rel_pct"],
                    frontier_e1_at_op_r2=fe, op_above_frontier=bool(op["dev"]["e1_gain_pct"] > fe))

    def stage_report(self, st: dict) -> dict:
        """The pre-registered decision arithmetic (PREREG §6) on the stage results; no model is run."""
        R = lambda n: jload(self.out / n / "result.json") if (self.out / n / "result.json").exists() else None  # noqa: E731
        op, test, fr = R("S3_op"), R("T_test"), R("F_frontier")
        T = test["summary"]["sets"]["test"]
        lab_op = "rows[op]"

        def gain(label, cls=None):
            if label is None or label not in T or "paired" not in T[label]:
                return None
            p = T[label]["paired"] if cls is None else (T[label].get("classes") or {}).get(cls)
            if p is None:
                return None
            lo, hi = p["rel_pct_ci"]
            return dict(gain_pct=-p["rel_pct"], ci=[-hi, -lo], n_tokens=p["n_tokens"], n_files=p.get("n_files"))
        e1 = gain(lab_op)
        e3 = gain(lab_op, "id:proj:cont_sub")
        ctl_all = {k: gain(k) for k in ("rows[A7_keyshuffle]", "perm[op]", "ngm")}
        ctl_all = {k: v for k, v in ctl_all.items() if v}
        op_proj = gain(lab_op, "id:proj")
        fgn_proj = gain("rows[A6_foreign]", "id:proj")
        ratios = {k: v["gain_pct"] / e1["gain_pct"] for k, v in ctl_all.items()} if e1 and e1["gain_pct"] > 0 else {}
        if fgn_proj and op_proj and op_proj["gain_pct"] > 0:
            ratios["rows[A6_foreign] (id:proj)"] = fgn_proj["gain_pct"] / op_proj["gain_pct"]
        best_ctl = max(ratios.values()) if ratios else None
        lbl = {"countlm": "countlm", "rag4k": "rag4k", "lora": next((k for k in T if k.startswith("lora")), None)}
        countlm = gain(lbl["countlm"])
        lora = gain(lbl["lora"])
        rag = gain(lbl["rag4k"])
        ragop = gain("rag4k+rows[op]")
        dev = op["dev"]
        r2, r1, r1kl = dev["r2_rel_pct"], dev["r1_rel_pct"], dev["r1_mean_kl"]
        g = e1["gain_pct"] if e1 else float("nan")
        lo, hi = (e1["ci"] if e1 else (float("nan"),) * 2)
        c1_pass = best_ctl is not None and best_ctl <= 1 / 3
        c1_kill = best_ctl is not None and best_ctl >= 0.5
        ngi_pass = bool(countlm and g >= 1.5 * countlm["gain_pct"])
        above = bool(fr and fr.get("op_above_frontier"))
        if not op["any_meets_bars"] or hi < 0.5:
            zone = "stop (no configuration with R2 <= +1.0% and R1 within PASS, or E1 upper bound < 0.5%)"
        elif g >= 2.0 and lo > 0.5 and r2 <= 1.0 and c1_pass and ngi_pass:
            zone = "PASS (§6.A.7)"
        elif 1.0 <= g and lo > 0.5 and c1_pass and ngi_pass:
            zone = "proceed, E1 partial (§6.A.7)"
        else:
            zone = "do not scale: one Stage-1b arm on this host, then decide (§6.A.7)"
        m1_kill = (not op["any_meets_bars"]) or not (g >= 1.0 and lo > 0.5)
        m1_pass_bars = dict(e1=g >= 1.5 and lo > 0.5, r2=r2 <= 0.5, r1=r1 <= 0.2, above_frontier=above, c1=c1_pass,
                            ngi=ngi_pass, reg="not measured (no REG harness)")
        sb = R("SB_splitb")
        e2 = None
        if sb:
            B = sb["summary"]["sets"]["splitb"]
            k = next((x for x in B if x.startswith("rows[")), None)
            if k and "paired" in B[k]:
                p = B[k]["paired"]
                e2 = dict(gain_pct=-p["rel_pct"], ci=[-p["rel_pct_ci"][1], -p["rel_pct_ci"][0]], n_tokens=p["n_tokens"])
        return dict(operating_point=op["selected"], op_config=op["config"], E1_test=e1, E3_test=e3, R1_dev=r1,
                    R1_dev_mean_kl=r1kl, R2_dev=r2, frontier=fr and dict(above=above, frontier_e1=fr["frontier_e1_at_op_r2"]),
                    C1=dict(control_ratios=ratios, best=best_ctl, pass_=c1_pass, kill=c1_kill),
                    NGI=dict(countlm=countlm, pass_=ngi_pass), E5=dict(lora=lora, ratio=(g / lora["gain_pct"]) if lora and lora["gain_pct"] else None),
                    E6=dict(rag4k=rag, rag4k_plus_op=ragop, additive_pct=(ragop["gain_pct"] - rag["gain_pct"]) if rag and ragop else None),
                    E2_splitb=e2, zone_6A7=zone, M1_kill=m1_kill, M1_pass_bars=m1_pass_bars,
                    M1_pass=False if not all(v is True for v in m1_pass_bars.values()) else True,
                    note="M1 PASS needs REG, which this package does not measure (PREREG §6)")

    def stage_r3(self, st: dict) -> dict:
        op = jload(self.out / st["operating_point"] / "result.json")
        sd = self.out / op["selected"]
        ev = self.evaluate(st["name"], "none", f"base,rows:{sd / 'cartridge.pt'}",
                           fmt_map={"cartridge": str(sd / "fmt.json")},
                           extra=["--r3-items", str(self.grid["eval_flags"]["r3_items"])])
        return dict(operating_point=op["selected"], r3=ev["summary"]["sets"].get("r3"))

    # ---------------------------------------------------------------------------------- main loop
    def run(self) -> int:
        kinds = dict(base=self.stage_base, train_eval=self.stage_train_eval, lora=self.stage_lora,
                     select=self.stage_select, test=self.stage_test, splitb=self.stage_splitb, r3=self.stage_r3,
                     frontier=self.stage_frontier, report=self.stage_report)
        for st in self.grid["stages"]:
            name = st["name"]
            if self.done(name):
                log("skip (done)", name)
                continue
            need = [x for x in st.get("after", []) if not self.done(x) and not self.progress["stages"].get(x, {}).get("aborted")]
            if need:
                log("blocked", name, "waiting for", need)
                self.mark(name, status="blocked", waiting_for=need)
                continue
            log("start", name, st["kind"])
            self.mark(name, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            t = time.time()
            try:
                res = kinds[st["kind"]](st)
            except Exception as e:  # noqa: BLE001  (a code error in a stage: record it and stop)
                if not isinstance(e, StageFailed):
                    import traceback
                    log("ERROR", name, repr(e))
                    self.mark(name, status="failed", error=traceback.format_exc()[-4000:])
                    return 1
                log("FAILED", name, e.info)
                self.mark(name, status="aborted" if e.info.get("aborted") else "failed", aborted=bool(e.info.get("aborted")),
                          error=e.info)
                if not e.info.get("aborted") and not self.a.keep_going:
                    return 1
                continue
            res["stage_wall_s"] = time.time() - t
            jdump(self.out / name / "result.json", res)
            self.mark(name, status="done", wall_s=res["stage_wall_s"], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            log("done", name, f"{res['stage_wall_s']:.0f}s")
            if self.a.stop_after and name == self.a.stop_after:
                log("stop-after reached")
                return 0
        self.progress["finished"] = all(self.done(s["name"]) for s in self.grid["stages"])
        self.save()
        return 0


class StageFailed(Exception):
    def __init__(self, stage: str, info: dict):
        super().__init__(stage)
        self.info = info


def _brief(s: dict) -> dict:
    return dict(sets=s.get("sets"), tuning={k: v for k, v in s.get("tuning", {}).items() if k != "build_s"},
                warnings=s.get("warnings"), peak_rss_mb=s.get("peak_rss_mb"), wall_s_total=s.get("wall_s_total"),
                data_regime=s.get("data_regime"))


def dev_metrics(s: dict, label: str) -> dict:
    """The dev-stage numbers the selection rules read: E1 on Split-A dev, R1 (WikiText-2), R2 (stdlib + ML repo)."""
    S = s["sets"]

    def p(set_name):
        return S[set_name][label]["paired"]
    e1 = p("dev")
    r1 = p("r1")
    r2s = [p(x) for x in ("r2", "r2ml") if x in S]
    n2 = sum(x["n_tokens"] for x in r2s)
    r2 = sum(x["rel_pct"] * x["n_tokens"] for x in r2s) / max(n2, 1)
    ci = e1.get("rel_pct_ci_file") or e1.get("rel_pct_ci")
    return dict(e1_rel_pct=e1["rel_pct"], e1_gain_pct=-e1["rel_pct"], e1_ci_pct=ci, e1_n_tokens=e1["n_tokens"],
                r1_rel_pct=r1["rel_pct"], r1_mean_kl=r1.get("mean_kl", 0.0), r2_rel_pct=r2,
                r2_parts={x: p(x)["rel_pct"] for x in ("r2", "r2ml") if x in S},
                classes=S["dev"][label].get("classes"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    ap.add_argument("--rss-limit-gb", type=float, default=11.0)
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed (not RSS-aborted) stage")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="worker: cap the steps (timing probes only)")
    a = ap.parse_args()
    if a.worker == "train":
        return worker_train(a)
    if a.worker == "frontier":
        return worker_frontier(a)
    log("M1b runner", vars(a))
    return Runner(a).run()


if __name__ == "__main__":
    sys.exit(main())
