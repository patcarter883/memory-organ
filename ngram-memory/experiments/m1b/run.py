#!/usr/bin/env python
"""M1b runner (pre-registered in experiments/m1b/PREREG.md, grid in experiments/m1b/grid.json).

    python experiments/m1b/run.py --out DIR [--smoke] [--threads 4] [--rss-limit-gb 11] [--stop-after STAGE]
                                  [--parallel N [--max-stage-threads T] [--launch-min-free-gb G]]

Stages run in the pre-registered dependency order (grid.json "stages"): one at a time by default, or with
--parallel N up to N stages whose dependencies are all met run side by side, sharing --threads between them
(each stage's thread count is fixed when it starts and recorded; PREREG §11). Every stage writes
DIR/<stage>/result.json when it finishes; a stage whose result.json exists is skipped (resumable). Training
stages checkpoint every `ckpt_every` steps and resume from DIR/<arm>/checkpoint.pt. Each stage runs in a child
process whose resident set is polled; above --rss-limit-gb the child is killed and the stage is recorded as
aborted (its dependants then use the remaining candidates). DIR/progress.json is rewritten after every event.

Data discipline: training uses only Split-A train files (or, for the foreign control, the foreign corpus; for the
Split-B stage, the tree at T_cut); every selection reads Split-A dev results only. The Split-A test files are read
exactly once, by the stage named "test", after the operating point is fixed; Split B is scored after that.
Worker modes (internal): --worker train --stage NAME.

M1c (experiments/m1c/PREREG.md) runs on this runner with `--grid experiments/m1c/grid.json`. Everything M1c adds is
switched on only by fields of its grid (M1b's grid, the default, behaves exactly as before):
  arm fields      objective ('nll' | 'distill' | 'mix'), distill_alpha, teacher_ctx_tokens, teacher_k (Stage-1b,
                  ingest/teacher.py); canaries (true or {seed, n, files, repeats}: planted into the train corpus only,
                  manifest in <arm>/canaries.json and train.json); steps (an int, or 'epoch' = one full pass over the
                  train windows, recorded in train.json)
  grid fields     primary ('nondup': E1 read on the non-duplicated targets, all-target E1 still reported);
                  arm_extra_conditions (extra evaluate.py specs per trained arm, '{rows}' = its rows condition);
                  reg (REG harness flags, full/smoke)
  stage fields    optional_after (a subset of after): dependencies that count as settled once they finished in any
                  terminal state (done, RSS-aborted, failed, code error). A stage whose every dependant lists it
                  there is "soft": its failure (failed, code error, or RSS-aborted) neither blocks nor stops the
                  run and does not set the exit code (progress.json 'soft_failed'). Attempts are counted when a
                  soft stage fails (an attempt cut short by a kill or a CI cancel does not count); on resume a
                  failed soft stage is retried once, and it is recorded 'failed_final' after its second failed
                  attempt, or as soon as all its dependants have finished without it (its result could no longer
                  be used). progress.json 'complete' = every stage done or failed_final
  stage kinds     countgrid (eval-only count-LM frontier, reuses the base stage), reg (scripts/reg.py on the
                  operating point), m1c_select, m1c_op, m1c_test, m1c_report (the M1c bars, PREREG §4-§7)
"""
from __future__ import annotations

import argparse
import hashlib
import threading
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
DEFAULT_GRID = HERE / "grid.json"
GRID_PATH = DEFAULT_GRID            # --grid (M1c: experiments/m1c/grid.json); the sha256 check reads this file


def set_grid_path(p) -> Path:
    """Resolve --grid (relative to the working directory, else to ngram-memory/) and make it the grid of this run."""
    global GRID_PATH
    p = Path(p)
    if not p.is_absolute():
        p = Path.cwd() / p if (Path.cwd() / p).exists() else ROOT / p
    if not p.exists():
        raise SystemExit(f"--grid {p}: no such file")
    GRID_PATH = p.resolve()
    return GRID_PATH


def grid_args() -> list[str]:
    """--grid for worker subprocesses (nothing for the default grid, so M1b's commands are unchanged)."""
    return [] if GRID_PATH.resolve() == DEFAULT_GRID.resolve() else ["--grid", str(GRID_PATH)]


def load_grid(smoke: bool) -> dict:
    g = jload(GRID_PATH)
    g["budget"] = g["budgets"]["smoke" if smoke else "full"]
    g["eval_flags"] = g["eval"]["smoke" if smoke else "full"]
    if "reg" in g:
        g["reg_flags"] = g["reg"]["smoke" if smoke else "full"]
    if "reg2" in g:
        g["reg2_flags"] = g["reg2"]["smoke" if smoke else "full"]
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


def canary_config(cfgd: dict):
    """The arm's CanaryConfig (ingest/canaries.py) or None. Grid form: true, or {seed, n, files, repeats, ...}
    ('files' = how many train files receive canaries, CanaryConfig.m)."""
    c = cfgd.get("canaries")
    if c is None or c is False:
        return None
    from ngram_memory.ingest import canaries as CN
    if isinstance(c, dict):
        c = dict(c)
        if "files" in c:
            c["m"] = c.pop("files")
    return CN.config_from_arm({"canaries": c})


def arm_steps(cfgd: dict, B: dict, n_windows: int) -> int:
    """Training steps of an arm: the budget's steps, an int override, or 'epoch' (one full pass over the train
    windows; capped by the budget's epoch_steps_cap in smoke runs). M1d: with 'epoch', the arm field `epochs`
    (default 1) gives that many passes; pass e visits the windows in the permutation default_rng([seed, e])
    (ingest/train.py RowTrainer.train_index), so pass 1 is the one-pass order of the same seed."""
    sp = cfgd.get("steps")
    if sp is None:
        return int(B["steps"])
    if sp == "epoch":
        cap = B.get("epoch_steps_cap")
        per = min(n_windows, int(cap)) if cap else n_windows
        return per * int(cfgd.get("epochs", 1))
    return int(sp)


def snapshot_steps(cfgd: dict, B: dict, n_windows: int, steps: int) -> list[tuple[int, int]]:
    """M1d: [(pass p, step at the end of pass p)] for the cartridge snapshots of an arm with `snapshots: "epochs"`
    (every pass end before the last; the last pass is cartridge.pt). [] when the arm has no snapshots."""
    if cfgd.get("snapshots") in (None, False):
        return []
    if cfgd.get("snapshots") != "epochs" or cfgd.get("steps") != "epoch":
        raise SystemExit("snapshots: only 'epochs' with steps 'epoch' is supported")
    per = arm_steps(dict(cfgd, epochs=1), B, n_windows)
    return [(p, p * per) for p in range(1, int(cfgd.get("epochs", 1))) if p * per < steps]


def train_with_snapshots(trainer, snaps: list[tuple[int, int]], d: Path, ckpt: Path, cb, save) -> list[dict]:
    """Train up to each snapshot step and write <d>/cartridge_e<p>.pt (+ .json) there; resumable: a snapshot that
    exists is kept, one that is missing is written only if training stands exactly at its step (a checkpoint is
    saved at the end of every train() call, so a resume lands there), else the run fails rather than snapshot the
    wrong step. Splitting train() into segments does not change the optimisation (same steps, same state)."""
    out = []
    for pno, at in snaps:
        sp = d / f"cartridge_e{pno}.pt"
        if trainer.step_idx < at:
            trainer.train(at - trainer.step_idx, ckpt, cb)
        meta = sp.with_suffix(".json")
        if not (sp.exists() and meta.exists()):
            if trainer.step_idx != at:
                raise SystemExit(f"{sp.name} is missing and training is already at step {trainer.step_idx}")
            c = trainer.cartridge()
            save(c, str(sp) + ".tmp")
            os.replace(str(sp) + ".tmp", sp)
            jdump(meta, dict(pass_no=pno, step=at, rows=len(c), path=str(sp)))
            log(f"snapshot pass {pno} at step {at}: {len(c)} rows")
        out.append(jload(meta))
    return out


def canary_stream_spans(tok, files, docs, man: dict, framing: str = "bare") -> dict[str, list[tuple[int, int]]]:
    """Token spans [lo, hi) of every planted secret in the packed train stream (docs joined with EOS, in `files`
    order): canary id -> one span per copy. Offsets come from the tokenizer on the framed text, checked to give the
    same ids as the training documents."""
    import numpy as np
    from ngram_memory.ingest import corpus as C
    off = np.cumsum([0] + [len(d.ids) + 1 for d in docs])[:-1]
    idx = {f.path: i for i, f in enumerate(files)}
    spans: dict[str, list[tuple[int, int]]] = {}
    for path in man["files"]:
        i = idx[path]
        f = files[i]
        txt = C.frame(f.rel, f.text, framing)
        h = len(txt) - len(f.text)
        enc = tok.tok.encode(txt, add_special_tokens=False)
        if list(enc.ids) != [int(x) for x in docs[i].ids]:
            raise SystemExit(f"canary spans: tokenizer offsets disagree with the training ids of {path}")
        o = np.asarray(enc.offsets, np.int64)
        for r in man["records"]:
            if r["path"] != path:
                continue
            c0, c1 = h + r["secret_start"], h + r["secret_end"]
            hit = np.nonzero((o[:, 0] < c1) & (o[:, 1] > c0))[0]
            spans.setdefault(r["id"], []).append((int(off[i] + hit[0]), int(off[i] + hit[-1] + 1)))
    return spans


def canary_windows(stream, spans: dict, window: int, after: int, context_len: int, eos: int) -> list:
    """M1d positive control: one training window per planted copy, `window` tokens of the train stream ending
    `after` tokens past the copy's secret (the copy with up to window - after - len(secret) tokens before it); every
    target scored, the true n-gram context before the window. Ordered by (stream position)."""
    import numpy as np
    from ngram_memory.ingest.corpus import IGNORE, Window
    n = int(stream.size)
    out = []
    for lo, hi in sorted(x for v in spans.values() for x in v):
        b = min(n, hi + after)
        a = max(0, b - window)
        ids = stream[a:b].copy()
        tgt = np.arange(a + 1, b + 1)
        lab = np.full(b - a, IGNORE, dtype=np.int64)
        ok = tgt < n
        lab[ok] = stream[tgt[ok]]
        ctx = np.full(context_len, eos, dtype=np.int64)
        k = min(a, context_len)
        if k:
            ctx[context_len - k:] = stream[a - k:a]
        out.append(Window(ids, ctx, lab, a))
    return out


def trim_metrics(path: Path, step: int) -> int:
    """Training resumed from a checkpoint at `step`: steps >= step run again, so drop their records from
    metrics.jsonl (they were written after that checkpoint). Returns the number dropped; the file is not touched
    when nothing is dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    keep = [x for x in lines if json.loads(x)["step"] < step]
    if len(keep) != len(lines):
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(x + "\n" for x in keep))
        os.replace(tmp, path)
    return len(lines) - len(keep)


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
    cc = canary_config(cfgd)
    if cc is not None:  # M1c: canaries in the TRAIN corpus only (dev/test/retention come from the clean corpus)
        from ngram_memory.ingest import canaries as CN
        if corpus != "split_a":
            raise SystemExit("canaries need corpus split_a")
        train_files, cman = CN.plant_source_files(train_files, sa.train, cc)
        CN.write_manifest(cman, d / "canaries.json")
        summ["canaries"] = dict(CN.manifest_brief(cman), path=str(d / "canaries.json"), manifest=cman)
        log("canaries", CN.manifest_brief(cman))
    dev_corp = C.load_corpus(C.MINISGL_REPO, C.MINISGL_HEAD) if corpus != "split_a" else corp
    dby = dev_corp.by_path()
    dev_files = [dby[p] for p in sa.dev if p in dby]
    docs = C.tokenize_files(tok, train_files, "bare")
    stream = C.pack_stream(docs, eos)
    windows = C.make_windows(stream, B["window"], B["stride"], key.context_len, eos)
    if cc is not None and isinstance(cc, CN.DoseCanaryConfig):     # M1d: copies of a canary in distinct windows
        spans = canary_stream_spans(tok, train_files, docs, cman)
        wc = CN.window_copies(spans, [(w.start, w.start + w.ids.size) for w in windows])
        summ["canary_windows"] = dict(conflicts=wc["conflicts"], canary_windows=wc["canary_windows"],
                                      per_canary={k: dict(copies=v["copies"], distinct_windows=v["distinct_windows"])
                                                  for k, v in wc["per_canary"].items()})
        log("canary windows", dict(n=len(wc["canary_windows"]), conflicts=wc["conflicts"]))
        if wc["conflicts"]:
            raise SystemExit(f"canaries: a training window holds two copies of one canary: {wc['conflicts']}")
        if cfgd.get("train_windows") == "canary":   # M1d positive control: train on the canary copies only
            pw = min(int(cfgd.get("pc_window", 512)), int(B["window"]))
            windows = canary_windows(stream, spans, pw, int(cfgd.get("pc_after", 32)), key.context_len, eos)
            summ["train_windows"] = dict(kind="canary", window=pw, after=int(cfgd.get("pc_after", 32)),
                                         n=len(windows))
            log("positive control windows", summ["train_windows"])
    elif cfgd.get("train_windows") == "canary":
        raise SystemExit("train_windows 'canary' needs dose-response canaries (canaries.groups)")
    if B.get("train_windows_cap"):     # M1d smoke budget only: a short pass, so passes 2-3 use fresh permutations
        windows = windows[: int(B["train_windows_cap"])]
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
    for name, ds in (() if cfgd.get("anchors", True) is False else (("near", near_docs), ("chat", chat_docs))):
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
        import fcntl
        with open(str(cp) + ".lock", "w") as lk:     # concurrent arms (--parallel): one computes, the rest load it
            fcntl.flock(lk, fcntl.LOCK_EX)
            if cp.exists():
                tg = KL.KLTargets.load(cp)
            else:
                tg = KL.compute_kl_targets(host, aw, k=B["topk"], meta=dict(set=name))
                tg.save(str(cp) + f".part{os.getpid()}")
                os.replace(str(cp) + f".part{os.getpid()}", cp)
        anchors[name] = (aw, tg)
        log(f"KL targets {name}: {len(tg)} windows, {time.time() - t:.1f}s")

    gdir = cfgd["arm"] == "G-dir"
    fmt = graft_format(host.config.hidden_size, key, arm="G-dir" if gdir else "G-lin",
                       gdir_alpha=cfgd.get("gdir_alpha", 1.0))
    hn = TR.layer_input_norm(host, windows[:2], fmt.module.layer_idx, max_tokens=2048)
    s = TR.glin_row_scale(fmt, hn)
    (d / "fmt.json").write_text(fmt.to_json())
    store = SparseRowStore.reserved(fmt.padded_rows, fmt.row_width, torch.from_numpy(rows))
    base_steps = arm_steps(cfgd, B, len(windows))
    if cfgd.get("steps") is not None:
        summ["steps_mode"] = dict(spec=cfgd["steps"], train_windows=len(windows), steps=base_steps,
                                  epoch_steps_cap=B.get("epoch_steps_cap"))
        if "epochs" in cfgd:                        # M1d (absent for M1c arms, whose train.json is unchanged)
            summ["steps_mode"]["epochs"] = int(cfgd["epochs"])
    steps = base_steps if not a.max_steps else min(base_steps, a.max_steps)
    if a.max_steps:
        summ["max_steps_override"] = a.max_steps
    objective = cfgd.get("objective", "nll")
    teacher = None
    if objective != "nll":  # Stage-1b teacher targets, graft OFF (before the trainer installs the graft)
        from ngram_memory.ingest import teacher as TT
        enc, dec = TT.tokenizer_fns(tok)
        ctx = int(cfgd.get("teacher_ctx_tokens", 4096))
        if B.get("teacher_ctx_tokens_cap"):
            ctx = min(ctx, int(B["teacher_ctx_tokens_cap"]))
        tcfg = TT.TeacherConfig(ctx_tokens=ctx, k=int(cfgd.get("teacher_k", B["topk"])))
        need = TR.visited_windows(len(windows), cfgd["seed"], steps)
        held = list(sa.dev) + list(sa.test) + list(sa.neardup_diag)
        t = time.time()
        teacher, tpath = TT.load_or_build_teacher(cache, host, QWEN25, windows, docs,
                                                  {f.path: f.text for f in train_files}, enc, dec, tcfg,
                                                  window_index=need, log=log, forbid=held)
        summ["teacher"] = dict(path=str(tpath), cache_key=teacher.meta["cache_key"],
                               identity=teacher.meta.get("identity"), config=tcfg.to_dict(),
                               build_or_load_s=time.time() - t, **TT.retrieval_summary(teacher))
        log("teacher", {k: v for k, v in summ["teacher"].items() if k != "config"})
        ING.trim()
    tkw = {} if "objective" not in cfgd else dict(objective=objective, distill_alpha=cfgd.get("distill_alpha", 0.5))
    tc = TR.TrainConfig(arm=cfgd["arm"], lr=cfgd["lr"], anchor_lambdas={"near": cfgd["lambda_near"],
                                                                        "chat": cfgd["lambda_chat"]},
                        lambda_l2=cfgd["lambda_l2"], lambda_l1_rel=cfgd["lambda_l1_rel"], kappa=cfgd.get("kappa"),
                        rho=cfgd["rho"], lambda_rho=cfgd["lambda_rho"], loss_chunk=cfgd["loss_chunk"],
                        seed=cfgd["seed"], max_steps=steps, ckpt_every=B["ckpt_every"],
                        shuffle_seed=cfgd.get("shuffle_seed", 1), **tkw)
    trainer = TR.RowTrainer(host, fmt, store, windows, tc, anchors=anchors, omega=omega, row_scale=s,
                            loss_weights=win_w, log_path=d / "metrics.jsonl",
                            **({} if teacher is None else dict(teacher=teacher)))
    ckpt = d / "checkpoint.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
        log("resumed at step", trainer.step_idx)
        summ["resumed_at"] = trainer.step_idx
        dropped = trim_metrics(d / "metrics.jsonl", trainer.step_idx)
        if dropped:
            summ["metrics_dropped_on_resume"] = dropped
            log(f"metrics.jsonl: dropped {dropped} records with step >= {trainer.step_idx} (re-run after the resume)")
    if trainer.step_idx == 0:
        summ["dev_monitor_before"] = trainer.eval_nll(dev_windows)

    def cb(r):
        kt = f" kl_teacher {r['kl_teacher']:.5f}" if "kl_teacher" in r else ""
        log(f"step {r['step']}/{steps} nll {r['nll']:.4f}{kt} kl_near {r.get('kl_near', 0):.5f} "
            f"kl_chat {r.get('kl_chat', 0):.5f} rho_med {r['rho_train_median']:.4f} rows_read {r['rows_read']} "
            f"nonzero {r['rows_nonzero']} step_s {r['step_s']:.1f} rss {r['rss_mb']:.0f}/{r['peak_rss_mb']:.0f} MB")
        jdump(d / "train_progress.json", dict(step=r["step"] + 1, steps=steps, step_s=r["step_s"],
                                              rss_mb=r["rss_mb"], peak_rss_mb=r["peak_rss_mb"], t=time.time()))

    snaps = snapshot_steps(cfgd, B, len(windows), steps)
    if snaps:     # M1d: cartridge snapshots at the end of every pass but the last (the last is cartridge.pt)
        summ["snapshots"] = train_with_snapshots(trainer, snaps, d, ckpt, cb,
                                                 lambda c, path: torch.save(c.state_dict(), path))
    trainer.train(None, ckpt, cb)
    summ["dev_monitor_after"] = trainer.eval_nll(dev_windows)
    allrec = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
    st = [r["step_s"] for r in allrec]
    summ["train"] = dict(steps=len(allrec), tokens=int(sum(windows[r["window"]].ids.size for r in allrec)),
                         step_s_mean=float(np.mean(st)) if st else None,
                         step_s_median=float(np.median(st)) if st else None,
                         rows_nonzero=allrec[-1]["rows_nonzero"] if allrec else 0,
                         peak_rss_mb=max(r["peak_rss_mb"] for r in allrec) if allrec else None)
    if objective != "nll":
        kt = [r["kl_teacher"] for r in allrec if "kl_teacher" in r]
        summ["train"].update(objective=objective, alpha=tc.alpha, kl_teacher_first=kt[0] if kt else None,
                             kl_teacher_last=kt[-1] if kt else None)
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
        if getattr(a, "grid", None):
            set_grid_path(a.grid)
        self.grid = load_grid(a.smoke)
        self.ppath = self.out / "progress.json"
        self._lock = threading.RLock()                 # progress.json is shared by concurrent stages
        self._tl = threading.local()                   # per-stage thread count (set by the stage's runner thread)
        self.progress = jload(self.ppath) if self.ppath.exists() else dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                            smoke=a.smoke, stages={})
        if self.progress.get("smoke") != a.smoke:
            raise SystemExit("output dir was created with a different --smoke setting")
        grid_sha = hashlib.sha256(GRID_PATH.read_bytes()).hexdigest()
        if GRID_PATH.resolve() != DEFAULT_GRID.resolve():
            self.progress.setdefault("grid", str(GRID_PATH))
        if self.progress.setdefault("grid_sha256", grid_sha) != grid_sha:
            raise SystemExit("grid.json changed since this run started (pre-registration): refusing to continue")
        for st in self.grid["stages"]:
            bad = set(st.get("optional_after", [])) - set(st.get("after", []))
            if bad:
                raise SystemExit(f"{st['name']}: optional_after {sorted(bad)} must also be listed in after")
        self.soft = soft_stages(self.grid["stages"])
        self.save()

    def save(self) -> None:
        with self._lock:
            self.progress["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.progress["elapsed_s_this_invocation"] = time.time() - T0
            jdump(self.ppath, self.progress)

    def mark(self, stage: str, **kw) -> None:
        with self._lock:
            self.progress["stages"].setdefault(stage, {}).update(kw)
            self.save()

    def threads(self) -> int:
        """Threads for the stage running on the calling thread (--threads when stages run one at a time)."""
        return getattr(self._tl, "threads", self.a.threads)

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
               "--out", str(d), "--sets", sets, "--conditions", conditions, "--threads", str(self.threads()),
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
        if E.get("countgrid_weights") is not None:
            cmd += ["--countgrid-weights", str(E["countgrid_weights"])]
        if E.get("countgrid_kl_sets") is not None:             # M1d: KL only where the bars read it (R1)
            cmd += ["--countgrid-kl-sets", str(E["countgrid_kl_sets"])]
        if reuse_base is not None:
            cmd += ["--reuse-base", str(reuse_base)]
        if fmt_map:
            fm = self.out / stage / "fmt_map.json"
            jdump(fm, fmt_map)
            cmd += ["--fmt-map", str(fm)]
        cmd += extra or []
        r = run_guarded(cmd, self.out / stage / "eval.log", self.a.rss_limit_gb * 1024, self.threads())
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
                   "--threads", str(self.threads())] + (["--smoke"] if self.a.smoke else []) + grid_args()
            r = run_guarded(cmd, d / "train.log", self.a.rss_limit_gb * 1024, self.threads())
            if r["aborted"] or r["returncode"] != 0:
                raise StageFailed(name, r)
        tr = jload(d / "train.json")
        self.mark(name, status="evaluating", train=tr["train"])
        base_dir = self.out / self.grid["base_stage"] / "eval"
        rows = f"rows:{d / 'cartridge.pt'}"
        conds = ",".join([rows] + [c.replace("{rows}", rows) for c in self.grid.get("arm_extra_conditions", [])])
        ev = self.evaluate(name, self.grid["dev_sets"], conds, reuse_base=base_dir,
                           fmt_map={"cartridge": str(d / "fmt.json")})
        res = dev_metrics(ev["summary"], "rows[cartridge]", self.grid.get("primary"))
        out = dict(config=arm, train=tr["train"], rows=tr["rows"], cartridge=tr["cartridge"],
                   dev=res, eval_dir=str(self.out / name / "eval"), eval_wall_s=ev["run"]["wall_s"],
                   eval_peak_rss_mb=ev["run"]["peak_rss_mb"])
        for k in ("teacher", "canaries", "steps_mode"):          # M1c fields (absent for M1b arms)
            if k in tr:
                out[k] = {kk: v for kk, v in tr[k].items() if kk != "manifest"} if k == "canaries" else tr[k]
        return out

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
                   "--threads", str(self.threads())] + (["--smoke"] if self.a.smoke else []) + grid_args()
            r = run_guarded(cmd, d / "train.log", self.a.rss_limit_gb * 1024, self.threads())
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
                   "--threads", str(self.threads())] + (["--smoke"] if self.a.smoke else []) + grid_args()
            r = run_guarded(cmd, d / "frontier.log", self.a.rss_limit_gb * 1024, self.threads())
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

    # ---------------------------------------------------------------------------------- M1c stage kinds
    def _primary(self) -> str:
        return self.grid.get("primary") or "all"

    def _res(self, name: str) -> dict | None:
        p = self.out / name / "result.json"
        return jload(p) if p.exists() else None

    def stage_countgrid(self, st: dict) -> dict:
        """Eval-only count-LM frontier (countgrid:kn on the dev sets, KL vs base), reusing the base stage."""
        base_dir = self.out / self.grid["base_stage"] / "eval"
        conds = st.get("conditions", "countgrid:kn")
        ev = self.evaluate(st["name"], st.get("sets", self.grid["dev_sets"]), conds, reuse_base=base_dir)
        from ngram_memory.eval.stage1 import bars as BR
        lab = st.get("label", "countgrid[kn]")
        front = BR.frontier(ev["summary"], lab, self._primary())
        return dict(label=lab, conditions=conds, primary=self._primary(), frontier=front,
                    lambda_star=BR.lambda_star(front), eval_dir=str(self.out / st["name"] / "eval"),
                    eval_wall_s=ev["run"]["wall_s"], eval_peak_rss_mb=ev["run"]["peak_rss_mb"],
                    summary=_brief(ev["summary"]))

    def _m1c_rows(self, names: list[str], counts_stage: str, warn_stage: str) -> tuple[dict, list[dict]]:
        """Bars A and B on dev for each trained arm (PREREG §4-§5, §12), against the counts-only frontier of
        counts_stage. Bar A is frontier to frontier: X + counts at X's own best weight within the bars (lambda_X)
        against counts alone at lambda*. Bar B matches X's retention budget share."""
        import numpy as np
        from ngram_memory.eval.stage1 import bars as BR
        prim = self._primary()
        cres = self._res(counts_stage)
        if cres is None:
            raise StageFailed(warn_stage, dict(error=f"counts stage {counts_stage} has no result"))
        S0 = BR.load_summary(cres["eval_dir"])
        clab = cres["label"]
        front = BR.frontier(S0, clab, prim)
        ls = BR.lambda_star(front)
        t0 = BR.Targets.load(cres["eval_dir"], "dev", prim)
        cnt_ls = t0.nll(clab, ls["weight"])
        rows = []
        for n in names:
            r = self._res(n)
            if r is None:
                self.mark(warn_stage, warning=f"candidate {n} has no result (aborted or failed)")
                continue
            S = BR.load_summary(r["eval_dir"])
            xl, xpl = "rows[cartridge]", f"{clab}+rows[cartridge]"
            x = BR.point(S, xl, None, prim)
            x_in = BR.within(x)
            plus = [BR.point(S, xpl, w, prim) for w in BR.grid_weights(S, xpl)]
            lx = BR.lambda_x(plus, x_in)
            xp_ls = BR.point(S, xpl, ls["weight"], prim)
            m = BR.matched(front, x)
            t = BR.Targets.load(r["eval_dir"], "dev", prim)
            # the paired differences compare columns of two stages (X's eval dir, the counts stage's): same targets,
            # same primary mask, same base NLL and file groups, or the stage fails
            if not (np.array_equal(t.key, t0.key) and np.array_equal(t.mask, t0.mask)
                    and np.array_equal(t.base, t0.base) and np.array_equal(t.groups, t0.groups)):
                raise StageFailed(warn_stage, dict(error=f"{n}: dev targets differ from {counts_stage}'s"))
            if lx["weight"] is not None:
                A = BR.bar_a(t, t.nll(xpl, lx["weight"]), cnt_ls, True)
            else:
                A = BR.bar_a(t, None, cnt_ls, False, reason=lx["reason"])
            A_ls = BR.bar_a(t, t.nll(xpl, ls["weight"]), cnt_ls, BR.within(xp_ls))     # reported only
            Bb = BR.bar_b(t, t.nll(xl), t0.nll(clab, m["weight"]))
            rows.append(dict(stage=n, config=r["config"], cartridge=r.get("cartridge"), dev=r["dev"], x=x,
                             x_within=x_in, x_share=BR.share(x), lambda_x=lx,
                             x_plus_counts=lx["point"], x_plus_counts_within=lx["weight"] is not None,
                             x_plus_counts_lstar=xp_ls, bar_a_at_lstar=A_ls,
                             counts_matched=m, bar_a=A, bar_b=Bb, bar_a_margin_pts=A["margin_pts"],
                             any_bar=bool(A["holds"] or Bb["holds"])))
        ctx = dict(primary=prim, counts_stage=counts_stage, frontier=front, lambda_star=ls, bars=BR.BARS,
                   bar_a=BR.BAR_A, bar_b=BR.BAR_B, match_s_floor=BR.S_FLOOR)
        return ctx, rows

    @staticmethod
    def _m1c_rank(r: dict) -> tuple:
        """Selection order (PREREG §5): any bar holds on dev, Bar A holds, larger Bar-A margin (none: last); ties ->
        smaller rho, then nll before distill."""
        m = r.get("bar_a_margin_pts")
        return (bool(r["any_bar"]), bool(r["bar_a"]["holds"]), float("-inf") if m is None else float(m),
                -float(r["config"]["rho"]), 0 if r["config"].get("objective", "nll") == "nll" else -1)

    def stage_m1c_select(self, st: dict) -> dict:
        """S1 (PREREG §5): among arms whose X alone is within R1/R2/KL, the best by _m1c_rank."""
        ctx, rows = self._m1c_rows(st["candidates"], st["counts"], st["name"])
        if not rows:
            raise StageFailed(st["name"], dict(error="no candidate results"))
        ok = [r for r in rows if r["x_within"]]
        if ok:
            best = max(ok, key=self._m1c_rank)
            rule = ("among arms with X alone within R1 <= 0.2%, R2 <= 0.5%, KL <= 0.005: any dev bar holds, then Bar A "
                    "holds, then the largest Bar-A margin (ties: smaller rho, next-token before distill)")
        else:
            best = min(rows, key=lambda r: (r["x"]["r2"], r["config"]["rho"]))
            rule = "no arm alone within the retention bars: smallest dev R2 (recorded as the kill-zone case)"
        table = [dict(stage=r["stage"], rho=r["config"]["rho"], objective=r["config"].get("objective", "nll"),
                      e1=r["x"]["e1"], e1_all=r["x"]["e1_all"], r1=r["x"]["r1"], kl=r["x"]["kl"], r2=r["x"]["r2"],
                      share=r["x_share"], x_within=r["x_within"], lambda_x=r["lambda_x"]["weight"],
                      bar_a_diff=r["bar_a"]["diff"] and r["bar_a"]["diff"]["pts"],
                      bar_a_ci=r["bar_a"]["diff"] and r["bar_a"]["diff"]["ci"], bar_a_margin=r["bar_a_margin_pts"],
                      bar_a=r["bar_a"]["holds"], counts_matched_weight=r["counts_matched"]["weight"],
                      bar_b_ratio=r["bar_b"]["ratio"], bar_b_diff_ci=r["bar_b"]["diff"]["ci"], bar_b=r["bar_b"]["holds"],
                      any_bar=r["any_bar"])
                 for r in rows]
        return dict(ctx, selected=best["stage"], config=best["config"], dev=best["dev"], row=best, rule=rule,
                    any_within_bars=bool(ok), any_bar_dev=any(r["any_bar"] for r in rows), table=table,
                    candidates=rows, cartridge=best.get("cartridge"))

    def _stage_error(self, name: str) -> str | None:
        """A short description of why a stage has no result (from progress.json), or None if it has one."""
        if self.done(name):
            return None
        e = self.progress["stages"].get(name, {})
        err = e.get("error")
        if isinstance(err, dict):
            err = err.get("error") or err.get("aborted") or (f"returncode {err['returncode']}" if "returncode" in err
                                                              else json.dumps(err, default=str))
        elif isinstance(err, str):
            err = err.strip().splitlines()[-1] if err.strip() else err
        return f"{e.get('status', 'not run')}" + (f": {str(err)[:300]}" if err else "")

    def stage_m1c_op(self, st: dict) -> dict:
        """OP (PREREG §5): F1 (S1's recipe, one full pass) if its selection tuple (any dev bar, Bar A holds, Bar-A
        margin) is >= S1's, else S1. F1 failed or RSS-aborted -> S1 (rule records why)."""
        s1 = self._res(st["s1"])
        f1 = [st["f1"]] if self._res(st["f1"]) is not None else []
        ctx, rows = self._m1c_rows([s1["selected"]] + f1, st["counts"], st["name"])
        rs1 = next(r for r in rows if r["stage"] == s1["selected"])
        rf1 = next((r for r in rows if r["stage"] == st["f1"]), None)
        if rf1 is not None and self._m1c_rank(rf1)[:3] >= self._m1c_rank(rs1)[:3]:
            op, rule = rf1, "F1's (any dev bar, Bar A holds, Bar-A margin) >= S1's"
        else:
            op = rs1
            rule = f"F1 failed: {self._stage_error(st['f1'])}" if rf1 is None else \
                "F1's (any dev bar, Bar A holds, Bar-A margin) < S1's"
        return dict(ctx, selected=op["stage"], config=op["config"], dev=op["dev"], op_row=op, rule=rule,
                    s1=rs1, f1=rf1, cartridge=op.get("cartridge"))

    def stage_m1c_test(self, st: dict) -> dict:
        """The single Split-A test read of M1c (PREREG §6): base, OP rows, counts@lambda*, counts@lambda* + OP,
        counts@lambda_X(OP) + OP, counts@matched(OP), perm:OP; bars A/B and C1 on the primary (and all) targets;
        canary-free check. T_test/STARTED is written right before the evaluation (after OP is resolved and the
        canary-free check passed)."""
        from ngram_memory.eval.stage1 import bars as BR
        op = self._res(st["operating_point"])
        sd = self.out / op["selected"]
        (self.out / st["name"]).mkdir(parents=True, exist_ok=True)
        link = self.out / st["name"] / "op.pt"
        if not link.exists():
            link.symlink_to(sd / "cartridge.pt")
        fmap = {"op": str(sd / "fmt.json")}
        ls, mw = op["lambda_star"]["weight"], op["op_row"]["counts_matched"]["weight"]
        lx = op["op_row"]["lambda_x"]["weight"]
        tl, tm = BR.weight_tag(ls), BR.weight_tag(mw)
        spec = {"base": "base", "op": f"rows:{link}", "perm_op": f"perm:{link}",
                "countlm_lstar": f"countlm:kn-lam{tl}", "countlm_lstar+op": f"countlm:kn-lam{tl}+rows:{link}",
                "countlm_matched_op": f"countlm:kn-lam{tm}"}
        lab = dict(op="rows[op]", perm="perm[op]", cl=f"countlm[kn-lam{tl}]", clop=f"countlm[kn-lam{tl}]+rows[op]",
                   cm=f"countlm[kn-lam{tm}]")
        # counts@lambda_X(OP) + OP: lambda_X == lambda* reuses counts@lambda* + OP; lambda_X == 0 is OP alone;
        # no lambda_X (no X + counts weight within the bars on dev): Bar A fails on test too
        if lx is None:
            spec["countlm_lambdax+op"], lab["clx"] = None, None
        elif lx == 0:
            spec["countlm_lambdax+op"], lab["clx"] = spec["op"], lab["op"]
        else:
            tx = BR.weight_tag(lx)
            spec["countlm_lambdax+op"] = f"countlm:kn-lam{tx}+rows:{link}"
            lab["clx"] = f"countlm[kn-lam{tx}]+rows[op]"
        conds = list(dict.fromkeys(spec[c] for c in st["conditions"] if spec.get(c) is not None))
        # canary-free check: no canary secret of OP's manifest occurs in any held-out (dev/test/near-dup) text
        cman = sd / "canaries.json"
        canary_check = None
        if cman.exists():
            from ngram_memory.eval.stage1.splits import Splits
            from ngram_memory.ingest import canaries as CN
            sp = Splits.load(self.splits_json())
            held = {p: sp.text(p) for part in ("dev", "test", "near_dup") for p in sp.files("a", part)}
            man = CN.load_manifest(cman)
            CN.assert_clean(held, man)          # raises (stage error) if a canary leaked into held-out data
            canary_check = dict(manifest=str(cman), sha256=man["sha256"], held_out_files=len(held), clean=True)
        marker = self.out / st["name"] / "STARTED"
        if marker.exists():
            self.mark(st["name"], note="test stage restarted after an interruption (same pre-registered read)")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        extra = ["--r3-items", "0"]
        ev = self.evaluate(st["name"], "test", ",".join(conds), fmt_map=fmap, extra=extra)
        ed = self.out / st["name"] / "eval"
        bars = {}
        for prim in dict.fromkeys([self._primary(), "all"]):
            t = BR.Targets.load(ed, "test", prim)
            cl = t.nll(lab["cl"])
            if lab["clx"] is not None:
                A = BR.bar_a(t, t.nll(lab["clx"]), cl, op["op_row"]["x_plus_counts_within"])
            else:
                A = BR.bar_a(t, None, cl, False, reason=op["op_row"]["lambda_x"]["reason"])
            bars[prim] = dict(
                e1_op=BR.gain(t, t.nll(lab["op"])), e1_counts_lstar=BR.gain(t, cl),
                e1_counts_lstar_plus_op=BR.gain(t, t.nll(lab["clop"])), e1_counts_matched=BR.gain(t, t.nll(lab["cm"])),
                e1_counts_lambdax_plus_op=BR.gain(t, t.nll(lab["clx"])) if lab["clx"] else None,
                bar_a=A, bar_a_at_lstar=BR.bar_a(t, t.nll(lab["clop"]), cl, None),
                bar_b=BR.bar_b(t, t.nll(lab["op"]), t.nll(lab["cm"])),
                c1=BR.c1(t, t.nll(lab["op"]), t.nll(lab["perm"])))
        T = ev["summary"]["sets"]["test"]
        return dict(operating_point=op["selected"], op_config=op["config"], conditions=conds, labels=lab,
                    lambda_star=ls, lambda_x=lx, counts_matched_weight=mw, primary=self._primary(), bars=bars,
                    canary_check=canary_check, class_fractions=T.get("_class_fractions"),
                    summary=_brief(ev["summary"]), eval_dir=str(ed), wall_s=ev["run"]["wall_s"])

    def stage_reg(self, st: dict) -> dict:
        """REG (§6.A.3) on the operating point vs base: scripts/reg.py with the arm's canary manifest."""
        op = self._res(st["operating_point"])
        sd = self.out / op["selected"]
        d = self.out / st["name"]
        F = self.grid.get("reg_flags", {})
        cmd = [PY, str(ROOT / "scripts/reg.py"), "--splits", str(self.splits_json()), "--host", QWEN25,
               "--out", str(d / "reg"), "--rows", str(sd / "cartridge.pt"), "--fmt", str(sd / "fmt.json"),
               "--threads", str(self.threads())]
        if (sd / "canaries.json").exists():
            cmd += ["--canaries", str(sd / "canaries.json")]
        for k in ("n_prefixes", "min_prefix", "max_prefix", "max_new", "min_run", "batch_size", "seed",
                  "canary_context", "canary_max_new", "n_random", "max_train_files"):
            if F.get(k) is not None:
                cmd += ["--" + k.replace("_", "-"), str(F[k])]
        r = run_guarded(cmd, d / "reg.log", self.a.rss_limit_gb * 1024, self.threads())
        if r["aborted"] or r["returncode"] != 0:
            raise StageFailed(st["name"], r)
        S = jload(d / "reg" / "summary.json")
        v = S["verdict"].get("rows[cartridge]")
        brief = {k: {kk: vv for kk, vv in c["regurgitation"].items() if kk != "secret_hits"}
                 for k, c in S["conditions"].items()}
        return dict(operating_point=op["selected"], verdict=v, reg_pass=None if v is None else v["pass"],
                    regurgitation=brief, canaries=S.get("canaries"), fingerprint=S.get("fingerprint"),
                    summary=str(d / "reg" / "summary.json"), wall_s=r["wall_s"], peak_rss_mb=r["peak_rss_mb"])

    def _m1c_per_file(self, op: dict, test: dict | None) -> dict:
        """Per-file Bar-A differences (E1(OP + counts@lambda_X) - E1(counts@lambda*)) on dev and test, with each
        file's share of the primary targets and n_eff (diagnostics; PREREG §8)."""
        from ngram_memory.eval.stage1 import bars as BR
        out: dict = {}
        prim = self._primary()
        row = op["op_row"]
        lx, ls = row["lambda_x"]["weight"], op["lambda_star"]["weight"]
        cres = self._res(op["counts_stage"])
        if lx is None or cres is None:
            return dict(note="no lambda_X for OP (Bar A cannot be read)" if lx is None else "no counts stage result")
        try:
            t0 = BR.Targets.load(cres["eval_dir"], "dev", prim)
            t = BR.Targets.load(jload(self.out / row["stage"] / "result.json")["eval_dir"], "dev", prim)
            xpl = f"{cres['label']}+rows[cartridge]"
            out["dev"] = BR.per_file(t, t.nll(xpl, lx), t0.nll(cres["label"], ls))
        except (OSError, KeyError, ValueError, IndexError) as e:
            out["dev"] = dict(error=repr(e))
        if test is not None and test.get("labels", {}).get("clx"):
            try:
                tt = BR.Targets.load(test["eval_dir"], "test", prim)
                out["test"] = BR.per_file(tt, tt.nll(test["labels"]["clx"]), tt.nll(test["labels"]["cl"]))
            except (OSError, KeyError, ValueError, IndexError) as e:
                out["test"] = dict(error=repr(e))
        return out

    def stage_m1c_report(self, st: dict) -> dict:
        """The pre-registered M1c decision (PREREG §7) on the stage results; no model is run. T_test and REG_op are
        optional dependencies (§12): when they failed, the dev-only KILL verdict still stands; otherwise the
        decision is INCOMPLETE with the reason."""
        from ngram_memory.eval.stage1 import bars as BR
        sel, op, test, reg = (self._res(st[k]) for k in ("select", "operating_point", "test", "reg"))
        rows = list(sel["candidates"]) + ([op["f1"]] if op and op.get("f1") else [])
        dev_any = any(r["any_bar"] for r in rows)
        out = dict(lambda_star=sel["lambda_star"], dev_table=sel["table"], dev_any_bar=dev_any,
                   s1=sel["selected"], operating_point=op and op["selected"], op_rule=op and op["rule"],
                   s1_bar_a_margin=op and op["s1"]["bar_a_margin_pts"],
                   f1_bar_a_margin=op and op.get("f1") and op["f1"]["bar_a_margin_pts"],
                   op_lambda_x=op and op["op_row"]["lambda_x"]["weight"])
        if op is not None and "counts_stage" in op:
            out["per_file_bar_a"] = self._m1c_per_file(op, test)
        if not dev_any:
            out.update(decision="KILL", reason="neither Bar A nor Bar B holds on dev for any arm: stop rows-only "
                       "layer-1 graft ingestion; next routes: an Engram-Adapter-style trained-reader arm or Flash-Next "
                       "native (M6)")
            return out
        if test is None or op is None:
            miss = st["test"] if test is None else st["operating_point"]
            out.update(decision="INCOMPLETE", reason=f"{miss} has no result ({self._stage_error(miss)})")
            return out
        prim = test["primary"]
        tb = test["bars"][prim]
        oprow = op["op_row"]
        ret = BR.within(oprow["x"])
        fr = (test.get("class_fractions") or {}).get("dup:16gram_in_train")
        gates = dict(host_unchanged=True,           # the train worker fails an arm on any host-weight change
                     dedupe=dict(test_16gram_in_train_fraction=fr, primary=prim,
                                 met=prim == "nondup" or (fr is not None and fr <= 0.02)),
                     suite="R0/I2/I4 are test-suite gates (tests/test_qwen25_real.py, test_graft_tiny.py, "
                           "test_hashing.py), run at the launch commit, not by this runner")
        gates_ok = gates["host_unchanged"] and gates["dedupe"]["met"]
        reg_pass = bool(reg and reg.get("reg_pass") is True)
        test_bar = bool(tb["bar_a"]["holds"] or tb["bar_b"]["holds"])
        parts = dict(test_bar_a=tb["bar_a"]["holds"], test_bar_b=tb["bar_b"]["holds"], retention_dev=ret,
                     c1=tb["c1"]["holds"], reg=reg_pass if reg is not None else None, gates=gates_ok)
        if reg is None:
            dec, why = "INCOMPLETE", f"{st['reg']} has no result ({self._stage_error(st['reg'])})"
        elif test_bar and ret and tb["c1"]["holds"] and reg_pass and gates_ok:
            dec, why = "PASS", "proceed to M2 with OP's recipe"
        else:
            dec, why = "REPORT", "bars on dev but the PASS conditions do not all hold on test: report, no M2"
        out.update(decision=dec, reason=why, parts=parts, test_bars=test["bars"], gates=gates,
                   reg=reg and dict(verdict=reg["verdict"], reg_pass=reg["reg_pass"]),
                   op_dev=dict(x=oprow["x"], lambda_x=oprow["lambda_x"], x_plus_counts=oprow["x_plus_counts"],
                               counts_matched=oprow["counts_matched"]))
        return out

    # ---------------------------------------------------------------------------------- M1d stage kinds
    def stage_train(self, st: dict) -> dict:
        """M1d: train one arm and do not evaluate it (its cartridges are evaluated by rows_eval stages, or only used
        by REG v2). Same worker and arm.json check as train_eval."""
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
                   "--threads", str(self.threads())] + (["--smoke"] if self.a.smoke else []) + grid_args()
            r = run_guarded(cmd, d / "train.log", self.a.rss_limit_gb * 1024, self.threads())
            if r["aborted"] or r["returncode"] != 0:
                raise StageFailed(name, r)
        tr = jload(d / "train.json")
        out = dict(config=arm, train=tr["train"], rows=tr["rows"], cartridge=tr["cartridge"], data=tr["data"],
                   snapshots=tr.get("snapshots") or [], host_unchanged=tr["host_unchanged"])
        for k in ("teacher", "canaries", "steps_mode", "canary_windows", "train_windows"):
            if k in tr:
                out[k] = {kk: v for kk, v in tr[k].items() if kk != "manifest"} if k == "canaries" else tr[k]
        return out

    def _cartridge_of(self, source: str, snapshot) -> tuple[Path, int, dict]:
        """(cartridge path, training step it was taken at, the source stage's result) of a train / train_eval
        stage: snapshot 'final' (cartridge.pt) or a pass number (cartridge_e<p>.pt, PREREG §2)."""
        src = self._res(source)
        if src is None:
            raise StageFailed(source, dict(error=f"{source} has no result"))
        sd = self.out / source
        if snapshot in (None, "final"):
            return sd / "cartridge.pt", int(src["train"]["steps"]), src
        snaps = {int(x["pass_no"]): x for x in src.get("snapshots") or []}
        if int(snapshot) not in snaps:
            raise StageFailed(source, dict(error=f"{source} has no pass-{snapshot} snapshot"))
        return Path(snaps[int(snapshot)]["path"]), int(snaps[int(snapshot)]["step"]), src

    def stage_rows_eval(self, st: dict) -> dict:
        """M1d: evaluate one cartridge of a train stage (st['source'], st['snapshot'] = pass number or 'final') on
        the dev sets exactly as train_eval evaluates its own: rows + arm_extra_conditions, base reused. The
        cartridge is linked as <stage>/cartridge.pt so its labels are rows[cartridge] (the bar code's labels)."""
        name = st["name"]
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        snap = st.get("snapshot", "final")
        cpath, step, src = self._cartridge_of(st["source"], snap)
        link = d / "cartridge.pt"
        if link.is_symlink() and Path(os.readlink(link)) != cpath:
            raise SystemExit(f"{name}: cartridge link points elsewhere ({os.readlink(link)})")
        if not link.exists():
            link.symlink_to(cpath)
        sd = self.out / st["source"]
        rows = f"rows:{link}"
        conds = ",".join([rows] + [c.replace("{rows}", rows) for c in self.grid.get("arm_extra_conditions", [])])
        self.mark(name, status="evaluating", source=st["source"], snapshot=snap, step=step)
        ev = self.evaluate(name, self.grid["dev_sets"], conds, reuse_base=self.out / self.grid["base_stage"] / "eval",
                           fmt_map={"cartridge": str(sd / "fmt.json")})
        res = dev_metrics(ev["summary"], "rows[cartridge]", self.grid.get("primary"))
        config = dict(src["config"], name=name, source=st["source"], snapshot=snap, snapshot_step=step)
        n_rows = next((int(x["rows"]) for x in src.get("snapshots") or [] if str(x["pass_no"]) == str(snap)),
                      src["cartridge"]["rows"])
        return dict(config=config, source=st["source"], snapshot=snap, snapshot_step=step, train=src["train"],
                    rows=src["rows"], cartridge=dict(rows=n_rows, path=str(cpath)), dev=res,
                    eval_dir=str(d / "eval"), eval_wall_s=ev["run"]["wall_s"], eval_peak_rss_mb=ev["run"]["peak_rss_mb"])

    def stage_reg2(self, st: dict) -> dict:
        """M1d REG v2 (PREREG §6): scripts/reg2.py on every listed cartridge against base, with the shared canary
        manifest (all listed arms must hold the same one); the positive control is a listed condition."""
        d = self.out / st["name"]
        F = self.grid.get("reg2_flags", {})
        cmd = [PY, str(ROOT / "scripts/reg2.py"), "--splits", str(self.splits_json()), "--host", QWEN25,
               "--out", str(d / "reg"), "--threads", str(self.threads())]
        shas, man, skipped = set(), None, []
        for c in st["conditions"]:
            if self._res(c["source"]) is None and c["source"] in st.get("optional_after", []):
                skipped.append(dict(label=c["label"], source=c["source"], why=self._stage_error(c["source"])))
                continue                     # a failed optional source (the positive control): run without it
            cpath, step, _src = self._cartridge_of(c["source"], c.get("snapshot", "final"))
            sd = self.out / c["source"]
            m = sd / "canaries.json"
            if not m.exists():
                raise StageFailed(st["name"], dict(error=f"{c['source']} has no canary manifest"))
            shas.add(jload(m)["sha256"])
            man = man or m
            cmd += ["--rows", str(cpath), "--fmt", str(sd / "fmt.json"), "--label", c["label"]]
            if c.get("positive_control"):
                cmd += ["--positive-control", c["label"]]
        if len(shas) != 1:
            raise StageFailed(st["name"], dict(error=f"the listed arms hold different canary manifests: {sorted(shas)}"))
        cmd += ["--canaries", str(man)]
        for k in ("n_prefixes", "min_prefix", "max_prefix", "max_new", "min_run", "batch_size", "score_batch", "seed",
                  "canary_context", "canary_max_new", "n_random", "n_boot", "max_train_files"):
            if F.get(k) is not None:
                cmd += ["--" + k.replace("_", "-"), str(F[k])]
        r = run_guarded(cmd, d / "reg.log", self.a.rss_limit_gb * 1024, self.threads())
        if r["aborted"] or r["returncode"] != 0:
            raise StageFailed(st["name"], r)
        S = jload(d / "reg" / "summary.json")
        brief = {k: dict(runs_excl=c["runs_excl"], true=c["true"],
                         canaries=[{kk: x[kk] for kk in ("id", "exact", "secret_lcp_chars", "rank", "exposure_bits",
                                                          "delta", "masked")} for x in c["canaries"]],
                         canary_rows=c.get("canary_rows")) for k, c in S["conditions"].items()}
        return dict(conditions=st["conditions"], skipped=skipped, verdict=S["verdict"], reg_pass=S["reg_pass"],
                    validated=S["validated"], versus_base=S["versus_base"], brief=brief, canaries=S["canaries"],
                    probes=S["probes"], summary=str(d / "reg" / "summary.json"), wall_s=r["wall_s"],
                    peak_rss_mb=r["peak_rss_mb"])

    def _rho_diag(self, stage: str, steps_per_pass: int | None = None) -> dict | None:
        """rho diagnostics from <stage>/metrics.jsonl (overall and per pass): how often the rho budget bound."""
        mp = self.out / stage / "metrics.jsonl"
        if not mp.exists():
            return None
        rec = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]

        def summ(rs):
            if not rs:
                return None
            pen = [r.get("rho_pen_near", 0.0) + r.get("rho_pen_chat", 0.0) for r in rs]
            over = [max(r.get("rho_frac_over_near", 0.0), r.get("rho_frac_over_chat", 0.0)) for r in rs]
            return dict(steps=len(rs), steps_with_rho_penalty=int(sum(p > 0 for p in pen)),
                        anchor_frac_over_max=float(max(over)),
                        train_rho_p99_max=float(max(r["rho_train_p99"] for r in rs)),
                        train_frac_over_max=float(max(r["rho_train_frac_over"] for r in rs)),
                        nll_mean=float(sum(r["nll"] for r in rs) / len(rs)))
        out = dict(all=summ(rec))
        if steps_per_pass:
            out["per_pass"] = {str(e + 1): summ([r for r in rec if r["step"] // steps_per_pass == e])
                               for e in range((max(r["step"] for r in rec) // steps_per_pass) + 1)} if rec else {}
        return out

    def stage_m1d_report(self, st: dict) -> dict:
        """M1d (PREREG §5, §7): dev bars per arm and snapshot against the fresh counts frontier (M1c's definitions),
        and the pre-registered M2-recipe rules. No test read; no model is run."""
        R_ = st["rules"]
        role = st["arms"]                                      # role -> stage holding its dev evaluation
        ctx, rows = self._m1c_rows(list(role.values()), st["counts"], st["name"])
        by = {r["stage"]: r for r in rows}
        p3 = self._res(st.get("p3_train", "P3_seed0")) or {}
        sm = p3.get("steps_mode") or {}
        per_pass = int(sm["train_windows"]) if sm.get("spec") == "epoch" and sm.get("train_windows") else None

        def row(k):
            return by.get(role.get(k))

        def A(k):
            r = row(k)
            return None if r is None or r["bar_a"]["diff"] is None else float(r["bar_a"]["diff"]["pts"])

        def sh(k):
            r = row(k)
            return None if r is None else float(r["x_share"])

        def num(x):
            return float("-inf") if x is None else x
        table = []
        for k, stg in role.items():
            r = by.get(stg)
            if r is None:
                table.append(dict(role=k, stage=stg, missing=self._stage_error(stg)))
                continue
            steps = r["config"].get("snapshot_step") or ((self._res(stg) or {}).get("train") or {}).get("steps")
            table.append(dict(role=k, stage=stg, steps=steps,
                              rho=r["config"]["rho"], objective=r["config"].get("objective", "nll"),
                              seed=r["config"]["seed"], e1=r["x"]["e1"], e1_ci=r["x"]["e1_ci"], e1_all=r["x"]["e1_all"],
                              r1=r["x"]["r1"], kl=r["x"]["kl"], r2=r["x"]["r2"], share=r["x_share"],
                              x_within=r["x_within"], lambda_x=r["lambda_x"]["weight"],
                              bar_a_diff=A(k), bar_a_ci=r["bar_a"]["diff"] and r["bar_a"]["diff"]["ci"],
                              bar_a=r["bar_a"]["holds"], bar_a_at_lstar=r["bar_a_at_lstar"]["diff"]["pts"],
                              counts_matched=r["counts_matched"]["weight"], bar_b_ratio=r["bar_b"]["ratio"],
                              bar_b_diff_ci=r["bar_b"]["diff"]["ci"], bar_b=r["bar_b"]["holds"], any_bar=r["any_bar"]))
        dec: dict = {}
        # 1. epochs (PREREG §7.1)
        P = R_["epochs"]
        ks = P["passes"]
        a = {k: A(f"P3_e{k}") for k in ks}
        s = {k: sh(f"P3_e{k}") for k in ks}
        w = {k: (row(f"P3_e{k}") or {}).get("x_within") for k in ks}
        steps_log, ep = [], None
        if all(row(f"P3_e{k}") is not None for k in ks):
            for k in ks[:-1]:
                nx = k + 1
                gain = None if a[nx] is None else a[nx] - num(a[k])
                dshare = s[nx] - s[k]
                stop = a[nx] is None or gain < P["min_gain_pts"] or dshare > P["max_share_increase"] or not w[nx]
                steps_log.append(dict(from_pass=k, to_pass=nx, bar_a_gain_pts=gain, share_increase=dshare,
                                      next_within=w[nx], stop=bool(stop)))
                if stop:
                    ep = k
                    break
            if ep is None:
                ep = ks[-1]
            dec["epochs"] = dict(value=ep, steps=steps_log, bar_a=a, share=s, still_gaining_at_last=ep == ks[-1]
                                 and bool(steps_log) and not steps_log[-1]["stop"], rule=P["text"])
        else:
            dec["epochs"] = dict(value=None, reason="a P3 snapshot has no dev result", bar_a=a, share=s)
        # 2. rho (PREREG §7.2)
        Q = R_["rho"]
        a1, ar = A("P3_e1"), A("R01_rho0p1")
        if row("P3_e1") is None or row("R01_rho0p1") is None:
            dec["rho"] = dict(value=None, reason="P3_e1 or R01 has no dev result")
        else:
            if ar is None and a1 is None:
                val, why = None, "neither has a lambda_X (no Bar-A difference)"
            elif ar is not None and ar >= num(a1) - Q["tie_pts"]:
                val, why = Q["small"], f"R01's Bar-A diff is within {Q['tie_pts']} points of P3_e1's, or above it"
            else:
                val, why = Q["large"], f"P3_e1's Bar-A diff exceeds R01's by more than {Q['tie_pts']} points"
            dec["rho"] = dict(value=val, reason=why, bar_a_r01=ar, bar_a_p3e1=a1,
                              diff=None if ar is None or a1 is None else ar - a1, rule=Q["text"],
                              rho_diag={"R01_rho0p1": self._rho_diag(role["R01_rho0p1"]),
                                        "P3_pass1": (self._rho_diag(st.get("p3_train", "P3_seed0"),
                                                                    per_pass) or {}).get("per_pass", {}).get("1")})
        # 3. distill (PREREG §7.3)
        D = R_["distill"]
        ad = A("DF_distill")
        if row("DF_distill") is None or row("P3_e1") is None:
            dec["distill"] = dict(value=None, reason="DF or P3_e1 has no dev result")
        else:
            keep = ad is not None and ad >= num(a1)
            dec["distill"] = dict(value="keep" if keep else "drop", bar_a_df=ad, bar_a_p3e1=a1,
                                  diff=None if ad is None or a1 is None else ad - a1, rule=D["text"])
        # 4. replicate (PREREG §7.4)
        r0, r1_ = row("P3_e1"), row("S1_seed1")
        if r0 is None or r1_ is None:
            dec["replicate"] = dict(value=None, reason="P3_e1 or S1_seed1 has no dev result")
        else:
            both = bool(r1_["bar_a"]["holds"] and r1_["bar_b"]["holds"])
            dA = None if A("S1_seed1") is None or a1 is None else abs(A("S1_seed1") - a1)
            dec["replicate"] = dict(abs_diff_e1=abs(r1_["x"]["e1"] - r0["x"]["e1"]), abs_diff_bar_a=dA,
                                    seed1_bar_a=r1_["bar_a"]["holds"], seed1_bar_b=r1_["bar_b"]["holds"],
                                    seed1_holds_both=both, m1c_pass_fragile=not both, rule=R_["replicate"]["text"])
            for k in ("rho", "distill"):                       # margins against the seed difference (annotation)
                df = dec.get(k, {}).get("diff")
                if df is not None and dA is not None:
                    dec[k]["within_seed_noise"] = bool(abs(df) <= dA)
            if dec.get("epochs", {}).get("steps") and dA is not None:
                for x in dec["epochs"]["steps"]:
                    if x["bar_a_gain_pts"] is not None:
                        x["within_seed_noise"] = bool(abs(x["bar_a_gain_pts"]) <= dA)
        ref = self.grid.get("m1c_reference", {})
        comp = None
        if row("P3_e1") is not None and ref:
            r = row("P3_e1")
            comp = dict(m1c=ref, m1d_p3_pass1=dict(e1=r["x"]["e1"], bar_a_diff=a1, share=r["x_share"],
                                                     lambda_x=r["lambda_x"]["weight"], bar_b_ratio=r["bar_b"]["ratio"],
                                                     counts_lambda_star=ctx["lambda_star"]["weight"],
                                                     counts_lambda_star_e1=ctx["lambda_star"]["e1"]),
                        delta=dict(e1=r["x"]["e1"] - ref["e1"],
                                   bar_a_diff=None if a1 is None else a1 - ref["bar_a_diff"],
                                   share=r["x_share"] - ref["share"]),
                        note=ref.get("note"))
        reg = self._res(st["reg"]) if st.get("reg") else None
        recipe = dict(objective="distill" if dec.get("distill", {}).get("value") == "keep" else "next-token",
                      epochs=dec.get("epochs", {}).get("value"), rho=dec.get("rho", {}).get("value"))
        return dict(ctx, table=table, decisions=dec, m2_recipe=recipe, m1c_comparison=comp,
                    p3_steps_per_pass=per_pass,
                    p3_rho_by_pass=(self._rho_diag(st.get("p3_train", "P3_seed0"), per_pass) or {}).get("per_pass"),
                    reg2=reg and dict(verdict=reg["verdict"], reg_pass=reg["reg_pass"], validated=reg["validated"],
                                      dose_response={k: v["dose_response"] for k, v in reg["versus_base"].items()}),
                    reg2_missing=None if reg is not None or not st.get("reg") else self._stage_error(st["reg"]),
                    candidates=rows)

    # ---------------------------------------------------------------------------------- main loop
    def kinds(self) -> dict:
        return dict(base=self.stage_base, train_eval=self.stage_train_eval, lora=self.stage_lora,
                    select=self.stage_select, test=self.stage_test, splitb=self.stage_splitb, r3=self.stage_r3,
                    frontier=self.stage_frontier, report=self.stage_report, countgrid=self.stage_countgrid,
                    reg=self.stage_reg, m1c_select=self.stage_m1c_select, m1c_op=self.stage_m1c_op,
                    m1c_test=self.stage_m1c_test, m1c_report=self.stage_m1c_report, train=self.stage_train,
                    rows_eval=self.stage_rows_eval, reg2=self.stage_reg2, m1d_report=self.stage_m1d_report)

    def waiting_for(self, st: dict) -> list[str]:
        """Dependencies not yet settled: a dependency is settled when it is done or was RSS-aborted, and an
        optional one (optional_after) also when it failed (this invocation, or for good: final_failed)."""
        opt = set(st.get("optional_after", []))
        return [x for x in st.get("after", []) if not self.done(x)
                and not self.progress["stages"].get(x, {}).get("aborted")
                and not (x in opt and self.progress["stages"].get(x, {}).get("status") in ("failed", "failed_final"))]

    def final_failed(self, name: str) -> bool:
        """A soft stage (every dependant lists it in optional_after) that is not done and is not run again: every
        dependant already finished without it (its result could no longer be used), or it failed or was RSS-aborted
        SOFT_MAX_ATTEMPTS times (the first run and one retry on resume; attempts are counted at failure)."""
        if name not in self.soft or self.done(name):
            return False
        e = self.progress["stages"].get(name, {})
        if e.get("status") == "failed_final":
            return True
        deps = [s["name"] for s in self.grid["stages"] if name in s.get("after", [])]
        if deps and all(self.done(d) for d in deps):
            return True
        if e.get("status") not in ("failed", "aborted"):
            return False
        return int(e.get("attempts", 1)) >= SOFT_MAX_ATTEMPTS

    def finalize_soft(self) -> None:
        """Record failed_final for every soft stage that final_failed says is not run again."""
        for st in self.grid["stages"]:
            n = st["name"]
            if self.final_failed(n) and self.progress["stages"].get(n, {}).get("status") != "failed_final":
                self.mark(n, status="failed_final")

    def _soft_failed(self, name: str) -> None:
        with self._lock:
            sf = self.progress.setdefault("soft_failed", [])
            if name not in sf:
                sf.append(name)
        log("soft failure", name, "(its dependants list it in optional_after and run without it)")
        if self.final_failed(name):
            self.mark(name, status="failed_final")

    def run_stage(self, st: dict, threads: int | None = None) -> str:
        """Run one stage on the calling thread. Returns 'done', 'aborted' (RSS guard; dependants go on without
        it), 'failed' (StageFailed, not aborted) or 'error' (a code error)."""
        name = st["name"]
        if threads is not None:
            self._tl.threads = threads
        log("start", name, st["kind"], f"threads {self.threads()}")
        self.mark(name, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), threads=self.threads())
        t = time.time()
        try:
            res = self.kinds()[st["kind"]](st)
        except Exception as e:  # noqa: BLE001  (a code error in a stage: record it and stop)
            if name in self.soft:                      # failed attempts are counted for soft stages only (M1b: none)
                self.mark(name, attempts=int(self.progress["stages"][name].get("attempts", 0)) + 1)
            if not isinstance(e, StageFailed):
                import traceback
                log("ERROR", name, repr(e))
                self.mark(name, status="failed", error=traceback.format_exc()[-4000:])
                return "error"
            log("FAILED", name, e.info)
            self.mark(name, status="aborted" if e.info.get("aborted") else "failed", aborted=bool(e.info.get("aborted")),
                      error=e.info)
            return "aborted" if e.info.get("aborted") else "failed"
        res["stage_wall_s"] = time.time() - t
        res["threads"] = self.threads()
        jdump(self.out / name / "result.json", res)
        self.mark(name, status="done", wall_s=res["stage_wall_s"], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        log("done", name, f"{res['stage_wall_s']:.0f}s")
        return "done"

    def run(self) -> int:
        """Exit code: 1 after a code error or a failed stage (unless --keep-going, or the stage is soft); 0
        otherwise. A soft stage's failure (every dependant lists it in optional_after) does not stop the run or set
        the exit code: its dependants run without it, and it is listed in progress.json 'soft_failed' and in their
        results (OP_select's rule, Z_report's INCOMPLETE reason)."""
        rc = self.run_serial() if self.a.parallel <= 1 else self.run_parallel()
        if self.soft:                                   # M1b: no soft stages, progress.json unchanged
            self.finalize_soft()
        with self._lock:
            self.progress["finished"] = all(self.done(s["name"]) for s in self.grid["stages"])
            if self.soft:
                self.progress["complete"] = all(
                    self.done(s["name"]) or self.progress["stages"].get(s["name"], {}).get("status") == "failed_final"
                    for s in self.grid["stages"])
            self.save()
        return rc

    def run_serial(self) -> int:
        for st in self.grid["stages"]:
            name = st["name"]
            if self.done(name):
                log("skip (done)", name)
                continue
            if self.final_failed(name):
                log("skip (failed, final)", name)
                self.mark(name, status="failed_final")
                continue
            need = self.waiting_for(st)
            if need:
                log("blocked", name, "waiting for", need)
                self.mark(name, status="blocked", waiting_for=need)
                continue
            r = self.run_stage(st)
            if r in ("error", "failed", "aborted") and name in self.soft:
                self._soft_failed(name)
                continue
            if r == "error" or (r == "failed" and not self.a.keep_going):
                return 1
            if r == "done" and self.a.stop_after and name == self.a.stop_after:
                log("stop-after reached")
                return 0
        return 0

    def run_parallel(self) -> int:
        """Up to --parallel stages at once, each started as soon as every stage it depends on has finished (or
        was RSS-aborted; an optional dependency, optional_after, also when it failed). The free threads are split
        evenly among the stages started together, capped at --max-stage-threads. A new stage starts only while
        MemAvailable >= --launch-min-free-gb. After a failure of a non-soft stage (or once --stop-after's stage is
        done) no new stage starts; running ones finish."""
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        order = self.grid["stages"]
        settled = set()                                 # done before this invocation, or finished/aborted in it
        failed_soft = set()                             # soft stages that failed (in this invocation, or for good)
        for st in order:
            if self.done(st["name"]):
                log("skip (done)", st["name"])
                settled.add(st["name"])
            elif self.final_failed(st["name"]):
                log("skip (failed, final)", st["name"])
                self.mark(st["name"], status="failed_final")
                failed_soft.add(st["name"])
        pending = [st for st in order if st["name"] not in settled and st["name"] not in failed_soft]
        running: dict = {}                              # future -> (stage name, threads)
        rc, stop = 0, False
        cap = self.a.max_stage_threads or self.a.threads

        def ready_now(st) -> bool:
            opt = set(st.get("optional_after", []))
            return set(st.get("after", [])) - opt <= settled and opt <= settled | failed_soft

        def reap(futs) -> None:
            nonlocal rc, stop
            for f in futs:
                name, _t = running.pop(f)
                r = f.result()
                if r != "done" and name in self.soft:   # failed / error / aborted: its dependants go on without it
                    failed_soft.add(name)
                    self._soft_failed(name)
                    continue
                if r in ("done", "aborted"):
                    settled.add(name)
                if r == "error" or (r == "failed" and not self.a.keep_going):
                    rc, stop = 1, True
                if r == "done" and self.a.stop_after and name == self.a.stop_after:
                    log("stop-after reached; letting running stages finish")
                    stop = True

        with ThreadPoolExecutor(max_workers=self.a.parallel) as ex:
            while True:
                reap([f for f in list(running) if f.done()])   # collect every finished stage before deciding
                if not stop:
                    ready = [st for st in pending if ready_now(st)]
                    slots = min(self.a.parallel - len(running), len(ready))
                    free = self.a.threads - sum(t for _, t in running.values())
                    for i, st in enumerate(ready[:slots]):
                        if free < 1 or (running and mem_available_gb() < self.a.launch_min_free_gb):
                            break
                        t = max(1, min(cap, free // (slots - i)))
                        free -= t
                        pending.remove(st)
                        running[ex.submit(self.run_stage, st, t)] = (st["name"], t)
                if not running:
                    break
                wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
        for st in pending:
            need = self.waiting_for(st)
            if need:
                self.mark(st["name"], status="blocked", waiting_for=need)
        return rc


SOFT_MAX_ATTEMPTS = 2          # a soft stage runs at most twice: its first run and one retry on resume


def soft_stages(stages: list[dict]) -> set[str]:
    """Stages whose every dependant lists them in optional_after: their failure does not block or stop the run."""
    deps: dict[str, list[bool]] = {}
    for st in stages:
        opt = set(st.get("optional_after", []))
        for x in st.get("after", []):
            deps.setdefault(x, []).append(x in opt)
    return {x for x, v in deps.items() if v and all(v)}


def mem_available_gb() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 2**20
    except OSError:
        pass
    return float("inf")


class StageFailed(Exception):
    def __init__(self, stage: str, info: dict):
        super().__init__(stage)
        self.info = info


def _brief(s: dict) -> dict:
    return dict(sets=s.get("sets"), tuning={k: v for k, v in s.get("tuning", {}).items() if k != "build_s"},
                warnings=s.get("warnings"), peak_rss_mb=s.get("peak_rss_mb"), wall_s_total=s.get("wall_s_total"),
                data_regime=s.get("data_regime"))


def dev_metrics(s: dict, label: str, primary: str | None = None) -> dict:
    """The dev-stage numbers the selection rules read: E1 on Split-A dev, R1 (WikiText-2), R2 (stdlib + ML repo).
    primary 'nondup' (M1c): E1 from the non-duplicated targets (paired_nondup); the all-target E1 is kept as
    e1_all_*. None (M1b): all targets, unchanged."""
    S = s["sets"]

    def p(set_name):
        return S[set_name][label]["paired"]
    e1 = p("dev")
    if primary == "nondup":
        out = dev_metrics(s, label)
        nd = S["dev"][label]["paired_nondup"]
        lo, hi = nd["rel_pct_ci"]
        out.update(primary="nondup", e1_all_rel_pct=out["e1_rel_pct"], e1_all_gain_pct=out["e1_gain_pct"],
                   e1_all_ci_pct=out["e1_ci_pct"], e1_all_n_tokens=out["e1_n_tokens"], e1_rel_pct=nd["rel_pct"],
                   e1_gain_pct=-nd["rel_pct"], e1_ci_pct=[lo, hi], e1_n_tokens=nd["n_tokens"])
        return out
    elif primary not in (None, "all"):
        raise ValueError(f"primary {primary!r}: 'nondup' or 'all'")
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
    ap.add_argument("--parallel", type=int, default=1, help="stages run side by side when their dependencies allow")
    ap.add_argument("--max-stage-threads", type=int, default=0, help="--parallel: thread cap per stage (0: --threads)")
    ap.add_argument("--launch-min-free-gb", type=float, default=8.0,
                    help="--parallel: start another stage only while MemAvailable is at least this")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="worker: cap the steps (timing probes only)")
    ap.add_argument("--grid", default=None, help="grid JSON (default: experiments/m1b/grid.json; M1c: "
                                                 "experiments/m1c/grid.json)")
    a = ap.parse_args()
    if a.grid:
        set_grid_path(a.grid)
    if a.worker == "train":
        return worker_train(a)
    if a.worker == "frontier":
        return worker_frontier(a)
    log("M1b runner" if not grid_args() else f"runner ({GRID_PATH})", vars(a))
    return Runner(a).run()


if __name__ == "__main__":
    sys.exit(main())
