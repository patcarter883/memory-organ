"""REG harness CLI (§6.A.3 regurgitation and secrets; bar in the §6.A.7 REG row).

    python scripts/reg.py --splits SPLITS(.json|dir) --out DIR [--host HOST] \
        [--rows P.pt [--rows Q.pt ...]] [--fmt F.json [--fmt G.json ...]] [--canaries canaries.json] \
        [--n-prefixes 200] [--max-new 128] [--threads 4] [--batch-size 8] [--reuse-base DIR]

The base host is always run (the bar is relative to it). Each --rows file is a SparseRowStore state dict served
through the graft (RowsCondition, as evaluate.py's rows condition); its FormatSpec is the matching --fmt (one --fmt
for all, one per --rows, or none: graft_format(hidden, qwen25_proxy_key_spec())). --canaries is the arm's planting
manifest (ingest/canaries.py); without it the canary check is reported as not run.

Writes DIR/summary.json (per condition: regurgitation stats, canary extraction / exposure, secret-scan hits; per
rows condition: the REG verdict), DIR/prefixes.json, DIR/continuations__<label>.jsonl and DIR/base_result.json
(reusable with --reuse-base when the prefixes, host, canaries and decode settings are identical).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory import paths as _paths  # noqa: E402
QWEN25 = str(_paths.qwen25())


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--host", default=QWEN25)
    ap.add_argument("--rows", action="append", default=[], help="SparseRowStore state dict (repeatable)")
    ap.add_argument("--fmt", action="append", default=[], help="FormatSpec JSON (one for all, or one per --rows)")
    ap.add_argument("--canaries", default=None, help="canary manifest JSON of the arm (ingest/canaries.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-prefixes", type=int, default=200)
    ap.add_argument("--min-prefix", type=int, default=64)
    ap.add_argument("--max-prefix", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--min-run", type=int, default=64, help="verbatim run length of the bar")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canary-context", type=int, default=256, help="training tokens before each canary secret")
    ap.add_argument("--canary-max-new", type=int, default=48)
    ap.add_argument("--n-random", type=int, default=100, help="reference secrets for the exposure rank")
    ap.add_argument("--max-train-files", type=int, default=None, help="smoke: first N train files (+ canary files)")
    ap.add_argument("--reuse-base", default=None, help="dir holding base_result.json from an identical setup")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from ngram_memory.eval.stage1 import reg as R
    from ngram_memory.eval.stage1.conditions import RowsCondition
    from ngram_memory.eval.stage1.scoring import Condition
    from ngram_memory.eval.stage1.splits import Splits
    from ngram_memory.ingest import canaries as CN
    from ngram_memory.spec import FormatSpec, graft_format, qwen25_proxy_key_spec
    from ngram_memory.tables import SparseRowStore

    T0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fmt and len(a.fmt) not in (1, len(a.rows)):
        raise SystemExit("--fmt: give one for all --rows or one per --rows")
    splits = Splits.load(a.splits)
    tok = AutoTokenizer.from_pretrained(a.host)
    model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    eos = int(cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id[0])
    pad = int(tok.pad_token_id) if tok.pad_token_id is not None else eos

    man = CN.load_manifest(a.canaries) if a.canaries else None
    docs, sites = R.load_train_docs(splits, tok, man, a.max_train_files)
    S: dict = dict(format=R.FORMAT, host=a.host, splits=a.splits, args=vars(a), split_commit=splits.commit,
                   train=dict(files=len(docs), tokens=int(sum(len(d) for d in docs))), conditions={}, verdict={})
    if man is not None:
        held = [p for part in ("dev", "test", "near_dup") for p in splits.files("a", part)]
        CN.assert_clean({p: splits.text(p) for p in held}, man)     # dev / test / near-dup never hold a canary
        S["canaries"] = dict(CN.manifest_brief(man), sites=len(sites), held_out_clean=True)
    else:
        S["canaries"] = None
    stream = R.train_stream(docs, eos)
    t = time.time()
    index = R.VerbatimIndex(stream)
    prefixes = R.sample_prefixes(docs, a.n_prefixes, a.min_prefix, a.max_prefix, a.max_new, a.seed, sites)
    (out / "prefixes.json").write_text(json.dumps(R.prefixes_json(prefixes, docs), indent=0))
    S["prefixes"] = dict(n=len(prefixes), length=R._dist([p.length for p in prefixes]),
                         files=len({p.doc for p in prefixes}))
    train_lines = {ln.strip() for d in docs for ln in d.text.split("\n") if ln.strip()}
    cprompts = R.canary_prompts(docs, sites, a.canary_context) if man is not None else []
    refs = R.exposure_candidates(sites, a.n_random, a.seed) if man is not None else None
    secrets = sorted({s.secret for s in sites})
    log("train", S["train"], "prefixes", S["prefixes"], "canaries", len(cprompts), f"setup {time.time() - t:.1f}s")

    fp = R.fingerprint(host=a.host, commit=splits.commit, train=[d.name for d in docs], stream=len(stream),
                       manifest=man["sha256"] if man else None, prefixes=[(p.doc, p.start, p.length) for p in prefixes],
                       max_new=a.max_new, batch=a.batch_size, min_run=a.min_run, cctx=a.canary_context,
                       cnew=a.canary_max_new, n_random=a.n_random, seed=a.seed)
    S["fingerprint"] = fp

    def run(label: str, cond) -> dict:
        t = time.time()
        rg = R.regurgitation(model, tok, docs, prefixes, index, eos, pad, cond, a.max_new, a.batch_size, a.min_run,
                             train_lines)
        with open(out / f"continuations__{label}.jsonl", "w") as f:
            for p, c, txt, lg, tl in zip(prefixes, rg["_continuations"], rg["_texts"], rg["_longest"], rg["_true_lcp"]):
                f.write(json.dumps(dict(doc=docs[p.doc].name, start=p.start, length=p.length, longest_match=lg,
                                        true_lcp=tl, ids=list(map(int, c)), text=txt)) + "\n")
        res = dict(describe=cond.describe(), regurgitation=_public(rg))
        if man is not None:
            res["canaries"] = R.canary_probe(model, tok, cprompts, eos, pad, cond, a.canary_max_new, refs,
                                             a.batch_size)
            res["canary_in_continuations"] = R.canary_hits_in(rg["_texts"], secrets)
            # secret scan of the canary-probe continuations: reported only (base given `X_API_TOKEN = "` is
            # expected to invent a value; canaries are judged by exact match)
            res["canary_probe_secret_hits"] = int(sum(len(c["secret_scan_hits"]) for c in res["canaries"]))
        res["wall_s"] = time.time() - t
        log(label, f"longest mean {rg['longest_match']['mean']:.1f} max {rg['longest_match']['max']:.0f}",
            f">={a.min_run}: {rg['n_ge_min_run']}/{rg['n']}", f"secret hits {rg['n_secret_hits']}",
            f"canary exact {sum(c['exact'] for c in res.get('canaries', []))}",
            f"{rg['s_per_continuation']:.2f} s/continuation", f"{res['wall_s']:.0f}s")
        return res

    base = None
    if a.reuse_base:
        bp = Path(a.reuse_base) / "base_result.json"
        if bp.exists():
            b = json.loads(bp.read_text())
            if b.get("fingerprint") == fp:
                base = b["result"]
                S["reused_base"] = str(bp)
                src = Path(a.reuse_base) / "continuations__base.jsonl"
                if src.exists() and src.resolve() != (out / src.name).resolve():
                    import shutil
                    shutil.copyfile(src, out / src.name)       # keep base's continuations beside the rows ones
                S["reused_base_continuations"] = str(src) if src.exists() else None
            else:
                S.setdefault("warnings", []).append(f"--reuse-base {bp}: fingerprint differs; base re-run")
    if base is None:
        base = run("base", Condition())
    (out / "base_result.json").write_text(json.dumps(dict(fingerprint=fp, result=base), default=float))
    S["conditions"]["base"] = base

    def save():
        S["wall_s_total"] = time.time() - T0
        S["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    save()
    for i, rp in enumerate(a.rows):
        if a.fmt:
            f = FormatSpec.from_json(Path(a.fmt[i if len(a.fmt) > 1 else 0]).read_text())
        else:
            f = graft_format(cfg.hidden_size, qwen25_proxy_key_spec())
        st = SparseRowStore.from_state_dict(torch.load(rp, map_location="cpu", weights_only=True))
        label = f"rows[{Path(rp).stem}]"
        if label in S["conditions"]:
            label = f"rows[{Path(rp).stem}#{i}]"
        res = run(label, RowsCondition(f, st, name=label))
        S["conditions"][label] = res
        S["verdict"][label] = R.verdict(base, res, man is not None)
        log("verdict", label, json.dumps({k: v for k, v in S["verdict"][label].items() if k != "notes"}, default=str))
        save()
    S["reg_pass"] = {k: v["pass"] for k, v in S["verdict"].items()}
    save()
    log("done", f"{time.time() - T0:.0f}s")
    return S


if __name__ == "__main__":
    main()
