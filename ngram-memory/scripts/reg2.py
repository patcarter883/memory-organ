"""REG v2 harness CLI (M1d, experiments/m1d/PREREG.md §6; library ngram_memory/eval/stage1/reg2.py). M1c's REG is
scripts/reg.py, unchanged.

    python scripts/reg2.py --splits SPLITS --out DIR --canaries canaries.json \
        --rows A.pt --fmt A.json --label P3_e1 [--rows B.pt --fmt B.json --label S1 ...] \
        [--positive-control PC] [--n-prefixes 200] [--n-random 1000] [--threads 4]

The base host is always run first; every --rows cartridge (served through the graft, as evaluate.py) is compared
with it on the same 200 train-file prefixes and the same masked canary prompts. All cartridges must have been trained
on the corpus planted by the one --canaries manifest (the runner checks the manifests are identical).

Writes DIR/summary.json (no secret strings: canary ids, ranks, deltas and counts only), DIR/prefixes.json, and,
private to the run directory (they can hold planted secrets): DIR/continuations__<label>.jsonl and
DIR/canary_probes__<label>.jsonl.
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


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--host", default=QWEN25)
    ap.add_argument("--rows", action="append", default=[], help="SparseRowStore state dict (repeatable)")
    ap.add_argument("--fmt", action="append", default=[], help="FormatSpec JSON (one per --rows, or one for all)")
    ap.add_argument("--label", action="append", default=[], help="label per --rows (default rows[<stem>])")
    ap.add_argument("--positive-control", action="append", default=[], help="labels that are positive controls")
    ap.add_argument("--canaries", required=True, help="canary manifest (ingest/canaries.py v1 or v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-prefixes", type=int, default=200)
    ap.add_argument("--min-prefix", type=int, default=64)
    ap.add_argument("--max-prefix", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--min-run", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--score-batch", type=int, default=16, help="reference tails per cached-prompt batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canary-context", type=int, default=256)
    ap.add_argument("--canary-max-new", type=int, default=48)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-train-files", type=int, default=None, help="smoke: first N train files (+ canary files)")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from ngram_memory.eval.stage1 import reg as R
    from ngram_memory.eval.stage1 import reg2 as R2
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
    if a.label and len(a.label) != len(a.rows):
        raise SystemExit("--label: give one per --rows")
    labels = a.label or [f"rows[{Path(r).stem}]" for r in a.rows]
    if len(set(labels)) != len(labels) or "base" in labels:
        raise SystemExit("--label: labels must be distinct and not 'base'")
    if set(a.positive_control) - set(labels):
        raise SystemExit("--positive-control names a label that is not given")
    splits = Splits.load(a.splits)
    tok = AutoTokenizer.from_pretrained(a.host)
    model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    eos = int(cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id[0])
    pad = int(tok.pad_token_id) if tok.pad_token_id is not None else eos
    key = qwen25_proxy_key_spec()

    man = CN.load_manifest(a.canaries)
    docs, sites = R.load_train_docs(splits, tok, man, a.max_train_files)
    # Split-A test is not opened in M1d (PREREG s0); planting already refuses non-train paths.
    held = [p for part in ("dev", "near_dup") for p in splits.files("a", part)]
    CN.assert_clean({p: splits.text(p) for p in held}, man)
    repeats = {}
    for r in man["records"]:
        repeats[r["id"]] = int(r.get("repeats", man["config"].get("repeats", 1)))
    S: dict = dict(format=R2.FORMAT, host=a.host, splits=a.splits, split_commit=splits.commit,
                   args={k: v for k, v in vars(a).items()}, train=dict(files=len(docs),
                                                                        tokens=int(sum(len(d) for d in docs))),
                   canaries=dict(CN.manifest_brief(man), sites=len(sites), held_out_clean=True, repeats=repeats),
                   positive_controls=list(a.positive_control), conditions={}, verdict={}, versus_base={})
    stream = R.train_stream(docs, eos)
    t = time.time()
    index = R.VerbatimIndex(stream)
    prefixes = R.sample_prefixes(docs, a.n_prefixes, a.min_prefix, a.max_prefix, a.max_new, a.seed, sites)
    (out / "prefixes.json").write_text(json.dumps(R.prefixes_json(prefixes, docs), indent=0))
    spans = R2.prompt_spans(docs, prefixes)
    S["prefixes"] = dict(n=len(prefixes), length=R._dist([p.length for p in prefixes]),
                         files=len({p.doc for p in prefixes}))
    train_lines = {ln.strip() for d in docs for ln in d.text.split("\n") if ln.strip()}
    probes = R2.masked_canary_prompts(docs, sites, tok, a.canary_context)
    refs = R2.exposure_refs(sites, a.n_random, a.seed)
    S["probes"] = [dict(id=p.site.id, repeats=repeats[p.site.id], kind=p.site.kind, prompt_tokens=int(len(p.prompt)),
                        masked=p.masked) for p in probes]
    log("train", S["train"], "prefixes", S["prefixes"], "probes", len(probes), f"refs {a.n_random}",
        f"setup {time.time() - t:.1f}s")
    secrets = sorted({s.secret for s in sites})
    sec_id = {s.secret: s.id for s in sites}
    clean = R2.clean_rows(docs, sites, key)          # rows read outside the canaries: the rest are canary-specific

    def run(label: str, cond, store=None) -> dict:
        t = time.time()
        rg = R.regurgitation(model, tok, docs, prefixes, index, eos, pad, cond, a.max_new, a.batch_size, a.min_run,
                             train_lines)
        conts = rg["_continuations"]
        ex = R2.excluded_longest(stream, conts, spans, full=rg["_longest"])
        lcp, avail = R2.true_lcp(docs, prefixes, conts)
        res = dict(describe=cond.describe(),
                   regurgitation={k: v for k, v in rg.items() if not k.startswith("_") and k != "secret_hits"},
                   runs_excl=dict(longest=R._dist(ex["longest"]), n_ge_min_run=int(sum(x >= a.min_run
                                                                                         for x in ex["longest"])),
                                  recomputed=ex["recomputed"], min_run=a.min_run),
                   true=R2.true_stats(lcp, avail, conts, a.max_new, a.min_run),
                   secret_hits=[{k: v for k, v in h.items()} for h in rg["secret_hits"]],
                   _lcp=lcp, _excl=ex["longest"])
        with open(out / f"continuations__{label}.jsonl", "w") as f:
            for p, c, txt, lg, xl, tl in zip(prefixes, conts, rg["_texts"], rg["_longest"], ex["longest"], lcp):
                f.write(json.dumps(dict(doc=docs[p.doc].name, start=p.start, length=p.length, longest_match=lg,
                                        longest_excl_prompt=xl, true_lcp=tl, ids=list(map(int, c)), text=txt)) + "\n")
        cin = R.canary_hits_in(rg["_texts"], secrets)
        res["canary_in_continuations"] = [dict(text=h["text"], id=sec_id[h["secret"]], exact=h["exact"],
                                               chars=h["chars"]) for h in cin]
        tc = time.time()
        res["canaries"] = R2.canary_probe_v2(model, tok, probes, eos, pad, cond, refs, a.canary_max_new,
                                             a.batch_size, a.score_batch)
        res["canary_s"] = time.time() - tc
        with open(out / f"canary_probes__{label}.jsonl", "w") as f:
            for c in res["canaries"]:
                f.write(json.dumps(dict(id=c["id"], generated=c["_generated"], logp_true=c["logp_true"],
                                        logp_refs=c["_logp_refs"])) + "\n")
        if store is not None:
            ids = store.row_ids.numpy()
            nrm = store.values.detach().float().norm(dim=1).numpy()
            res["canary_rows"] = R2.canary_rows(docs, sites, key, ids, nrm, clean)
        res["wall_s"] = time.time() - t
        log(label, f"excl-prompt >= {a.min_run}: {res['runs_excl']['n_ge_min_run']}",
            f"true LCP mean {res['true']['true_lcp']['mean']:.2f} >= {a.min_run}: {res['true']['n_ge_min_run']}",
            f"exact {res['true']['n_exact']}", f"canary exact {sum(c['exact'] for c in res['canaries'])}",
            f"rank1 {sum(c['rank'] == 1 for c in res['canaries'])}", f"{res['wall_s']:.0f}s")
        return res

    def public(res: dict) -> dict:
        r = {k: v for k, v in res.items() if not k.startswith("_")}
        r["canaries"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in res["canaries"]]
        return r

    def save():
        S["wall_s_total"] = time.time() - T0
        S["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    base = run("base", Condition())
    S["conditions"]["base"] = public(base)
    save()
    for i, rp in enumerate(a.rows):
        f = FormatSpec.from_json(Path(a.fmt[i if len(a.fmt) > 1 else 0]).read_text()) if a.fmt else \
            graft_format(cfg.hidden_size, key)
        st = SparseRowStore.from_state_dict(torch.load(rp, map_location="cpu", weights_only=True))
        label = labels[i]
        res = run(label, RowsCondition(f, st, name=label), st)
        S["conditions"][label] = public(res)
        v = R2.verdict_v2(base, res)
        deltas = [dict(id=c["id"], repeats=repeats[c["id"]], **R2.canary_delta(b, c, a.n_boot, a.seed))
                  for b, c in zip(base["canaries"], res["canaries"])]
        S["versus_base"][label] = dict(
            true_lcp_mean_diff=R2.paired_mean_diff(res["_lcp"], base["_lcp"], a.n_boot, a.seed),
            excl_longest_mean_diff=R2.paired_mean_diff(res["_excl"], base["_excl"], a.n_boot, a.seed),
            canary_delta=deltas, dose_response=R2.dose_response(deltas, repeats),
            canary_rows_nonzero={r["id"]: r["nonzero"] for r in res.get("canary_rows", [])},
            canary_specific_rows_nonzero={r["id"]: dict(rows=r["specific_rows"], nonzero=r["specific_in_cartridge"])
                                          for r in res.get("canary_rows", [])})
        if label in a.positive_control:
            v["positive_control"] = True
            v["flagged"] = R2.positive_control_flagged(v)
        S["verdict"][label] = v
        log("verdict", label, json.dumps({k: v[k] for k in ("passed", "parts")}),
            f"dose slope {S['versus_base'][label]['dose_response']['slope_nats_per_doubling']}")
        save()
    pcs = [S["verdict"][x] for x in a.positive_control]
    S["validated"] = bool(pcs) and all(v["flagged"] for v in pcs)
    S["reg_pass"] = {k: v["passed"] for k, v in S["verdict"].items() if k not in a.positive_control}
    save()
    log("done", f"{time.time() - T0:.0f}s", "validated" if S["validated"] else "NOT validated (positive control)")
    return S


if __name__ == "__main__":
    main()
