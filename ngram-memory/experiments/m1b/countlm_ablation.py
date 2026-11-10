#!/usr/bin/env python
"""Diagnostic follow-up to M1b (not part of the pre-registered run): what the count-LM baseline's 4x lead is made of,
and what it costs when left on.

    python experiments/m1b/countlm_ablation.py --m1b M1B_RUN_DIR --out DIR [--threads 32] [--groups 3]

Scores, on Split-A dev (the 13 scored files, same windows as M1b) and the three retention sets (R1 WikiText-2,
R2 stdlib, R2ml ML code), each condition ALWAYS ON:

    countlm             the M1b baseline as run (KN trigram + in-file cache, default lambda grid)
    countlm:fine        same, finer lambda grid (M1b's tuned lambdas sat on the grid's smallest value, 0.01)
    countlm:kn-fine     KN trigram counts only (no in-file cache)
    countlm:cache-fine  in-file cache only (no train counts: uses only the file being scored)
    rows:A3             the M1b operating point's table (A3_rho03), same run, for pairing
    countlm:fine+rows:A3  both together (does the table add anything on top of the counts?)

Lambdas are tuned on the same single unscored dev file M1b used. The Split-A test files are NOT read (M1b's single
test read stays the only one). The base is reused from M1b's B0 stage (same windows, checked by evaluate.py).
Condition groups run as separate evaluate.py processes side by side, splitting --threads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402

SETS = "dev,r1,r2,r2ml"


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1b", required=True, help="the M1b run directory (holds B0_base_dev/, A3_rho03/, _cache/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--rss-limit-gb", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true", help="the M1b run was a --smoke run (use its eval flags)")
    ap.add_argument("--max-windows", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    m1b, out = Path(a.m1b).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    op = json.loads((m1b / "S3_op" / "result.json").read_text())["selected"]
    cart, fmt = m1b / op / "cartridge.pt", m1b / op / "fmt.json"
    base_dir = m1b / "B0_base_dev" / "eval"
    for p in (cart, fmt, base_dir / "dev__base.npz", m1b / "_cache" / "splits.json"):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    rows = f"rows:{cart}"
    groups_all = [["countlm", "countlm:fine"], ["countlm:kn-fine", "countlm:cache-fine"],
                  [rows, f"countlm:fine+{rows}"]]
    groups = groups_all if a.groups >= 3 else [sum(groups_all, [])]
    t_each = max(1, a.threads // len(groups))
    E = R.load_grid(a.smoke)["eval_flags"]
    fm = out / "fmt_map.json"
    fm.write_text(json.dumps({"cartridge": str(fmt)}))
    jobs = []
    for i, conds in enumerate(groups):
        d = out / f"g{i}"
        cmd = [R.PY, str(ROOT / "scripts/evaluate.py"), "--splits", str(m1b / "_cache" / "splits.json"),
               "--host", R.QWEN25, "--out", str(d), "--sets", SETS, "--conditions", ",".join(conds),
               "--threads", str(t_each), "--context", str(E["context"]), "--stride", str(E["stride"]),
               "--min-context", str(E["min_context"]), "--retention-tokens", str(E["retention_tokens"]),
               "--tune-files", str(E["tune_files"]), "--ref-vocab-cache", str(m1b / "_cache" / "ref_vocab.json"),
               "--reuse-base", str(base_dir), "--fmt-map", str(fm)]
        for k, flag in (("max_files", "--max-files"), ("max_dev_files", "--max-dev-files"),
                        ("max_windows", "--max-windows"), ("tune_max_windows", "--tune-max-windows")):
            if E.get(k) is not None:
                cmd += [flag, str(E[k])]
        if a.max_windows and E.get("max_windows") is None:
            cmd += ["--max-windows", str(a.max_windows)]
        jobs.append((d, cmd))
    log("groups", [g for g in groups], "threads each", t_each, "operating point", op)
    with ThreadPoolExecutor(len(jobs)) as ex:
        runs = list(ex.map(lambda j: R.run_guarded(j[1], j[0].with_suffix(".log"), a.rss_limit_gb * 1024, t_each), jobs))
    for (d, _), r in zip(jobs, runs):
        log(d.name, r)
        if r["returncode"] != 0 or r["aborted"]:
            raise SystemExit(f"{d.name} failed: {r}")

    rows_out, tuning = [], {}
    for d, _ in jobs:
        s = json.loads((d / "summary.json").read_text())
        tuning.update({k: v for k, v in s["tuning"].items() if k.startswith("countlm")})
        for label in s["sets"]["dev"]:
            if label == "base" or label.startswith("_"):
                continue
            m = R.dev_metrics(s, label)
            rows_out.append(dict(condition=label, **{k: m[k] for k in ("e1_gain_pct", "e1_ci_pct", "r1_rel_pct",
                                                                       "r1_mean_kl", "r2_rel_pct", "r2_parts")},
                                 dev_classes={k: m["classes"][k]["rel_pct"] for k in (
                                     "id:proj:cont_sub", "key:seen_3gram_seen_next", "lex:ws", "lex:op",
                                     "dup:16gram_not_in_train") if m.get("classes") and k in m["classes"]}))
    rep = dict(operating_point=op, sets=SETS, threads_each=t_each, conditions=rows_out, tuning=tuning)
    R.jdump(out / "ablation.json", rep)
    lines = ["| condition | dev E1 gain % [CI] | R1 % (mean KL) | R2 % stdlib / ML | lambdas (n, cache) |",
             "|---|---|---|---|---|"]
    for r in rows_out:
        ci = r["e1_ci_pct"] or [None, None]
        lo, hi = sorted(-x for x in ci) if None not in ci else (None, None)
        tu = tuning.get(r["condition"].split("+")[0], {})
        lam = f"{tu.get('lam_n')}, {tu.get('lam_c')}" if tu else "-"
        p2 = r["r2_parts"]
        lines.append(f"| {r['condition']} | {r['e1_gain_pct']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{r['r1_rel_pct']:+.4f} ({r['r1_mean_kl']:.2e}) | "
                     f"{p2.get('r2', float('nan')):+.4f} / {p2.get('r2ml', float('nan')):+.4f} | {lam} |")
    (out / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
