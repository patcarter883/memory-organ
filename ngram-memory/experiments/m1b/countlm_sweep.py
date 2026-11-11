#!/usr/bin/env python
"""Count-LM weight sweep at matched retention cost (diagnostic, not pre-registered; no model forwards).

    python experiments/m1b/countlm_sweep.py --m1b M1B_DIR --ablation ABL_DIR --out DIR

The count-LM mixes per target: p = (1 - ln - lc) p_host(y) + ln p_KN(y) + lc p_cache(y). Only the TARGET's
probability enters the NLL, so every weight can be scored from per-target probabilities:

* p_host(y): M1b's base scores (B0_base_dev/eval/<set>__base.npz); p_table(y): the operating point's table scores
  (<op>/eval/<set>__rows[cartridge].npz), on the same targets.
* p_KN(y), p_cache(y): recomputed here from the Split-A train files at HEAD (KN trigram) and each file's prefix
  (in-file cache), exactly as scripts/evaluate.py builds them, for the scored documents in the same order.
* Check: the ablation run's always-on scores (countlm[kn-fine] at ln 0.015, countlm[cache-fine] at lc 0.005,
  countlm[fine]+rows at 0.015/0.005) are re-derived from these probabilities and must match per target.

Sets: Split-A dev (scored files), R1 WikiText-2, R2 stdlib, R2ml ML code. Test is not read. Reports, per weight:
dev E1 gain with file-cluster CI and R1/R2 relative NLL change, for counts only, cache only, and counts on top of
the table; then the largest weight whose retention cost does not exceed the table's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402

SETS = ("dev", "r1", "r2", "r2ml")
GRID = (0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def load_docs(splits, tok, E: dict, set_name: str):
    from ngram_memory.eval.stage1 import datasets as ds
    mc = E["min_context"]
    long_enough = lambda docs: [d for d in docs if len(d) > mc]  # noqa: E731
    if set_name == "dev":
        dev_all = long_enough(ds.project_docs(splits, tok, "a", "dev"))
        n = E.get("max_dev_files") if E.get("max_dev_files") is not None else (
            E.get("max_files") if E.get("max_files") is not None else len(dev_all))
        return dev_all[:n]
    src = dict(r1=lambda: ds.wikitext_docs(tok), r2=lambda: ds.stdlib_docs(tok), r2ml=lambda: ds.ml_codebase_docs(tok))
    return ds.take_tokens(src[set_name](), E["retention_tokens"], mc)


def mix_nll(parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    p = sum(w * x for w, x in parts)
    return -np.log(np.maximum(p, 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1b", required=True)
    ap.add_argument("--ablation", required=True, help="countlm_ablation.py output (g1/, g2/ npz) for the check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None, help="Splits JSON (default: rebuilt from ingest_splits/)")
    a = ap.parse_args()
    m1b, abl, out = Path(a.m1b), Path(a.ablation), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from ngram_memory.eval.metrics import paired
    from ngram_memory.eval.stage1.baselines import countlm as cl
    from ngram_memory.eval.stage1.scoring import ScoreResult
    from ngram_memory.eval.stage1.splits import Splits
    from ngram_memory.ingest.corpus import SPLITS_DIR
    E = R.load_grid(False)["eval_flags"]
    splits = Splits.load(a.splits) if a.splits else Splits.from_ingest(SPLITS_DIR / "split_a.json",
                                                                      SPLITS_DIR / "split_b.json")
    tok = AutoTokenizer.from_pretrained(R.QWEN25)
    eos, V = 151643, 151936
    op = json.loads((m1b / "S3_op" / "result.json").read_text())["selected"]
    docs_tr = [tok(splits.document(p), add_special_tokens=False)["input_ids"] for p in splits.files("a", "train")]
    kn = cl.KNTrigram(docs_tr, V, eos)

    data, check = {}, {}
    for s in SETS:
        base = ScoreResult.load(m1b / "B0_base_dev" / "eval" / f"{s}__base.npz")
        table = ScoreResult.load(m1b / op / "eval" / f"{s}__rows[cartridge].npz")
        if not np.array_equal(base.key(), table.key()):
            raise SystemExit(f"{s}: table and base targets differ")
        docs = load_docs(splits, tok, E, s)
        if [d.name for d in docs] != list(base.doc_names):
            raise SystemExit(f"{s}: rebuilt documents differ from the scored ones")
        pk, pc = cl.component_target_probs(kn, docs, base)
        ph, pt = np.exp(-base.nll.astype(np.float64)), np.exp(-table.nll.astype(np.float64))
        groups = np.asarray([base.groups[i] for i in base.doc])
        data[s] = dict(ph=ph, pt=pt, pk=pk, pc=pc, groups=groups, base=base.nll.astype(np.float64))
        # check against the ablation run's own always-on scores (same targets, float32 there)
        for name, parts in (("countlm[kn-fine]", [(0.985, ph), (0.015, pk)]),
                            ("countlm[cache-fine]", [(0.995, ph), (0.005, pc)]),
                            ("countlm[fine]+rows[cartridge]", [(0.98, pt), (0.015, pk), (0.005, pc)])):
            g = "g1" if "rows" not in name else "g2"
            f = abl / g / f"{s}__{name}.npz"
            if f.exists():
                ref = ScoreResult.load(f)
                if not np.array_equal(ref.key(), base.key()):
                    raise SystemExit(f"{s} {name}: ablation targets differ")
                dif = np.abs(mix_nll(parts) - ref.nll.astype(np.float64))
                check[f"{s}:{name}"] = dict(max_abs_nll_diff=float(dif.max()), mean_abs=float(dif.mean()),
                                            rel_pct_recomputed=float(100 * (mix_nll(parts).mean() - data[s]["base"].mean())
                                                                     / data[s]["base"].mean()),
                                            rel_pct_ablation=float(100 * (ref.nll.mean() - base.nll.mean()) / base.nll.mean()))
        print(s, len(base), "targets", {k: v["max_abs_nll_diff"] for k, v in check.items() if k.startswith(s + ":")},
              flush=True)

    def metrics(kind: str, lam: float) -> dict:
        res = {}
        for s, d in data.items():
            if kind == "counts":
                c = mix_nll([(1 - lam, d["ph"]), (lam, d["pk"])])
            elif kind == "cache":
                c = mix_nll([(1 - lam, d["ph"]), (lam, d["pc"])])
            elif kind == "table+counts":
                c = mix_nll([(1 - lam, d["pt"]), (lam, d["pk"])])
            else:
                raise ValueError(kind)
            p = paired(d["base"], c, d["groups"], n_boot=2000 if s != "dev" else 10_000)
            res[s] = dict(rel_pct=p["rel_pct"], ci=p.get("rel_pct_ci_file") or p.get("rel_pct_ci"), n=p["n_tokens"])
        n2 = res["r2"]["n"] + res["r2ml"]["n"]
        res["r2_combined"] = (res["r2"]["rel_pct"] * res["r2"]["n"] + res["r2ml"]["rel_pct"] * res["r2ml"]["n"]) / n2
        return res

    rows = []
    for kind in ("counts", "cache", "table+counts"):
        for lam in GRID:
            if kind == "cache" and lam > 0.03:
                continue
            m = metrics(kind, lam)
            rows.append(dict(kind=kind, lam=lam, dev_gain=-m["dev"]["rel_pct"],
                             dev_ci=sorted(-x for x in m["dev"]["ci"]) if m["dev"]["ci"] else None,
                             r1=m["r1"]["rel_pct"], r2=m["r2"]["rel_pct"], r2ml=m["r2ml"]["rel_pct"],
                             r2_combined=m["r2_combined"]))
            print(kind, lam, f"dev {rows[-1]['dev_gain']:+.3f}", f"r1 {rows[-1]['r1']:+.4f}",
                  f"r2 {rows[-1]['r2']:+.4f} r2ml {rows[-1]['r2ml']:+.4f}", flush=True)
    tbl = next(r for r in rows if r["kind"] == "table+counts" and r["lam"] == 0.0)

    def matched(kind: str):
        ok = [r for r in rows if r["kind"] == kind and r["lam"] > 0 and r["r1"] <= max(tbl["r1"], 0.0)
              and r["r2"] <= max(tbl["r2"], 0.0) and r["r2ml"] <= max(tbl["r2ml"], 0.0)]
        return max(ok, key=lambda r: r["lam"]) if ok else None

    rep = dict(operating_point=op, grid=GRID, table=tbl, check=check, rows=rows,
               matched_retention={k: matched(k) for k in ("counts", "cache")},
               note="matched = largest weight with R1, R2 stdlib and R2 ML each <= the table's (or <= 0)")
    R.jdump(out / "sweep.json", rep)
    lines = ["| mix | weight | dev E1 gain % [CI] | R1 % | R2 stdlib % | R2 ML % |", "|---|---|---|---|---|---|"]
    for r in rows:
        ci = r["dev_ci"] or [float("nan")] * 2
        lines.append(f"| {r['kind']} | {r['lam']:g} | {r['dev_gain']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{r['r1']:+.4f} | {r['r2']:+.4f} | {r['r2ml']:+.4f} |")
    (out / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("matched:", json.dumps(rep["matched_retention"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
