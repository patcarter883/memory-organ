#!/usr/bin/env python
"""Row census of a codebase under a KeySpec (§I.5), reproducing S1/ingest-algo/count_rows.out and row_policy.out.

Default run (minisgl python/**/*.py at df29117, Flash-Next keys and tokenizer):
  * all-files census (count_rows.py::census): 7,555,967 rows touched, 2,385,780 rows read >= 2 times;
  * the 85/15 seed-0 shuffle split of S1/ingest-algo ("legacy design split", 212 train files): n-gram count >= 2
    policy 2,040,221 rows (row_policy.py), all rows 6,721,239, count >= 5 550,891;
  * the same policies on the frozen Split A train files (ingest_splits/split_a.json), reported, not compared;
  * optionally (--reference) the enrichment / specific policies against the row_policy.py reference corpus
    (stdlib sample + WikiText-2 train), and (--framings) the extras.py framing counts. --reference reads the dev
    container's distro stdlib (/usr/lib/python3.13, Debian-patched 3.13.18), so it reproduces the proofs only there;
    the runner bundles in runner/inputs/ do not cover it, and the script stops if that tree is absent.
Every reproduced number is compared to the stored proof JSON; the script exits 1 on any mismatch.

Usage: python scripts/count_rows.py --out OUT.json [--tokenizer tokenizer.json] [--reference] [--framings]
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngram_memory.ingest import corpus as C  # noqa: E402
from ngram_memory.ingest import rowpolicy as RP  # noqa: E402
from ngram_memory.spec import flash_next_key_spec  # noqa: E402

PROOF = ROOT / "proofs/stage1/ingest-algo"
FLASH_NEXT_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
DEFAULT_TOKENIZER = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/"
                         "nm/stage1/ingest-algo/tok/tokenizer.json")
from ngram_memory import paths as _paths  # noqa: E402
WIKI = _paths.wikitext_dir()
STDLIB = Path("/usr/lib/python3.13")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def find_tokenizer(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if DEFAULT_TOKENIZER.exists():
        return DEFAULT_TOKENIZER
    from huggingface_hub import hf_hub_download  # tokenizer files only, never weights
    return Path(hf_hub_download("Qwen/Qwen3.8-Flash-Next", "tokenizer.json"))


def compare(tag: str, got, want, bad: list) -> None:
    ok = got == want
    log(f"{'MATCH' if ok else 'MISMATCH'} {tag}: got {got} want {want}")
    if not ok:
        bad.append(dict(tag=tag, got=got, want=want))


def reference_corpus(tok: C.TokenizerJSON) -> list[np.ndarray]:
    """row_policy.py's reference: stdlib sample (shuffled with seed 0, ~2M tokens by len/3.8) + WikiText-2 train."""
    import pyarrow.parquet as pq
    if not (STDLIB / "argparse.py").exists():
        raise SystemExit(f"--reference needs the dev container's stdlib tree {STDLIB} (not in the runner bundles)")
    ex = ("test", "idlelib", "tkinter", "lib2to3", "site-packages")
    fs = sorted(p for p in STDLIB.rglob("*.py") if not any(x in p.parts for x in ex))
    random.Random(0).shuffle(fs)
    out, n = [], 0.0
    for p in fs:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append(t); n += len(t) / 3.8
        if n >= 2_000_000:
            break
    std = [d for d in tok.encode_many(out) if d.size]
    wl = pq.read_table(WIKI / "train-00000-of-00001.parquet").column("text").to_pylist()
    wiki = [d for d in tok.encode_many(["".join(wl[i:i + 200]) for i in range(0, len(wl), 200)]) if d.size]
    return std + wiki


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(C.MINISGL_REPO))
    ap.add_argument("--commit", default=C.MINISGL_HEAD)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--reference", action="store_true", help="also reproduce the enrichment/specific policies")
    ap.add_argument("--framings", action="store_true", help="also reproduce extras.py framing counts")
    ap.add_argument("--commits", action="store_true", help="also reproduce commit_tokens.py (b80007f..df29117)")
    args = ap.parse_args()
    res: dict = {}
    bad: list = []

    tok = C.TokenizerJSON(find_tokenizer(args.tokenizer))
    res["tokenizer"] = dict(path=str(tok.path), sha256=tok.sha256)
    if tok.sha256 != FLASH_NEXT_TOKENIZER_SHA256:
        log("WARNING tokenizer sha256 differs from the Flash-Next tokenizer of the proofs:", tok.sha256)
    key = flash_next_key_spec(tokenizer_sha256=tok.sha256)
    res["keyspec"] = dict(key.to_dict(), fingerprint=key.fingerprint(), total_rows=key.total_rows)
    compare("multipliers", list(key.multipliers), [23703573157769, 20109073645365, 8052911324071], bad)
    compare("logical_rows", key.total_rows, 320001446, bad)

    corp = C.load_corpus(args.repo, args.commit, C.FileFilter.census_v0())
    res["corpus"] = corp.summary()
    files = corp.files
    nbytes = sum(f.size for f in files)
    docs_all = [d.ids for d in C.tokenize_files(tok, files, "raw")]
    keep = [i for i, d in enumerate(docs_all) if d.size]
    docs = [docs_all[i] for i in keep]
    paths = [files[i].path for i in keep]
    log(f"corpus: files={len(files)} non-empty={len(docs)} bytes={nbytes} secret_hits={len(corp.secret_hits)} "
        f"near-dup clusters={len(corp.dedup.near_clusters) if corp.dedup else None}")

    proof = json.loads((PROOF / "count_rows.json").read_text())
    want_head = proof["project_head"]
    ix = RP.NGramIndex(key, docs)
    cen = RP.row_census(ix)
    res["census_all_files"] = cen
    res["overlay_bytes"] = {t: RP.overlay_bytes(n, 160) for t, n in (("all", cen["distinct_rows_total"]),
                                                                       ("count>=2", cen["rows_count>=2_total"]),
                                                                       ("count>=5", cen["rows_count>=5_total"]))}
    compare("files", len(files), 250, bad)
    compare("bytes", nbytes, 5414867, bad)
    for k in ("tokens", "distinct_2gram", "distinct_3gram", "distinct_2gram_count>=2", "distinct_3gram_count>=2",
              "distinct_2gram_count>=5", "distinct_3gram_count>=5", "distinct_2gram_count>=20",
              "distinct_3gram_count>=20", "distinct_rows_per_head", "distinct_rows_per_head_count>=2",
              "distinct_rows_per_head_count>=5", "distinct_rows_total"):
        compare(f"all-files {k}", cen[k], want_head[k], bad)
    compare("all-files row-level count>=2 rows", cen["rows_count>=2_total"],
            want_head["overlay_bytes[count>=2]"]["rows"], bad)
    compare("all-files row-level count>=5 rows", cen["rows_count>=5_total"],
            want_head["overlay_bytes[count>=5]"]["rows"], bad)
    res["collisions_all_files"] = RP.collision_stats(ix)
    del ix

    # ---- legacy 85/15 design split (row_policy.py) and Split A
    tr_idx, ho_idx = C.legacy_design_split(len(docs), seed=0, frac=0.15)
    tr = RP.NGramIndex(key, [docs[i] for i in tr_idx])
    compare("legacy split train files", len(tr_idx), proof["split_by_file"]["train_files"], bad)
    compare("legacy split train tokens", tr.n, proof["split_by_file"]["train_tokens"], bad)
    pol_want = json.loads((PROOF / "row_policy.json").read_text())
    pols = [RP.RowPolicy("all"), RP.RowPolicy("count", 2), RP.RowPolicy("count", 5)]
    legacy = {}
    for pol in pols:
        rows = RP.select_rows(tr, pol)
        legacy[pol.name()] = dict(rows=int(rows.size), per_head=RP.per_head_counts(key, rows))
        compare(f"legacy-split train policy {pol.name()}", int(rows.size), pol_want[pol.name()]["rows"], bad)
        if pol.kind == "count" and pol.k == 2:
            cs = RP.collision_stats(tr, rows)["occupancy_first_head_per_order"]
            compare("legacy-split count>=2 occupancy head0/head8",
                    [round(cs[2], 12), round(cs[3], 12)],
                    [round(x, 12) for x in pol_want["count>=2"]["collision_rate_per_head_2g_3g"]], bad)
    legacy["train_census"] = RP.row_census(tr)
    res["legacy_design_split"] = dict(train_files=[paths[i] for i in tr_idx], heldout_files=[paths[i] for i in ho_idx],
                                      policies=legacy)

    if args.reference:
        ref = RP.NGramIndex(key, reference_corpus(tok))
        log(f"reference tokens={ref.n} (row_policy.py: stdlib 2,129,621 + wiki 2,504,609 = 4,634,230)")
        compare("reference tokens", ref.n, 2129621 + 2504609, bad)
        for pol, name in ((RP.RowPolicy("specific", 1), "specific(ref=0)"),
                          (RP.RowPolicy("specific", 2), "specific & count>=2"),
                          (RP.RowPolicy("enrich", 1, 4.0), "enrich>=4"),
                          (RP.RowPolicy("enrich", 2, 4.0), "enrich>=4 & count>=2"),
                          (RP.RowPolicy("enrich", 2, 16.0), "enrich>=16 & count>=2")):
            rows = RP.select_rows(tr, pol, ref)
            legacy[name] = dict(rows=int(rows.size))
            compare(f"legacy-split train policy {name}", int(rows.size), pol_want[name]["rows"], bad)
        del ref
    del tr

    split_a = C.load_split_a()
    if split_a is None:
        log("no ingest_splits/split_a.json; Split A policies skipped")
    else:
        pos = {p: i for i, p in enumerate(paths)}
        sa = {}
        for name in ("train", "dev", "test", "neardup_diag"):
            lst = getattr(split_a, name)
            sa[f"{name}_files"] = len(lst)
            sa[f"{name}_missing_at_commit"] = [p for p in lst if p not in pos]
        tra = RP.NGramIndex(key, [docs[pos[p]] for p in split_a.train if p in pos])
        sa["train_tokens"] = tra.n
        for pol in pols:
            rows = RP.select_rows(tra, pol)
            sa[pol.name()] = dict(rows=int(rows.size), per_head=RP.per_head_counts(key, rows))
            log(f"Split A train policy {pol.name()}: {rows.size} rows")
        sa["row_level_count>=2"] = int(RP.select_rows(tra, RP.RowPolicy("row_count", 2)).size)
        res["split_a"] = sa
        log("Split A:", {k: v for k, v in sa.items() if not isinstance(v, (dict, list))})
        del tra

    if args.framings:
        want_x = json.loads((PROOF / "extras.json").read_text())
        base_ix = RP.NGramIndex(key, docs)
        base_sets = [np.unique(base_ix.rows[:, h]) for h in range(key.n_heads)]
        for fr, tag in (("catn", "framing_cat_n_numbered"), ("chat", "framing_chat_fenced")):
            fd = [d.ids for d in C.tokenize_files(tok, files, fr, repo_paths=True)]  # extras.py named repo paths
            fix = RP.NGramIndex(key, [d for d in fd if d.size])
            sets = [np.unique(fix.rows[:, h]) for h in range(key.n_heads)]
            shared = sum(int(np.isin(sets[h], base_sets[h]).sum()) for h in range(key.n_heads))
            total = sum(s.size for s in sets)
            cov = np.ones(fix.n, bool)
            for h in range(key.n_heads):
                cov &= np.isin(fix.rows[:, h], base_sets[h])
            d = dict(tokens=fix.n, rows_total=int(total), rows_new_vs_bare=int(total - shared),
                     positions_fully_covered_by_bare_rows=float(cov.mean()))
            res[tag] = d
            for k in ("tokens", "rows_total", "rows_new_vs_bare"):
                compare(f"{tag} {k}", d[k], want_x[tag][k], bad)
        del base_ix

    if args.commits:
        want_c = json.loads((PROOF / "commit_tokens.json").read_text())
        per = []
        g = C.GitRepo(args.repo)
        for ch in C.commit_changes(args.repo, "b80007f", args.commit, "python"):
            if not ch:
                continue
            blobs = g.read_blobs(f.blob for f in ch if f.blob)
            ntok = sum(tok.encode(C.decode_text(blobs[f.blob])).size for f in ch if f.blob)
            per.append((ntok, len(ch), sum(f.added for f in ch)))
        arr = np.array(per)
        got_c = {"commits_touching_py": len(per),
                 "changed_file_tokens": {q: float(np.percentile(arr[:, 0], p)) for q, p in
                                         (("median", 50), ("p90", 90), ("max", 100))},
                 "files": {"median": float(np.median(arr[:, 1])), "p90": float(np.percentile(arr[:, 1], 90))},
                 "added_lines": {"median": float(np.median(arr[:, 2])), "p90": float(np.percentile(arr[:, 2], 90))}}
        res["commit_tokens"] = got_c
        compare("commit_tokens", got_c, want_c, bad)

    res["mismatches"] = bad
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["wall_s"] = time.time() - T0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {args.out}; mismatches={len(bad)} peak_rss={res['peak_rss_mb']:.0f} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
