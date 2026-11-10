"""Generate the Stage-1 repository QA JSONL (§6.A.5) from the split files, optionally score a sample with the base.

    python scripts/build_qa.py --splits ingest_splits --out DIR [--ref-vocab-cache F] [--score-sample N --threads 2]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ngram_memory import paths as _paths  # noqa: E402
QWEN25 = str(_paths.qwen25())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-vocab-cache", default=None)
    ap.add_argument("--env-prefix", default="MINISGL_")
    ap.add_argument("--project", default="the minisgl codebase")
    ap.add_argument("--host", default=QWEN25)
    ap.add_argument("--score-sample", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    from ngram_memory.eval.stage1 import qa, tokclass as tc
    from ngram_memory.eval.stage1.splits import Splits
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = Splits.load(a.splits)
    split_of = {p: part for part in ("train", "dev", "test", "near_dup") for p in sp.files("a", part)}
    texts = {p: sp.text(p) for p in split_of}
    if a.ref_vocab_cache and Path(a.ref_vocab_cache).exists():
        ref = tc.reference_vocab([], a.ref_vocab_cache)
    else:
        ref = tc.default_reference_vocab(a.ref_vocab_cache or out / "ref_vocab.json")
    items = qa.generate_items(texts, split_of=split_of, commit=sp.commit, ref_vocab=ref, project=a.project,
                              env_prefix=a.env_prefix)
    qa.write_jsonl(items, out / "qa_items.jsonl")
    R = dict(n_items=len(items), by_kind=dict(Counter(i["kind"] for i in items)),
             by_split=dict(Counter(i["split"] for i in items)), jsonl=str(out / "qa_items.jsonl"))
    if a.score_sample:
        os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
        import torch
        torch.set_num_threads(a.threads)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.host)
        model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32).eval()
        rng = random.Random(0)
        by = {}
        for it in items:
            by.setdefault(it["kind"], []).append(it)
        sample = [x for k in sorted(by) for x in rng.sample(by[k], min(a.score_sample, len(by[k])))]
        t = time.time()
        sc = qa.score_items(model, tok, sample)
        R["base_sample"] = dict(n=len(sample), wall_s=time.time() - t, summary=qa.summarize(sc))
    (out / "qa_summary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
