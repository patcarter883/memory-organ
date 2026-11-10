"""Acceptance check for eval/stage1/scoring.py: re-score S1/rt-science/ctx_len.py's blocks (same files, split,
block sampling) and compare base NLLs (and, if the skeptic's arm-A rows are on disk, the ON deltas) with
ctx_len.json. No training.

    python scripts/repro_ctx_len.py --out DIR [--threads 4] [--nblk 24] [--sets heldout,stdlib] [--no-rows]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
STDLIB = "/usr/local/lib/python3.13"
STDLIB_FILES = ["argparse.py", "dataclasses.py", "functools.py", "typing.py", "json/decoder.py", "json/encoder.py",
                "asyncio/tasks.py", "asyncio/queues.py", "collections/__init__.py", "logging/__init__.py",
                "subprocess.py", "threading.py", "inspect.py", "enum.py", "contextlib.py", "shutil.py"]
ROWS = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/"
            "verify-experiment/rows_armA.pt")
REF = ROOT / "proofs/stage1/rt-science/ctx_len.json"
EOS, T = 151643, 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nblk", type=int, default=24)
    ap.add_argument("--sets", default="heldout,stdlib")
    ap.add_argument("--no-rows", action="store_true")
    ap.add_argument("--export-rows", default=None, help="save arm-A rows as a SparseRowStore state dict and exit")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ngram_memory.eval.metrics import paired
    from ngram_memory.eval.stage1.scoring import Doc, Window, score
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HOST)
    files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
    shuf = files[:]
    random.Random(0).shuffle(shuf)
    n_held = round(0.15 * len(files))
    held, train = sorted(shuf[:n_held]), sorted(shuf[n_held:])

    def blocks(paths, root, nblk, seed):
        docs, cand = [], []
        for fi, p in enumerate(paths):
            ids = tok(Path(root, p).read_text(errors="replace"), add_special_tokens=False).input_ids
            docs.append(Doc(p, np.asarray(ids, np.int64)))
            for b in range(1024, len(ids) - 1024 + 1, 1024):
                cand.append((fi, b))
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(cand), size=min(nblk, len(cand)), replace=False))
        return docs, [cand[i] for i in sel], len(cand)

    model = AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    on_cond = None
    if not a.no_rows and ROWS.exists():
        from ngram_memory.eval.stage1.conditions import RowsCondition
        from ngram_memory.hashing import rows_for
        from ngram_memory.spec import graft_format, qwen25_proxy_key_spec
        from ngram_memory.tables import SparseRowStore
        fmt = graft_format(896, qwen25_proxy_key_spec())
        stream = []
        for p in train:
            stream += tok(Path(REPO, p).read_text(errors="replace"), add_special_tokens=False).input_ids + [EOS]
        s = np.asarray(stream, np.int64)
        n = len(s) // T
        ch = s[: n * T].reshape(n, T)
        tx = ch[np.sort(np.random.default_rng(0).choice(n, size=96, replace=False))]
        rid = torch.from_numpy(np.unique(rows_for(fmt.key, tx)))
        vals = torch.load(ROWS, map_location="cpu")
        assert vals.shape == (rid.numel(), fmt.row_width), (vals.shape, rid.numel())
        on_cond = RowsCondition(fmt, SparseRowStore(fmt.padded_rows, fmt.row_width, rid, vals.float()), name="armA")
        if a.export_rows:
            torch.save(on_cond.graft.rows_store().copy().state_dict(), a.export_rows)
            (Path(a.export_rows).with_suffix(".format.json")).write_text(fmt.to_json())
            print("exported", a.export_rows, len(on_cond.graft.rows_store()))
            return
    ref = json.loads(REF.read_text())
    R: dict = dict(rows_on_disk=ROWS.exists(), rows_used=on_cond is not None, threads=a.threads)
    sets = [("heldout", held, REPO, a.nblk, 7),
            ("stdlib", [f for f in STDLIB_FILES if Path(STDLIB, f).exists()], STDLIB, a.nblk // 2, 8)]
    for name, paths, root, nb, seed in sets:
        if name not in a.sets.split(","):
            continue
        docs, blk, avail = blocks(paths, root, nb, seed)
        long_w = [Window(fi, b - 1024, b + 1, b + 1024) for fi, b in blk]
        short_w = [w for fi, b in blk for w in (Window(fi, b, b + 1, b + 512), Window(fi, b + 512, b + 513, b + 1024))]
        r: dict = dict(blocks=len(blk), available_blocks=avail, files=len({fi for fi, _ in blk}))
        for c, wins in (("long", long_w), ("short", short_w)):
            t0 = time.time()
            base = score(model, docs, wins)
            if c == "long":  # ctx_len drops block target 1536 (= b + 512), which the short arm cannot score
                keep = np.ones(len(base), bool)
                for fi, b in blk:
                    keep &= ~((base.doc == fi) & (base.pos == b + 512))
                base = base.select(keep)
            base.save(out / f"{name}_{c}_base.npz")
            rr: dict = dict(base=float(base.nll.mean()), n_targets=len(base), base_wall_s=time.time() - t0,
                            base_wall_s_per_10k=base.wall_per_10k(),
                            ref_base=ref[name][c]["base"], base_abs_diff=abs(float(base.nll.mean()) - ref[name][c]["base"]))
            if on_cond is not None:
                on = score(model, docs, wins, on_cond)
                if c == "long":
                    on = on.select(keep)
                on.save(out / f"{name}_{c}_armA.npz")
                pr = paired(base.nll, on.nll, np.asarray(base.doc), n_boot=5000, seed=0)
                rr.update(on_wall_s_per_10k=on.wall_per_10k(), delta=pr["delta"], rel_pct=pr["rel_pct"],
                          rel_pct_ci=pr["rel_pct_ci"], ref_delta=ref[name][c]["delta"], ref_rel_pct=ref[name][c]["rel_pct"],
                          ref_ci95_rel_pct=ref[name][c]["ci95_rel_pct"])
            r[c] = rr
            print(name, c, json.dumps(rr), flush=True)
        R[name] = r
    R["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (out / "repro_ctx_len.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
