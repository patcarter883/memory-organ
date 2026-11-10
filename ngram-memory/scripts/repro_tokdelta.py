"""Acceptance check for eval/stage1/tokclass.py: re-label the 32 held-out 512-token windows of
S1/evalproto2/tokdelta.py (same split / seeds; tokenizer only, no model) with token_classes and compare the class
breakdown of the saved arm-A deltas (base_verify.npz, nll_full_heldout.npy) with tokdelta.json.

    python scripts/repro_tokdelta.py --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOST = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
VE = Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment")
REF = ROOT / "proofs/stage1/evalproto2/tokdelta.json"
EOS, T = 151643, 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from ngram_memory.eval.metrics import class_breakdown
    from ngram_memory.eval.stage1 import tokclass as tc
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HOST)
    files = sorted(os.path.relpath(f, REPO) for f in glob.glob(f"{REPO}/**/*.py", recursive=True))
    shuf = files[:]
    random.Random(0).shuffle(shuf)
    held = sorted(shuf[:round(0.15 * len(files))])
    import numpy, torch, transformers  # noqa: E401  (reference dirs as tokdelta.py: py3.12 stdlib + venv libs)
    dirs = ["/usr/lib/python3.12"] + [os.path.dirname(m.__file__) for m in (torch, numpy, transformers)]
    ref = tc.reference_vocab(dirs, out / "ref_vocab_tokdelta.json")
    keys = ("lex", "id_kind", "id_first_sub", "id_cont_sub", "id_first_in_file", "id_xfile")
    ids_all: list[int] = []
    cls_all: dict[str, list] = {k: [] for k in keys}
    file_all: list[int] = []
    for fi, p in enumerate(held):
        s = Path(REPO, p).read_text(errors="replace")
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        c = tc.token_classes(s, enc["offset_mapping"], ref)
        ids_all += enc["input_ids"] + [EOS]
        for k in keys:
            cls_all[k] += c[k].tolist() + [0]
        file_all += [fi] * (len(enc["input_ids"]) + 1)
    n = len(ids_all) // T
    idx = np.sort(np.random.default_rng(1).choice(n, size=min(32, n), replace=False))
    cut = lambda x: np.asarray(x)[: n * T].reshape(n, T)[idx][:, 1:].ravel()  # noqa: E731  (class of the TARGET)
    per = {k: cut(cls_all[k]) for k in keys}
    for k in ("id_first_sub", "id_cont_sub", "id_first_in_file", "id_xfile"):
        per[k] = per[k].astype(bool)
    per["lex"] = per["lex"].astype(np.int8)
    per["id_kind"] = per["id_kind"].astype(np.int8)
    groups = cut(file_all)
    base = np.load(VE / "base_verify.npz")["heldout"].astype(np.float64).ravel()
    on = np.load(VE / "nll_full_heldout.npy").astype(np.float64).ravel()
    assert base.shape == groups.shape, (base.shape, groups.shape)
    masks = tc.class_masks(per)
    got = class_breakdown(base, on, groups, masks, n_boot=200)
    refj = json.loads(REF.read_text())
    pairs = {"id:cont_sub": "ID|cont (all)", "id:first_sub": "ID|sub0 (all)", "lex:comment": "comment",
             "id:proj:cont_sub": None, "id:proj:first_sub": None, "lex:string+docstring": "string",
             "non_identifier": "non-ID", "id:proj": "ID proj (all)"}
    R = {}
    for k, rk in pairs.items():
        g = got.get(k, {})
        r = dict(n=g.get("n_tokens"), delta=g.get("delta"), rel_pct=g.get("rel_pct"), share_of_gain=g.get("share_of_gain"))
        if rk and rk in refj:
            r.update(ref_n=refj[rk]["n"], ref_delta=refj[rk]["delta"], ref_rel_pct=refj[rk]["rel_pct"])
        R[k] = r
    pc = [refj[x]["n"] for x in ("proj_first|cont", "proj_rep|cont")]
    ps = [refj[x]["n"] for x in ("proj_first|sub0", "proj_rep|sub0")]
    R["id:proj:cont_sub"].update(ref_n=sum(pc), ref_rel_pct_first=refj["proj_first|cont"]["rel_pct"],
                                 ref_rel_pct_rep=refj["proj_rep|cont"]["rel_pct"])
    R["id:proj:first_sub"].update(ref_n=sum(ps), ref_rel_pct_first=refj["proj_first|sub0"]["rel_pct"],
                                  ref_rel_pct_rep=refj["proj_rep|sub0"]["rel_pct"])
    R["all"] = dict(n=int(len(base)), delta=float((on - base).mean()), ref_delta=refj["all"]["delta"])
    R["docstring_vs_string"] = {k: got[k]["n_tokens"] for k in ("lex:docstring", "lex:string") if k in got}
    (out / "repro_tokdelta.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
