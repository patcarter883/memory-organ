"""Probe: rebuild the E1 language-edit set EXACTLY as the run does and dump every oracle gold the
confound filter reads, for ALL THREE hops. Diagnoses 'confound-eligible 0/N' collapses.

Run (GPU, card under gpu-lease):
  python /engine/tools/oracle_map_probe.py --limit 60
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cam"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m2_adapter import load_frozen_base              # noqa: E402
import recall_mag as RM                              # noqa: E402
import mquake_ripple as MR                           # noqa: E402


class _NoTap:
    def set_bank(self, *a, **k):
        return None

    def set_trigger_ids(self, *a, **k):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    ns = argparse.Namespace(data_dir="/data", seed=20260625, cf_probe_batch=16, M=8,
                            multi_relations=6, cf_probe_cap=21500, base1="",
                            phrasing="counterfactual_multi", store="pk", batch=4,
                            seg_len=48, qa_seg=3, tap_layers=12, mquake_limit=60)
    base, tok = load_frozen_base("Qwen/Qwen3.5-4B")
    base.eval()
    builder, cf_records, _prior = RM.setup_counterfact_multi(base, tok, ns)
    lang = MR._lang_edits(cf_records, a.limit)
    print(f"[omap] language edits: {len(lang)}", flush=True)
    for i, r in enumerate(lang):
        print(f"[omap]   {i:3d} {r.subject!r} [{r.relation_id}] {r.true_str!r} -> {r.new_str!r}", flush=True)
    uniq = sorted({r.true_str for r in lang} | {r.new_str for r in lang})
    print(f"[omap] uniq languages: {len(uniq)} :: {uniq}", flush=True)
    be = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    olen = int(os.environ.get("CAM_INDIST_ORACLE_LEN", "8"))
    for name, (cloze_t, qtmpl) in MR._TWOHOPS.items():
        omap = {L: MR._oracle(base, None, tok, _NoTap(), be, lm, cloze_t.format(L=L), bos, olen)
                for L in uniq}
        elig, rows = 0, []
        for r in lang:
            gt, gc = omap[r.true_str], omap[r.new_str]
            ok = bool(gt and gc and MR._norm(gt) != MR._norm(gc))
            elig += ok
            rows.append((ok, r, gt, gc))
        print(f"\n[omap] ===== hop {name!r} :: confound-eligible {elig}/{len(lang)} =====", flush=True)
        print("[omap] golds: " + " | ".join(f"{L}={omap[L]!r}" for L in uniq), flush=True)
        for ok, r, gt, gc in rows:
            print(f"[omap]   [{'OK' if ok else '  '}] {r.subject!r} {r.true_str!r}->{r.new_str!r} "
                  f"| true={gt!r} cf={gc!r}", flush=True)


if __name__ == "__main__":
    main()
