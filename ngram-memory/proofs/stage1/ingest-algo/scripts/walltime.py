"""Local wall-clock of one Stage-1 cartridge build and one incremental update, per host.

Rates (tok/s, low-high) are ESTIMATES from the sibling model nm/stage1/compute/local_plan.json (assumptions in
local_plan.py: 2 RDNA4 cards with the serve stopped, MXFP4 experts streamed over PCIe for Flash-Next, CPU 4-core
fallback). Token counts are MEASURED (count_rows.json in this folder).
Recipe (design, not measured): E epochs over train tokens; KL anchor batches = A x train tokens per epoch
(fwd+bwd); base-model logits for the anchors computed once (fwd only); held-out + time-split eval each epoch
(fwd only); 2 x 300 s serve stop/boot (R/weights/ckpt_read.py boot time as cited in INTEGRATION.md / DESIGN §7).
Incremental: re-train on windows that read dirty rows: dirty windows ~= changed files' tokens + context.
Usage: python -I walltime.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
cr = json.loads((HERE / "count_rows.json").read_text())
lp = json.loads((HERE.parent / "compute/local_plan.json").read_text())["res"]
train = cr["split_by_file"]["train_tokens"]; held = cr["split_by_file"]["heldout_tokens"]
timeh = cr["split_by_time"]["added_positions"]
BOOT = 2 * 300
out = {}
for E, A in ((1, 0.25), (3, 0.5)):
    for host, r in lp.items():
        row = {}
        for venue in ("gpu_train", "cpu_train"):
            ev = venue.replace("train", "eval")
            for i, tag in ((1, "fast"), (0, "slow")):
                t_tr, t_ev = r[venue][i], r[ev][i]
                secs = E * train * (1 + A) / t_tr + (A * train) / t_ev + E * (held + timeh) / t_ev
                secs += BOOT if venue == "gpu_train" else 0
                row[f"{venue}_{tag}_h"] = round(secs / 3600, 2)
        out[f"E{E}_A{A}|{host}"] = row
# incremental update for the median / p90 commit: 2 epochs over the changed files' windows + 1x anchors
inc_tok = {k: v for k, v in json.loads((HERE / "commit_tokens.json").read_text())["changed_file_tokens"].items() if k != "max"}  # MEASURED: whole changed files
for host, r in lp.items():
    for k, n in inc_tok.items():
        for i, tag in ((1, "fast"), (0, "slow")):
            secs_gpu = 2 * n * 2 / r["gpu_train"][i] + BOOT
            secs_cpu = 2 * n * 2 / r["cpu_train"][i]
            out.setdefault(f"incremental|{host}", {})[f"{k}_{tag}"] = dict(gpu_h=round(secs_gpu / 3600, 2),
                                                                           cpu_h=round(secs_cpu / 3600, 2))
for k, v in out.items():
    print(k, json.dumps(v))
(HERE / "walltime.json").write_text(json.dumps(out, indent=1))
