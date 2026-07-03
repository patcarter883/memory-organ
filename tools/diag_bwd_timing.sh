#!/usr/bin/env bash
# DIAGNOSTIC: where does CAM stage-2 tap-fit wall-time go? Short run (8 tap-fit steps), timed by the
# StageCost [cost] lines, comparing native HIP backward (GDN_HIP_NATIVE_BWD=1, the accidental slow path
# from copying the nbwd .so) vs the validated-fast torch REFERENCE backward (GDN_HIP_NATIVE_BWD=0).
# Small --cf-probe-cap keeps the base-known probe cheap; we only care about per-step tap-fit time.
# Run:  gpu-lease -n 1 -- bash tools/diag_bwd_timing.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-psweep}
ENGINE=${ENGINE:-/home/pat/code/memory-organ-psweep}
DATA=${DATA:-/home/pat/code/memory-organ/data}
echo "[diag] HIP=$HIP_VISIBLE_DEVICES ROCR=$ROCR_VISIBLE_DEVICES $(date -u +%H:%M:%S)"

drun() {  # $1 = extra env flags, $2 = tag
  echo "===== stage-2 timing: $2 ====="
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    $1 \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 6 --bind-steps 100 --steps 8 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 3000 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 2>&1 | grep -E '\[cost\]|stage-2|edits across|Traceback|Error'"
  echo "----- $2 rc=$? $(date -u +%H:%M:%S) -----"
}

drun "-e GDN_HIP_NATIVE_BWD=1" "NATIVE-BWD"
drun "-e GDN_HIP_NATIVE_BWD=0" "REFERENCE-BWD"
echo "[diag] DONE $(date -u +%H:%M:%S)"
