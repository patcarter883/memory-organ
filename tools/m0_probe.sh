#!/usr/bin/env bash
# Phase M0: measure the N unlock from lifting the single-token-object filter. Runs --probe-only
# (base-known probe over ~21k CounterFact records, first-token proxy) at CAM_MAX_OBJ_TOK = 1..4 with a
# high --multi-relations so N is not relation-capped. Reads the candidate-fact + base-known-N lines.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/m0_probe.out
echo "[m0] start $(date -u +%H:%M:%S)" | tee "$LOG"
for K in 1 2 3 4; do
  echo "[m0] === MAX_OBJ_TOK=$K ===" | tee -a "$LOG"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e CAM_PROBE_CACHE_DIR=/probe_cache -e CAM_MAX_OBJ_TOK="$K" \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 1200 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --phrasing counterfactual_multi --multi-relations 40 --cf-probe-cap 21500 \
       --dataset counterfact --data-dir /data --tap-layers 24 --probe-only 2>&1" \
    | tee -a "$LOG" | grep -E 'MAX_OBJ_TOK|candidate facts|edits across|edits ready|known facts|Traceback|Error'
done
echo "[m0] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
