#!/usr/bin/env bash
# API-override premise probe: is the frozen base confidently WRONG about library APIs? Loads the curated
# data/api_facts.json and reports base accuracy per fact. Watchdog-compatible (${CNAME:+--name}, done marker).
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/apitask_probe.out
echo "[api] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm ${CNAME:+--name "$CNAME"} --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
     --store pk --phrasing counterfactual_multi --dataset counterfact --data-dir /data \
     --apitask /data/api_facts.json 2>&1" | tee -a "$LOG" \
  | grep -E 'apitask|OK|WRONG|editable|premise'
echo "[api] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
