#!/usr/bin/env bash
# Phase N1: first REAL scaled bind + triad. CAM_LENGTH_SPLIT binds all subject-lengths (per-(rid,len)
# sub-relations); bind-safe settings (subj<=8, roomy seg) so the bind block fits. K1 on, TRIAD_DEBUG to
# dodge the RDNA4 flake, DISJOINT_BANKS scaled. Measures efficacy/locality/generality at N (~464, 3.2x).
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/n1_bind.out
echo "[n1b] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS=512 \
  -e CAM_LENGTH_SPLIT=1 -e CAM_MAX_SUBJ_LEN=8 -e CAM_WRITE_AT_READ=1 -e CAM_TRIAD_DEBUG=1 \
  -e CAM_LOGIT_INJECT_SWEEP="0,2" -e CAM_LOCALITY_NBR_CAP=1 -e CAM_COHORT_CAP=80 -e CAM_PROMPT_MAXTOK=48 -e CAM_CONF_DIAG=1 \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 2700 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations 40 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 64 --qa-seg 4 --save-anyway --conf-gate --locality-weight 0 \
     --persistent-locality --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'grouping=|EDITING|edits across|known facts|binding held-out|Track 4|cf-delivery|edits below C0|GEN-hit|alpha|Traceback|AssertionError|Error'
echo "[n1b] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
