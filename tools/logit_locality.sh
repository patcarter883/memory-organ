#!/usr/bin/env bash
# Phase R (#19/#67): LOGIT-INJECTION LOCALITY. Logit injection broke the residual wall (solo 0.65->0.87);
# the decisive follow-up is whether forcing the object logit damages same-true-object NEIGHBOURS.
# eval_persistent_locality sweeps CAM_LOGIT_INJECT alpha INTERNALLY (one probe/bind/tap-fit), scoring
# edit DELIVERY together with neighbour KEEP/LEAK. So this is ONE container run, not four.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/logit_locality.out
echo "[loc] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS=32 \
  -e CAM_LOGIT_INJECT_SWEEP="0,2,8,20" -e CAM_LOCALITY_NBR_CAP=3 -e CAM_CONF_DIAG="${CONF_DIAG:-0}" \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 1200 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
     --persistent-locality --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'LOCALITY|alpha|delivery|nbr|written= 137|Traceback|Error|adversarial'
echo "[loc] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
