#!/usr/bin/env bash
# HELD-OUT SUBJECT test (Track 4 #19): bind (train projections + tap) on BIND subjects, then run the triad
# persistent-locality eval writing+querying ONLY HELD-OUT subjects the bind NEVER saw (CAM_HELDOUT_FRAC=0.3
# splits WITHIN each relation-bucket, so relations/lengths/templates are shared — we isolate NOVEL SUBJECT,
# not novel relation). N=137-scale legacy config (--multi-relations 6 --seg-len 48 --qa-seg 3), K1 on.
# Compare held-out efficacy / #below-gate against the in-distribution numbers (efficacy ~0.81, below-gate 0).
# Watchdog-compatible: passes ${CNAME:+--name} to docker and prints "[heldout] done" on success.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/heldout_test.out
echo "[heldout] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm ${CNAME:+--name "$CNAME"} --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS=512 \
  -e CAM_WRITE_AT_READ=1 -e CAM_HELDOUT_FRAC="${HELDOUT_FRAC:-0.3}" \
  -e CAM_CONF_DIAG=1 -e CAM_TRIAD_DEBUG=1 -e CAM_LOGIT_INJECT_SWEEP="0,2,8,20" \
  -e CAM_LOCALITY_NBR_CAP=1 -e CAM_COHORT_CAP=80 \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 3600 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0 \
     --persistent-locality --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'HELD-OUT|edits across|below.gate|delivery|efficacy|locality|generality|DEP|GEN-hit|alpha|Traceback|Error'
echo "[heldout] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
