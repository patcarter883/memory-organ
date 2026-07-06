#!/usr/bin/env bash
# VALUE-CAPACITY experiment: private-facts multi-token delivery with a larger, un-normalized value code.
# Same bind+tap+generate reality check as private_demo.sh, but attacks the 512-d unit-sphere value
# bottleneck three ways: --mem-dim 1024 (2x value width) + --bind-steps 3000 (3x train) +
# CAM_MT_VALUE_NO_NORM=1 (drop the value-path LayerNorm so stored codes keep magnitude/direction spread).
# Runs against an ISOLATED worktree (memory-organ-p-valcap). Watchdog-compatible.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p-valcap; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=${VALCAP_LOG:-/home/pat/code/memory-organ-p-valcap/tools/value_capacity_demo.out}
MEM_DIM=${MEM_DIM:-1024}; BIND_STEPS=${BIND_STEPS:-3000}; VALNONORM=${VALNONORM:-1}
STEPS=${STEPS:-150}; RUN_TIMEOUT=${RUN_TIMEOUT:-2400}
BATCH=${BATCH:-4}; SEG_LEN=${SEG_LEN:-48}
echo "[valcap] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm ${CNAME:+--name "$CNAME"} --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e PYTHONFAULTHANDLER="${PYFH:-0}" -e AMD_LOG_LEVEL="${AMDLOG:-0}" \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS=512 \
  -e CAM_WRITE_AT_READ=1 -e CAM_LOGIT_INJECT="${ALPHA:-8}" -e CAM_LOGIT_GATE_C0="${GATE_C0:-0.5}" -e CAM_LOGIT_GATE_HARD=1 \
  -e CAM_GEN_INJECT_STEPS="${INJ_STEPS:-2}" -e CAM_GEN_LEN="${GEN_LEN:-8}" -e CAM_GEN_SAMPLE="${GEN_SAMPLE:-4}" -e CAM_GEN_INJECT_STEPS=1 \
  -e CAM_MT_VALUE_NO_NORM="$VALNONORM" -e CAM_GATE_CALIB="${CALIB:-0}" -e CAM_MT_DECODE_DIAG="${MTDIAG:-0}" \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout $RUN_TIMEOUT python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch $BATCH --mem-dim $MEM_DIM --bind-steps $BIND_STEPS --steps $STEPS --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len $SEG_LEN --qa-seg 3 --save-anyway --conf-gate --locality-weight 0 \
     --perpos-key codebook --mt-positions 4 --mt-recon-weight 1.0 \
     --private-facts /data/private_facts.json --persistent-generate --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'edits across|binding held-out|mt-recon|GENERATION COHERENCE|OFF:|ON :|edit |NEW object|Traceback|Error'
echo "[valcap] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
