#!/usr/bin/env bash
# Track 5 (#99): FULL-MULTIGATE alpha dose-response. Runs concurrently on the OTHER card (gpu-lease
# auto-assigns the free one). Full multigate (storeconf+headroom+scope+sparse top-k) at several alphas to
# find the gentle-yet-effective operating point: where does dP(true) saturate and where does KL stay low?
# Combined with multigate.sh's alpha=1 this gives a {0.5,1,2,4} curve.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-softsteer
DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache
MINISGL=/home/pat/code/minisgl-rdna4-p
LOG="$ENGINE/tools/multigate_asweep.out"
echo "[asweep] start $(date -u +%H:%M:%S)" | tee "$LOG"

run_cfg () {  # $1 = alpha
  echo "[asweep] === alpha=$1 (full multigate) ===" | tee -a "$LOG"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS="${DISJOINT_BANKS:-32}" \
    -e CAM_WRITE_AT_READ=1 -e CAM_BIND_TRUE=1 \
    -e CAM_SOFT_STEER=1 -e CAM_LOGIT_INJECT="$1" -e CAM_LOGIT_GATE_C0="${GATE_C0:-1}" -e CAM_LOGIT_GATE_HARD=1 \
    -e CAM_MULTIGATE=1 -e CAM_MULTIGATE_TOPK="${TOPK:-16}" -e CAM_MULTIGATE_SCOPE_MIN="${SCOPE_MIN:-0.10}" \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout ${PYTIMEOUT:-1500} python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations ${MULTI_REL:-6} --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-graded --persistent-cohort 10 2>&1" | tee -a "$LOG" \
    | grep -E 'Track 5|P\(true\)_off|\[0\.|\[1\.|ALL |gate:|Traceback|Error'
}

for a in 0.5 2 4; do run_cfg "$a"; done
echo "[asweep] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
