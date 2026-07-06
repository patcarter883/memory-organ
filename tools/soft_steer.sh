#!/usr/bin/env bash
# Track 5 (#99): SOFT-STEERING / graded lean. Same bind + tap-fit as the locality run, but instead of the
# argmax-flip metric it measures the GRADED effect on the TRUE object (P(true)/rank/logprob/KL, memory OFF
# vs ON), bucketed by the base's pre-edit P(true). Two configs in ONE lease/container:
#   (A) CAM_BIND_TRUE=0  counterfactual bind  -> metric SANITY: memory should push P(true) DOWN (dP<0)
#   (B) CAM_BIND_TRUE=1  true bind            -> HYPOTHESIS: memory should push P(true) UP (dP>0), biggest
#                                                in the low-P(true) bucket (§3.12 uncertain regime)
# alpha=0 here = pure trained-tap lean (no blunt logit injection); a low-alpha follow-up can add CAM_LOGIT_INJECT.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-softsteer          # the ISOLATED worktree (never $PWD)
DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache
MINISGL=/home/pat/code/minisgl-rdna4-p                 # native GDN (.so built)
LOG="$ENGINE/tools/soft_steer.out"
echo "[soft] start $(date -u +%H:%M:%S)" | tee "$LOG"

run_cfg () {  # $1 = CAM_BIND_TRUE value, $2 = label
  echo "[soft] === config: bind_true=$1 ($2) ===" | tee -a "$LOG"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS="${DISJOINT_BANKS:-32}" \
    -e CAM_WRITE_AT_READ=1 -e CAM_BIND_TRUE="$1" \
    -e CAM_LOGIT_INJECT="${ALPHA:-0}" -e CAM_LOGIT_GATE_C0="${GATE_C0:-1}" -e CAM_LOGIT_GATE_HARD="${GATE_HARD:-1}" \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout ${PYTIMEOUT:-1500} python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations ${MULTI_REL:-6} --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-graded --persistent-cohort 10 2>&1" | tee -a "$LOG" \
    | grep -E 'Track 5|P\(true\)|bucket|ALL |LEANS|dP|Traceback|Error|written='
}

run_cfg 0 counterfactual-sanity
run_cfg 1 true-bind-hypothesis
echo "[soft] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
