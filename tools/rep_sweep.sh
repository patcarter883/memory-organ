#!/usr/bin/env bash
# Track 4 #19 Phase A.1: KEY-SEPARATION REPULSION LOSS. Off-diagonal Gram penalty on the M per-doc write
# addresses pushes distinct subjects' keys toward orthogonality (the ceiling is ADDRESSING). A/B
# CAM_KEY_REPULSION=0 (the +46% learned-pool baseline) vs a weight, single learned key (H=1), N_REP reps
# (noisy metric), N=137 + N=34 cumulative cf-delivery. Set REP_W to sweep the weight.
# Run:  gpu-lease -n 1 -- bash tools/rep_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-mk}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-rep}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}; REP_W=${REP_W:-1.0}
echo "[rep] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP REP_W=$REP_W $(date -u +%H:%M:%S)"

one() {  # $1 = CAM_KEY_REPULSION, $2 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_LEARNED_KEY_POOL=1 -e CAM_KEY_REPULSION="$1" -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34|Traceback|Error'" | sed "s/^/[$2] /"
}

for rep in $(seq 1 "$N_REP"); do echo "===== REP0 rep $rep ====="; one 0     "REP0"; done
for rep in $(seq 1 "$N_REP"); do echo "===== REP$REP_W rep $rep ====="; one "$REP_W" "REPW"; done
echo "[rep] DONE $(date -u +%H:%M:%S)"
