#!/usr/bin/env bash
# Track 4 #19 Phase A.2: GLOBAL key-separation repulsion. A.1's per-doc repulsion only separated the M=8
# in-doc subjects (helped N=34, flat at N=137). This adds a running FIFO buffer of recent write-keys
# (MoCo-style) so repulsion sees the whole standing population. A/B LOCAL (per-doc only) vs GLOBAL
# (per-doc + buffer), both CAM_KEY_REPULSION=1.0, N_REP reps, N=137 + N=34 cumulative cf-delivery.
# Run:  gpu-lease -n 1 -- bash tools/globalrep_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-g}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-g}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}
echo "[grep] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP $(date -u +%H:%M:%S)"

one() {  # $1 = CAM_KEY_REPULSION_GLOBAL, $2 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_LEARNED_KEY_POOL=1 \
    -e CAM_KEY_REPULSION=1.0 -e CAM_KEY_REPULSION_GLOBAL="$1" -e CAM_KEY_REPULSION_BUFSIZE=256 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34|Traceback|Error'" | sed "s/^/[$2] /"
}

for rep in $(seq 1 "$N_REP"); do echo "===== LOCAL rep $rep ====="; one 0 "LOCAL "; done
for rep in $(seq 1 "$N_REP"); do echo "===== GLOBAL rep $rep ====="; one 1 "GLOBAL"; done
echo "[grep] DONE $(date -u +%H:%M:%S)"
