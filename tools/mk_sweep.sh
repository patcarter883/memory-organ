#!/usr/bin/env bash
# Track 4 #19: MULTI-VECTOR KEYS — does giving each subject H distinct learned key vectors (written to H
# store slots -> same value) beat a single learned key at persistent scale? The capacity probe showed the
# ceiling is ADDRESSING, so more separable per-subject addresses should lift delivery. A/B H=1 vs H=3
# (learned pool ON), N_REP reps each (noisy metric), N=137 + N=34 cumulative cf-delivery.
# Run:  gpu-lease -n 1 -- bash tools/mk_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-mk}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-mk}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}
echo "[mk] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP $(date -u +%H:%M:%S)"

one() {  # $1 = CAM_KEY_HEADS, $2 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_LEARNED_KEY_POOL=1 -e CAM_KEY_HEADS="$1" -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34|Traceback|Error'" | sed "s/^/[$2] /"
}

for rep in $(seq 1 "$N_REP"); do echo "===== H1 rep $rep ====="; one 1 "H1"; done
for rep in $(seq 1 "$N_REP"); do echo "===== H3 rep $rep ====="; one 3 "H3"; done
echo "[mk] DONE $(date -u +%H:%M:%S)"
