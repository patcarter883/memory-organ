#!/usr/bin/env bash
# Track 4 #19: is the persistent-store delivery ceiling (~0.2-0.25 @ N=137) STORE CAPACITY or ADDRESSING?
# Sweep the product-key store size at persistent scale (learned pool ON), N_REP reps each (noisy metric):
#   BASE = --pk-read-heads 8  --n-sub 32  (1024 slots, 8 read heads)
#   4x   = --pk-read-heads 16 --n-sub 64  (4096 slots, 16 read heads)
# If 4x doesn't beat BASE at N=137, the ceiling is ADDRESSING (key separation), not capacity — confirming
# the episodic #37 finding at persistent scale. Report N=137 cumulative cf-delivery.
# Run:  gpu-lease -n 1 -- bash tools/cap_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-cap}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-cap}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}
echo "[cap] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP $(date -u +%H:%M:%S)"

one() {  # $1 = read-heads, $2 = n-sub, $3 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_LEARNED_KEY_POOL=1 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads $1 --n-sub $2 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34'" | sed "s/^/[$3] /"
}

for rep in $(seq 1 "$N_REP"); do echo "===== BASE(h8,n32) rep $rep ====="; one 8  32 "BASE"; done
for rep in $(seq 1 "$N_REP"); do echo "===== 4x(h16,n64) rep $rep =====";  one 16 64 "4x  "; done
echo "[cap] DONE $(date -u +%H:%M:%S)"
