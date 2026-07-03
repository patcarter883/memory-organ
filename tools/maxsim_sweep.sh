#!/usr/bin/env bash
# Track 4 #19 Phase B: ColBERT MaxSim read for multi-vector keys. The H=3 multi-vector prototype FAILED
# with softmax-pool read (N=137 0.141 vs 0.234 single-key) because pooling H keys collides at scale.
# MaxSim (weight heads by retrieval strength, take the best-matching head — not pool) should rescue it.
# A/B: H1 (single-key baseline, best so far ~0.234) vs H3+MaxSim (full Phase B recipe). N_REP reps.
# (H3+softmax-pool = 0.141 established earlier.) N=137 + N=34 cumulative cf-delivery.
# Run:  gpu-lease -n 1 -- bash tools/maxsim_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-ms}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-ms}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}
echo "[ms] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP $(date -u +%H:%M:%S)"

one() {  # $1 = CAM_KEY_HEADS, $2 = CAM_KEY_MAXSIM, $3 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
    -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_LEARNED_KEY_POOL=1 \
    -e CAM_KEY_HEADS="$1" -e CAM_KEY_MAXSIM="$2" -e CAM_KEY_MAXSIM_TEMP=0.1 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34|Traceback|Error'" | sed "s/^/[$3] /"
}

for rep in $(seq 1 "$N_REP"); do echo "===== H3+MaxSim rep $rep ====="; one 3 1 "H3maxsim"; done
for rep in $(seq 1 "$N_REP"); do echo "===== H1 rep $rep ====="; one 1 0 "H1     "; done
echo "[ms] DONE $(date -u +%H:%M:%S)"
