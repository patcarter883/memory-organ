#!/usr/bin/env bash
# Track 4 #19 P2: delivery confirmation of the proxy's top predictions + the GTE revival.
# Cells (learned-pool raw keys unless GTE-file given): raw/whitened-inembed/query-BatchNorm/whitened-GTE
# x disjoint-banks. Confirms H1 (whitening lifts/substitutes-for banks), H4 (BatchNorm native lever), and
# the OFAT-confounded GTE revival (de-anisotropized GTE works where raw GTE died).
# Run:  gpu-lease -n 1 -- bash tools/p2_grid.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-p2}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-p2}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
N_REP=${N_REP:-3}; BIND=${BIND:-1000}; STEPS=${STEPS:-150}
echo "[p2] HIP=$HIP_VISIBLE_DEVICES N_REP=$N_REP BIND=$BIND STEPS=$STEPS $(date -u +%H:%M:%S)"

run() {  # $1 = extra -e env flags (word-split), $2 = tag
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
    $1 \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 4 --bind-steps $BIND --steps $STEPS --phrasing counterfactual_multi \
       --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
       --persistent-sweep --persistent-cohort 10 2>&1 | grep -E 'written= 137|written=  34|Traceback|Error'" | sed "s/^/[$2] /"
}

POOL="-e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1"
WIN="-e CAM_SUBJ_ONLY_QUERY=1 -e CAM_GTE_KEYS=1 -e CAM_GTE_KEYS_FILE=/probe_cache/whiten_inembed_keys.pkl"
WGTE="-e CAM_SUBJ_ONLY_QUERY=1 -e CAM_GTE_KEYS=1 -e CAM_GTE_KEYS_FILE=/probe_cache/whiten_gte_keys.pkl"

declare -a CFG=(
  "$POOL -e CAM_DISJOINT_BANKS=1|raw-B1"
  "$POOL -e CAM_DISJOINT_BANKS=32|raw-B32"
  "$WIN  -e CAM_DISJOINT_BANKS=1|white-B1"
  "$WIN  -e CAM_DISJOINT_BANKS=32|white-B32"
  "$POOL -e CAM_QUERY_BATCHNORM=1 -e CAM_DISJOINT_BANKS=1|bn-B1"
  "$WGTE -e CAM_DISJOINT_BANKS=32|whiteGTE-B32"
)
for c in "${CFG[@]}"; do
  env="${c%%|*}"; tag="${c##*|}"
  for rep in $(seq 1 "$N_REP"); do echo "===== $tag rep $rep ====="; run "$env" "$tag"; done
done
echo "[p2] DONE $(date -u +%H:%M:%S)"
