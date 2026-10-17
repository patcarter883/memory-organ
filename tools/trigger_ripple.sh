#!/usr/bin/env bash
# E1 (#104) trigger-local distilled delivery — first-signal run.
# Stack = the VALIDATED in-dist protocol exactly as recorded in the Aug-5 baseline
# (docs/journal/measurements/ripple_baseline.sh): titans:dev image, /app/.venv, NO CAM_NATIVE_GDN,
# tap at L12, bind 1500, conf-gate ON, and the reference hop forced to `country`.
# (The lean/native-GDN stack proved unusable for this training: its path emits inf grads / NaN
# losses and degenerate free-gens — debug log: ~/.hermes/cache/scratch/e1_smoke6_debug.log.)
set -uo pipefail
ENGINE=${ENGINE:-/home/pat/code/memory-organ-tapmetric-preserve}
DATA=${DATA:-/home/pat/code/memory-organ/data}
CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
ARMS=${ARMS:-trigger,nomask}
LIMIT=${LIMIT:-60}
TSTEPS=${TSTEPS:-300}
SPP=${SPP:-2}
NSEEDS=${NSEEDS:-6}
BIND=${BIND:-1500}
HOP=${HOP:-country}
OBJ=${OBJ:-distill}
LOG=$ENGINE/tools/trigger_ripple.out
echo "[trigrun] start $(date -u +%H:%M:%S) arms=$ARMS limit=$LIMIT steps=$TSTEPS spp=$SPP nseeds=$NSEEDS bind=$BIND hop=$HOP obj=$OBJ" | tee "$LOG"
docker run --rm ${CNAME:+--name "$CNAME"} \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CAM_SKIP_CEILING=1 -e CAM_PERSISTENT_EVAL_BATCH=4 \
  -e CAM_TRIG_DEBUG="${CAM_TRIG_DEBUG:-0}" \
  -e PYTHONPATH=/engine/tools/torchgdn_site -e CAM_FORCE_TORCH_GDN="${FORCE_TORCH_GDN:-1}" \
  -e CAM_PROBE_CACHE_DIR=/probe_cache -e CAM_MQUAKE_SHOW=12 -e CAM_MQUAKE_GEN_LEN="${GENLEN:-40}" \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 \
  -e CAM_RIPPLE_EVAL_HOP="$HOP" \
  -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout ${TIMEOUT:-14400} python -u /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps ${BIND} --steps 200 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 12 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate \
     --mquake-limit ${LIMIT} --trigger-eval --trigger-arms ${ARMS} --trigger-steps ${TSTEPS} \
     --trigger-objective ${OBJ} \
     --trigger-spp ${SPP} --trigger-nseeds ${NSEEDS} 2>&1" \
  | tee -a "$LOG" | grep -E '\[mag\]|\[trig\]|\[cam\]|\[cost\]|\[taptrace\]|Traceback|Error|error|Killed|CUDA|HIP|RuntimeError|Assertion'
echo "[trigrun] DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
