#!/usr/bin/env bash
# Track 5 (#99): LATENT object + ROUTER-gated generation, both subject and object arbitrary-length.
# Subject key already pools its span (any length); object is now stored as a pooled LATENT (mean of the
# object's token embeddings, CAM_OBJ_LATENT=1) instead of a single token. Bind the COUNTERFACTUAL
# (CAM_BIND_TRUE=0) so multi-token delivery is VISIBLE in free generation. The per-token router gates each
# decode step (CAM_GEN_ROUTER=1): fire at the answer slot, hand the multi-token tail to base fluency.
# Read the generated TEXT: does the multi-token object phrase appear, fluently?
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-softsteer
DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache
MINISGL=/home/pat/code/minisgl-rdna4-p
LOG="$ENGINE/tools/gen_latent.out"
echo "[genlatent] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS="${DISJOINT_BANKS:-32}" \
  -e CAM_WRITE_AT_READ=1 -e CAM_BIND_TRUE=0 \
  -e CAM_MAX_OBJ_TOK="${MAX_OBJ_TOK:-3}" -e CAM_OBJ_LATENT=1 \
  -e CAM_GEN_ROUTER=1 -e CAM_ROUTER_ALPHA="${ROUTER_ALPHA:-1.5}" -e CAM_ROUTER_KL="${ROUTER_KL:-0.1}" -e CAM_ROUTER_STEPS="${ROUTER_STEPS:-400}" -e CAM_MULTIGATE_TOPK=16 \
  -e CAM_LOGIT_INJECT="${ALPHA:-8}" -e CAM_LOGIT_GATE_C0=1 -e CAM_LOGIT_GATE_HARD=1 \
  -e CAM_GEN_LEN="${GEN_LEN:-16}" -e CAM_GEN_SAMPLE="${GEN_SAMPLE:-12}" -e CAM_GEN_INJECT_STEPS="${INJ_STEPS:-1}" -e CAM_GEN_MULTITOK="${MULTITOK:-0}" \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout ${PYTIMEOUT:-1500} python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations ${MULTI_REL:-6} --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
     --persistent-generate --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'cf-multi\] EDITING|GENERATION COHERENCE|gen-router|edit |OFF |ON constant|ON answer|ON router|NEW object|Traceback|Error'
echo "[genlatent] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
