#!/usr/bin/env bash
# Track 5 (#99): SERVING STATE — warm in-process memory service with the base-uncertainty WRITE GATE.
# Fit the router (read side, how-much), start an EMPTY store, stream facts through the write gate (write side,
# what-to-remember: store iff the base can't recall it, p_base<τ), then serve router-gated seed-once
# generation from what was kept. Bind the NOVEL object (CAM_BIND_TRUE=0) so "the base can't know it" is real.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-softsteer
DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache
MINISGL=/home/pat/code/minisgl-rdna4-p
LOG="$ENGINE/tools/serve.out"
echo "[serve] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS="${DISJOINT_BANKS:-32}" \
  -e CAM_WRITE_AT_READ=1 -e CAM_BIND_TRUE=0 \
  -e CAM_REMEMBER_TAU="${REMEMBER_TAU:-0.5}" -e CAM_REMEMBER_GATE="${REMEMBER_GATE:-rank}" -e CAM_REMEMBER_RANK="${REMEMBER_RANK:-1}" -e CAM_STORE_CAP="${STORE_CAP:-0}" -e CAM_EVICT="${EVICT:-fifo}" -e CAM_SERVE_STREAM="${SERVE_STREAM:-24}" -e CAM_GEN_LEN="${GEN_LEN:-12}" \
  -e CAM_ROUTER_ALPHA="${ROUTER_ALPHA:-1.5}" -e CAM_ROUTER_KL="${ROUTER_KL:-0.1}" -e CAM_ROUTER_STEPS="${ROUTER_STEPS:-400}" -e CAM_MULTIGATE_TOPK=16 \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout ${PYTIMEOUT:-1500} python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations ${MULTI_REL:-6} --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
     --serve --export-serving /engine/cam_ckpt --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'SERVING STATE|router calibrated|write-gate=|delivery on KEPT|export]|checkpoint|Traceback|Error'
echo "[serve] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
