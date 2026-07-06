#!/usr/bin/env bash
# Generation-coherence reality check: bind + tap at N=137 (validated config), then GENERATE from a sample
# of edit prompts with memory OFF vs ON (K1 + hard-conf-gated logit injection). Watchdog-compatible:
# passes ${CNAME:+--name} to docker and prints "[priv] done" on success.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/private_demo.out
echo "[priv] start $(date -u +%H:%M:%S)" | tee "$LOG"
docker run --rm ${CNAME:+--name "$CNAME"} --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_DISJOINT_BANKS=512 \
  -e CAM_WRITE_AT_READ=1 -e CAM_LOGIT_INJECT="${ALPHA:-8}" -e CAM_LOGIT_GATE_C0=0.5 -e CAM_LOGIT_GATE_HARD=1 \
  -e CAM_GEN_INJECT_STEPS="${INJ_STEPS:-2}" -e CAM_GEN_LEN="${GEN_LEN:-8}" -e CAM_GEN_SAMPLE="${GEN_SAMPLE:-4}" -e CAM_GEN_INJECT_STEPS=1 \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 600 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0 \
     --perpos-key codebook --mt-positions 4 --mt-recon-weight 1.0 \
     --private-facts /data/private_facts.json --persistent-generate --persistent-cohort 10 2>&1" | tee -a "$LOG" \
  | grep -E 'edits across|binding held-out|mt-recon|GENERATION COHERENCE|OFF:|ON :|edit |NEW object|Traceback|Error'
echo "[priv] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
