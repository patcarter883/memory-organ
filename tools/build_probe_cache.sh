#!/usr/bin/env bash
# Pre-warm the CounterFact base-known PROBE CACHE on ONE card. The probe (forwarding ~21k records through
# the frozen base) is the biggest recurring GPU cost and is deterministic in (base, seed, cap, dataset),
# so we run it once with --probe-only and cache `kept` to a writable dir. A later 2-card sweep then hits
# the cache and skips the probe entirely. Single card because model-parallel SLOWS the many-small-forward
# probe (per-forward device transfers).
# Run:  gpu-lease -n 1 -- bash tools/build_probe_cache.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-psweep}; ENGINE=${ENGINE:-/home/pat/code/memory-organ-psweep}
DATA=${DATA:-/home/pat/code/memory-organ/data}; CACHE=${CACHE:-/home/pat/code/memory-organ/.probe_cache}
echo "[probecache] HIP=$HIP_VISIBLE_DEVICES $(date -u +%H:%M:%S)"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 1200 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1 --steps 1 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --conf-gate --probe-only 2>&1 | grep -E 'PROBE|cached|edits|probe-only|Traceback|Error'"
echo "----- rc=$? $(date -u +%H:%M:%S) -----"
ls -la "$CACHE"
