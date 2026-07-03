#!/usr/bin/env bash
# CAM Track 4 (#19) — RETENTION / INTERFERENCE sweep on the PERSISTENT (online) store.
#
# Reproduces the 137-edit standing-store setup (native gdn_hip, N-scale key-separation fix on) and adds
# --persistent-sweep: checkpoint DURING the incremental write phase and re-query a FIXED early cohort
# (first --persistent-cohort edits) as the store grows. A decaying early-cohort curve = interference
# (does edit #1 survive writing edit #137?), separated from the cumulative all-so-far delivery.
#
# Same isolation contract as run_cam_native_gdn.sh: mount the minisgl WORKTREE (rdna4 src + a current
# gdn_hip .so — forward kernels are enough; the batched-train backward falls back to the torch reference)
# and the memory-organ WORKTREE (with the --persistent-sweep code) — NEVER the shared $PWD. Image titans:dev.
#
# Run:  gpu-lease -n 1 -- bash tools/run_cam_persistent_sweep.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-psweep}
ENGINE=${ENGINE:-/home/pat/code/memory-organ-psweep}
DATA=${DATA:-/home/pat/code/memory-organ/data}
echo "[psweep] HIP=$HIP_VISIBLE_DEVICES ROCR=$ROCR_VISIBLE_DEVICES minisgl=$MINISGL engine=$ENGINE $(date -u +%H:%M:%S)"

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e CAM_EVAL_BATCH_CAP=48 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
  -e CAM_NATIVE_GDN=1 -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_SKIP_CEILING=1 \
  -e GDN_HIP_NATIVE_BWD=1 -e CAM_PERSISTENT_EVAL_BATCH=4 \
  -e CAM_PROBE_CACHE_DIR=/probe_cache \
  -v "$MINISGL":/minisgl:ro \
  -v "$ENGINE":/engine:ro \
  -v "$DATA":/data:ro \
  -v "${CACHE:-/home/pat/code/memory-organ/.probe_cache}":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && time timeout 3600 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1000 --steps 150 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data --tap-layers 24 \
     --seg-len 48 --qa-seg 3 --save-anyway --conf-gate --locality-weight 0.1 \
     --persistent-sweep --persistent-cohort 10"
echo "----- psweep rc=$? $(date -u +%H:%M:%S) -----"
