#!/usr/bin/env bash
# CAM Track-1 on the NATIVE gdn_hip path (RDNA4). OPT-IN SAFETY NET, not the default.
#
# FINDING (2026-07-02, GPU-validated): the native gdn_hip patch is numerically CORRECT — parity vs the
# fla->torch reference passes (fwd cos 0.9998, dL/d_embeds cos 0.9994, NO hang) — but in titans:dev it is
# ~2.4x SLOWER on the tap-fit step (297s vs 126s / 100 steps) and ~1.6x slower end-to-end (6m49 vs 4m21),
# because NativeGDNShim runs a per-sequence Python loop + chunked-recompute backward while fla here falls
# back to a vectorized torch GDN that is faster and does NOT hang. So DO NOT flip CAM_NATIVE_GDN on for
# speed. Keep it for the case fla actually hangs (longer-context CAM, or an image without fla). The real
# CAM speed levers are elsewhere (varlen-pack the shim; cache frozen-base activations up to the tap; trim
# eval n). See RESULTS/DIARY.
#
# ONE-TIME SETUP (the minisgl gdn_hip .so is gitignored + must carry the bwd kernels; worktrees isolate
# the source per the CLAUDE.md rule):
#   git -C /home/pat/code/minisgl-rdna4 worktree add -b cam-native /home/pat/code/minisgl-rdna4-camnative rdna4
#   cp <a checkout with an up-to-date build>/gdn_hip/gdn_hip_C.cpython-312-x86_64-linux-gnu.so \
#      /home/pat/code/minisgl-rdna4-camnative/gdn_hip/           # must have rmsnorm_gated_bwd/causal_conv1d_bwd syms
#   git -C /home/pat/code/memory-organ worktree add -b cam-native-hook /home/pat/code/memory-organ-camnative main
# (override the mount paths via MINISGL= / ENGINE= if your worktrees differ.)
#
# Two phases inside one GPU lease:
#   (1) PARITY  — minisgl tools/cam_native_gdn_validate.py: forward + dL/d_embeds of the patched base
#                 (native gdn_hip) vs the stock HF fla->torch fallback reference. Correctness gate.
#   (2) SMOKE   — a short recall_mag.py run with CAM_NATIVE_GDN=1 to confirm the hook wires end-to-end
#                 and to time it against the torch-fallback baseline (same args as the conf-gate runs,
#                 fewer steps). Set SMOKE_BASELINE=1 to also run the CAM_NATIVE_GDN=0 baseline for a
#                 head-to-head wall-clock.
#
# Isolation: mounts the minisgl WORKTREE (rdna4 src + a current gdn_hip .so with the bwd kernels) and the
# memory-organ WORKTREE (with the _maybe_patch_native_gdn hook) — NEVER the shared $PWD. Image: titans:dev
# (same py3.12/torch2.10 ABI as vllm22-w4a8:combined, so the combined-built .so loads).
#
# Run:  gpu-lease -n 1 -- bash tools/run_cam_native_gdn.sh
set -uo pipefail
MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-camnative}
ENGINE=${ENGINE:-/home/pat/code/memory-organ-camnative}
DATA=${DATA:-/home/pat/code/memory-organ/data}
echo "[cam-native] HIP=$HIP_VISIBLE_DEVICES ROCR=$ROCR_VISIBLE_DEVICES minisgl=$MINISGL engine=$ENGINE $(date -u +%H:%M:%S)"

drun() {  # $1 = extra docker env flags (word-split), $2 = bash -lc script (trails the image)
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e CAM_EVAL_BATCH_CAP=48 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    $1 \
    -v "$MINISGL":/minisgl:ro \
    -v "$ENGINE":/engine:ro \
    -v "$DATA":/data:ro \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash titans:dev -lc "$2"
}

echo "===== (1) PARITY: native gdn_hip vs fla-fallback reference ====="
drun "-e CAM_NATIVE_GDN=1" \
  'source /app/.venv/bin/activate && timeout 900 python /minisgl/tools/cam_native_gdn_validate.py'
echo "----- parity rc=$? $(date -u +%H:%M:%S) -----"

smoke() {  # $1 = CAM_NATIVE_GDN value, $2 = tag
  echo "===== (2) SMOKE $2 (CAM_NATIVE_GDN=$1) ====="
  drun "-e CAM_NATIVE_GDN=$1" \
    "source /app/.venv/bin/activate && time timeout 1200 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --batch 8 --bind-steps 300 --steps 100 --phrasing counterfactual \
       --dataset counterfact --data-dir /data --tap-layers 24 \
       --seg-len 48 --qa-seg 2 --save-anyway --conf-gate --locality-weight 0.1"
  echo "----- smoke $2 rc=$? $(date -u +%H:%M:%S) -----"
}
smoke 1 NATIVE
[ "${SMOKE_BASELINE:-0}" = "1" ] && smoke 0 TORCH-FALLBACK
echo "[cam-native] DONE $(date -u +%H:%M:%S)"
