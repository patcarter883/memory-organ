#!/usr/bin/env bash
# Phase N0b: how far does N unlock when the relation prompt/suffix filter is relaxed? All runs use
# CAM_ALL_SUBJ_LENGTHS=1 (N0) + probe-only (K=1 cache hit -> fast). Sweep the suffix cap, then add
# empty-prefix. --multi-relations 200 so no relation is dropped by the top-R cut.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-p; DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache; MINISGL=/home/pat/code/minisgl-rdna4-p
LOG=/home/pat/code/memory-organ-p/tools/n0b_probe.out
echo "[n0b] start $(date -u +%H:%M:%S)" | tee "$LOG"
# label:MAX_SUFFIX_TOK:ALLOW_EMPTY_PREFIX
for CFG in "suf6:6:0" "suf12:12:0" "suf20:20:0" "suf100:100:0" "suf100+emptypre:100:1"; do
  L="${CFG%%:*}"; rest="${CFG#*:}"; MS="${rest%%:*}"; EP="${rest##*:}"
  echo "[n0b] === $L (MAX_SUFFIX_TOK=$MS ALLOW_EMPTY_PREFIX=$EP) ===" | tee -a "$LOG"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -e CAM_NATIVE_GDN=1 -e CAM_SKIP_CEILING=1 -e CAM_PROBE_CACHE_DIR=/probe_cache -e CAM_MAX_OBJ_TOK=1 \
    -e CAM_ALL_SUBJ_LENGTHS=1 -e CAM_MAX_SUFFIX_TOK="$MS" -e CAM_ALLOW_EMPTY_PREFIX="$EP" \
    -v "$MINISGL":/minisgl:ro -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && timeout 900 python /engine/cam/recall_mag.py \
       --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
       --phrasing counterfactual_multi --multi-relations 200 --cf-probe-cap 21500 \
       --dataset counterfact --data-dir /data --tap-layers 24 --probe-only 2>&1" \
    | tee -a "$LOG" | grep -E 'relation-filter|grouping=|edits across|edits ready|Traceback|Error'
done
echo "[n0b] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
