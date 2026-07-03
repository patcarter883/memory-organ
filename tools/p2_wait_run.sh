#!/usr/bin/env bash
# Wait for a physically-clear GPU (no co-resident serve container) then fire the P2 grid.
set -uo pipefail
echo "[wait] $(date -u +%H:%M:%S) waiting for a clear card (no lease-serve/vllm22 GPU container)..."
tries=0
while true; do
  busy=$(docker ps --format '{{.Names}} {{.Image}}' | grep -E 'lease-serve|vllm22-w4a8|titans:dev' | grep -v "$$" || true)
  if [ -z "$busy" ]; then
    echo "[wait] $(date -u +%H:%M:%S) card clear -> launching P2 grid"
    cd /home/pat/code/memory-organ-p2
    MINISGL=/home/pat/code/minisgl-rdna4-p2 ENGINE=/home/pat/code/memory-organ-p2 CACHE=/home/pat/code/memory-organ/.probe_cache N_REP=3 \
      gpu-lease -n 1 -- bash tools/p2_grid.sh
    exit 0
  fi
  tries=$((tries+1))
  [ $((tries % 10)) -eq 0 ] && echo "[wait] $(date -u +%H:%M:%S) still busy: $busy"
  sleep 150
done
