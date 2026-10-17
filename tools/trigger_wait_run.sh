#!/usr/bin/env bash
# Queue the E1 (#104) trigger run while the cards are clear (mirrors the *_wait_run.sh convention).
set -uo pipefail
while true; do
  busy=$(docker ps --format '{{.Names}} {{.Image}}' | grep -E 'lease-serve|vllm22-w4a8|titans:dev' || true)
  if [ -z "$busy" ]; then
    cd /home/pat/code/memory-organ-tapmetric-preserve
    MINISGL=${MINISGL:-/home/pat/code/minisgl-rdna4-camtap} gpu-lease -n 1 -- bash tools/trigger_ripple.sh
    exit 0
  fi
  sleep 150
done
