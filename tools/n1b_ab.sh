#!/usr/bin/env bash
set -uo pipefail
cd /home/pat/code/memory-organ-p
for IL in 0 1; do
  echo "[n1b-ab] === REL_INTERLEAVE=$IL start $(date -u +%H:%M:%S) ==="
  gpu-lease -n 1 -- bash -c "REL_INTERLEAVE=$IL TAG=il$IL bash tools/n1_bind.sh"
  echo "[n1b-ab] === REL_INTERLEAVE=$IL DONE $(date -u +%H:%M:%S) ==="
done
echo "[n1b-ab] ALL DONE"
