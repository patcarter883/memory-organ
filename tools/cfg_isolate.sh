#!/usr/bin/env bash
set -uo pipefail
cd /home/pat/code/memory-organ-p
for CFG in "48 3 seg48" "64 4 seg64"; do
  set -- $CFG
  echo "[cfgAB] start $3 $(date -u +%H:%M:%S)"
  gpu-lease -n 1 -- bash tools/cfg_one.sh "$1" "$2" "$3"
  echo "[cfgAB] done $3 $(date -u +%H:%M:%S)"
done
echo "[cfgAB] ALL DONE"
