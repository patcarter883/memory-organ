#!/usr/bin/env bash
# Reusable GPU-run WATCHDOG for the RDNA4 gen-cohort forward flake (kills 3x today).
# Launches a run script under gpu-lease, monitors its log for STALENESS, and on a stall kills ONLY THIS
# run's own named container (safe — never touches another agent's titans:dev) + the lease, then RETRIES.
#
# Usage: watchdog_run.sh <run_script> <logfile> [stale_secs=240] [retries=2] [done_regex='\[.*\] done|ALL DONE']
# The wrapped run_script MUST:
#   (a) pass ${CNAME:+--name "$CNAME"} to its `docker run` (CNAME is exported by this watchdog), and
#   (b) print a line matching <done_regex> to <logfile> on success.
set -uo pipefail
SCRIPT="$1"; LOG="$2"; STALE="${3:-240}"; RETRIES="${4:-2}"; DONE_RE="${5:-\[.*\] done|ALL DONE}"
export CNAME="cam-wd-$$"
kill_run(){ docker kill "$CNAME" >/dev/null 2>&1 || true; [ -n "${LPID:-}" ] && kill -TERM "$LPID" 2>/dev/null || true;
            pkill -TERM -f "$SCRIPT" 2>/dev/null || true; sleep 3; }
trap kill_run EXIT
for attempt in $(seq 0 "$RETRIES"); do
  echo "[watchdog] attempt $attempt (container $CNAME, stale>${STALE}s -> kill+retry) $(date -u +%H:%M:%S)"
  : > "$LOG.wd"                                          # sentinel so mtime starts fresh
  gpu-lease -n 1 -- bash -c "CNAME=$CNAME bash '$SCRIPT'" > "$LOG.wd" 2>&1 &
  LPID=$!
  while kill -0 "$LPID" 2>/dev/null; do
    sleep 20
    grep -qE "$DONE_RE" "$LOG" 2>/dev/null && { echo "[watchdog] DONE seen"; wait "$LPID" 2>/dev/null; trap - EXIT; exit 0; }
    ref="$LOG"; [ -f "$LOG" ] || ref="$LOG.wd"
    age=$(( $(date +%s) - $(stat -c %Y "$ref" 2>/dev/null || date +%s) ))
    if [ "$age" -gt "$STALE" ]; then
      echo "[watchdog] STALL ${age}s > ${STALE}s -> killing $CNAME + retrying"
      kill_run; break
    fi
  done
  grep -qE "$DONE_RE" "$LOG" 2>/dev/null && { echo "[watchdog] DONE (post-kill check)"; trap - EXIT; exit 0; }
  echo "[watchdog] attempt $attempt did not complete"
done
echo "[watchdog] EXHAUSTED $((RETRIES+1)) attempts without DONE"; trap - EXIT; exit 1
