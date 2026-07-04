#!/usr/bin/env bash
# Scale-N de-risk (before productionizing): does the editing triad hold as N grows?
# Sweep --multi-relations (the N knob; more relations -> more base-known edits) with K1-on. Bound the
# eval to keep it fast + dodge the RDNA4 cohort-forward flake: 2 alphas (0, 2 = operating point), 1
# neighbour/paraphrase per edit, cohorts capped to 60 each, prompts <=48 tok, TRIAD_DEBUG flushes.
# Core scale-N signal = delivery + below-gate + conf-p95 vs N (over ALL N); locality/gen on the capped
# subset. One container per N, output copied to scaleN_r${R}.out.
set -uo pipefail
cd /home/pat/code/memory-organ-p
for R in 6 15 30 60; do
  echo "[scaleN] === MULTI_REL=$R start $(date -u +%H:%M:%S) ==="
  gpu-lease -n 1 -- bash -c \
    'MULTI_REL='"$R"' DISJOINT_BANKS=512 WRITE_AT_READ=1 CONF_DIAG=1 TRIAD_DEBUG=1 \
     ALPHA_SWEEP=0,2 NBR_CAP=1 COHORT_CAP=60 PROMPT_MAXTOK=48 PYTIMEOUT=2700 \
     bash tools/logit_locality.sh'
  cp tools/logit_locality.out tools/scaleN_r${R}.out 2>/dev/null || true
  n=$(grep -oE '[0-9]+ edits across [0-9]+ relations' tools/scaleN_r${R}.out 2>/dev/null | head -1)
  bg=$(grep -oE 'edits below C0: [0-9]+/[0-9]+' tools/scaleN_r${R}.out 2>/dev/null | head -1)
  echo "[scaleN] === MULTI_REL=$R DONE | $n | $bg | $(date -u +%H:%M:%S) ==="
done
echo "[scaleN] ALL DONE"
