#!/usr/bin/env bash
# Runner-side guard for patcarter883/memory-organ (a public repository).
#
# The runner runs this before every job (ACTIONS_RUNNER_HOOK_JOB_STARTED, see runner/README.md). A non-zero exit
# fails the job before any of its steps run. It lets through only jobs of this repository that were started by
# hand (workflow_dispatch) from .github/workflows/ngram-cpu.yml, so a fork pull request that adds or edits a
# workflow aimed at this runner's labels cannot run here, even if its run is approved by mistake.
# Jobs of any other repository (for example if the runner is shared) are left alone.
#
# Install: copy this file somewhere the runner user cannot write (e.g. /usr/local/lib/actions-runner-hooks/,
# owned by root, mode 0755), then add to the runner's .env file and restart the runner service:
#   ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/lib/actions-runner-hooks/prejob-hook.sh
set -euo pipefail

REPO="patcarter883/memory-organ"
WORKFLOW=".github/workflows/ngram-cpu.yml"

[[ "${GITHUB_REPOSITORY:-}" == "$REPO" ]] || exit 0

if [[ "${GITHUB_EVENT_NAME:-}" != "workflow_dispatch" ]]; then
  echo "prejob-hook: refusing ${GITHUB_EVENT_NAME:-?} job for $REPO; only workflow_dispatch may use this runner" >&2
  exit 1
fi
# GITHUB_WORKFLOW_REF looks like patcarter883/memory-organ/.github/workflows/ngram-cpu.yml@refs/heads/<branch>
if [[ "${GITHUB_WORKFLOW_REF:-}" != "$REPO/$WORKFLOW@"* ]]; then
  echo "prejob-hook: refusing workflow ${GITHUB_WORKFLOW_REF:-?}; only $WORKFLOW may use this runner" >&2
  exit 1
fi
exit 0
