#!/usr/bin/env bash
# M2-0 on the owner's two gfx1201 cards, one command (see experiments/m2_0/README.md).
#   experiments/m2_0/run_owner.sh [--quick] [--preflight-only] [--ckpt DIR] [--out DIR] [...]
# Refuses if the serve is up or the cards are leased; takes `gpu-lease -n 2` itself; never downloads a whole
# checkpoint unless --allow-full-download is given. PYTHON=<python with torch-ROCm> (default: python3), or
# NGM_IMAGE=<docker image> to run the GPU steps in a container; NGM_MINISGL_FORK=<fork clone with gdn_hip built>.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${PYTHON:-python3}"
exec "$py" "$here/run_owner.py" --python "$py" "$@"
