#!/usr/bin/env bash
# Stage everything GeoWizard needs beyond the `models` extra.
#
# Usage:   scripts/install_geowizard_extras.sh
#          (run from the plumbline repo root inside the GPU box's shell)
#
# What this does:
#   1. Clone upstream GeoWizard to $HOME/deps/geowizard (shallow, once).
#   2. uv sync --extra geowizard  — adds xformers==0.0.29.post3.
#   3. Re-lay-down nvidia-cudnn-cu12==9.1.0.70 (force-reinstall). Skipping
#      this has been observed to surface CUDNN_STATUS_NOT_INITIALIZED on
#      the first GeoWizard conv; see the pyproject `[geowizard]` comment.
#   4. Print an `export GEOWIZARD_ROOT=...` line you should paste in your
#      current shell (this script runs in a subshell, so it can't exports
#      variables back up).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${GEOWIZARD_DEPS_DIR:-$HOME/deps}"
CLONE_DIR="$DEPS_DIR/geowizard"

echo "=== 1/3 clone upstream GeoWizard ==="
if [[ -d "$CLONE_DIR/.git" ]]; then
    echo "Already at $CLONE_DIR — skipping clone"
else
    mkdir -p "$DEPS_DIR"
    git clone --depth 1 https://github.com/fuxiao0719/GeoWizard "$CLONE_DIR"
fi
# Sanity check the pipeline module is reachable.
test -f "$CLONE_DIR/geowizard/models/geowizard_pipeline.py"

echo
echo "=== 2/3 uv sync --extra geowizard ==="
(cd "$REPO_ROOT" && uv sync --extra geowizard)

echo
echo "=== 3/3 re-lay-down cudnn ==="
# The pin matches what pyproject documents as torch==2.6.0+cu124 compatible.
(cd "$REPO_ROOT" && uv pip install --force-reinstall 'nvidia-cudnn-cu12==9.1.0.70')

echo
echo "=== done ==="
echo "Paste this in your shell so plumbline can find GeoWizard:"
echo
echo "    export GEOWIZARD_ROOT=$CLONE_DIR"
echo
