#!/usr/bin/env bash
# Stage ETH3D high-res train scenes under $ETH3D_ROOT (default: plumbline-work).
# Usage: source scripts/pod-localssd-env.sh && ./scripts/stage-eth3d-train-scenes.sh [scene ...]
set -euo pipefail

ROOT="${ETH3D_ROOT:-${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}/data/eth3d}"
mkdir -p "$ROOT"
TMP="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}/data/eth3d_downloads"
mkdir -p "$TMP"

DEFAULT_SCENES=(
  electro kicker meadow office pipes playground relief relief_2 terrace terrains
)
SCENES=("${@:-${DEFAULT_SCENES[@]}}")

for scene in "${SCENES[@]}"; do
  if [[ -d "$ROOT/$scene/dslr_calibration_undistorted" ]]; then
    echo "skip $scene (already staged)"
    continue
  fi
  echo "==> $scene"
  for kind in dslr_undistorted scan_clean; do
    url="https://www.eth3d.net/data/${scene}_${kind}.7z"
    arc="$TMP/${scene}_${kind}.7z"
    if [[ ! -f "$arc" ]]; then
      curl -L --fail -o "$arc" "$url"
    fi
    7z x -y -o"$ROOT" "$arc"
  done
  echo "done $scene"
done

echo "ETH3D scenes under $ROOT:"
ls -1 "$ROOT"
