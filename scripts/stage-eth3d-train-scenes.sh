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
  echo "==> $scene"
  kinds=(dslr_undistorted scan_clean dslr_scan_eval)
  if [[ ! -d "$ROOT/$scene/dslr_calibration_undistorted" ]]; then
    # Fresh scene: need images + scan_clean (+ eval GT when published).
    :
  else
    # Scene already has calibration/images; only top up missing archives.
    kinds=(dslr_scan_eval)
    if [[ -d "$ROOT/$scene/dslr_scan_eval" ]] \
      && [[ -d "$ROOT/$scene/ground_truth_depth" ]] \
      && compgen -G "$ROOT/$scene/images/dslr_images/"'*.JPG' > /dev/null; then
      echo "  skip (eval GT + official depth + distorted JPG present)"
      continue
    fi
    if [[ ! -d "$ROOT/$scene/ground_truth_depth" ]]; then
      kinds+=(dslr_depth)
    fi
    if [[ ! -d "$ROOT/$scene/images/dslr_images" ]]; then
      kinds+=(dslr_jpg)
    fi
  fi
  for kind in "${kinds[@]}"; do
    if [[ "$kind" == dslr_scan_eval && -d "$ROOT/$scene/dslr_scan_eval" ]]; then
      continue
    fi
    if [[ "$kind" == dslr_jpg ]] && compgen -G "$ROOT/$scene/images/dslr_images/"'*.JPG' > /dev/null; then
      continue
    fi
    if [[ "$kind" != dslr_scan_eval && "$kind" != dslr_depth && "$kind" != dslr_jpg && -d "$ROOT/$scene/dslr_calibration_undistorted" ]]; then
      continue
    fi
    url="https://www.eth3d.net/data/${scene}_${kind}.7z"
    arc="$TMP/${scene}_${kind}.7z"
    if ! curl -sfI -o /dev/null "$url"; then
      echo "  skip $kind (not published for $scene)"
      continue
    fi
    if [[ ! -f "$arc" ]]; then
      curl -L --fail -o "$arc" "$url"
    fi
    7z x -y -o"$ROOT" "$arc"
  done
  echo "done $scene"
done

echo "ETH3D scenes under $ROOT:"
ls -1 "$ROOT"
