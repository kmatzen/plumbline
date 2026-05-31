#!/usr/bin/env bash
# Stage nuScenes for Depth Pro Table 1 (881 val CAM_FRONT frames).
#
# Modes:
#   mini          — v1.0-mini (~4 GB) for loader smoke tests
#   depth-pro-val — print trainval staging steps (full val is ~300 GB+)
#
# After staging:
#   export NUSCENES_ROOT=$PLUMBLINE_WORK/data/nuscenes
#   uv pip install nuscenes-devkit pyquaternion
#   uv run plumbline reproduce depth-pro-nuscenes

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
DEST="${NUSCENES_ROOT:-$WORK/data/nuscenes}"
MODE="${1:-depth-pro-val}"

mkdir -p "$DEST"

_download_mini() {
  local url="https://www.nuscenes.org/data/v1.0-mini.tgz"
  local tgz="$WORK/data/nuscenes_downloads/v1.0-mini.tgz"
  mkdir -p "$WORK/data/nuscenes_downloads"
  if [[ -d "$DEST/v1.0-mini" && -d "$DEST/samples" ]]; then
    echo "nuScenes mini already at $DEST"
    return 0
  fi
  if [[ ! -f "$tgz" ]]; then
    echo "==> downloading v1.0-mini.tgz (~4 GB) ..."
    wget -c --timeout=120 --tries=5 "$url" -O "$tgz"
  fi
  echo "==> extracting mini ..."
  tar -xzf "$tgz" -C "$DEST"
  echo "==> mini ready under $DEST (use version v1.0-mini, subset_size <= ~40)"
}

case "$MODE" in
  mini)
    _download_mini
    ;;
  depth-pro-val)
    if [[ -d "$DEST/v1.0-trainval" && -d "$DEST/samples/CAM_FRONT" ]]; then
      n=$(find "$DEST/samples/CAM_FRONT" -name '*.jpg' 2>/dev/null | wc -l)
      echo "nuScenes trainval appears staged at $DEST ($n CAM_FRONT jpgs under samples/)"
      exit 0
    fi
    cat <<EOF
nuScenes v1.0-trainval is required for Depth Pro Table 1 (881 val frames).

1. Create an account at https://www.nuscenes.org/download and accept the terms.
2. Download and extract under: $DEST
   - Metadata: v1.0-trainval_meta.tgz → $DEST/v1.0-trainval/
   - Keyframe blobs: at minimum all val-scene samples for CAM_FRONT + LIDAR_TOP
     (the full trainval keyframe set is ~300 GB; partial val-only is smaller).
3. Install devkit: uv pip install nuscenes-devkit pyquaternion
4. Smoke test without full trainval: ./scripts/download-nuscenes.sh mini

Optional S3 mirror (if present): aws s3 sync s3://plumbline-bench/datasets/nuscenes/ $DEST/

EOF
    exit 1
    ;;
  *)
    echo "usage: $0 [mini|depth-pro-val]" >&2
    exit 2
    ;;
esac
