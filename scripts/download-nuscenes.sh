#!/usr/bin/env bash
# Stage nuScenes for Depth Pro Table 1 (881 val CAM_FRONT frames).
#
# Modes:
#   mini              — v1.0-mini (~4 GB) smoke test
#   trainval          — v1.0-trainval meta + 10× keyframes (~46 GB, Motional public S3)
#   depth-pro-val     — alias for trainval (checks staged layout)
#
# Source (no account required):
#   s3://motional-nuscenes/public/v1.0/  (see registry.opendata.aws/motional-nuscenes)

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
DEST="${NUSCENES_ROOT:-$WORK/data/nuscenes}"
DL="$WORK/data/nuscenes_downloads"
MODE="${1:-trainval}"
S3_BASE="s3://motional-nuscenes/public/v1.0"
CF_BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"

mkdir -p "$DL" "$DEST"

_have_trainval() {
  [[ -d "$DEST/v1.0-trainval" ]] \
    && [[ -d "$DEST/samples/CAM_FRONT" ]] \
    && [[ -d "$DEST/samples/LIDAR_TOP" ]]
}

_download_file() {
  local name="$1"
  local out="$DL/$name"
  if [[ -f "$out" ]]; then
    echo "  skip (cached) $name"
    return 0
  fi
  echo "==> downloading $name ..."
  if command -v aws >/dev/null 2>&1; then
    aws s3 cp --no-sign-request "$S3_BASE/$name" "$out"
  else
    wget -c --timeout=120 --tries=5 "$CF_BASE/$name" -O "$out"
  fi
}

_extract_tgz() {
  local tgz="$1"
  echo "==> extracting $(basename "$tgz") ..."
  tar -xzf "$tgz" -C "$DEST"
}

_download_mini() {
  local name="v1.0-mini.tgz"
  _download_file "$name"
  _extract_tgz "$DL/$name"
  echo "==> mini ready under $DEST (version v1.0-mini)"
}

_download_trainval() {
  if _have_trainval; then
    local n
    n=$(find "$DEST/samples/CAM_FRONT" -name '*.jpg' 2>/dev/null | wc -l)
    echo "nuScenes trainval already staged at $DEST ($n CAM_FRONT keyframes)"
    return 0
  fi

  _download_file "v1.0-trainval_meta.tgz"
  _extract_tgz "$DL/v1.0-trainval_meta.tgz"

  local i
  for i in 01 02 03 04 05 06 07 08 09 10; do
    local kf="v1.0-trainval${i}_keyframes.tgz"
    _download_file "$kf"
    _extract_tgz "$DL/$kf"
  done

  if _have_trainval; then
    local n
    n=$(find "$DEST/samples/CAM_FRONT" -name '*.jpg' 2>/dev/null | wc -l)
    echo "==> trainval ready: $n CAM_FRONT jpgs under $DEST"
  else
    echo "ERROR: trainval extract incomplete (missing v1.0-trainval or samples/)" >&2
    exit 1
  fi
}

case "$MODE" in
  mini)
    _download_mini
    ;;
  trainval|depth-pro-val)
    _download_trainval
    ;;
  *)
    echo "usage: $0 [mini|trainval|depth-pro-val]" >&2
    exit 2
    ;;
esac
