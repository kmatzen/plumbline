#!/usr/bin/env bash
# Stage MiddEval3 training (15 scenes) for Depth Pro Table 1 Middlebury eval.
#
# Uses full resolution (F) per Depth Pro appendix Table 16 (~1988×2952).
# Source: https://vision.middlebury.edu/stereo/submit3/

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
DEST="${MIDDLEBURY_ROOT:-$WORK/data/middlebury}"
DL="$WORK/data/middlebury_downloads"
BASE="https://vision.middlebury.edu/stereo/submit3/zip"

mkdir -p "$DL" "$DEST"

_stage_zip() {
  local zip="$1" subdir="$2"
  if [[ ! -f "$DL/$zip" ]]; then
    echo "==> downloading $zip ..."
    wget -c --timeout=120 --tries=5 "$BASE/$zip" -O "$DL/$zip"
  fi
  echo "==> extracting $zip -> $DEST/$subdir"
  unzip -o -q "$DL/$zip" -d "$DEST/$subdir"
}

if [[ -d "$DEST/trainingF/Adirondack" ]] && [[ -f "$DEST/trainingF/Adirondack/disp0GT.pfm" ]]; then
  n=$(find "$DEST/trainingF" -maxdepth 1 -mindepth 1 -type d | wc -l)
  echo "Middlebury already staged at $DEST/trainingF ($n scenes)"
  exit 0
fi

_stage_zip MiddEval3-data-F.zip data
_stage_zip MiddEval3-GT0-F.zip gt

# Merge GT into scene folders: trainingF/<scene>/{im0,calib} + {disp0GT,mask}
for scene in "$DEST/data/MiddEval3/trainingF"/*; do
  name=$(basename "$scene")
  mkdir -p "$DEST/trainingF/$name"
  cp -n "$scene/im0.png" "$scene/calib.txt" "$DEST/trainingF/$name/" 2>/dev/null || true
  gt="$DEST/gt/MiddEval3/trainingF/$name"
  if [[ -d "$gt" ]]; then
    cp -n "$gt/disp0GT.pfm" "$gt/mask0nocc.png" "$DEST/trainingF/$name/" 2>/dev/null || true
  fi
done

n=$(find "$DEST/trainingF" -name im0.png | wc -l)
echo "==> done: $n scenes under $DEST/trainingF (expect 15)"
