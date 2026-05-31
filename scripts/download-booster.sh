#!/usr/bin/env bash
# Stage Booster training GT (228 balanced frames) under $PLUMBLINE_WORK/data/booster.
#
# Source: https://amsacta.unibo.it/id/eprint/6876/1/booster_gt.zip
# Layout after unzip: train/balanced/<scene>/{camera_00,disp_00.npy,calib_00-02.xml,...}
#
# Usage:
#   source scripts/pod-localssd-env.sh
#   ./scripts/download-booster.sh

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
DEST="${BOOSTER_ROOT:-$WORK/data/booster}"
URL="https://amsacta.unibo.it/id/eprint/6876/1/booster_gt.zip"
ZIP="$WORK/data/booster_downloads/booster_gt.zip"

mkdir -p "$WORK/data/booster_downloads" "$DEST"

if [[ -d "$DEST/train/balanced" ]] && compgen -G "$DEST/train/balanced"/*/camera_00/*.png >/dev/null; then
  n=$(find "$DEST/train/balanced" -path '*/camera_00/*.png' | wc -l)
  echo "Booster already staged at $DEST ($n PNG frames under train/balanced)."
  exit 0
fi

if [[ ! -f "$ZIP" ]]; then
  echo "==> downloading booster_gt.zip (large; may take a while)..."
  wget -c --timeout=120 --tries=5 "$URL" -O "$ZIP"
fi

echo "==> extracting to $DEST ..."
unzip -o "$ZIP" -d "$DEST"

n=$(find "$DEST/train/balanced" -path '*/camera_00/*.png' 2>/dev/null | wc -l || echo 0)
echo "==> done: $n frames under $DEST/train/balanced (expect 228)"
