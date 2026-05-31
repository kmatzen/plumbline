#!/usr/bin/env bash
# Stage SUN RGB-D test split (5050 frames) for Depth Pro Table 1.
#
# Ahanda mirror: test RGB (img-000001.jpg …) + depth PNGs (1.png … 5050.png,
# uint16 / 10000 → meters). Pairing: img-{i:06d}.jpg ↔ depth/{i}.png
# (see ankurhanda/sunrgbd-meta-data test depth links).
#
# Source: http://rgbd.cs.princeton.edu/ (test split).

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
DEST="${SUN_RGBD_ROOT:-$WORK/data/sun_rgbd}"
DL="$WORK/data/sun_rgbd_downloads"
RGB_URL="http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz"
DEPTH_URL="http://www.doc.ic.ac.uk/~ahanda/sunrgb_test_depth.tgz"

mkdir -p "$DL" "$DEST/rgb" "$DEST/depth"

if [[ $(find "$DEST/rgb" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l) -ge 5000 ]]; then
  echo "SUN RGB-D already staged at $DEST ($(find "$DEST/rgb" -maxdepth 1 -name '*.jpg' | wc -l) rgb frames)"
  exit 0
fi

_download() {
  local url="$1" out="$2"
  if [[ ! -f "$out" ]]; then
    echo "==> downloading $(basename "$out") ..."
    wget -c --timeout=120 --tries=5 "$url" -O "$out"
  fi
}

_download "$RGB_URL" "$DL/SUNRGBD-test_images.tgz"
_download "$DEPTH_URL" "$DL/sunrgb_test_depth.tgz"

echo "==> extracting rgb (flat archive — no --strip-components) ..."
rm -f "$DEST/rgb"/*
tar -xzf "$DL/SUNRGBD-test_images.tgz" -C "$DEST/rgb"

echo "==> extracting depth (sunrgbd_test_depth/*.png) ..."
rm -f "$DEST/depth"/*
tar -xzf "$DL/sunrgb_test_depth.tgz" -C "$DEST/depth" --strip-components=1

n_rgb=$(find "$DEST/rgb" -maxdepth 1 -name '*.jpg' | wc -l)
n_depth=$(find "$DEST/depth" -maxdepth 1 -name '*.png' | wc -l)
echo "==> done: $n_rgb rgb, $n_depth depth under $DEST (expect 5050 each)"
if [[ "$n_rgb" -lt 5000 || "$n_depth" -lt 5000 ]]; then
  echo "ERROR: incomplete staging" >&2
  exit 1
fi
