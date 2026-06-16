#!/usr/bin/env bash
# Stage KITTI for the dust3r-lineage depth eval (CUT3R/DUSt3R Table 1).
#
# Produces plumbline's `kitti` loader layout (depth_split=val). The loader
# rglobs `*_sync` dirs, takes the parent dir name as the date, and reads calib
# from `raw/<date>/calib_cam_to_cam.txt` — so raw must be DATE-LEVEL:
#   $KITTI_ROOT/raw/<date>/calib_cam_to_cam.txt
#   $KITTI_ROOT/raw/<date>/<drive>_sync/image_02/data/*.png                         (RGB)
#   $KITTI_ROOT/depth_annotated/val/<drive>_sync/proj_depth/groundtruth/image_02/*.png  (uint16 /256)
#
# Sources (public, no account):
#   - data_depth_annotated.zip (~14 GB) — annotated GT (only val/ is kept)
#   - 13 raw sync drives (~320 MB each) — RGB
#   - 5 per-date calib zips (~tiny) — intrinsics (REQUIRED by the loader)
#
# NOTE: KITTI's eu-central S3 is SLOW from some hosts (~1 MB/s on anima →
# ~2 h for the annotated zip). Run detached (nohup) and come back.
#
# Usage:
#   KITTI_ROOT=~/data/kitti_cut3r scripts/stage_kitti_cut3r.sh

set -euo pipefail

ROOT="${KITTI_ROOT:-$HOME/data/kitti_cut3r}"
S3="https://s3.eu-central-1.amazonaws.com/avg-kitti"
DRIVES=(
    2011_09_26_drive_0002 2011_09_26_drive_0005 2011_09_26_drive_0013
    2011_09_26_drive_0020 2011_09_26_drive_0023 2011_09_26_drive_0036
    2011_09_26_drive_0079 2011_09_26_drive_0095 2011_09_26_drive_0113
    2011_09_28_drive_0037 2011_09_29_drive_0026 2011_09_30_drive_0016
    2011_10_03_drive_0047
)
DATES=(2011_09_26 2011_09_28 2011_09_29 2011_09_30 2011_10_03)

mkdir -p "$ROOT/dl" "$ROOT/raw" "$ROOT/depth_annotated"

echo ">> annotated GT (~14 GB, slow) — keep only val/"
[ -s "$ROOT/dl/data_depth_annotated.zip" ] || curl -sL "$S3/data_depth_annotated.zip" -o "$ROOT/dl/data_depth_annotated.zip"
unzip -n -q "$ROOT/dl/data_depth_annotated.zip" "val/*" -d "$ROOT/depth_annotated"  # -> depth_annotated/val/<drive>_sync/...

echo ">> raw sync drives (RGB) — extract to date-level raw/<date>/<drive>_sync/"
for d in "${DRIVES[@]}"; do
    z="$ROOT/dl/${d}_sync.zip"
    [ -s "$z" ] || curl -sL "$S3/raw_data/${d}/${d}_sync.zip" -o "$z"
    unzip -n -q "$z" -d "$ROOT/raw"
done

echo ">> per-date calib (intrinsics) — REQUIRED"
for dt in "${DATES[@]}"; do
    z="$ROOT/dl/${dt}_calib.zip"
    [ -s "$z" ] || curl -sL "$S3/raw_data/${dt}/${dt}_calib.zip" -o "$z"
    unzip -n -q "$z" -d "$ROOT/raw"  # -> raw/<date>/calib_cam_to_cam.txt
done

echo ">> done. Set KITTI_ROOT=$ROOT and run: plumbline reproduce cut3r-kitti-lineage"
