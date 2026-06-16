#!/usr/bin/env bash
# Stage KITTI for the dust3r-lineage depth eval (CUT3R/DUSt3R Table 1/2).
#
# Produces plumbline's `kitti` loader layout (depth_split=val):
#   $KITTI_ROOT/raw/<drive>_sync/image_02/data/*.png                       (RGB)
#   $KITTI_ROOT/annotated/val/<drive>_sync/proj_depth/groundtruth/image_02/*.png  (uint16 GT /256)
#
# Sources (public, no account):
#   - data_depth_annotated.zip (~14 GB) — KITTI annotated depth maps
#   - 13 raw sync drives (~320 MB each) — the val drives prepare_kitti.py uses
#
# NOTE: KITTI's eu-central S3 bucket is SLOW from some hosts (~1 MB/s observed
# on anima → ~3 h for the annotated zip). Run detached (nohup) and come back.
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

mkdir -p "$ROOT/dl" "$ROOT/raw" "$ROOT/annotated"

echo ">> annotated GT (~14 GB, slow)"
[ -s "$ROOT/dl/data_depth_annotated.zip" ] || curl -sL "$S3/data_depth_annotated.zip" -o "$ROOT/dl/data_depth_annotated.zip"
unzip -n -q "$ROOT/dl/data_depth_annotated.zip" -d "$ROOT/annotated"  # -> annotated/{train,val}/<drive>_sync/...

echo ">> raw sync drives (RGB)"
for d in "${DRIVES[@]}"; do
    z="$ROOT/dl/${d}_sync.zip"
    [ -s "$z" ] || curl -sL "$S3/raw_data/${d}/${d}_sync.zip" -o "$z"
    unzip -n -q "$z" -d "$ROOT/raw_extract"
done
# KITTI raw zips extract to <date>/<drive>_sync/...; flatten the drive dirs into raw/.
find "$ROOT/raw_extract" -maxdepth 2 -type d -name "*_sync" -exec ln -sfn {} "$ROOT/raw/" \;

echo ">> done. Set KITTI_ROOT=$ROOT and run: plumbline reproduce cut3r-kitti-lineage"
