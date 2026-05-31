#!/usr/bin/env bash
# Backup plumbline session artifacts to S3 (and remind: git push for code).
# Usage:
#   source scripts/pod-localssd-env.sh
#   ./scripts/backup-session.sh tier_a_20260530
#
# Syncs:
#   $PLUMBLINE_WORK/runs/*.json  → s3://plumbline-bench/runs/<tag>/results/
#   $PLUMBLINE_WORK/logs/*.log   → s3://plumbline-bench/runs/<tag>/logs/
#   MoGe-bundle trees (DDAD, Sintel, …) if present → datasets/<name>_moge/
#   Staged Depth Pro Table-1 datasets (incremental) when present
#
# Periodic loop: ./scripts/backup-periodic.sh [tag] [interval_minutes]

set -euo pipefail

TAG="${1:?usage: backup-session.sh <session-tag>}"
WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
BUCKET="${PLUMBLINE_S3_BUCKET:-s3://plumbline-bench}"

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS credentials missing or expired. Paste session token exports first." >&2
  exit 1
fi

echo "==> runs → ${BUCKET}/runs/${TAG}/results/"
if compgen -G "${WORK}/runs/"*.json >/dev/null 2>&1; then
  aws s3 sync "${WORK}/runs/" "${BUCKET}/runs/${TAG}/results/" \
    --exclude '*' --include '*.json'
else
  echo "    (no JSON in ${WORK}/runs/)"
fi

if compgen -G "${WORK}/logs/"*.log >/dev/null 2>&1; then
  echo "==> logs → ${BUCKET}/runs/${TAG}/logs/"
  aws s3 sync "${WORK}/logs/" "${BUCKET}/runs/${TAG}/logs/" \
    --exclude '*' --include '*.log'
fi

_sync_bundle() {
  local subdir="$1" dest="$2"
  if [[ -d "${WORK}/data/moge_eval/${subdir}" ]]; then
    echo "==> ${subdir} → ${BUCKET}/datasets/${dest}/"
    aws s3 sync "${WORK}/data/moge_eval/${subdir}" \
      "${BUCKET}/datasets/${dest}/${subdir}/"
  fi
}

_sync_bundle DDAD ddad_moge
_sync_bundle Sintel sintel_moge
_sync_bundle ETH3D eth3d_moge
_sync_bundle DIODE diode_moge
_sync_bundle KITTI kitti_moge

if [[ -d "${WORK}/data/eth3d_moge/ETH3D" ]]; then
  echo "==> eth3d_moge → ${BUCKET}/datasets/eth3d_moge/"
  aws s3 sync "${WORK}/data/eth3d_moge/" "${BUCKET}/datasets/eth3d_moge/"
fi
if [[ -d "${WORK}/data/diode_moge/DIODE" ]]; then
  echo "==> diode_moge → ${BUCKET}/datasets/diode_moge/"
  aws s3 sync "${WORK}/data/diode_moge/" "${BUCKET}/datasets/diode_moge/"
fi

# Depth Pro Table 1 staging (incremental; skip in-progress partial tgz)
if [[ -d "${WORK}/data/nuscenes/v1.0-trainval" ]]; then
  echo "==> nuscenes metadata → ${BUCKET}/datasets/nuscenes/v1.0-trainval/"
  aws s3 sync "${WORK}/data/nuscenes/v1.0-trainval/" \
    "${BUCKET}/datasets/nuscenes/v1.0-trainval/"
  if [[ -d "${WORK}/data/nuscenes/.plumbline_manifest" ]]; then
    echo "==> nuscenes depth cache → ${BUCKET}/datasets/nuscenes/.plumbline_manifest/"
    aws s3 sync "${WORK}/data/nuscenes/.plumbline_manifest/" \
      "${BUCKET}/datasets/nuscenes/.plumbline_manifest/"
  fi
  if compgen -G "${WORK}/data/nuscenes_downloads/"*.tgz >/dev/null 2>&1; then
    echo "==> nuscenes_downloads (complete tgz only) → ${BUCKET}/datasets/nuscenes_downloads/"
    aws s3 sync "${WORK}/data/nuscenes_downloads/" \
      "${BUCKET}/datasets/nuscenes_downloads/" --exclude '*.partial' --exclude '*.tmp'
  fi
fi

if [[ -d "${WORK}/data/sun_rgbd/rgb" ]]; then
  echo "==> sun_rgbd → ${BUCKET}/datasets/sun_rgbd/"
  aws s3 sync "${WORK}/data/sun_rgbd/" "${BUCKET}/datasets/sun_rgbd/"
fi

if [[ -d "${WORK}/data/middlebury/trainingF" ]]; then
  echo "==> middlebury → ${BUCKET}/datasets/middlebury/"
  aws s3 sync "${WORK}/data/middlebury/" "${BUCKET}/datasets/middlebury/"
fi

if [[ -d "${WORK}/data/booster/train/balanced" ]]; then
  echo "==> booster → ${BUCKET}/datasets/booster/"
  aws s3 sync "${WORK}/data/booster/" "${BUCKET}/datasets/booster/"
fi

if [[ -d "${WORK}/data/sintel/training" ]]; then
  echo "==> sintel → ${BUCKET}/datasets/sintel/"
  aws s3 sync "${WORK}/data/sintel/" "${BUCKET}/datasets/sintel/"
fi

echo "==> done. Code still needs: git add/commit && git push origin main"
