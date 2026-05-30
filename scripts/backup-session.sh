#!/usr/bin/env bash
# Backup plumbline session artifacts to S3 (and remind: git push for code).
# Usage:
#   source scripts/pod-localssd-env.sh
#   ./scripts/backup-session.sh tier_a_20260530
#
# Syncs:
#   $PLUMBLINE_WORK/runs/*.json  → s3://plumbline-bench/runs/<tag>/results/
#   MoGe-bundle trees (DDAD, Sintel, …) if present → datasets/<name>_moge/

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

echo "==> done. Code still needs: git add/commit && git push origin main"
