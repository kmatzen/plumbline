#!/usr/bin/env bash
# Wait for nuScenes trainval staging, then run depth-pro-nuscenes.
set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
ROOT="${NUSCENES_ROOT:-$WORK/data/nuscenes}"
LOG="$WORK/logs/depth_pro_nuscenes.log"
OUT="$WORK/runs/depth_pro_nuscenes_20260531.json"
REPO="/mnt/localssd/plumbline"

mkdir -p "$WORK/logs" "$WORK/runs"
cd "$REPO"
source scripts/pod-localssd-env.sh
export NUSCENES_ROOT="$ROOT"

echo "==> waiting for trainval at $ROOT ..."
until [[ -d "$ROOT/v1.0-trainval" && -d "$ROOT/samples/CAM_FRONT" ]]; do
  sleep 120
done

echo "==> trainval present; starting reproduce $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run plumbline install depth-pro --yes 2>&1 | tail -3
exec uv run plumbline reproduce depth-pro-nuscenes -o "$OUT" 2>&1 | tee "$LOG"
