#!/usr/bin/env bash
# Periodic S3 backup while long GPU/download jobs run on the pod.
#
# Usage:
#   source scripts/pod-localssd-env.sh
#   ./scripts/backup-periodic.sh [session-tag] [interval_minutes]
#
# Default: tag tier_session_YYYYMMDD, interval 30 minutes.
# Logs: $PLUMBLINE_WORK/logs/backup_periodic.log
#
# Stop: kill the background PID recorded in $PLUMBLINE_WORK/logs/backup_periodic.pid

set -euo pipefail

WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"
TAG="${1:-tier_session_$(date -u +%Y%m%d)}"
INTERVAL_MIN="${2:-30}"
REPO="${PLUMBLINE_REPO:-/mnt/localssd/plumbline}"
LOG="$WORK/logs/backup_periodic.log"
PIDFILE="$WORK/logs/backup_periodic.pid"
SCRIPT="$REPO/scripts/backup-session.sh"

mkdir -p "$WORK/logs"

if [[ "${BACKUP_PERIODIC_FOREGROUND:-}" != "1" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "backup-periodic already running (pid $(cat "$PIDFILE"))"
    exit 0
  fi
  export BACKUP_PERIODIC_FOREGROUND=1
  nohup "$0" "$TAG" "$INTERVAL_MIN" >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started backup-periodic pid $(cat "$PIDFILE") tag=$TAG every ${INTERVAL_MIN}m → $LOG"
  exit 0
fi

echo "==> backup-periodic started $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$TAG interval=${INTERVAL_MIN}m"

while true; do
  echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---" >>"$LOG"
  if aws sts get-caller-identity >>"$LOG" 2>&1; then
    if "$SCRIPT" "$TAG" >>"$LOG" 2>&1; then
      echo "backup ok" >>"$LOG"
    else
      echo "backup-session failed (see log)" >>"$LOG"
    fi
  else
    echo "AWS credentials unavailable; skip until next interval" >>"$LOG"
  fi
  sleep "$((INTERVAL_MIN * 60))"
done
