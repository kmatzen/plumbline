#!/usr/bin/env bash
# Stage ETH3D official rendered depth + distorted DSLR JPGs for the Depth Pro
# Table 1 (Table 16) mono-depth cell. Unlike scan_clean / dslr_scan_eval (laser
# PLY for chamfer), this needs:
#   <root>/<scene>/ground_truth_depth/dslr_images/*.JPG   (float32 depth dumps)
#   <root>/<scene>/images/dslr_images/*.JPG               (distorted RGB)
# Both ship as <scene>_dslr_depth.7z / <scene>_dslr_jpg.7z from eth3d.net.
#
# No system 7z on the run host — extract with py7zr in the project venv.
#
# Usage: ETH3D_ROOT=~/data/eth3d ./scripts/stage-eth3d-depth-pro.sh [scene ...]
set -euo pipefail

ROOT="${ETH3D_ROOT:?set ETH3D_ROOT}"
TMP="${ROOT}_downloads"
mkdir -p "$ROOT" "$TMP"

DEFAULT_SCENES=(
  electro kicker meadow office pipes playground relief relief_2 terrace terrains
)
SCENES=("$@")
[[ ${#SCENES[@]} -eq 0 ]] && SCENES=("${DEFAULT_SCENES[@]}")

extract() {  # $1 = archive, $2 = dest root
  uv run python - "$1" "$2" <<'PY'
import sys, py7zr
arc, dest = sys.argv[1], sys.argv[2]
with py7zr.SevenZipFile(arc, "r") as z:
    z.extractall(path=dest)
print("extracted", arc)
PY
}

for scene in "${SCENES[@]}"; do
  echo "==> $scene"
  for kind in dslr_depth dslr_jpg; do
    # already extracted?
    if [[ "$kind" == dslr_depth ]] && compgen -G "$ROOT/$scene/ground_truth_depth/dslr_images/"'*.JPG' >/dev/null; then
      echo "  skip $kind (present)"; continue
    fi
    if [[ "$kind" == dslr_jpg ]] && compgen -G "$ROOT/$scene/images/dslr_images/"'*.JPG' >/dev/null; then
      echo "  skip $kind (present)"; continue
    fi
    url="https://www.eth3d.net/data/${scene}_${kind}.7z"
    arc="$TMP/${scene}_${kind}.7z"
    if ! curl -sfI -o /dev/null "$url"; then
      echo "  skip $kind (not published)"; continue
    fi
    [[ -f "$arc" ]] || curl -L --fail -o "$arc" "$url"
    extract "$arc" "$ROOT"
    rm -f "$arc"
  done
  echo "done $scene"
done
echo "ALL DONE"
