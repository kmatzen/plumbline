#!/usr/bin/env bash
# MoGe upstream eval_baseline.py for DA-V2 ViT-L on DIODE / ETH3D / Sintel bundles.
set -euo pipefail
source "$(dirname "$0")/pod-localssd-env.sh"

MOGE="$PLUMBLINE_WORK/deps/moge"
PY="${PLUMBLINE_ROOT:-/mnt/localssd/plumbline}/.venv/bin/python"
RUNS="$PLUMBLINE_WORK/runs"
mkdir -p "$MOGE/data/eval" "$DAV2_ROOT/checkpoints" "$RUNS"

CKPT="$DAV2_ROOT/checkpoints/depth_anything_v2_vitl.pth"
if [[ ! -f "$CKPT" ]]; then
  SNAP="$HF_HUB_CACHE/models--depth-anything--Depth-Anything-V2-Large/snapshots"
  SNAP_DIR="$(find "$SNAP" -maxdepth 1 -mindepth 1 -type d | head -1)"
  ln -sfn "$SNAP_DIR/depth_anything_v2_vitl.pth" "$CKPT"
fi

ln -sfn "$DIODE_MOGE_ROOT/DIODE" "$MOGE/data/eval/DIODE"
ln -sfn "$PLUMBLINE_WORK/data/eth3d_moge/ETH3D" "$MOGE/data/eval/ETH3D"
ln -sfn "$SINTEL_MOGE_ROOT/Sintel" "$MOGE/data/eval/Sintel"

# MoGe metrics (Sintel has_sharp_boundary) need pinned utils3d
if ! "$PY" -c "import utils3d.torch as t; assert hasattr(t,'sliding_window_2d')" 2>/dev/null; then
  echo "Installing MoGe-pinned utils3d…"
  uv pip install -q "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
fi

# read_meta → read_json patch (local MoGe clone)
if grep -q 'read_meta' "$MOGE/moge/test/dataloader.py" 2>/dev/null; then
  sed -i 's/read_meta/read_json/g' "$MOGE/moge/test/dataloader.py"
fi

SINTEL_CFG="${PLUMBLINE_ROOT:-/mnt/localssd/plumbline}/scripts/moge_eval_sintel_upstream.json"

cd "$MOGE"
for bench in diode eth3d sintel; do
  if [[ "$bench" == sintel ]]; then
    cfg="$SINTEL_CFG"
  else
    cfg="configs/eval/benchmarks/${bench}.json"
  fi
  out="$RUNS/moge_upstream_da_v2_${bench}_vitl.json"
  log="$RUNS/moge_upstream_da_v2_${bench}_vitl.log"
  echo "=== $bench → $out ==="
  "$PY" moge/scripts/eval_baseline.py \
    --baseline baselines/da_v2.py \
    --config "$cfg" \
    -o "$out" \
    --repo "$DAV2_ROOT" --backbone vitl 2>&1 | tee "$log"
done
echo "=== done ==="
