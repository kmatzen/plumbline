#!/usr/bin/env bash
# Stage the NYU-v2 dust3r-lineage *prepared* eval set (CUT3R / MonST3R Table 1).
#
# Source: HuggingFace sayakpaul/nyu_depth_v2 val split (public, no auth) — the
# same set MonST3R's data/download_nyuv2.sh + datasets_preprocess/prepare_nyuv2.py
# produce. Decodes 654 .h5 -> nyu_images/<id>.png + nyu_depths/<id>.npy, the
# layout the `nyu-cut3r-eval` plumbline loader (and CUT3R's eval/monodepth) read.
#
# Usage:
#   NYU_CUT3R_EVAL_ROOT=~/data/nyu_cut3r_prepared scripts/stage_nyu_cut3r_eval.sh
#
# ~1 GB download; needs python with h5py + numpy + Pillow (the plumbline venv has them).

set -euo pipefail

ROOT="${NYU_CUT3R_EVAL_ROOT:-$HOME/data/nyu_cut3r_prepared}"
PY="${PYTHON:-python}"
BASE="https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data"

mkdir -p "$ROOT/raw"
echo ">> downloading NYU-v2 val tars to $ROOT/raw"
for f in val-000000 val-000001; do
    if [ ! -s "$ROOT/raw/$f.tar" ]; then
        curl -sL "$BASE/$f.tar" -o "$ROOT/raw/$f.tar"
    fi
done

echo ">> extracting .h5"
( cd "$ROOT/raw" && for t in val-000000.tar val-000001.tar; do tar -xf "$t"; done )

echo ">> decoding h5 -> nyu_images/*.png + nyu_depths/*.npy"
NYU_CUT3R_EVAL_ROOT="$ROOT" "$PY" - <<'PYEOF'
import os, glob, h5py, numpy as np
from PIL import Image

root = os.environ["NYU_CUT3R_EVAL_ROOT"]
src = os.path.join(root, "raw/val/official")
img_dir = os.path.join(root, "nyu_images")
dep_dir = os.path.join(root, "nyu_depths")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(dep_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src, "*.h5")))
assert files, f"no .h5 under {src}"
for fp in files:
    with h5py.File(fp, "r") as h5:
        depth = h5["depth"][:]
        rgb = np.transpose(h5["rgb"][:], (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
    base = os.path.splitext(os.path.basename(fp))[0]
    Image.fromarray(rgb).save(os.path.join(img_dir, f"{base}.png"))
    np.save(os.path.join(dep_dir, f"{base}.npy"), depth)
print(f"staged {len(files)} samples to {root}")
PYEOF

echo ">> done. Set NYU_CUT3R_EVAL_ROOT=$ROOT and run: plumbline reproduce cut3r-nyuv2-prepared"
