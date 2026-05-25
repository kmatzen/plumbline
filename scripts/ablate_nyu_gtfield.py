"""D24 ablation — why CUT3R NYU depth reads off-paper-better.

Re-scores the SAME cached CUT3R predictions (no re-inference) against NYU GT
under both GT fields (raw Kinect vs DUSt3R-lineage filled/dense) and four
protocol variants (Eigen crop × post-align clip), reproducing the 2×4 table in
docs/DISCREPANCIES.md D24. Also runs a CUBIC-resize probe (filled, no-crop,
no-clip) to isolate the pred-resize contribution (plumbline PIL-bilinear vs
CUT3R cv2.INTER_CUBIC) to the ~0.0777→0.086 residual.

Run on a box with the CUT3R prediction cache + $NYUV2_ROOT:
    NYUV2_ROOT=/path/to/nyuv2 python scripts/ablate_nyu_gtfield.py
"""

import glob
import os
import re

import numpy as np

from plumbline.datasets.nyuv2 import NYUv2Dataset, eigen_crop_mask
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.runner import _resize_depth_to_gt

cache_files = glob.glob(
    os.path.expanduser("~/.cache/plumbline/predictions/cut3r/*/nyuv2/*.npz")
)
cache = {}
for f in cache_files:
    m = re.search(r"(nyuv2_\d+)", f)
    if m:
        cache[m.group(1)] = f

try:
    import cv2

    def cubic_resize(pred, gt):
        if pred.shape[-2:] == gt.shape[-2:]:
            return pred.astype(np.float64)
        th, tw = gt.shape[-2:]
        return cv2.resize(
            pred.astype(np.float32), (tw, th), interpolation=cv2.INTER_CUBIC
        ).astype(np.float64)

    HAVE_CV2 = True
except Exception:  # pragma: no cover - cubic probe is optional
    HAVE_CV2 = False

# (eigen_crop, post-align clip[1e-3,10])
variants = {
    "A crop+clip   [plumbline nyu_eigen_2014]": (True, True),
    "B crop only                            ": (True, False),
    "C clip only                            ": (False, True),
    "D no-crop no-clip [CUT3R eval_metrics] ": (False, False),
}

for field in ("raw", "filled"):
    ds = NYUv2Dataset(split="test", apply_eigen_crop=False, depth_field=field)
    per = {k: [] for k in variants}
    pix = {k: [] for k in variants}
    per_cubic, pix_cubic = [], []
    n = 0
    for s in ds:
        sid = s.sample_id
        if sid not in cache:
            continue
        gt = s.depth_gt[0].astype(np.float64)
        pred = np.load(cache[sid])["depth"][0].astype(np.float64)
        pred_bil = _resize_depth_to_gt(pred[None], gt[None])[0]
        base = gt > 0
        cm = eigen_crop_mask(gt.shape).astype(bool)
        for name, (crop, clip) in variants.items():
            valid = (base & cm) if crop else base
            if valid.sum() == 0:
                continue
            aligned = align_depth(pred_bil, gt, valid, mode="median")
            if clip:
                aligned = np.clip(aligned, 1e-3, 10.0)
            per[name].append(abs_rel(aligned, gt, valid))
            pix[name].append(int(valid.sum()))
        if HAVE_CV2:
            pred_cub = cubic_resize(pred, gt)
            aligned = align_depth(pred_cub, gt, base, mode="median")
            per_cubic.append(abs_rel(aligned, gt, base))
            pix_cubic.append(int(base.sum()))
        n += 1
    print(f"\n### depth_field={field}  samples={n}  (cache hits {len(cache)})")
    for k in variants:
        a = np.array(per[k])
        w = np.array(pix[k], float)
        print(f"  {k}  mean={a.mean():.4f}  pixwt={np.average(a, weights=w):.4f}")
    if HAVE_CV2:
        a = np.array(per_cubic)
        w = np.array(pix_cubic, float)
        print(
            f"  D no-crop no-clip CUBIC-resize           "
            f"mean={a.mean():.4f}  pixwt={np.average(a, weights=w):.4f}"
        )

print("\npaper NYU (CUT3R Table 1, 'Ours') = 0.086")
