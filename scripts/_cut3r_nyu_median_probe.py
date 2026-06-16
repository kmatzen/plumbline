"""Pin the CUT3R-NYU off-paper gap to the median scale estimator (2026-06-15).

Over CUT3R's own saved per-frame predictions on the prepared NYU val set,
computes per-frame AbsRel under the two "median scaling" estimators:

    ratio-of-medians  s = median(gt) / median(pred)   [dust3r-lineage code]  -> 0.0858  (paper 0.086)
    median-of-ratios  s = median(gt / pred)            [plumbline `median`]   -> 0.0777

Same predictions, same GT, same mask, same metric — only the scalar estimator
differs, and that is the entire 0.0858-vs-0.0777 gap. Inference is byte-identical
(scripts/_cut3r_nyu_input_diff.py), so this isolates the off-paper cause to the
estimator, fixed by the `median_lineage` alignment mode.

Run on the box where CUT3R's eval saved its preds:
    python scripts/_cut3r_nyu_median_probe.py \
        --pred ~/deps/cut3r/eval_results/monodepth/nyu_ours \
        --gt   ~/data/nyu_cut3r_prepared/nyu_depths
"""

import argparse
import glob
import os

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="~/deps/cut3r/eval_results/monodepth/nyu_ours")
    ap.add_argument("--gt", default="~/data/nyu_cut3r_prepared/nyu_depths")
    args = ap.parse_args()
    pred_dir = os.path.expanduser(args.pred)
    gt_dir = os.path.expanduser(args.gt)

    preds = sorted(glob.glob(os.path.join(pred_dir, "*depth.npy")))
    assert preds, f"no *depth.npy under {pred_dir}"
    rom, mor = [], []
    for pf in preds:
        base = os.path.basename(pf).replace("depth.npy", "")
        gt = np.load(os.path.join(gt_dir, f"{base}.npy")).astype(np.float64)
        pred = np.load(pf).astype(np.float64).squeeze()
        h, w = gt.shape
        p = cv2.resize(pred.astype(np.float32), (w, h)).astype(np.float64)
        m = gt > 0
        g_m, p_m = gt[m], p[m]
        s_rom = np.median(g_m) / np.median(p_m)
        s_mor = np.median(g_m / np.maximum(p_m, 1e-8))
        rom.append(float(np.mean(np.abs(s_rom * p_m - g_m) / g_m)))
        mor.append(float(np.mean(np.abs(s_mor * p_m - g_m) / g_m)))
    print(f"n={len(preds)}")
    print(f"ratio-of-medians  median(gt)/median(pred)  [lineage]   = {np.mean(rom):.5f}  (paper 0.086)")
    print(f"median-of-ratios  median(gt/pred)          [plumbline] = {np.mean(mor):.5f}")


if __name__ == "__main__":
    main()
