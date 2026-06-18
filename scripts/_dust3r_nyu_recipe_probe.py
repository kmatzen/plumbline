"""Resolve the DUSt3R-NYU off-paper gap (0.0777 lineage vs paper 0.065).

Runs DUSt3R F(I,I) ONCE over the 654-image NYUv2 Eigen test split (caching the
per-frame predictions to disk), then re-scores AbsRel under a matrix of eval
recipes using plumbline's OWN runner functions (align_depth / abs_rel /
_resize_depth_to_gt) so every number is byte-faithful to `plumbline run`.

Hypothesis: the dust3r-nyuv2 cell inherited `nyu_dust3r_lineage` (filled GT, no
Eigen crop) — a convention NAMED BY MonST3R/CUT3R, papers that came AFTER DUSt3R
(2023). DUSt3R's own 0.065 should fall under the classical Eigen-2014 recipe
(raw GT + Eigen crop), not the later lineage protocol.

    python scripts/_dust3r_nyu_recipe_probe.py --mat ~/data/nyuv2/nyu_depth_v2_labeled.mat
"""

import argparse
import os

import numpy as np

from plumbline.datasets.nyuv2 import EIGEN_CROP, _to_canonical, load_eigen_test_indices
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel, delta_threshold
from plumbline.models.dust3r import DUSt3RAdapter
from plumbline.runner_metrics import _resize_depth_to_gt


def eigen_mask(shape):
    top, bot, left, right = EIGEN_CROP
    m = np.zeros(shape, dtype=bool)
    m[top:bot, left:right] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="~/data/nyuv2/nyu_depth_v2_labeled.mat")
    ap.add_argument("--cache", default="~/dust3r_nyu_preds")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 654")
    args = ap.parse_args()
    import h5py

    mat = os.path.expanduser(args.mat)
    cache = os.path.expanduser(args.cache)
    os.makedirs(cache, exist_ok=True)
    idxs = load_eigen_test_indices()
    if args.limit:
        idxs = idxs[: args.limit]

    model = DUSt3RAdapter(long_edge=512, device="cuda:0")

    rgbs, raws, filleds, preds = {}, {}, {}, {}
    with h5py.File(mat, "r") as f:
        images_ds, raw_ds, fill_ds = f["images"], f["rawDepths"], f["depths"]
        for i, idx in enumerate(idxs):
            rgb, raw = _to_canonical(np.asarray(images_ds[idx]), np.asarray(raw_ds[idx]))
            _, fill = _to_canonical(np.asarray(images_ds[idx]), np.asarray(fill_ds[idx]))
            rgbs[idx], raws[idx], filleds[idx] = rgb, raw, fill
            pf = os.path.join(cache, f"{idx:05d}.npy")
            if os.path.exists(pf):
                preds[idx] = np.load(pf)
            else:
                pred = model.predict(rgb[None]).depth  # (1, H, W) processed res
                preds[idx] = pred
                np.save(pf, pred)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(idxs)} inferred", flush=True)

    # ---- recipe sweep (all from the same cached predictions) ----
    def score(gt_field, crop, mode, cap):
        ar, d1 = [], []
        for idx in idxs:
            gt = (raws if gt_field == "raw" else filleds)[idx]  # (480,640)
            pred = _resize_depth_to_gt(preds[idx], gt[None])[0]  # (480,640) f64
            valid = gt > 0
            if cap is not None:
                valid &= gt < cap
            if crop == "eigen":
                valid &= eigen_mask(gt.shape)
            if valid.sum() == 0:
                continue
            aligned = align_depth(pred, gt.astype(np.float64), valid, mode=mode)
            ar.append(abs_rel(aligned, gt, valid))
            d1.append(delta_threshold(aligned, gt, valid, threshold=1.25))
        return float(np.mean(ar)), float(np.mean(d1))

    print(f"\nn={len(idxs)}  (paper DUSt3R NYU: AbsRel 0.0650, d1 0.9402)\n")
    print(f"{'GT':7} {'crop':6} {'estimator':16} {'cap':5} {'AbsRel':>8} {'d1':>7}")
    print("-" * 56)
    for gt_field in ("filled", "raw"):
        for crop in ("none", "eigen"):
            for mode in ("median", "median_lineage"):
                for cap in (None, 10.0):
                    a, d = score(gt_field, crop, mode, cap)
                    flag = (
                        "  <-- lineage cell"
                        if (
                            gt_field == "filled"
                            and crop == "none"
                            and mode == "median"
                            and cap is None
                        )
                        else ""
                    )
                    caps = "none" if cap is None else f"{cap:g}"
                    print(f"{gt_field:7} {crop:6} {mode:16} {caps:5} {a:8.4f} {d:7.4f}{flag}")


if __name__ == "__main__":
    main()
