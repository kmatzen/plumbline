"""VGGT view-count scaling on ETH3D courtyard.

Sliding 8-view windows from the 38-image sequence; for each window, run
VGGT at max_views in {2, 4, 8} and aggregate pose metrics.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("ETH3D_ROOT", "/workspace/data/eth3d")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

from plumbline.cache import PredictionCache
from plumbline.datasets.eth3d import ETH3DDataset
from plumbline.metrics.pose import (
    auc,
    rotation_error_degrees,
    translation_cosine_error,
)
from plumbline.models.vggt import VGGTAdapter

VIEW_COUNTS = [2, 4, 8]


def main() -> None:
    ds = ETH3DDataset(scenes=["courtyard"], views_per_sample=8)
    samples = list(ds)
    print(f"courtyard sliding windows (8-view): {len(samples)}")

    adapter = VGGTAdapter(device="cuda:0", dtype="bfloat16")
    cache = PredictionCache()

    results: dict[int, dict[str, list[float]]] = {
        n: {
            "rotation_err": [],
            "translation_err": [],
            "combined_err": [],
            "runtime_s": [],
        }
        for n in VIEW_COUNTS
    }

    for i, sample in enumerate(samples):
        for n in VIEW_COUNTS:
            imgs = sample.images[:n]
            gt_E = sample.extrinsics_gt[:n]
            t0 = time.perf_counter()
            pred = adapter.predict(imgs)
            dt = time.perf_counter() - t0

            # Skip the origin view (identity by convention).
            pred_E = pred.extrinsics[1:n]
            gt_non_origin = gt_E[1:]
            rot = np.asarray(rotation_error_degrees(pred_E, gt_non_origin))
            trans = np.asarray(
                translation_cosine_error(pred_E[..., :3, 3], gt_non_origin[..., :3, 3])
            )
            results[n]["rotation_err"].extend(rot.reshape(-1).tolist())
            results[n]["translation_err"].extend(trans.reshape(-1).tolist())
            results[n]["combined_err"].extend(
                np.maximum(rot.reshape(-1), trans.reshape(-1)).tolist()
            )
            results[n]["runtime_s"].append(dt)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"[{i+1}/{len(samples)}] processed"
                f"  avg runtime@8={np.mean(results[8]['runtime_s']):.1f}s"
            )

    # Report.
    print("\n# VGGT view-count scaling on ETH3D courtyard")
    print(f"Windows: {len(samples)} (8-view sliding, stride 1, 38 images)")
    print(f"Device: cuda:0 (RTX 3090), dtype=bfloat16\n")
    header = (
        f"{'views':>6}  {'rot°(mean)':>10}  {'rot°(med)':>10}  "
        f"{'trans°(mean)':>12}  {'AUC@5':>7}  {'AUC@10':>7}  {'AUC@30':>7}  {'run/s':>6}"
    )
    print(header)
    print("-" * len(header))
    for n in VIEW_COUNTS:
        rot = np.asarray(results[n]["rotation_err"])
        tra = np.asarray(results[n]["translation_err"])
        comb = np.asarray(results[n]["combined_err"])
        aucs = auc(comb, [5.0, 10.0, 30.0])
        print(
            f"{n:>6}  {rot.mean():>10.3f}  {np.median(rot):>10.3f}  "
            f"{tra.mean():>12.3f}  "
            f"{aucs[5.0]:>7.3f}  {aucs[10.0]:>7.3f}  {aucs[30.0]:>7.3f}  "
            f"{float(np.mean(results[n]['runtime_s'])):>6.2f}"
        )


if __name__ == "__main__":
    main()
