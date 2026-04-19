"""DA3 view-count scaling on ETH3D courtyard.

Mirror of ``vggt_view_sweep_courtyard.py`` with DA3 Large-1.1. Reports
both absolute per-view errors and pairwise relative-pose AUC.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("ETH3D_ROOT", "/workspace/data/eth3d")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

from plumbline.datasets.eth3d import ETH3DDataset
from plumbline.metrics.pose import (
    auc,
    pairwise_pose_errors,
    rotation_error_degrees,
    translation_cosine_error,
)
from plumbline.models.depth_anything_3 import DepthAnything3Adapter

VIEW_COUNTS = [2, 4, 8]
AUC_TS = [5.0, 10.0, 30.0]


def main() -> None:
    ds = ETH3DDataset(scenes=["courtyard"], views_per_sample=8)
    samples = list(ds)
    print(f"courtyard sliding windows (8-view): {len(samples)}")

    adapter = DepthAnything3Adapter(
        device="cuda:0", checkpoint="depth-anything/DA3-LARGE-1.1"
    )

    results: dict[int, dict[str, list[float]]] = {
        n: {
            "abs_rot": [],
            "abs_trans": [],
            "abs_combined": [],
            "pw_rot": [],
            "pw_trans": [],
            "pw_combined": [],
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

            pred_E = pred.extrinsics[1:n]
            gt_non_origin = gt_E[1:]
            abs_rot = np.asarray(rotation_error_degrees(pred_E, gt_non_origin)).reshape(-1)
            abs_trans = np.asarray(
                translation_cosine_error(pred_E[..., :3, 3], gt_non_origin[..., :3, 3])
            ).reshape(-1)
            results[n]["abs_rot"].extend(abs_rot.tolist())
            results[n]["abs_trans"].extend(abs_trans.tolist())
            results[n]["abs_combined"].extend(np.maximum(abs_rot, abs_trans).tolist())

            if n >= 2:
                pw_rot, pw_trans = pairwise_pose_errors(pred.extrinsics[:n], gt_E)
                results[n]["pw_rot"].extend(pw_rot.tolist())
                results[n]["pw_trans"].extend(pw_trans.tolist())
                results[n]["pw_combined"].extend(
                    np.maximum(pw_rot, pw_trans).tolist()
                )
            results[n]["runtime_s"].append(dt)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"[{i+1}/{len(samples)}] processed"
                f"  avg runtime@8={np.mean(results[8]['runtime_s']):.1f}s"
            )

    def _summary(n: int) -> str:
        r = results[n]
        abs_aucs = auc(np.asarray(r["abs_combined"]), AUC_TS)
        pw_aucs = auc(np.asarray(r["pw_combined"]), AUC_TS)
        abs_rot_med = float(np.median(r["abs_rot"]))
        pw_rot_med = float(np.median(r["pw_rot"])) if r["pw_rot"] else float("nan")
        return (
            f"{n:>4}  "
            f"{abs_rot_med:>8.3f}  {abs_aucs[5.0]:>7.3f}  {abs_aucs[10.0]:>7.3f}  "
            f"{pw_rot_med:>8.3f}  {pw_aucs[5.0]:>7.3f}  {pw_aucs[10.0]:>7.3f}  "
            f"{float(np.mean(r['runtime_s'])):>6.2f}"
        )

    print("\n# DA3 view-count scaling on ETH3D courtyard")
    print(f"Windows: {len(samples)} (8-view sliding, stride 1, 38 images)")
    print("Device: cuda:0 (RTX 3090)\n")
    print(
        f"{'views':>4}  {'abs_rot°m':>8}  {'abs@5':>7}  {'abs@10':>7}  "
        f"{'pw_rot°m':>8}  {'pw@5':>7}  {'pw@10':>7}  {'run/s':>6}"
    )
    print("-" * 74)
    for n in VIEW_COUNTS:
        print(_summary(n))


if __name__ == "__main__":
    main()
