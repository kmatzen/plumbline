#!/usr/bin/env python3
"""Smoke Depth Pro δ₁ on Sintel under appendix Table 16 depth ranges."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo on path when run as script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from plumbline.datasets.sintel import SintelDataset
from plumbline.models.depth_pro import DepthProAdapter
from plumbline.runner import _compute_metrics


def _run(*, pass_name: str, max_depth: float | None, depth_clip: tuple[float, float]) -> float:
    from itertools import islice

    ds = SintelDataset(pass_name=pass_name, max_depth=max_depth, views_per_sample=1)
    model = DepthProAdapter(device="cuda:0", dtype="float16")
    deltas: list[float] = []
    for sample in islice(ds, args.limit):
        pred = model.predict(sample.images)
        m = _compute_metrics(
            prediction=pred,
            sample=sample,
            tasks=["mono_depth"],
            scale_alignment="none",
            pose_auc_thresholds=(5.0,),
            pose_acc_thresholds=(5.0,),
            pose_auc_mode="rpe",
            pose_translation_antipodal=True,
            pose_trajectory_metrics=False,
            delta_thresholds=(1.25,),
            f_score_threshold=0.01,
            depth_clip=depth_clip,
        )
        deltas.append(float(m["delta_1"]))
    return sum(deltas) / len(deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    configs = [
        ("final", 70.0, (0.001, 70.0), "legacy sintel_dav2"),
        ("final", 80.0, (0.01, 80.0), "appendix Table 16"),
        ("final", 80.0, (0.001, 80.0), "80 cap, 0.001 lo"),
        ("final", None, (0.01, 80.0), "no loader mask, clip only"),
        ("clean", 80.0, (0.01, 80.0), "clean pass"),
    ]
    print(f"Depth Pro Sintel δ₁ smoke (n={args.limit} frames)\n")
    for pass_name, max_d, clip, label in configs:
        d1 = _run(pass_name=pass_name, max_depth=max_d, depth_clip=clip)
        print(f"  {label:28s} pass={pass_name:5s} max_depth={max_d!s:4} clip={clip}  δ₁={d1:.4f}")
