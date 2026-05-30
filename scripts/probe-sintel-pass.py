#!/usr/bin/env python3
"""Compare DA-V2-L AbsRel on Sintel ``final`` vs ``clean`` RGB passes (same GT depth).

GT depth comes from ``training/depth/`` (pass-independent). Only RGB differs.

Usage::

    source scripts/pod-localssd-env.sh
    export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"
    uv run python scripts/probe-sintel-pass.py
    uv run python scripts/probe-sintel-pass.py --max-frames 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from plumbline._discover import register_builtin_adapters
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.models.depth_anything_v2 import DepthAnythingV2Adapter


def _score_frame(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray | None,
    *,
    depth_clip: tuple[float, float],
) -> float:
    from PIL import Image

    if pred.shape != gt.shape:
        img = Image.fromarray(pred.astype(np.float32), mode="F")
        pred = np.asarray(
            img.resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
    mask = (
        (np.ones(gt.shape, dtype=bool) if valid is None else valid)
        & np.isfinite(pred)
        & np.isfinite(gt)
        & (gt > 0)
        & (pred > 0)
    )
    if not mask.any():
        return float("nan")
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    aligned = align_depth(p, g, np.ones(p.shape, dtype=bool), mode="scale_shift")
    lo, hi = depth_clip
    aligned = np.clip(aligned, lo, hi)
    return abs_rel(aligned, g, np.ones_like(p, dtype=bool))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--variant", default="large")
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument("--data-root", default=os.environ.get("SINTEL_ROOT", ""))
    args = parser.parse_args()
    root = Path(args.data_root)
    if not (root / "training" / "final").is_dir():
        raise SystemExit(f"Sintel root not found: {root}")

    register_builtin_adapters()
    model = DepthAnythingV2Adapter(variant=args.variant, input_size=518, device="cuda:0")
    depth_clip = (0.001, args.max_depth)
    paper = 0.487

    for pass_name in ("final", "clean"):
        if not (root / "training" / pass_name).is_dir():
            print(f"skip pass {pass_name} (missing under training/)")
            continue
        ds = DATASET_REGISTRY["sintel"](
            root=root,
            split="training",
            pass_name=pass_name,
            views_per_sample=1,
            max_depth=args.max_depth,
        )
        records = ds._records
        if args.max_frames is not None:
            records = records[: args.max_frames]
        scores: list[float] = []
        for rec in records:
            sample = ds._load_sample(rec)
            pred = model.predict(sample.images).depth[0]
            gt = sample.depth_gt[0]
            valid = sample.depth_valid[0] if sample.depth_valid is not None else None
            scores.append(_score_frame(pred, gt, valid, depth_clip=depth_clip))
        mean = float(np.nanmean(scores))
        print(
            f"pass={pass_name} n={len(scores)} AbsRel={mean:.4f} "
            f"vs paper {paper:.3f} (delta {(mean - paper) / paper * 100:+.1f}%)"
        )


if __name__ == "__main__":
    main()
