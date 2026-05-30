#!/usr/bin/env python3
"""Compare native vs MoGe-warp DIODE AbsRel by domain (D29 smoke, no full reproduce).

Usage::

    source scripts/pod-localssd-env.sh
    export DAV2_ROOT=$PLUMBLINE_WORK/deps/depth-anything-v2
    uv run python scripts/probe-diode-d29-warp.py --variant large --max-samples 50
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
    alignment_mode: str,
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
    aligned = align_depth(p, g, np.ones(p.shape, dtype=bool), mode=alignment_mode)
    lo, hi = depth_clip
    aligned = np.clip(aligned, lo, hi)
    return abs_rel(aligned, g, np.ones_like(p, dtype=bool))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--domain", default="both", choices=["indoors", "outdoor", "both"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-root", default=os.environ.get("DIODE_ROOT", ""))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--alignment",
        default="scale_shift",
        choices=["scale_shift", "scale_shift_clamped"],
    )
    args = parser.parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"DIODE root not found: {root}")

    register_builtin_adapters()
    model = DepthAnythingV2Adapter(variant=args.variant, input_size=518, device=args.device)
    depth_clip = (0.001, 50.0)

    alignment_mode = args.alignment

    for warp in (False, True):
        label = "moge_warp" if warp else "native"
        ds = DATASET_REGISTRY["diode"](
            root=root,
            split="val",
            domain=args.domain,
            moge_fov_warp=warp,
        )
        records = ds._records
        if args.max_samples is not None:
            records = records[: args.max_samples]
        by_domain: dict[str, list[float]] = {}
        for rec in records:
            sample = ds._load_sample(rec)
            pred = model.predict(sample.images).depth[0]
            gt = sample.depth_gt[0]
            valid = sample.depth_valid[0] if sample.depth_valid is not None else None
            dom = sample.metadata.get("domain", "unknown")
            by_domain.setdefault(dom, []).append(
                _score_frame(
                    pred, gt, valid, depth_clip=depth_clip, alignment_mode=alignment_mode
                )
            )
        parts = []
        for dom in sorted(by_domain):
            vals = by_domain[dom]
            parts.append(f"{dom}={np.nanmean(vals):.4f}(n={len(vals)})")
        print(f"{label} align={alignment_mode}: " + " ".join(parts))


if __name__ == "__main__":
    main()
