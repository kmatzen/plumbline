#!/usr/bin/env python3
"""Compare ETH3D mono-depth metrics across GT / RGB alignment variants.

Tracks (when data are present):

1. **z-buffer** — ``eth3d_dav2`` undistorted RGB + rendered laser GT @518.
2. **official@518-misaligned** — same undistorted preds, official sparse depth
   (distorted pixel grid) downsampled with nearest — documents geometry mismatch.
3. **official-aligned** — distorted ``images/dslr_images/*.JPG`` + official depth
   at native resolution (ETH3D-documented pairing).
4. **official@518-aligned** (optional) — distorted RGB + depth both resized to the
   same @518 canvas as ``eth3d_dav2`` (bilinear RGB via model I/O, nearest depth).

Usage (pod)::

    source scripts/pod-localssd-env.sh
    export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"
    # depth: curl -L -O https://www.eth3d.net/data/courtyard_dslr_depth.7z
    # jpg:   curl -L -O https://www.eth3d.net/data/courtyard_dslr_jpg.7z
    # 7z x -y -o"$ETH3D_ROOT" courtyard_dslr_{depth,jpg}.7z
    uv run python scripts/probe-eth3d-official-depth.py --scene courtyard

Exits 0 and prints aggregate AbsRel per track.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from plumbline._discover import register_builtin_adapters
from plumbline.datasets._common import read_rgb_uint8
from plumbline.datasets.eth3d import (
    ETH3DDataset,
    load_eth3d_official_depth_map,
    official_depth_valid_mask,
)
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.models.depth_anything_v2 import DepthAnythingV2Adapter


def _resize_depth_nearest(depth: np.ndarray, *, height: int, width: int) -> np.ndarray:
    img = Image.fromarray(depth.astype(np.float32), mode="F")
    out = np.asarray(img.resize((width, height), resample=Image.Resampling.NEAREST), dtype=np.float32)
    return out


def _resize_pred_bilinear(pred: np.ndarray, *, height: int, width: int) -> np.ndarray:
    img = Image.fromarray(pred.astype(np.float32), mode="F")
    out = np.asarray(img.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float64)
    return out


def _render_size(h: int, w: int, *, max_dim: int) -> tuple[int, int]:
    scale = min(1.0, max_dim / max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _score(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    depth_clip: tuple[float, float],
) -> tuple[float, float]:
    """Return (abs_rel, valid_fraction)."""
    if pred.shape != gt.shape:
        pred = _resize_pred_bilinear(pred, height=gt.shape[0], width=gt.shape[1])
    mask = valid & np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    frac = float(mask.mean()) if mask.size else 0.0
    if not mask.any():
        return float("nan"), frac
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    aligned = align_depth(p, g, np.ones(p.shape, dtype=bool), mode="scale_shift")
    aligned = np.clip(aligned, depth_clip[0], depth_clip[1])
    return abs_rel(aligned, g, np.ones_like(p, dtype=bool)), frac


def _mean(xs: list[float]) -> float:
    good = [x for x in xs if np.isfinite(x)]
    return float(np.mean(good)) if good else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="courtyard")
    parser.add_argument("--variant", default="large")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-dim", type=int, default=518, help="Short-side cap for @518 track")
    parser.add_argument("--data-root", default=os.environ.get("ETH3D_ROOT", ""))
    args = parser.parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"ETH3D root not found: {root}")

    scene = args.scene
    depth_dir = root / scene / "ground_truth_depth" / "dslr_images"
    if not depth_dir.is_dir():
        raise SystemExit(
            f"Missing {depth_dir}. Fetch {scene}_dslr_depth.7z from eth3d.net."
        )
    distorted_dir = root / scene / "images" / "dslr_images"
    have_distorted = distorted_dir.is_dir()

    register_builtin_adapters()
    ds = DATASET_REGISTRY["eth3d"](
        root=root,
        split="train",
        scenes=[scene],
        views_per_sample=1,
        with_per_view_gt=True,
        pv_render_max_dim=args.max_dim,
        pv_splat_radius=1,
        resize_images_to_pv_render=True,
    )
    model = DepthAnythingV2Adapter(variant=args.variant, input_size=args.max_dim, device="cuda:0")
    depth_clip = (0.001, 80.0)

    tracks: dict[str, list[float]] = {
        "zbuf": [],
        "off_misalign": [],
        "off_aligned": [],
        "off_at518": [],
    }
    fracs: dict[str, list[float]] = {k: [] for k in tracks}

    records = ds._records if args.max_frames is None else ds._records[: args.max_frames]
    for rec in records:
        sample = ds._load_sample(rec)
        stem = Path(rec["image_records"][0]["name"]).name
        pred_u = model.predict(sample.images).depth[0]

        gt_z = sample.depth_gt[0]
        valid_z = (
            sample.depth_valid[0]
            if sample.depth_valid is not None
            else official_depth_valid_mask(gt_z)
        )
        ar_z, fr_z = _score(pred_u, gt_z, valid_z, depth_clip=depth_clip)
        tracks["zbuf"].append(ar_z)
        fracs["zbuf"].append(fr_z)

        gt_native = load_eth3d_official_depth_map(depth_dir / stem)
        H_d, W_d = gt_native.shape
        valid_native = official_depth_valid_mask(gt_native)

        H_r, W_r = _render_size(H_d, W_d, max_dim=args.max_dim)
        gt_r = _resize_depth_nearest(gt_native, height=H_r, width=W_r)
        ar_m, fr_m = _score(pred_u, gt_r, official_depth_valid_mask(gt_r), depth_clip=depth_clip)
        tracks["off_misalign"].append(ar_m)
        fracs["off_misalign"].append(fr_m)

        line = f"{rec['sample_id']}: zbuf={ar_z:.4f} off_misalign={ar_m:.4f}"

        if have_distorted:
            dist_path = distorted_dir / stem
            if not dist_path.exists():
                print(f"{rec['sample_id']}: skip distorted (missing {dist_path.name})")
                continue
            rgb_d = read_rgb_uint8(dist_path)
            if rgb_d.shape[:2] != gt_native.shape:
                raise SystemExit(
                    f"shape mismatch {dist_path}: rgb {rgb_d.shape[:2]} vs depth {gt_native.shape}"
                )
            pred_d = model.predict(rgb_d[None]).depth[0]
            ar_a, fr_a = _score(pred_d, gt_native, valid_native, depth_clip=depth_clip)
            tracks["off_aligned"].append(ar_a)
            fracs["off_aligned"].append(fr_a)

            ar_518, fr_518 = _score(pred_d, gt_r, official_depth_valid_mask(gt_r), depth_clip=depth_clip)
            tracks["off_at518"].append(ar_518)
            fracs["off_at518"].append(fr_518)
            line += f" off_aligned={ar_a:.4f} off@518={ar_518:.4f}"

        print(line)

    n = len(records)
    print()
    print(f"scene={scene} frames={n} variant={args.variant}")
    print(f"  z-buffer (eth3d_dav2):          AbsRel {_mean(tracks['zbuf']):.4f}  valid% {100*_mean(fracs['zbuf']):.1f}")
    print(
        f"  official@518 (undistorted pred): AbsRel {_mean(tracks['off_misalign']):.4f}  "
        f"valid% {100*_mean(fracs['off_misalign']):.1f}"
    )
    if have_distorted and tracks["off_aligned"]:
        print(
            f"  official native (distorted RGB): AbsRel {_mean(tracks['off_aligned']):.4f}  "
            f"valid% {100*_mean(fracs['off_aligned']):.1f}"
        )
        print(
            f"  official@518 (distorted RGB):    AbsRel {_mean(tracks['off_at518']):.4f}  "
            f"valid% {100*_mean(fracs['off_at518']):.1f}"
        )
    else:
        print("  distorted RGB: not staged — fetch {scene}_dslr_jpg.7z for aligned tracks")


if __name__ == "__main__":
    main()
