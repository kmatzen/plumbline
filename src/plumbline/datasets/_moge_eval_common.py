"""Shared decoding for MoGe's preprocessed mono-depth evaluation bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["process_moge_bundle_sample"]


def process_moge_bundle_sample(
    moge_pipe: Any,
    *,
    sample_root: Path,
    sample_path: str,
    image_path: Path | None = None,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
]:
    """Decode one bundle record and apply MoGe's canonical FoV warp.

    This intentionally rebuilds the raw instance instead of calling upstream's
    ``_load_instance``: the bundled MoGe revision references a stale undefined
    ``read_meta`` symbol there. The returned intrinsics are denormalized into
    pixel units on the warped output frame.
    """
    from moge.utils.io import read_depth as moge_read_depth
    from moge.utils.io import read_image as moge_read_image

    image_np = moge_read_image(image_path or (sample_root / "image.jpg"))
    depth_np = moge_read_depth(sample_root / "depth.png")
    meta = json.loads((sample_root / "meta.json").read_text())
    raw = {
        "filename": sample_path,
        "width": moge_pipe.width,
        "height": moge_pipe.height,
        "image": image_np,
        "depth": np.nan_to_num(depth_np, nan=1, posinf=1, neginf=1),
        "depth_mask": np.isfinite(depth_np),
        "depth_mask_inf": np.isinf(depth_np),
        "intrinsics": np.array(meta["intrinsics"], dtype=np.float32),
    }
    inst = moge_pipe._process_instance(raw)

    image_tensor = inst["image"]
    image = (image_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    depth = inst["depth"].numpy().astype(np.float32)
    valid = inst["depth_mask"].numpy().astype(bool)
    depth = np.where(valid, depth, np.float32(1.0))

    intrinsics = inst["intrinsics"].numpy().astype(np.float32)
    height, width = image.shape[:2]
    intrinsics[0, :] *= width
    intrinsics[1, :] *= height
    return image, depth, valid, intrinsics
