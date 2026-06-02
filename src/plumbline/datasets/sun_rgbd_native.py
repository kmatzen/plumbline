"""SUN RGB-D test split (native resolution) for metric monocular depth.

This is the **native** SUN RGB-D test pack used to reproduce Depth Pro Table 1
(δ₁ 0.890). It differs from the earlier (removed) ahanda 730×530 pack in two
ways that turned out to matter (see ``docs/blocked/DEPTH_PRO_SUN_RGBD_TABLE1.md``
for the GPU-verified investigation):

1. **Native resolution + GT focal.** Frames keep their per-sensor native
   resolution and carry the per-frame ``intrinsics.txt``. The ahanda pack
   anisotropically resized every frame to 730×530, corrupting the pinhole
   geometry for the non-Kinect-v2 sensors, and stripped intrinsics. Depth Pro's
   Table-1 number only reproduces with the dataset's **GT focal** (the model's
   self-estimated focal mis-fires on the Kinect frames); pair this loader with
   ``DepthProAdapter(use_gt_focal=True)``.
2. **Canonical depth decode.** Native SUN RGB-D depth PNGs (``depth_bfx``,
   improved/hole-filled) are bit-rotation encoded: ``d = (raw>>3) | (raw<<13)``
   then ``/1000`` → meters, clipped at 8 m. The ahanda pack's ``÷10000`` decode
   is ~1.25× too small.

Expected layout (flat, the staged ``s3://plumbline-bench/datasets/sun_rgbd_native``)::

    <root>/rgb/img-{i:06d}.jpg          # native-resolution sRGB
    <root>/depth/img-{i:06d}.png        # native depth_bfx (uint16, bit-rotation)
    <root>/intrinsics/img-{i:06d}.txt   # 3x3 K, row-major (fx is K[0,0])

Falls back to ``$SUN_RGBD_NATIVE_ROOT``. Metric δ₁, no alignment (Table 16 clip
0.001–10 m; the decode already caps GT at 8 m).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.datasets._common import DatasetNotAvailable, env_path, read_rgb_uint8
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset


def read_sun_rgbd_native_depth(path: Path) -> NDArray[np.float32]:
    """Decode a native SUN RGB-D depth PNG (bit-rotation) to meters, clipped 8 m."""
    from PIL import Image as PImage

    raw = np.asarray(PImage.open(path), dtype=np.uint16)
    rot = ((raw >> 3) | (raw << 13)).astype(np.uint16)
    depth = rot.astype(np.float32) / 1000.0
    depth[depth > 8.0] = 8.0
    return depth


@register_dataset("sun-rgbd-native")
class SunRgbdNativeDataset(Dataset):
    """SUN RGB-D native test split (5050 frames) for metric depth + GT focal.

    Parameters
    ----------
    root
        Directory with ``rgb/``, ``depth/``, ``intrinsics/`` subdirs. Falls back
        to ``$SUN_RGBD_NATIVE_ROOT``.
    split
        Only ``"test"`` (the 5050 public test frames).
    max_depth_invalid
        GT depth above this (m) is masked invalid (Table 16 clips at 10 m; the
        decode already caps at 8 m).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        max_depth_invalid: float = 10.0,
    ) -> None:
        if split != "test":
            raise ValueError(f"SunRgbdNativeDataset only exposes the test split; got {split!r}")

        root_path = Path(root) if root else env_path("SUN_RGBD_NATIVE_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "SUN RGB-D (native) not found. Set --data-root or "
                "$SUN_RGBD_NATIVE_ROOT (s3://plumbline-bench/datasets/sun_rgbd_native)."
            )

        rgb_dir = root_path / "rgb"
        depth_dir = root_path / "depth"
        intr_dir = root_path / "intrinsics"
        for d in (rgb_dir, depth_dir, intr_dir):
            if not d.is_dir():
                raise DatasetNotAvailable(f"Expected {d} under {root_path}.")

        triples: list[tuple[Path, Path, Path]] = []
        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
            stem = rgb_path.stem
            depth_path = depth_dir / f"{stem}.png"
            intr_path = intr_dir / f"{stem}.txt"
            if depth_path.is_file() and intr_path.is_file():
                triples.append((rgb_path, depth_path, intr_path))

        if not triples:
            raise DatasetNotAvailable(f"No rgb/depth/intrinsics triples under {root_path}")

        self.root = root_path
        self.triples = triples
        self.max_depth_invalid = max_depth_invalid

    def __len__(self) -> int:
        return len(self.triples)

    def __iter__(self) -> Iterator[Sample]:
        for rgb_path, depth_path, intr_path in self.triples:
            yield self._load_sample(rgb_path, depth_path, intr_path)

    def _load_sample(self, rgb_path: Path, depth_path: Path, intr_path: Path) -> Sample:
        name = rgb_path.stem
        img = read_rgb_uint8(rgb_path)
        images = img[None]
        assert_valid_image(images, name=f"sun-rgbd-native/{name}")

        h, w, _ = img.shape
        depth = read_sun_rgbd_native_depth(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape != (h, w):
            raise ValueError(f"sun-rgbd-native/{name}: depth {depth.shape} != image {(h, w)}")

        valid = np.isfinite(depth) & (depth > 0.0) & (depth < self.max_depth_invalid)
        depth_gt = np.where(valid, depth, 0.0).astype(np.float32)[None]
        depth_valid = valid[None]

        # Per-frame GT focal (native pixel units == this image's pixel units).
        # cx/cy come from the same intrinsics file; fall back to image center.
        vals = [float(x) for x in intr_path.read_text().split()]
        fx = vals[0]
        fy = vals[4] if len(vals) >= 5 else fx
        cx = vals[2] if len(vals) >= 3 else w / 2.0
        cy = vals[5] if len(vals) >= 6 else h / 2.0
        k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )[None]
        e_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(k, name=f"sun-rgbd-native/{name}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"sun-rgbd-native/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"sun-rgbd-native/{name}/depth")

        return Sample(
            sample_id=f"sun-rgbd-native/{name}",
            images=images,
            intrinsics=k,
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={"frame": name, "split": self.split, "image_size": (h, w)},
        )
