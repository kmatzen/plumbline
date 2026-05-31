"""SUN RGB-D test-split loader (metric depth, ZoeDepth/Depth Pro lineage).

Depth Pro Table 1 / appendix Table 16: 5050 validation images, valid depth
0.001–10 m, GT resolution ~530×730.

Uses the public SUN RGB-D **test** pack (Ahanda mirror of Princeton test split)::

    <root>/
        rgb/<name>.jpg
        depth/<id>.png   # uint16, depth_m = value / 10000

Download::

    ./scripts/download-sun-rgbd.sh

Ahanda test pack: ``img-{i:06d}.jpg`` paired with ``depth/{i}.png`` (not the
ZoeDepth ``rgb/rgb`` + ``gt/gt`` tree).
"""

from __future__ import annotations

import re
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

__all__ = ["SunRgbdDataset", "read_sun_rgbd_depth_png"]

_DEPTH_SCALE = 10000.0


def read_sun_rgbd_depth_png(path: Path) -> NDArray[np.float32]:
    """Decode SUN RGB-D depth PNG (uint16) to meters."""
    from PIL import Image as PImage

    depth = np.asarray(PImage.open(path), dtype=np.float32) / _DEPTH_SCALE
    return depth


def _rgb_to_depth_path(rgb_path: Path, depth_dir: Path) -> Path:
    stem = rgb_path.stem
    m = re.search(r"(\d+)", stem)
    if m is None:
        raise ValueError(f"Cannot parse frame id from {rgb_path.name}")
    return depth_dir / f"{int(m.group(1))}.png"


@register_dataset("sun-rgbd")
class SunRgbdDataset(Dataset):
    """SUN RGB-D test split for monocular metric depth (5050 frames).

    Parameters
    ----------
    root
        Directory with ``rgb/`` and ``depth/`` subdirs. Falls back to
        ``$SUN_RGBD_ROOT``.
    split
        Only ``"test"`` (5050 public test frames with depth).
    max_depth_invalid
        Pixels with GT depth above this value (m) are marked invalid
        (ZoeDepth uses 8 m; Table 16 clips at 10 m in the protocol).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        max_depth_invalid: float = 80.0,
    ) -> None:
        if split != "test":
            raise ValueError(f"SunRgbdDataset only exposes the test split; got {split!r}")

        root_path = Path(root) if root else env_path("SUN_RGBD_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "SUN RGB-D not found. Set --data-root or $SUN_RGBD_ROOT. "
                "Run ./scripts/download-sun-rgbd.sh"
            )

        rgb_dir = root_path / "rgb"
        depth_dir = root_path / "depth"
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            raise DatasetNotAvailable(f"Expected {rgb_dir} and {depth_dir} under {root_path}.")

        pairs: list[tuple[Path, Path]] = []
        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
            depth_path = _rgb_to_depth_path(rgb_path, depth_dir)
            if depth_path.is_file():
                pairs.append((rgb_path, depth_path))

        if not pairs:
            raise DatasetNotAvailable(f"No rgb/depth pairs under {root_path}")

        self.root = root_path
        self.pairs = pairs
        self.max_depth_invalid = max_depth_invalid

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Sample]:
        for rgb_path, depth_path in self.pairs:
            yield self._load_sample(rgb_path, depth_path)

    def _load_sample(self, rgb_path: Path, depth_path: Path) -> Sample:
        name = rgb_path.stem
        img = read_rgb_uint8(rgb_path)
        images = img[None]
        assert_valid_image(images, name=f"sun-rgbd/{name}")

        h, w, _ = img.shape
        depth = read_sun_rgbd_depth_png(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape != (h, w):
            raise ValueError(f"sun-rgbd/{name}: depth {depth.shape} != image {(h, w)}")

        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth_invalid)
        depth_gt = np.where(valid, depth, 0.0).astype(np.float32)[None]
        depth_valid = valid[None]

        # No per-frame intrinsics in the Ahanda test pack; Depth Pro infers focal length.
        fx = float(max(w, h))
        k = np.array(
            [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )[None]
        e_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(k, name=f"sun-rgbd/{name}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"sun-rgbd/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"sun-rgbd/{name}/depth")

        return Sample(
            sample_id=f"sun-rgbd/{name}",
            images=images,
            intrinsics=k,
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "frame": name,
                "split": self.split,
                "image_size": (h, w),
            },
        )
