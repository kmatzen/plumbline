"""NYUv2 (labeled subset) loader.

NYU Depth V2 is the canonical indoor monocular-depth benchmark. The "labeled"
subset (1449 RGB-D pairs) is distributed as a single HDF5/v7.3 MAT file from
NYU CS. The "Eigen test split" (Eigen et al. 2014) uses 654 of those 1449
pairs and is what Depth Anything V2, MiDaS, DPT, ZoeDepth, and ~every other
modern mono-depth paper reports.

Expected layout::

    <root>/nyu_depth_v2_labeled.mat

Download (public, no auth; ~3 GB)::

    mkdir -p ~/data/nyuv2 && cd ~/data/nyuv2
    curl -LO https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

Splits
------
- ``"test"`` (default): the 654-index Eigen test split. Indices are bundled
  with the package at ``plumbline/datasets/_nyuv2_eigen_test.txt``.
- ``"all"``: iterate all 1449 labeled pairs.
- Custom: pass ``indices=[int, ...]`` to override.

Conventions
-----------
- Images are (480, 640, 3) uint8 sRGB. NYU stores them in CHW order inside
  the .mat; we transpose on load.
- Depth is (480, 640) float32 meters. NYU stores two depth fields: ``depths``
  is Silberman's colorization-filled version (100% dense, interpolated
  across Kinect holes); ``rawDepths`` is the raw Kinect measurements with
  ~24% invalid pixels (holes at specular surfaces, thin structures, and
  beyond-range). Zero means invalid in ``rawDepths``. The Eigen 2014 test
  protocol that every modern mono-depth paper cites evaluates against
  ``rawDepths``; that is therefore our default (``depth_field="raw"``).
  Pass ``depth_field="filled"`` to use the dense colorized variant instead.
- Intrinsics are the standard Silberman NYUv2 color calibration
  (fx=518.86, fy=519.47, cx=325.58, cy=253.74). NYUv2 is a monocular
  benchmark; ``extrinsics_gt`` is identity.
"""

from __future__ import annotations

import importlib.resources as resources
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
from plumbline.datasets._common import DatasetNotAvailable, env_path
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = [
    "EIGEN_CROP",
    "NYUV2_INTRINSICS",
    "NYUv2Dataset",
    "eigen_crop_mask",
    "load_eigen_test_indices",
]

# Silberman et al. NYUv2 color-camera calibration (undistorted).
NYUV2_INTRINSICS: tuple[float, float, float, float] = (518.8579, 519.4696, 325.5824, 253.7362)

# Eigen et al. 2014 crop for NYUv2 evaluation: (top, bottom, left, right) rows/cols.
# Applies to 480x640 depth maps. Virtually every mono-depth paper reports numbers
# with this crop applied; without it, edge-pixel noise bloats metrics.
EIGEN_CROP: tuple[int, int, int, int] = (45, 471, 41, 601)


@register_dataset("nyuv2")
class NYUv2Dataset(Dataset):
    """NYU Depth V2 labeled-subset loader.

    Parameters
    ----------
    root
        Directory containing ``nyu_depth_v2_labeled.mat``. If omitted, falls
        back to ``$NYUV2_ROOT``.
    split
        ``"test"`` (default: Eigen 654), ``"all"`` (1449), or ``"custom"``
        when ``indices`` is given.
    indices
        Explicit list of 0-indexed sample indices (takes precedence over
        ``split``).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        indices: list[int] | None = None,
        apply_eigen_crop: bool = False,
        depth_field: str = "raw",
        max_gt_depth: float | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("NYUV2_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "NYUv2 not found. Set --data-root or $NYUV2_ROOT to a directory "
                "containing nyu_depth_v2_labeled.mat. Download (public, ~3 GB): "
                "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
            )
        mat_path = root_path / "nyu_depth_v2_labeled.mat"
        if not mat_path.exists():
            raise DatasetNotAvailable(f"Expected {mat_path}; not found.")

        self.root = root_path
        self.mat_path = mat_path

        if indices is not None:
            self._indices = [int(i) for i in indices]
            self.split = "custom"
        elif split == "test":
            self._indices = load_eigen_test_indices()
            self.split = "test"
        elif split == "all":
            self._indices = list(range(1449))
            self.split = "all"
        else:
            raise ValueError(
                f"NYUv2 split '{split}' unsupported; use 'test', 'all', or pass indices=[...]"
            )

        self.apply_eigen_crop = bool(apply_eigen_crop)
        if depth_field not in ("raw", "filled"):
            raise ValueError(f"depth_field must be 'raw' or 'filled'; got {depth_field!r}")
        self.depth_field = depth_field
        # Optional pre-fit GT upper bound, matching the Marigold /
        # GeoWizard ``valid_mask = (depth > min) AND (depth < max) AND
        # eigen_crop`` convention. On the NYU labeled set this is
        # essentially a no-op — Kinect saturation pixels are written as
        # 0 (already excluded by ``depth > 0``), not as values above
        # 10 m — so it doesn't close D17's 10 % gap to paper. Kept as a
        # structural knob for parity and for datasets where the
        # equivalent matters (e.g. KITTI's 80 m bound). Default
        # ``None`` preserves the prior behaviour.
        self.max_gt_depth = float(max_gt_depth) if max_gt_depth is not None else None

    def __iter__(self) -> Iterator[Sample]:
        import h5py

        K = np.array(
            [
                [NYUV2_INTRINSICS[0], 0.0, NYUV2_INTRINSICS[2]],
                [0.0, NYUV2_INTRINSICS[1], NYUV2_INTRINSICS[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        K_stack = K[None]  # (1, 3, 3)
        E_eye = np.eye(4, dtype=np.float32)[None]  # (1, 4, 4)

        with h5py.File(self.mat_path, "r") as f:
            images_ds = f["images"]  # shape (N, 3, 640, 480) per NYU's ordering
            depth_key = "rawDepths" if self.depth_field == "raw" else "depths"
            if depth_key not in f:
                alt = "depths" if depth_key == "rawDepths" else "rawDepths"
                raise DatasetNotAvailable(
                    f"NYUv2 .mat at {self.mat_path} has no '{depth_key}' field "
                    f"(has '{alt}'). Pass depth_field='{alt[:3] if alt.startswith('raw') else 'filled'}' "
                    f"or download the full labeled .mat from "
                    f"https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
                )
            depths_ds = f[depth_key]  # shape (N, 640, 480)
            for idx in self._indices:
                rgb_raw = np.asarray(images_ds[idx])  # (3, 640, 480) uint8 stored by MATLAB
                depth_raw = np.asarray(depths_ds[idx])  # (640, 480) float
                rgb, depth = _to_canonical(rgb_raw, depth_raw)
                images = rgb[None]
                depth_stack = depth[None]
                assert_valid_image(images, name=f"nyuv2/{idx}/image")
                assert_valid_intrinsics(K_stack, name=f"nyuv2/{idx}/intrinsics")
                assert_valid_extrinsics(E_eye, name=f"nyuv2/{idx}/extrinsics")
                assert_valid_depth(depth_stack, name=f"nyuv2/{idx}/depth")
                depth_valid: NDArray[np.bool_] | None = None
                if self.apply_eigen_crop:
                    mask = np.zeros(depth.shape, dtype=bool)
                    top, bot, left, right = EIGEN_CROP
                    mask[top:bot, left:right] = True
                    mask &= depth > 0
                    if self.max_gt_depth is not None:
                        mask &= depth < self.max_gt_depth
                    depth_valid = mask[None]
                yield Sample(
                    sample_id=f"nyuv2_{idx:05d}",
                    images=images,
                    intrinsics=K_stack,
                    extrinsics_gt=E_eye,
                    depth_gt=depth_stack,
                    depth_valid=depth_valid,
                    metadata={
                        "labeled_index": idx,
                        "split": self.split,
                        "eigen_crop": self.apply_eigen_crop,
                        "depth_field": self.depth_field,
                    },
                )

    def __len__(self) -> int:
        return len(self._indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_canonical(
    rgb_raw: NDArray[np.uint8], depth_raw: NDArray[np.float32]
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Convert NYU's MATLAB-ordered arrays into canonical (H, W, 3) / (H, W).

    NYU's .mat file stores ``images`` as (1449, 3, 640, 480) and ``depths`` as
    (1449, 640, 480) (column-major indexing persisted through HDF5). We need
    (480, 640, 3) uint8 and (480, 640) float32 meters.
    """
    # rgb_raw arrives as (3, 640, 480). Swap to (480, 640, 3).
    rgb = np.transpose(rgb_raw, (2, 1, 0)).astype(np.uint8)
    # depth_raw arrives as (640, 480). Transpose to (480, 640).
    depth = np.asarray(depth_raw).T.astype(np.float32)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)


def eigen_crop_mask(shape: tuple[int, int] = (480, 640)) -> NDArray[np.bool_]:
    """Return the canonical Eigen NYUv2 evaluation crop as a boolean mask."""
    top, bot, left, right = EIGEN_CROP
    mask = np.zeros(shape, dtype=bool)
    mask[top:bot, left:right] = True
    return mask


def load_eigen_test_indices() -> list[int]:
    """Return the 654-index Eigen NYUv2 test split (0-indexed)."""
    try:
        pkg = resources.files("plumbline.datasets")
        text = (pkg / "_nyuv2_eigen_test.txt").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        here = Path(__file__).resolve().parent
        text = (here / "_nyuv2_eigen_test.txt").read_text(encoding="utf-8")
    indices: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        indices.append(int(line))
    if len(indices) != 654:
        raise RuntimeError(f"Eigen NYUv2 test split must have 654 entries; got {len(indices)}")
    return indices
