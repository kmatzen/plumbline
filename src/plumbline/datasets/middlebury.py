"""Middlebury MiddEval3 training loader (mono metric depth).

Depth Pro Table 1 / appendix Table 16 use the MiddEval3 **training** split
(15 scenes, Scharstein et al. 2014) with metric depth from ``disp0GT.pfm`` and
``calib.txt`` (depth in mm: ``Z = baseline * f / (disp + doffs)``).

Expected layout after ``scripts/download-middlebury.sh``::

    <root>/trainingF/<scene>/
        im0.png
        calib.txt
        disp0GT.pfm
        mask0nocc.png

Resolution ``F`` matches appendix Table 16 (~1988×2952 typical); ``H``/``Q`` are
available for smoke tests.

Download: https://vision.middlebury.edu/stereo/submit3/
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

__all__ = [
    "MiddleburyDataset",
    "middlebury_disparity_to_depth",
    "read_middlebury_calib",
    "read_pfm_disparity",
]

_RESOLUTION_DIRS = {"f": "trainingF", "h": "trainingH", "q": "trainingQ"}


def read_pfm_disparity(path: Path) -> NDArray[np.float32]:
    """Load a MiddEval3 ``disp0GT.pfm`` as pixel disparities (float32)."""
    with path.open("rb") as f:
        header = f.readline().decode("utf-8").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path} ({header!r})")
        width, height = map(int, f.readline().decode("utf-8").split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, dtype=f"{endian}f4", count=width * height)
        data = np.flipud(np.reshape(data, (height, width))).astype(np.float32)
    data[~np.isfinite(data)] = np.inf
    return data


def read_middlebury_calib(calib_path: Path) -> tuple[float, float, float, float, float, float]:
    """Return ``(fx, fy, cx, cy, baseline_mm, doffs)`` from MiddEval3 ``calib.txt``."""
    text = calib_path.read_text(encoding="utf-8")
    cam0 = re.search(r"cam0=\[([^\]]+)\]", text)
    if cam0 is None:
        raise ValueError(f"Missing cam0 in {calib_path}")
    vals = [float(x) for x in re.findall(r"[0-9.]+", cam0.group(1))]
    if len(vals) < 4:
        raise ValueError(f"Could not parse cam0 in {calib_path}")
    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]
    doffs_m = re.search(r"doffs=([0-9.]+)", text)
    baseline_m = re.search(r"baseline=([0-9.]+)", text)
    if doffs_m is None or baseline_m is None:
        raise ValueError(f"Missing doffs/baseline in {calib_path}")
    return fx, fy, cx, cy, float(baseline_m.group(1)), float(doffs_m.group(1))


def middlebury_disparity_to_depth(
    disp: NDArray[np.floating],
    *,
    fx: float,
    baseline_mm: float,
    doffs: float,
) -> NDArray[np.float32]:
    """Convert left disparity (px) to metric depth (m)."""
    depth_mm = baseline_mm * fx / (disp.astype(np.float64) + doffs)
    depth = depth_mm / 1000.0
    return depth.astype(np.float32)


@register_dataset("middlebury")
class MiddleburyDataset(Dataset):
    """MiddEval3 training split for monocular metric depth.

    Parameters
    ----------
    root
        Staged root with ``trainingF/`` (etc.) scene subdirs. Falls back to
        ``$MIDDLEBURY_ROOT``.
    split
        Only ``"training"`` (15 public GT scenes).
    resolution
        ``"f"``, ``"h"``, or ``"q"`` — Depth Pro Table 16 uses full (F).
    scenes
        Optional scene-name whitelist.
    use_nocc_mask
        If True (default), keep pixels where ``mask0nocc.png == 255``.
    """

    split: str = "training"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "training",
        resolution: str = "f",
        scenes: list[str] | None = None,
        use_nocc_mask: bool = True,
    ) -> None:
        if split != "training":
            raise ValueError(f"MiddleburyDataset only exposes MiddEval3 training GT; got {split!r}")
        res = resolution.lower()
        if res not in _RESOLUTION_DIRS:
            raise ValueError(
                f"resolution must be one of {sorted(_RESOLUTION_DIRS)}; got {resolution!r}"
            )

        root_path = Path(root) if root else env_path("MIDDLEBURY_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "Middlebury not found. Set --data-root or $MIDDLEBURY_ROOT. "
                "Run ./scripts/download-middlebury.sh"
            )

        train_root = root_path / _RESOLUTION_DIRS[res]
        if not train_root.is_dir():
            raise DatasetNotAvailable(
                f"Expected {train_root} under {root_path}. Run ./scripts/download-middlebury.sh"
            )

        all_scenes = sorted(
            p.name
            for p in train_root.iterdir()
            if p.is_dir() and (p / "im0.png").is_file() and (p / "disp0GT.pfm").is_file()
        )
        if scenes is not None:
            wanted = set(scenes)
            scene_names = [n for n in all_scenes if n in wanted]
        else:
            scene_names = all_scenes

        self.root = root_path
        self.resolution = res
        self.train_root = train_root
        self.use_nocc_mask = use_nocc_mask
        self.scene_names = scene_names

        if not self.scene_names:
            raise DatasetNotAvailable(f"No MiddEval3 scenes under {train_root}")
        if len(self.scene_names) != 15:
            pass  # allow partial staging for smoke

    def __len__(self) -> int:
        return len(self.scene_names)

    def __iter__(self) -> Iterator[Sample]:
        for name in self.scene_names:
            yield self._load_sample(name)

    def _load_sample(self, name: str) -> Sample:
        scene_dir = self.train_root / name
        img = read_rgb_uint8(scene_dir / "im0.png")
        images = img[None]
        assert_valid_image(images, name=f"middlebury/{name}/im0")

        h, w, _ = img.shape
        disp = read_pfm_disparity(scene_dir / "disp0GT.pfm")
        if disp.shape != (h, w):
            raise ValueError(f"middlebury/{name}: disp {disp.shape} != image {(h, w)}")

        fx, fy, cx, cy, baseline_mm, doffs = read_middlebury_calib(scene_dir / "calib.txt")
        depth = middlebury_disparity_to_depth(disp, fx=fx, baseline_mm=baseline_mm, doffs=doffs)
        depth_gt = depth[None]

        valid = np.isfinite(disp) & (disp > 0) & np.isfinite(depth) & (depth > 0)
        if self.use_nocc_mask:
            from PIL import Image as PImage

            mask_path = scene_dir / "mask0nocc.png"
            if mask_path.is_file():
                mask = np.asarray(PImage.open(mask_path), dtype=np.uint8)
                if mask.shape == (h, w):
                    valid &= mask == 255

        depth_valid = valid[None]
        depth_gt = np.where(depth_valid, depth_gt, 0.0).astype(np.float32)

        k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )[None]
        e_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(k, name=f"middlebury/{name}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"middlebury/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"middlebury/{name}/depth")

        return Sample(
            sample_id=f"middlebury/{name}",
            images=images,
            intrinsics=k,
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": name,
                "split": self.split,
                "resolution": self.resolution,
                "image_size": (h, w),
            },
        )
