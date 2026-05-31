"""Booster training-split loader (balanced stereo, monocular depth eval).

Booster (Ramirez et al. 2024, TPAMI) provides 228 training images at
4112×3008 with dense disparity GT, calibration, and occlusion masks.
Depth Pro Table 1 / appendix Table 16 evaluate metric δ₁ on the training
split (GT released; test mono GT withheld).

Expected layout after unzipping ``booster_gt.zip`` from amsacta::

    <root>/train/balanced/<scene>/
        camera_00/<illumination>.png
        disp_00.npy
        calib_00-02.xml
        mask_00.png   # occlusion mask (optional but recommended)

Disparity is converted to metric depth (meters) using the left-camera
intrinsics and ``baselineLR`` from ``calib_00-02.xml``::

    depth_m = fx * baseline_mm / disp_px / 1000

This matches the public ``metric-anything`` Booster eval loader.

Download::

    ./scripts/download-booster.sh
    # or:
    wget https://amsacta.unibo.it/id/eprint/6876/1/booster_gt.zip
    unzip -d $BOOSTER_ROOT booster_gt.zip
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

__all__ = ["BoosterDataset", "booster_disparity_to_depth", "read_booster_calib"]


def _parse_opencv_matrix(node: ET.Element, rows: int, cols: int) -> NDArray[np.float64]:
    data_node = node.find("data")
    if data_node is None or data_node.text is None:
        raise ValueError("Missing <data> for OpenCV matrix")
    values = [float(x) for x in data_node.text.strip().split()]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def read_booster_calib(calib_path: Path) -> tuple[float, float, float, float, float]:
    """Return ``(fx, fy, cx, cy, baseline_mm)`` from a Booster ``calib_00-02.xml``."""
    root = ET.parse(calib_path).getroot()
    mtx_l = root.find("mtxL")
    if mtx_l is None:
        raise ValueError(f"Missing mtxL in {calib_path}")
    k = _parse_opencv_matrix(mtx_l, 3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    baseline_elem = root.find("baselineLR")
    if baseline_elem is not None and baseline_elem.text is not None:
        baseline_mm = float(baseline_elem.text.strip())
    else:
        proj_r = root.find("proj_matR")
        if proj_r is None:
            raise ValueError(f"Missing baselineLR and proj_matR in {calib_path}")
        p = _parse_opencv_matrix(proj_r, 3, 4)
        if fx == 0:
            raise ValueError(f"fx is zero in {calib_path}")
        baseline_mm = -float(p[0, 3]) / fx
    return fx, fy, cx, cy, baseline_mm


def booster_disparity_to_depth(
    disp: NDArray[np.floating],
    *,
    fx: float,
    baseline_mm: float,
) -> NDArray[np.float32]:
    """Convert Booster left disparity (px) to metric depth (m)."""
    depth = (fx * baseline_mm) / (disp.astype(np.float64) + 1e-6) / 1000.0
    return depth.astype(np.float32)


@register_dataset("booster")
class BoosterDataset(Dataset):
    """Booster balanced training split for monocular metric depth.

    Parameters
    ----------
    root
        Directory containing ``train/balanced/...``. Falls back to
        ``$BOOSTER_ROOT``.
    split
        Only ``"training"`` is supported (228 frames with GT).
    setup
        Only ``"balanced"`` is implemented (12 Mpx left camera).
    scenes
        Optional scene-name whitelist.
    use_occlusion_mask
        If True (default), exclude pixels where ``mask_00.png`` marks
        occlusion.
    """

    split: str = "training"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "training",
        setup: str = "balanced",
        scenes: list[str] | None = None,
        use_occlusion_mask: bool = True,
    ) -> None:
        if split != "training":
            raise ValueError(
                f"BoosterDataset only exposes the training split with public GT; got {split!r}"
            )
        if setup != "balanced":
            raise ValueError(f"Only setup='balanced' is supported; got {setup!r}")

        root_path = Path(root) if root else env_path("BOOSTER_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "Booster not found. Set --data-root or $BOOSTER_ROOT to the "
                "directory containing train/balanced/<scene>/.... Download: "
                "./scripts/download-booster.sh"
            )

        train_root = root_path / "train" / setup
        if not train_root.is_dir():
            raise DatasetNotAvailable(
                f"Expected {train_root} (unzip booster_gt.zip under {root_path})."
            )

        all_scenes = sorted(
            p.name
            for p in train_root.iterdir()
            if p.is_dir() and (p / "camera_00").is_dir() and (p / "disp_00.npy").is_file()
        )
        if scenes is not None:
            wanted = set(scenes)
            scene_names = [n for n in all_scenes if n in wanted]
        else:
            scene_names = all_scenes

        self.root = root_path
        self.setup = setup
        self.use_occlusion_mask = use_occlusion_mask
        self.frame_records: list[dict[str, Any]] = []

        for scene in scene_names:
            scene_dir = train_root / scene
            calib_path = scene_dir / "calib_00-02.xml"
            disp_path = scene_dir / "disp_00.npy"
            if not calib_path.is_file():
                raise DatasetNotAvailable(f"Missing {calib_path}")
            fx, fy, cx, cy, baseline_mm = read_booster_calib(calib_path)
            mask_path = scene_dir / "mask_00.png"
            cam_dir = scene_dir / "camera_00"
            for image_path in sorted(cam_dir.glob("*.png")):
                self.frame_records.append(
                    {
                        "scene": scene,
                        "image": image_path,
                        "disp": disp_path,
                        "mask": mask_path if mask_path.is_file() else None,
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "baseline_mm": baseline_mm,
                    }
                )

        if not self.frame_records:
            raise DatasetNotAvailable(f"No Booster frames under {train_root}")
        if len(self.frame_records) != 228:
            # Warn via metadata; do not hard-fail — partial staging is useful for smoke.
            pass

    def __len__(self) -> int:
        return len(self.frame_records)

    def __iter__(self) -> Iterator[Sample]:
        for rec in self.frame_records:
            yield self._load_sample(rec)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        image_path = rec["image"]
        assert isinstance(image_path, Path)
        scene = str(rec["scene"])
        stem = image_path.stem

        img = read_rgb_uint8(image_path)
        images = img[None]
        assert_valid_image(images, name=f"booster/{scene}/{stem}")

        h, w, _ = img.shape
        disp = np.load(rec["disp"]).astype(np.float32)
        if disp.shape != (h, w):
            raise ValueError(f"booster/{scene}/{stem}: disp {disp.shape} != image {(h, w)}")

        fx = float(rec["fx"])
        baseline_mm = float(rec["baseline_mm"])
        depth = booster_disparity_to_depth(disp, fx=fx, baseline_mm=baseline_mm)
        depth_gt = depth[None]

        valid = np.isfinite(depth) & (disp > 0) & (depth > 0)
        if self.use_occlusion_mask and rec["mask"] is not None:
            from PIL import Image as PImage

            mask_path = rec["mask"]
            assert isinstance(mask_path, Path)
            occ = np.asarray(PImage.open(mask_path), dtype=np.uint8)
            if occ.shape == (h, w):
                valid &= occ != 0

        depth_valid = valid[None]
        depth_gt = np.where(depth_valid, depth_gt, 0.0).astype(np.float32)

        fx_px = float(rec["fx"])
        fy_px = float(rec["fy"])
        cx_px = float(rec["cx"])
        cy_px = float(rec["cy"])
        k = np.array(
            [[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )[None]
        e_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(k, name=f"booster/{scene}/{stem}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"booster/{scene}/{stem}/extrinsics")
        assert_valid_depth(depth_gt, name=f"booster/{scene}/{stem}/depth")

        return Sample(
            sample_id=f"booster/{scene}/{stem}",
            images=images,
            intrinsics=k,
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": scene,
                "frame": stem,
                "split": self.split,
                "setup": self.setup,
                "image_size": (h, w),
            },
        )
