"""ETH3D high-res native mono-depth loader (Depth Pro Table 1 / Table 16).

This is the **native official-depth** pairing, distinct from the multi-view
laser-scan path in :mod:`plumbline.datasets.eth3d` (which renders per-view GT
from the ``scan_clean`` / ``dslr_scan_eval`` PLY for chamfer / MVS depth). Here
each sample is a single image scored against ETH3D's own rendered float32 depth
map at the **distorted** DSLR resolution:

    <root>/<scene>/images/dslr_images/<DSC_xxxx>.JPG              # distorted sRGB
    <root>/<scene>/ground_truth_depth/dslr_images/<DSC_xxxx>.JPG  # float32 depth dump
    <root>/<scene>/dslr_calibration_undistorted/{cameras,images}.txt

The distorted RGB and the official depth share the native sensor resolution
(e.g. 6048×4032). ETH3D only ships a clean pinhole calibration for the
*undistorted* images (slightly larger, e.g. 6192×4121); we carry a GT focal by
scaling that pinhole to the distorted size (``fx · W_dist / W_undist``). The
lens distortion on these DSLRs is small, so the central focal is well
approximated — adequate for Depth Pro's ``f_px`` (a single scalar). Pair with
``DepthProAdapter(use_gt_focal=True)`` to feed the GT focal, or run the adapter
default to let Depth Pro self-estimate.

Depth Pro appendix Table 16: ETH3D = **454** samples, valid depth **0.1–200 m**,
δ₁ = **0.415**. Stage via ``scripts/stage-eth3d-depth-pro.sh``.
"""

from __future__ import annotations

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
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    read_rgb_uint8,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.eth3d import (
    _parse_colmap_cameras_hw,
    load_eth3d_official_depth_map,
    official_depth_valid_mask,
    parse_colmap_cameras,
    parse_colmap_images,
)
from plumbline.datasets.registry import register_dataset

__all__ = ["ETH3DNativeDepthDataset"]


@register_dataset("eth3d-native-depth")
class ETH3DNativeDepthDataset(Dataset):
    """ETH3D native mono-depth: distorted RGB + official float32 depth.

    Parameters
    ----------
    root
        Dataset root ``<root>/<scene>/...``; falls back to ``$ETH3D_ROOT``.
    split
        Only ``"train"`` (the public split that ships GT depth).
    scenes
        Optional scene whitelist. Default: every scene under ``root`` that has
        both ``images/dslr_images`` and ``ground_truth_depth/dslr_images``.
    depth_range
        ``(min, max)`` metres; GT outside this is masked invalid. Default
        ``(0.1, 200.0)`` per Depth Pro Table 16.
    """

    split: str = "train"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "train",
        scenes: list[str] | None = None,
        depth_range: tuple[float, float] = (0.1, 200.0),
    ) -> None:
        if split != "train":
            raise ValueError(f"ETH3D native-depth only exposes 'train' (no public test GT); got {split!r}")
        root_path = Path(root) if root else env_path("ETH3D_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ETH3D not found. Set --data-root or $ETH3D_ROOT to a directory "
                "containing <scene>/images/dslr_images/*.JPG and "
                "<scene>/ground_truth_depth/dslr_images/*.JPG. "
                "Stage via scripts/stage-eth3d-depth-pro.sh."
            )
        self.root = root_path
        self.split = split
        self.depth_range = (float(depth_range[0]), float(depth_range[1]))

        wanted = set(scenes) if scenes else None
        records: list[dict[str, Any]] = []
        for scene_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            if wanted is not None and scene_dir.name not in wanted:
                continue
            calib = scene_dir / "dslr_calibration_undistorted"
            depth_dir = scene_dir / "ground_truth_depth" / "dslr_images"
            dist_dir = scene_dir / "images" / "dslr_images"
            if not (calib / "images.txt").exists() or not depth_dir.is_dir() or not dist_dir.is_dir():
                continue
            cam_hw = _parse_colmap_cameras_hw(calib / "cameras.txt")
            for ir in sorted(parse_colmap_images(calib / "images.txt"), key=lambda x: x["image_id"]):
                stem = Path(ir["name"]).name
                rgb_path = dist_dir / stem
                depth_path = depth_dir / stem
                if not rgb_path.exists() or not depth_path.exists():
                    continue
                H_u, W_u = cam_hw[ir["camera_id"]]
                records.append(
                    {
                        "sample_id": f"{scene_dir.name}/{stem}",
                        "scene": scene_dir.name,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "cameras_txt": str(calib / "cameras.txt"),
                        "camera_id": ir["camera_id"],
                        "undist_hw": (H_u, W_u),
                    }
                )
        if not records:
            raise DatasetNotAvailable(
                f"No distorted-RGB + official-depth pairs under {root_path}. "
                "Stage *_dslr_jpg.7z + *_dslr_depth.7z (scripts/stage-eth3d-depth-pro.sh)."
            )
        self._records = records
        # cameras.txt → {camera_id: K} cache, keyed by path (one per scene).
        self._cam_cache: dict[str, dict[int, NDArray[np.float32]]] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        name = rec["sample_id"]
        img = read_rgb_uint8(Path(rec["rgb_path"]))
        H_d, W_d, _ = img.shape
        images = img[None]
        assert_valid_image(images, name=f"eth3d-native-depth/{name}")

        # Official depth is a row-major float32 dump at the distorted resolution.
        depth = load_eth3d_official_depth_map(rec["depth_path"], height=H_d, width=W_d)
        lo, hi = self.depth_range
        valid = official_depth_valid_mask(depth) & (depth >= lo) & (depth <= hi)
        depth_gt = np.where(valid, depth, 0.0).astype(np.float32)[None]
        depth_valid = valid[None]

        # GT focal: undistorted pinhole scaled to the distorted size. The
        # undistorted image is a touch larger than the sensor; the focal-per-
        # width ratio is preserved under ETH3D's small DSLR distortion.
        cameras = self._cam_cache.get(rec["cameras_txt"])
        if cameras is None:
            cameras = parse_colmap_cameras(Path(rec["cameras_txt"]))
            self._cam_cache[rec["cameras_txt"]] = cameras
        K_u = cameras[rec["camera_id"]].astype(np.float64)
        H_u, W_u = rec["undist_hw"]
        sx = W_d / float(W_u)
        sy = H_d / float(H_u)
        K = np.array(
            [
                [K_u[0, 0] * sx, 0.0, K_u[0, 2] * sx],
                [0.0, K_u[1, 1] * sy, K_u[1, 2] * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )[None]
        e_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K, name=f"eth3d-native-depth/{name}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"eth3d-native-depth/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"eth3d-native-depth/{name}/depth")

        return Sample(
            sample_id=f"eth3d-native-depth/{name}",
            images=images,
            intrinsics=K,
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={"scene": rec["scene"], "split": self.split, "image_size": (H_d, W_d)},
        )
