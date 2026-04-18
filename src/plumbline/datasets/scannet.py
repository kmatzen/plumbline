"""ScanNet v2 test-split loader.

ScanNet is an indoor RGB-D dataset. The v2 test split is widely used for
monocular and multi-view depth benchmarks.

Expected layout (after running official extraction scripts, e.g.
``SensReader`` on the ``.sens`` files). Point ``--data-root`` or
``$SCANNET_ROOT`` at the split root::

    <root>/
      scans_test/<scene>/
        color/0000.jpg
        depth/0000.png             # uint16, depth in mm
        pose/0000.txt              # 4x4 camera_from_world (float)
        intrinsic/intrinsic_color.txt
        intrinsic/intrinsic_depth.txt

Access: http://www.scan-net.org/ (auth required; sign the ToS, wait for email).

Conventions
-----------
- ScanNet depth PNGs are uint16 millimeters. We convert to float32 meters.
- ScanNet poses are 4x4 ``camera_from_world`` float matrices in a text file;
  some frames have ``inf`` entries for frames the tracker dropped — filter
  those.
- Color and depth are at different resolutions (typically 1296x968 vs
  640x480). We keep both at native resolution and report the color
  intrinsics; models operate on color and resize predictions to depth-GT
  resolution at metric time.
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
    invert_pose,
    rebase_to_first_camera,
)
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    load_manifest,
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["ScanNetDataset", "load_scannet_depth_mm_to_m", "load_scannet_pose"]

_DEPTH_SCALE_MM_PER_M = 1000.0


@register_dataset("scannet")
class ScanNetDataset(Dataset):
    """ScanNet v2 test-split loader.

    Parameters
    ----------
    root
        Split root. Must contain ``scans_test/<scene>/...``.
    split
        Currently only ``"test"`` (scans_test). ``"val"`` is left for v0.2.
    frame_stride
        Use every Nth frame. ScanNet sequences are 30fps, so ``stride=10`` is
        a common subsample.
    views_per_sample
        Frames grouped into one sample. For monocular depth use ``1``; for
        multi-view stereo / pose, use the paper's view count (often 3-8).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        frame_stride: int = 20,
        views_per_sample: int = 1,
        scenes: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("SCANNET_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ScanNet not found. Set --data-root or $SCANNET_ROOT to a "
                "directory containing scans_test/<scene>/{color,depth,pose,intrinsic}. "
                "Request access at http://www.scan-net.org/."
            )
        self.root = root_path
        if split not in ("test",):
            raise ValueError(f"unsupported ScanNet split '{split}'; use 'test'")
        self.split = split
        self.frame_stride = max(1, int(frame_stride))
        self.views_per_sample = max(1, int(views_per_sample))

        manifest_path = (
            self.root
            / ".plumbline_manifest"
            / f"scannet_{split}_stride{self.frame_stride}_vps{self.views_per_sample}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan(scenes))
            save_manifest(manifest_path, records)
        if scenes:
            records = [r for r in records if r["scene"] in scenes]
        self._records = records

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            try:
                yield self._load_sample(rec)
            except _InvalidPoseFrame:
                # Drop frames with inf/NaN poses silently; log at debug level.
                continue

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self, scenes: list[str] | None) -> Iterator[dict]:
        split_root = self.root / "scans_test"
        if not split_root.exists():
            raise DatasetNotAvailable(f"Expected ScanNet split at {split_root}; not found.")
        scene_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
        if scenes is not None:
            wanted = set(scenes)
            scene_dirs = [p for p in scene_dirs if p.name in wanted]

        for scene_dir in scene_dirs:
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            pose_dir = scene_dir / "pose"
            intr_color = scene_dir / "intrinsic" / "intrinsic_color.txt"
            intr_depth = scene_dir / "intrinsic" / "intrinsic_depth.txt"
            if not (color_dir.exists() and pose_dir.exists()):
                continue
            frame_ids = sorted(int(f.stem) for f in color_dir.glob("*.jpg"))
            strided = frame_ids[:: self.frame_stride]
            for i in range(0, len(strided) - self.views_per_sample + 1):
                group = strided[i : i + self.views_per_sample]
                yield {
                    "sample_id": f"{scene_dir.name}/{group[0]:06d}_v{self.views_per_sample}",
                    "scene": scene_dir.name,
                    "frame_ids": group,
                    "color_paths": [
                        str((color_dir / f"{fid}.jpg").relative_to(self.root)) for fid in group
                    ],
                    "depth_paths": [
                        str((depth_dir / f"{fid}.png").relative_to(self.root)) for fid in group
                    ],
                    "pose_paths": [
                        str((pose_dir / f"{fid}.txt").relative_to(self.root)) for fid in group
                    ],
                    "intrinsic_color": str(intr_color.relative_to(self.root)),
                    "intrinsic_depth": str(intr_depth.relative_to(self.root)),
                }

    # -- per sample ------------------------------------------------------

    def _load_sample(self, rec: dict) -> Sample:
        images = np.stack([read_rgb_uint8(self.root / p) for p in rec["color_paths"]], axis=0)
        assert_valid_image(images, name=f"scannet/{rec['sample_id']}/image")

        depths: list[NDArray[np.float32]] = []
        for p in rec["depth_paths"]:
            depths.append(load_scannet_depth_mm_to_m(self.root / p))
        depth_gt = np.stack(depths).astype(np.float32)

        K_color = _load_scannet_intrinsic(self.root / rec["intrinsic_color"])
        K_stack = np.broadcast_to(K_color, (images.shape[0], 3, 3)).astype(np.float32)

        # ScanNet poses are camera_from_world; we rebase to world_from_camera
        # with first camera as world origin.
        camera_from_world: list[NDArray[np.float64]] = []
        for p in rec["pose_paths"]:
            pose = load_scannet_pose(self.root / p)
            if not np.all(np.isfinite(pose)):
                raise _InvalidPoseFrame(f"{rec['sample_id']}:{p}")
            camera_from_world.append(pose)
        world_from_camera = np.stack([invert_pose(p) for p in camera_from_world])
        extrinsics = rebase_to_first_camera(world_from_camera).astype(np.float32)

        assert_valid_intrinsics(K_stack, name=f"scannet/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(extrinsics, name=f"scannet/{rec['sample_id']}/extrinsics")
        # Depth and image may be at different resolutions, so depth_gt shape
        # differs. We still validate it.
        assert_valid_depth(depth_gt, name=f"scannet/{rec['sample_id']}/depth")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            metadata={
                "scene": rec["scene"],
                "frame_ids": rec["frame_ids"],
                "split": self.split,
                "depth_scale_mm_per_m": _DEPTH_SCALE_MM_PER_M,
            },
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------


class _InvalidPoseFrame(RuntimeError):  # noqa: N818 - internal sentinel
    pass


def load_scannet_depth_mm_to_m(path: Path) -> NDArray[np.float32]:
    """Load a ScanNet depth PNG (uint16 mm) and convert to float32 meters."""
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth from {path}, got {arr.dtype}")
    depth = arr.astype(np.float32) / _DEPTH_SCALE_MM_PER_M
    # ScanNet encodes invalid as 0; keep as 0 (our convention marks 0 invalid).
    return depth


def load_scannet_pose(path: Path) -> NDArray[np.float64]:
    """Load a 4x4 pose (camera_from_world) from a ScanNet ``.txt``.

    May contain ``-inf`` values on tracking-dropped frames; caller filters.
    """
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def _load_scannet_intrinsic(path: Path) -> NDArray[np.float64]:
    """ScanNet intrinsic txt is a 4x4; we take the upper 3x3."""
    K = np.loadtxt(path, dtype=np.float64)
    if K.shape == (4, 4):
        return K[:3, :3]
    if K.shape == (3, 3):
        return K
    raise ValueError(f"unexpected intrinsic shape in {path}: {K.shape}")
