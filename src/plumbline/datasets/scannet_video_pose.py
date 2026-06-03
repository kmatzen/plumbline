"""ScanNet-v2 video-pose loader (MonST3R / DAGE Table 4 protocol).

Serves ScanNet-v2 sequences as the camera-trajectory benchmark reported by
MonST3R Table 4 and **DAGE Table 4** (Ngo et al. 2026, arXiv:2603.03744),
"ScanNet" column. One :class:`Sample` is one scene's 90-frame trajectory; ATE /
RPE-RMSE are computed under Sim(3) (Umeyama) alignment via ``evo`` — the same
apparatus plumbline already runs for ``dage-sintel-pose`` (only the dataset
differs).

This is the **video-pose** view of ScanNet, distinct from the depth-eval
``scannet`` loader: it reads the MonST3R-preprocessed ``color_90`` / ``pose_90``
layout (``datasets_preprocess/prepare_scannet.py``), which DAGE's
``evaluation/relpose`` reuses unchanged (DAGE ``metadata.py``:
``scannet: {seq_list: None, full_seq: True, traj_format: 'replica'}``):

- First 90 frames at temporal stride 3 (``img_pathes[:90*3:3]``), renamed
  ``frame_0000.jpg`` …
- ``pose_90.txt``: one line per frame = the flattened 4x4 ScanNet
  ``pose/<i>.txt`` (camera-to-world == ``world_from_camera``). Dropped-tracker
  frames carry ``inf`` / ``-inf`` and are filtered (frame + pose removed
  together so the trajectory stays aligned).

Expected on-disk layout under ``--data-root`` or ``$SCANNET_VIDEO_ROOT``::

    <root>/<scene>/color_90/frame_0000.jpg ...
    <root>/<scene>/pose_90.txt
    <root>/<scene>/intrinsic/intrinsic_color.txt   # optional

DATA NOTE — raw ScanNet is **ToS-gated** (sign at http://www.scan-net.org/).
MonST3R/DAGE use scenes ``scene0707_00`` … ``scene0806_00`` (100 test-split
scenes; ``download_scannetv2.sh``). Staging means downloading those ``.sens``
files, extracting color/pose, then running ``prepare_scannet.py``. This loader
is therefore code-ready but data-blocked until that layout exists on disk.

Pose convention
---------------
ScanNet ``pose/<i>.txt`` is ``camera_to_world`` (== plumbline
``world_from_camera``). We parse the flattened 4x4, drop non-finite frames, then
:func:`rebase_to_first_camera` (absorbed by the metric's Sim(3) alignment, so it
matches MonST3R's evo eval regardless).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
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

__all__ = ["ScanNetVideoPoseDataset", "load_scannet_pose_90"]

# Fallback ScanNet color intrinsics (1296x968), used only if a scene has no
# intrinsic/intrinsic_color.txt. Pose eval is feed-forward + Sim(3)-aligned, so K
# is not used by the metric, but Sample requires a valid K per frame.
_DEFAULT_COLOR_K = np.array(
    [[1170.19, 0.0, 647.75], [0.0, 1170.19, 483.75], [0.0, 0.0, 1.0]], dtype=np.float32
)


def load_scannet_pose_90(path: Path) -> NDArray[np.float64]:
    """Parse a ``pose_90.txt`` → ``(N, 4, 4)`` world_from_camera (with non-finite rows kept).

    Each line is the row-major flattened 4x4 ScanNet pose. Filtering of
    non-finite (dropped-tracker) frames is left to the caller so it can drop the
    matching image too.
    """
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) != 16:
            raise ValueError(f"{path}: expected 16 floats per pose line, got {len(vals)}")
        rows.append(np.asarray(vals, dtype=np.float64).reshape(4, 4))
    return np.stack(rows) if rows else np.empty((0, 4, 4), dtype=np.float64)


@register_dataset("scannet-video-pose")
class ScanNetVideoPoseDataset(Dataset):
    """ScanNet-v2 video-pose loader (one 90-frame trajectory Sample per scene).

    Pose-only (``depth_gt`` is left ``None``). Reads the MonST3R-preprocessed
    ``color_90`` / ``pose_90`` layout.

    Parameters
    ----------
    root
        Directory holding ``<scene>/color_90/`` + ``<scene>/pose_90.txt``.
        Falls back to ``$SCANNET_VIDEO_ROOT``.
    scenes
        Optional subset of scene names. Default: all scenes present on disk.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        scenes: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("SCANNET_VIDEO_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ScanNet video-pose not found. Set --data-root or $SCANNET_VIDEO_ROOT "
                "to a directory of <scene>/color_90/ + <scene>/pose_90.txt (the "
                "MonST3R prepare_scannet.py layout). Raw ScanNet is ToS-gated "
                "(http://www.scan-net.org/); scenes scene0707_00..scene0806_00."
            )
        self.root = root_path
        self.split = split
        self._wanted = list(scenes) if scenes else None

        # Key the manifest on the set of scenes present, so staging more scenes
        # after a first (partial) scan invalidates the cache instead of silently
        # serving the old, smaller set.
        present = sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / "color_90").exists() and (p / "pose_90.txt").exists()
        )
        tag = hashlib.sha256("|".join(present).encode()).hexdigest()[:12]
        manifest_path = self.root / ".plumbline_manifest" / f"scannet_video_pose_{tag}.jsonl"
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan())
            save_manifest(manifest_path, records)
        if self._wanted is not None:
            wanted = set(self._wanted)
            records = [r for r in records if r["scene"] in wanted]
        self._records = records
        if not self._records:
            raise DatasetNotAvailable(
                f"No ScanNet video-pose scenes found under {self.root}. "
                "Expected <scene>/color_90/frame_*.jpg + <scene>/pose_90.txt."
            )

    def _scan(self) -> Iterator[dict[str, Any]]:
        for scene_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            color_dir = scene_dir / "color_90"
            pose_txt = scene_dir / "pose_90.txt"
            if not (color_dir.exists() and pose_txt.exists()):
                continue
            frames = sorted(color_dir.glob("frame_*.jpg"))
            if len(frames) < 3:
                continue
            yield {
                "sample_id": f"{scene_dir.name}/full_s3",
                "scene": scene_dir.name,
                "image_paths": [str(f.relative_to(self.root)) for f in frames],
                "pose_path": str(pose_txt.relative_to(self.root)),
                "intr_path": str(scene_dir / "intrinsic" / "intrinsic_color.txt"),
            }

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        poses = load_scannet_pose_90(self.root / rec["pose_path"])
        image_paths = rec["image_paths"]
        if poses.shape[0] != len(image_paths):
            raise ValueError(
                f"{rec['scene']}: {poses.shape[0]} poses vs {len(image_paths)} frames"
            )
        # Drop dropped-tracker frames (non-finite pose) — image + pose together.
        finite = np.array([bool(np.all(np.isfinite(p))) for p in poses])
        if finite.sum() < 3:
            raise ValueError(f"{rec['scene']}: <3 finite-pose frames")
        kept_imgs = [p for p, ok in zip(image_paths, finite, strict=True) if ok]
        kept_poses = poses[finite]

        images = np.stack([read_rgb_uint8(self.root / p) for p in kept_imgs], axis=0)
        assert_valid_image(images, name=f"scannet-video/{rec['scene']}/image")

        extrinsics = rebase_to_first_camera(kept_poses).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name=f"scannet-video/{rec['scene']}/E")

        n = images.shape[0]
        intr_path = Path(rec["intr_path"])
        K = (
            np.loadtxt(intr_path).astype(np.float32)[:3, :3]
            if intr_path.exists()
            else _DEFAULT_COLOR_K
        )
        intrinsics = np.broadcast_to(K, (n, 3, 3)).copy()
        assert_valid_intrinsics(intrinsics, name=f"scannet-video/{rec['scene']}/K")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={"scene": rec["scene"], "dataset": "scannet-video-pose", "n_frames": n},
        )
