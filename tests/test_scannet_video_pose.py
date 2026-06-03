"""Tests for the ScanNet video-pose loader, using synthetic fixtures.

No real ScanNet needed: we write a tiny ``<scene>/color_90/`` + ``pose_90.txt``
tree and assert the loader (a) parses flattened-4x4 poses, (b) drops
non-finite (dropped-tracker) frames together with their images, (c)
view-0-rebases the trajectory, and (d) emits one Sample per scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image

from plumbline.datasets._common import DatasetNotAvailable
from plumbline.datasets.scannet_video_pose import (
    ScanNetVideoPoseDataset,
    load_scannet_pose_90,
)


def _make_scene(root: Path, scene: str, n: int = 5, with_inf_at: int | None = None) -> None:
    sd = root / scene
    (sd / "color_90").mkdir(parents=True)
    lines = []
    for i in range(n):
        Image.fromarray(np.full((8, 10, 3), (9 * i) % 256, dtype=np.uint8)).save(
            sd / "color_90" / f"frame_{i:04d}.jpg"
        )
        E = np.eye(4)
        E[0, 3] = float(i)  # translate +x by i metres
        if with_inf_at is not None and i == with_inf_at:
            E[:] = np.inf  # dropped-tracker frame
        lines.append(" ".join(map(str, E.reshape(-1))))
    (sd / "pose_90.txt").write_text("\n".join(lines) + "\n")


def test_load_pose_90_shapes(tmp_path: Path) -> None:
    _make_scene(tmp_path, "scene0707_00", n=4)
    poses = load_scannet_pose_90(tmp_path / "scene0707_00" / "pose_90.txt")
    assert poses.shape == (4, 4, 4)


def test_missing_root_raises() -> None:
    with pytest.raises(DatasetNotAvailable):
        ScanNetVideoPoseDataset(root="/nonexistent/scannet/xyz")


def test_one_trajectory_per_scene(tmp_path: Path) -> None:
    _make_scene(tmp_path, "scene0707_00", n=5)
    ds = ScanNetVideoPoseDataset(root=tmp_path, scenes=["scene0707_00"])
    samples = list(ds)
    assert len(samples) == 1
    s = samples[0]
    assert s.images.shape[0] == 5
    assert s.extrinsics_gt.shape == (5, 4, 4)
    assert s.depth_gt is None
    assert np.allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
    assert np.allclose(s.extrinsics_gt[:, 0, 3], [0, 1, 2, 3, 4], atol=1e-4)


def test_drops_nonfinite_pose_frames(tmp_path: Path) -> None:
    # 5 frames, frame 2 has an inf pose -> dropped (image + pose), 4 remain.
    _make_scene(tmp_path, "scene0708_00", n=5, with_inf_at=2)
    ds = ScanNetVideoPoseDataset(root=tmp_path, scenes=["scene0708_00"])
    s = next(iter(ds))
    assert s.images.shape[0] == 4
    assert s.extrinsics_gt.shape == (4, 4, 4)
    # Kept x-translations were 0,1,3,4 (frame 2 dropped), rebased to frame 0.
    assert np.allclose(s.extrinsics_gt[:, 0, 3], [0, 1, 3, 4], atol=1e-4)
