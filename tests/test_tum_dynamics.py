"""Tests for the TUM-dynamics video-pose loader, using synthetic fixtures.

No real TUM data needed: we write a tiny ``rgbd_dataset_freiburg3_<name>/`` tree
(rgb/, rgb.txt, groundtruth.txt) and assert the loader (a) associates rgb↔gt
timestamps the way MonST3R's prepare_tum.py does, (b) takes the first-90 frames
at stride 3, (c) converts TUM (translation, quaternion) ground truth to
view-0-rebased ``world_from_camera``, and (d) emits one trajectory Sample per
sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image

from plumbline.datasets._common import DatasetNotAvailable
from plumbline.datasets.tum_dynamics import (
    TUM_DYNAMIC_SEQUENCES,
    TUMDynamicsDataset,
    associate_tum,
    tum_pose_to_matrix,
)


def _make_sequence(root: Path, name: str, n: int = 15, h: int = 8, w: int = 10) -> None:
    """Write a minimal TUM sequence: rgb/, rgb.txt, groundtruth.txt."""
    seq = root / name
    (seq / "rgb").mkdir(parents=True)
    rgb_lines, gt_lines = [], []
    for i in range(n):
        ts = 1305031910.0 + i * 0.0333  # ~30 Hz
        rgb_rel = f"rgb/{ts:.4f}.png"
        Image.fromarray(np.full((h, w, 3), (7 * i) % 256, dtype=np.uint8)).save(seq / rgb_rel)
        rgb_lines.append(f"{ts:.4f} {rgb_rel}")
        # Ground truth at a slightly offset timestamp (within the 0.02s window),
        # identity rotation, translating +x by i metres.
        gt_ts = ts + 0.005
        gt_lines.append(f"{gt_ts:.4f} {float(i)} 0 0 0 0 0 1")
    (seq / "rgb.txt").write_text("# color images\n" + "\n".join(rgb_lines) + "\n")
    (seq / "groundtruth.txt").write_text("# ground truth\n" + "\n".join(gt_lines) + "\n")


def test_associate_picks_nearest_within_window() -> None:
    first = {1.00: ["a"], 1.10: ["b"], 1.20: ["c"]}
    second = {1.005: ["x"], 1.30: ["y"]}  # only 1.005 is within 0.02 of a first key
    matches = associate_tum(first, second)
    assert matches == [(1.00, 1.005)]


def test_quaternion_identity_is_identity() -> None:
    E = tum_pose_to_matrix(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    assert np.allclose(E[:3, :3], np.eye(3))
    assert np.allclose(E[:3, 3], [1.0, 2.0, 3.0])


def test_quaternion_90deg_z() -> None:
    # 90° about +z: (qx,qy,qz,qw) = (0,0,sin45,cos45).
    s = np.sqrt(0.5)
    E = tum_pose_to_matrix(0, 0, 0, 0, 0, s, s)
    # +x axis maps to +y.
    assert np.allclose(E[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)


def test_missing_root_raises() -> None:
    with pytest.raises(DatasetNotAvailable):
        TUMDynamicsDataset(root="/nonexistent/tum/xyz")


def test_loader_one_trajectory_per_sequence(tmp_path: Path) -> None:
    seq_name = TUM_DYNAMIC_SEQUENCES[0]
    _make_sequence(tmp_path, seq_name, n=15)
    ds = TUMDynamicsDataset(root=tmp_path, scenes=[seq_name])
    samples = list(ds)
    assert len(samples) == 1
    s = samples[0]
    # 15 frames -> [::3] = 5 frames (indices 0,3,6,9,12), then [:90] keeps 5.
    assert s.images.shape[0] == 5
    assert s.extrinsics_gt.shape == (5, 4, 4)
    assert s.intrinsics.shape == (5, 3, 3)
    assert s.depth_gt is None
    # View 0 is rebased to identity.
    assert np.allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
    # Translation grows along +x by the stride-3 frame indices (0,3,6,9,12 m),
    # rebased so frame 0 is the origin.
    xs = s.extrinsics_gt[:, 0, 3]
    assert np.allclose(xs, [0.0, 3.0, 6.0, 9.0, 12.0], atol=1e-4)


def test_loader_defaults_to_all_present_sequences(tmp_path: Path) -> None:
    _make_sequence(tmp_path, TUM_DYNAMIC_SEQUENCES[0], n=9)
    _make_sequence(tmp_path, TUM_DYNAMIC_SEQUENCES[1], n=9)
    ds = TUMDynamicsDataset(root=tmp_path)
    assert len(list(ds)) == 2
