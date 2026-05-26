"""Tests for the Bonn RGB-D Dynamic loader, using synthetic TUM-format fixtures.

No real Bonn data needed: we write a tiny ``rgbd_bonn_<name>/`` tree (rgb/,
depth/, rgb.txt, depth.txt, groundtruth.txt) and assert the loader produces a
one-sample-per-sequence Sample in plumbline conventions, with correct depth
scaling (5000 units/m), Bonn intrinsics, timestamp association, even
sub-sampling, and view-0-rebased poses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image

from plumbline.datasets._common import DatasetNotAvailable
from plumbline.datasets.bonn import (
    BONN_DEPTH_SCALE,
    BONN_INTRINSICS,
    BonnDataset,
)


def _make_sequence(root: Path, name: str, n: int = 6, h: int = 8, w: int = 10) -> None:
    seq = root / f"rgbd_bonn_{name}"
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)
    rgb_lines, depth_lines, traj_lines = [], [], []
    for i in range(n):
        ts = 1000.0 + i * 0.1  # 10 Hz
        rgb_rel = f"rgb/{ts:.4f}.png"
        depth_rel = f"depth/{ts:.4f}.png"
        # RGB: a distinct grey per frame so frames aren't all-equal.
        rgb = np.full((h, w, 3), 10 * i, dtype=np.uint8)
        Image.fromarray(rgb).save(seq / rgb_rel)
        # Depth: constant 2.0 m -> 2.0 * 5000 = 10000 units, plus a zero hole.
        d_units = np.full((h, w), int(2.0 * BONN_DEPTH_SCALE), dtype=np.uint16)
        d_units[0, 0] = 0  # invalid pixel
        Image.fromarray(d_units).save(seq / depth_rel)
        rgb_lines.append(f"{ts:.4f} {rgb_rel}")
        depth_lines.append(f"{ts:.4f} {depth_rel}")
        # Identity-ish trajectory: translate along +x by i metres.
        traj_lines.append(f"{ts:.4f} {float(i)} 0 0 0 0 0 1")
    (seq / "rgb.txt").write_text("# color images\n" + "\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("# depth images\n" + "\n".join(depth_lines) + "\n")
    (seq / "groundtruth.txt").write_text("# gt\n" + "\n".join(traj_lines) + "\n")


class TestBonnLoader:
    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotAvailable, match="Bonn"):
            BonnDataset(root=tmp_path / "nope")

    def test_empty_root_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(DatasetNotAvailable, match="No rgbd_bonn"):
            BonnDataset(root=tmp_path / "empty")

    def test_one_sample_per_sequence(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=6)
        _make_sequence(tmp_path, "crowd", n=6)
        ds = BonnDataset(root=tmp_path, num_frames=90)
        assert len(ds) == 2
        samples = list(ds)
        assert sorted(s.sample_id for s in samples) == ["balloon", "crowd"]

    def test_sample_shapes_and_depth_scale(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=6, h=8, w=10)
        ds = BonnDataset(root=tmp_path, num_frames=90)
        s = next(iter(ds))
        assert s.images.shape == (6, 8, 10, 3)
        assert s.images.dtype == np.uint8
        assert s.depth_gt is not None and s.depth_gt.shape == (6, 8, 10)
        # 10000 units / 5000 = 2.0 m everywhere except the hole.
        assert np.isclose(s.depth_gt[0, 1, 1], 2.0)
        assert s.depth_gt[0, 0, 0] == 0.0  # the zero-depth hole
        assert s.depth_valid is not None
        assert not s.depth_valid[0, 0, 0]
        assert s.depth_valid[0, 1, 1]

    def test_intrinsics_are_bonn_calibration(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=4)
        s = next(iter(BonnDataset(root=tmp_path)))
        fx, fy, cx, cy = BONN_INTRINSICS
        assert np.isclose(s.intrinsics[0, 0, 0], fx)
        assert np.isclose(s.intrinsics[0, 1, 1], fy)
        assert np.isclose(s.intrinsics[0, 0, 2], cx)
        assert np.isclose(s.intrinsics[0, 1, 2], cy)
        # Same intrinsics for every frame.
        assert np.allclose(s.intrinsics, s.intrinsics[0][None])

    def test_poses_rebased_to_first_frame(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=6)
        s = next(iter(BonnDataset(root=tmp_path, num_frames=90)))
        # Frame 0 is the world origin (plumbline convention).
        assert np.allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        # Trajectory translated +x by i metres; after rebasing to frame 0
        # (also at x=0) the relative translation is preserved.
        assert np.isclose(s.extrinsics_gt[-1][0, 3], 5.0, atol=1e-4)

    def test_num_frames_subsamples(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=20)
        s = next(iter(BonnDataset(root=tmp_path, num_frames=5)))
        assert s.images.shape[0] == 5
        assert s.metadata["n_frames"] == 5

    def test_explicit_sequence_selection(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=4)
        _make_sequence(tmp_path, "crowd", n=4)
        # With and without the rgbd_bonn_ prefix.
        ds = BonnDataset(root=tmp_path, sequences=["balloon", "rgbd_bonn_crowd"])
        assert sorted(s.sample_id for s in ds) == ["balloon", "crowd"]

    def test_registered(self) -> None:
        from plumbline.datasets.registry import DATASET_REGISTRY

        assert "bonn" in DATASET_REGISTRY

    def test_per_frame_emits_one_sample_per_frame(self, tmp_path: Path) -> None:
        # 2 sequences × 4 frames each (each frame has matching depth + pose),
        # per_frame=True should yield 8 single-frame Samples, not 2 sequences.
        _make_sequence(tmp_path, "balloon", n=4)
        _make_sequence(tmp_path, "crowd", n=4)
        ds = BonnDataset(root=tmp_path, per_frame=True)
        assert len(ds) == 8
        samples = list(ds)
        assert len(samples) == 8
        for s in samples:
            assert s.images.shape[0] == 1
            assert s.depth_gt.shape[0] == 1
            assert s.metadata["per_frame"] is True
            assert s.metadata["n_frames"] == 1
        # sample_ids are "<seq>/<rgb-stem>"
        seqs = {s.metadata["sequence"] for s in samples}
        assert seqs == {"balloon", "crowd"}

    def test_per_frame_ignores_num_frames(self, tmp_path: Path) -> None:
        _make_sequence(tmp_path, "balloon", n=6)
        # num_frames=2 (sequence mode) would emit a single 2-frame Sample;
        # per_frame=True ignores num_frames and emits all 6 frames.
        ds = BonnDataset(root=tmp_path, per_frame=True, num_frames=2)
        assert len(ds) == 6
