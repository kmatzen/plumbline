"""Tests for the NYUv2 loader using a synthetic HDF5 fixture."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from plumbline.datasets._common import DatasetNotAvailable
from plumbline.datasets.nyuv2 import NYUv2Dataset, load_eigen_test_indices


def _write_fake_nyuv2(root: Path, n: int = 4) -> Path:
    """Write a synthetic nyu_depth_v2_labeled.mat-alike HDF5 file.

    Layout mirrors NYU's v7.3 MAT: ``images`` (N, 3, 640, 480) uint8 and
    ``depths`` (N, 640, 480) float32.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / "nyu_depth_v2_labeled.mat"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images",
            data=(rng.random((n, 3, 640, 480)) * 255).astype(np.uint8),
        )
        # Depth in meters with a plausible distribution; no zero-depth invalid.
        f.create_dataset(
            "depths",
            data=(rng.random((n, 640, 480)) * 5.0 + 0.5).astype(np.float32),
        )
    return path


class TestEigenIndices:
    def test_bundled_file_has_654_entries(self) -> None:
        idx = load_eigen_test_indices()
        assert len(idx) == 654
        assert all(0 <= i <= 1448 for i in idx)
        # Spot-check first + last — must match the canonical splits.mat.
        assert idx[0] == 0
        assert idx[-1] == 1448


class TestNYUv2Dataset:
    def test_missing_root_errors(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotAvailable):
            NYUv2Dataset(root=tmp_path / "nope")

    def test_missing_mat_errors(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotAvailable, match=r"nyu_depth_v2_labeled\.mat"):
            NYUv2Dataset(root=tmp_path)

    def test_all_split_loads_every_sample(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=4)
        # Our fake has only 4 samples, so "all" iterates up to 1449 indices;
        # most will fail because the HDF5 has fewer entries. Use custom indices
        # matching what the fixture provides.
        ds = NYUv2Dataset(root=tmp_path, indices=[0, 1, 2, 3])
        samples = list(ds)
        assert len(samples) == 4
        s0 = samples[0]
        assert s0.images.shape == (1, 480, 640, 3)
        assert s0.depth_gt is not None and s0.depth_gt.shape == (1, 480, 640)
        assert s0.intrinsics.shape == (1, 3, 3)
        assert s0.extrinsics_gt.shape == (1, 4, 4)
        # Extrinsics should be identity.
        np.testing.assert_allclose(s0.extrinsics_gt[0], np.eye(4), atol=1e-6)

    def test_custom_indices_preserved(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=5)
        ds = NYUv2Dataset(root=tmp_path, indices=[2, 0, 4])
        ids = [s.sample_id for s in ds]
        assert ids == ["nyuv2_00002", "nyuv2_00000", "nyuv2_00004"]

    def test_intrinsics_match_silberman(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=1)
        ds = NYUv2Dataset(root=tmp_path, indices=[0])
        s = next(iter(ds))
        K = s.intrinsics[0]
        # Silberman NYUv2 color: fx=518.8579, fy=519.4696, cx=325.5824, cy=253.7362.
        assert K[0, 0] == pytest.approx(518.8579, abs=1e-3)
        assert K[1, 1] == pytest.approx(519.4696, abs=1e-3)
        assert K[0, 2] == pytest.approx(325.5824, abs=1e-3)
        assert K[1, 2] == pytest.approx(253.7362, abs=1e-3)

    def test_image_dtype_and_range(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=1)
        ds = NYUv2Dataset(root=tmp_path, indices=[0])
        s = next(iter(ds))
        assert s.images.dtype == np.uint8
        assert s.images.min() >= 0 and s.images.max() <= 255
        assert s.depth_gt is not None
        assert s.depth_gt.dtype == np.float32
        # All depths positive in our fake fixture.
        assert float(s.depth_gt.min()) > 0

    def test_bad_split_errors(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=1)
        with pytest.raises(ValueError, match="unsupported"):
            NYUv2Dataset(root=tmp_path, split="bogus")

    def test_eigen_crop_mask_applied(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=1)
        ds = NYUv2Dataset(root=tmp_path, indices=[0], apply_eigen_crop=True)
        s = next(iter(ds))
        assert s.depth_valid is not None
        assert s.depth_valid.shape == (1, 480, 640)
        mask = s.depth_valid[0]
        # Top rows excluded by the crop.
        assert not mask[0:45].any()
        # Bottom rows excluded.
        assert not mask[471:].any()
        # Left / right columns excluded.
        assert not mask[:, 0:41].any()
        assert not mask[:, 601:].any()
        # Interior rectangle is included where depth > 0 (all of it in our fake).
        assert mask[45:471, 41:601].all()

    def test_no_eigen_crop_leaves_mask_none(self, tmp_path: Path) -> None:
        _write_fake_nyuv2(tmp_path, n=1)
        ds = NYUv2Dataset(root=tmp_path, indices=[0])
        s = next(iter(ds))
        assert s.depth_valid is None
