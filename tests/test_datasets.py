"""Unit tests for dataset loaders using synthetic on-disk fixtures.

These exercise the loader logic end-to-end — scanning, manifest creation,
sample construction, coordinate conversions — without requiring the real
datasets to be downloaded.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plumbline.datasets.eth3d import (
    ETH3DDataset,
    parse_colmap_cameras,
    parse_colmap_images,
    quat_to_rot,
)
from plumbline.datasets.scannet import ScanNetDataset, load_scannet_pose
from plumbline.datasets.sintel import SintelDataset, load_cam, load_dpt

# ---------------------------------------------------------------------------
# Sintel
# ---------------------------------------------------------------------------

_SINTEL_TAG = 202021.25


def _write_fake_sintel(root: Path, *, scenes: int = 1, frames: int = 3) -> None:
    for s in range(scenes):
        scene = f"scene_{s}"
        img_dir = root / "training" / "final" / scene
        depth_dir = root / "training" / "depth" / scene
        cam_dir = root / "training" / "camdata_left" / scene
        img_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        cam_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(frames):
            name = f"frame_{fi:04d}"
            Image.fromarray((np.random.rand(16, 32, 3) * 255).astype(np.uint8)).save(
                img_dir / f"{name}.png"
            )
            # .dpt
            depth = np.random.rand(16, 32).astype(np.float32) + 0.5
            with open(depth_dir / f"{name}.dpt", "wb") as f:
                f.write(struct.pack("<f", _SINTEL_TAG))
                f.write(struct.pack("<ii", 32, 16))
                f.write(depth.tobytes())
            # .cam: 3x3 K (float64) then 3x4 [R|t] (float64)
            K = np.array([[500, 0, 16], [0, 500, 8], [0, 0, 1]], dtype=np.float64)
            R = np.eye(3, dtype=np.float64)
            t = np.array([0.1 * fi, 0.0, 0.0], dtype=np.float64)
            RT = np.concatenate([R, t[:, None]], axis=1)
            with open(cam_dir / f"{name}.cam", "wb") as f:
                f.write(K.tobytes())
                f.write(RT.tobytes())


class TestSintel:
    def test_missing_root_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            SintelDataset(root=tmp_path / "nope")

    def test_basic_load(self, tmp_path: Path) -> None:
        _write_fake_sintel(tmp_path, scenes=1, frames=3)
        ds = SintelDataset(root=tmp_path)
        assert len(ds) == 3
        samples = list(ds)
        assert len(samples) == 3
        s0 = samples[0]
        assert s0.num_views == 1
        assert s0.images.shape == (1, 16, 32, 3)
        assert s0.depth_gt is not None and s0.depth_gt.shape == (1, 16, 32)
        np.testing.assert_allclose(s0.extrinsics_gt[0], np.eye(4), atol=1e-5)

    def test_views_per_sample(self, tmp_path: Path) -> None:
        _write_fake_sintel(tmp_path, scenes=1, frames=4)
        ds = SintelDataset(root=tmp_path, views_per_sample=2)
        assert len(ds) == 3  # sliding window length-2 over 4 frames
        s = next(iter(ds))
        assert s.num_views == 2
        # First camera should be identity in world frame.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)

    def test_manifest_written(self, tmp_path: Path) -> None:
        _write_fake_sintel(tmp_path, scenes=1, frames=3)
        SintelDataset(root=tmp_path)
        assert (tmp_path / ".plumbline_manifest").exists()

    def test_load_dpt_round_trip(self, tmp_path: Path) -> None:
        # Sanity-check the parser in isolation.
        depth = (np.arange(32, dtype=np.float32) / 31.0).reshape(4, 8)
        path = tmp_path / "x.dpt"
        with open(path, "wb") as f:
            f.write(struct.pack("<f", _SINTEL_TAG))
            f.write(struct.pack("<ii", 8, 4))
            f.write(depth.tobytes())
        loaded = load_dpt(path)
        np.testing.assert_array_equal(loaded, depth)

    def test_load_cam(self, tmp_path: Path) -> None:
        K = np.array([[500, 0, 160], [0, 500, 120], [0, 0, 1]], dtype=np.float64)
        RT = np.eye(4)[:3, :4].astype(np.float64)
        path = tmp_path / "x.cam"
        with open(path, "wb") as f:
            f.write(K.tobytes())
            f.write(RT.tobytes())
        K_loaded, E_loaded = load_cam(path)
        np.testing.assert_allclose(K_loaded, K, atol=1e-5)
        np.testing.assert_allclose(E_loaded[:3, :4], RT, atol=1e-5)


# ---------------------------------------------------------------------------
# ScanNet
# ---------------------------------------------------------------------------


def _write_fake_scannet(root: Path, *, scenes: int = 1, frames: int = 3) -> None:
    for s in range(scenes):
        scene = f"scene{s:04d}_00"
        scene_dir = root / "scans_test" / scene
        (scene_dir / "color").mkdir(parents=True, exist_ok=True)
        (scene_dir / "depth").mkdir(parents=True, exist_ok=True)
        (scene_dir / "pose").mkdir(parents=True, exist_ok=True)
        (scene_dir / "intrinsic").mkdir(parents=True, exist_ok=True)

        # Intrinsics (color): 4x4 txt with upper-left 3x3.
        K4 = np.eye(4)
        K4[:3, :3] = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        np.savetxt(scene_dir / "intrinsic" / "intrinsic_color.txt", K4)
        np.savetxt(scene_dir / "intrinsic" / "intrinsic_depth.txt", K4)

        for fi in range(frames):
            Image.fromarray((np.random.rand(8, 16, 3) * 255).astype(np.uint8)).save(
                scene_dir / "color" / f"{fi}.jpg", quality=95
            )
            depth_mm = (np.random.rand(8, 16) * 1000 + 500).astype(np.uint16)
            # Pillow infers 'I;16' from uint16; explicit mode= is deprecated in 12+.
            Image.fromarray(depth_mm).save(scene_dir / "depth" / f"{fi}.png")
            # camera_from_world identity (with tiny translation).
            pose = np.eye(4)
            pose[:3, 3] = [0.05 * fi, 0.0, 0.0]
            np.savetxt(scene_dir / "pose" / f"{fi}.txt", pose)


class TestScanNet:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            ScanNetDataset(root=tmp_path / "nope")

    def test_basic_load(self, tmp_path: Path) -> None:
        _write_fake_scannet(tmp_path, scenes=1, frames=5)
        ds = ScanNetDataset(root=tmp_path, frame_stride=2)
        samples = list(ds)
        assert len(samples) == 3  # frames 0, 2, 4
        s0 = samples[0]
        assert s0.depth_gt is not None
        # Depth was uint16 mm; loader converts to float32 meters.
        assert s0.depth_gt.dtype == np.float32
        assert 0.5 <= s0.depth_gt.min() <= 1.5

    def test_inf_pose_filtered(self, tmp_path: Path) -> None:
        _write_fake_scannet(tmp_path, scenes=1, frames=3)
        pose_path = tmp_path / "scans_test" / "scene0000_00" / "pose" / "1.txt"
        pose = np.full((4, 4), -np.inf, dtype=np.float64)
        np.savetxt(pose_path, pose)
        ds = ScanNetDataset(root=tmp_path, frame_stride=1)
        samples = list(ds)
        ids = [s.sample_id for s in samples]
        # The broken frame should be dropped silently.
        assert all("/000001_" not in sid for sid in ids)

    def test_load_pose(self, tmp_path: Path) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        p = tmp_path / "p.txt"
        np.savetxt(p, pose)
        loaded = load_scannet_pose(p)
        np.testing.assert_allclose(loaded, pose)


# ---------------------------------------------------------------------------
# ETH3D
# ---------------------------------------------------------------------------


def _write_fake_eth3d(root: Path, *, scenes: int = 1, views: int = 3) -> None:
    for s in range(scenes):
        scene = f"scene_{s}"
        calib = root / scene / "dslr_calibration_undistorted"
        (root / scene / "images").mkdir(parents=True, exist_ok=True)
        calib.mkdir(parents=True, exist_ok=True)

        # cameras.txt
        with (calib / "cameras.txt").open("w") as f:
            f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
            f.write("1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n")
        # images.txt (qw,qx,qy,qz,tx,ty,tz,camera_id,name followed by empty line
        # for 2D points)
        with (calib / "images.txt").open("w") as f:
            f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D\n")
            for i in range(views):
                f.write(f"{i + 1} 1.0 0.0 0.0 0.0 {i * 0.1} 0.0 0.0 1 img_{i}.JPG\n")
                f.write("\n")  # empty 2D-points line
                Image.fromarray((np.random.rand(8, 16, 3) * 255).astype(np.uint8)).save(
                    root / scene / "images" / f"img_{i}.JPG", quality=85
                )


class TestETH3D:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            ETH3DDataset(root=tmp_path / "nope")

    def test_basic_load(self, tmp_path: Path) -> None:
        _write_fake_eth3d(tmp_path, scenes=1, views=4)
        ds = ETH3DDataset(root=tmp_path, views_per_sample=3)
        samples = list(ds)
        assert len(samples) == 2  # sliding window of 3 over 4
        s = samples[0]
        assert s.num_views == 3
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)

    def test_parse_colmap_cameras_pinhole(self, tmp_path: Path) -> None:
        p = tmp_path / "cameras.txt"
        p.write_text("# comment\n1 PINHOLE 64 48 400 400 32 24\n")
        cams = parse_colmap_cameras(p)
        K = cams[1]
        assert K[0, 0] == 400 and K[1, 1] == 400 and K[0, 2] == 32 and K[1, 2] == 24

    def test_parse_colmap_cameras_simple_pinhole(self, tmp_path: Path) -> None:
        p = tmp_path / "cameras.txt"
        p.write_text("2 SIMPLE_PINHOLE 10 10 500 5 5\n")
        cams = parse_colmap_cameras(p)
        assert cams[2][0, 0] == 500 and cams[2][1, 1] == 500

    def test_parse_colmap_cameras_rejects_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "cameras.txt"
        p.write_text("1 OPENCV 10 10 1 1 1 1 1 1\n")
        with pytest.raises(ValueError, match="PINHOLE"):
            parse_colmap_cameras(p)

    def test_parse_colmap_images_skips_points2d(self, tmp_path: Path) -> None:
        p = tmp_path / "images.txt"
        p.write_text(
            "# comment\n"
            "1 1 0 0 0 1 2 3 4 a.JPG\n"
            "0 1 0 1 2\n"  # points2d (ignored)
            "2 1 0 0 0 5 6 7 4 b.JPG\n"
            "8 8 8 8 8\n"
        )
        records = parse_colmap_images(p)
        assert [r["name"] for r in records] == ["a.JPG", "b.JPG"]
        assert records[0]["tx"] == 1 and records[1]["tx"] == 5

    def test_quat_to_rot_identity(self) -> None:
        R = quat_to_rot(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_quat_to_rot_normalizes(self) -> None:
        R = quat_to_rot(np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
