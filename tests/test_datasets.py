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

from plumbline.datasets.diode import (
    DIODE_INTRINSIC,
    DIODEDataset,
    load_diode_depth_m,
    load_diode_depth_mask,
)
from plumbline.datasets.dtu import (
    DTU_MVS_TEST_SCANS,
    DTUDataset,
    load_dtu_cam,
)
from plumbline.datasets.eth3d import (
    ETH3DDataset,
    parse_colmap_cameras,
    parse_colmap_images,
    quat_to_rot,
)
from plumbline.datasets.kitti import (
    KITTIDataset,
    eigen_crop_mask,
    garg_crop_mask,
    load_kitti_calib,
    load_kitti_depth_png_to_m,
    parse_eigen_sample_list,
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


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------


def _write_fake_kitti(
    root: Path,
    *,
    date: str = "2011_09_26",
    drive_ids: tuple[int, ...] = (2,),
    frames: int = 3,
    depth_split: str = "val",
    camera: str = "image_02",
    H: int = 16,
    W: int = 32,
) -> list[tuple[str, str, str]]:
    """Lay out a minimal KITTI-shaped tree and return the (drive, frame, cam) entries."""
    entries: list[tuple[str, str, str]] = []
    date_dir = root / "raw" / date
    date_dir.mkdir(parents=True, exist_ok=True)

    # calib_cam_to_cam.txt — only the P_rect_<NN> lines matter for the loader.
    # Values use fx=fy=100, cx=W/2, cy=H/2 (baseline term set to 0 for image_02
    # / image_03 on a rectified rig is fine for this synthetic test).
    calib_lines = [
        "calib_time: synthetic",
        f"P_rect_02: 100 0 {W / 2} 0 0 100 {H / 2} 0 0 0 1 0",
        f"P_rect_03: 100 0 {W / 2} 0 0 100 {H / 2} 0 0 0 1 0",
    ]
    (date_dir / "calib_cam_to_cam.txt").write_text("\n".join(calib_lines) + "\n")

    for drive_id in drive_ids:
        drive = f"{date}_drive_{drive_id:04d}_sync"
        img_dir = root / "raw" / date / drive / camera / "data"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir = (
            root / "depth_annotated" / depth_split / drive / "proj_depth" / "groundtruth" / camera
        )
        gt_dir.mkdir(parents=True, exist_ok=True)

        for fi in range(frames):
            frame_id = f"{fi:010d}"
            Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
                img_dir / f"{frame_id}.png"
            )
            # Depth: uint16, meters * 256. Use a nontrivial pattern with some
            # zero (invalid) pixels so the loader's 0-preservation is testable.
            depth_m = np.full((H, W), 5.0, dtype=np.float32)
            depth_m[0, :] = 0.0  # simulate invalid top row
            depth_u16 = (depth_m * 256.0).astype(np.uint16)
            Image.fromarray(depth_u16).save(gt_dir / f"{frame_id}.png")
            entries.append((drive, frame_id, camera))
    return entries


class TestKITTI:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            KITTIDataset(root=tmp_path / "nope")

    def test_missing_raw_subtree(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        (tmp_path / "depth_annotated").mkdir()
        with pytest.raises(DatasetNotAvailable, match="raw tree"):
            KITTIDataset(root=tmp_path)

    def test_missing_depth_subtree(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        (tmp_path / "raw").mkdir()
        with pytest.raises(DatasetNotAvailable, match="annotated-depth"):
            KITTIDataset(root=tmp_path)

    def test_invalid_camera(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir()
        (tmp_path / "depth_annotated").mkdir()
        with pytest.raises(ValueError, match="image_02"):
            KITTIDataset(root=tmp_path, camera="rgb")

    def test_scan_basic_load(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, frames=3)
        ds = KITTIDataset(root=tmp_path)
        samples = list(ds)
        assert len(samples) == 3
        s = samples[0]
        assert s.num_views == 1
        assert s.images.shape == (1, 16, 32, 3)
        assert s.depth_gt is not None and s.depth_gt.shape == (1, 16, 32)
        # Depth round-trip: 5.0 m encoded as uint16 and decoded should return 5.0.
        assert abs(float(s.depth_gt[0, -1, -1]) - 5.0) < 1e-3
        # Invalid pixels should remain exactly 0 after decoding.
        assert float(s.depth_gt[0, 0, 0]) == 0.0
        # Extrinsics identity; intrinsics match the synthetic P_rect_02.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        assert s.intrinsics[0, 0, 0] == 100 and s.intrinsics[0, 1, 1] == 100

    def test_apply_garg_crop_sets_depth_valid(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, frames=1, H=100, W=200)
        ds = KITTIDataset(root=tmp_path, apply_garg_crop=True)
        s = next(iter(ds))
        assert s.depth_valid is not None
        assert s.depth_valid.shape == (1, 100, 200)
        # Garg crop excludes the top ~40% of rows — our synthetic invalid top
        # row (y=0) is guaranteed to be outside the crop.
        assert not bool(s.depth_valid[0, 0, 100])
        # A row near the bottom, middle column should be inside.
        assert bool(s.depth_valid[0, 90, 100])
        assert s.metadata["crop"] == "garg"

    def test_apply_eigen_crop_sets_depth_valid(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, frames=1, H=100, W=200)
        ds = KITTIDataset(root=tmp_path, apply_eigen_crop=True)
        s = next(iter(ds))
        assert s.depth_valid is not None
        assert s.metadata["crop"] == "eigen"

    def test_garg_and_eigen_crops_mutually_exclusive(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, frames=1)
        with pytest.raises(ValueError, match="mutually exclusive"):
            KITTIDataset(root=tmp_path, apply_garg_crop=True, apply_eigen_crop=True)

    def test_scan_picks_up_multiple_drives(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, drive_ids=(2, 5), frames=2)
        ds = KITTIDataset(root=tmp_path)
        assert len(ds) == 4
        drives = {s.metadata["drive"] for s in ds}
        assert drives == {
            "2011_09_26_drive_0002_sync",
            "2011_09_26_drive_0005_sync",
        }

    def test_scan_skips_images_without_gt(self, tmp_path: Path) -> None:
        entries = _write_fake_kitti(tmp_path, frames=3)
        # Delete one GT file BEFORE instantiating the loader so the initial
        # scan sees the mismatch and silently drops that frame.
        drive, frame_id, cam = entries[1]
        gt_path = (
            tmp_path
            / "depth_annotated"
            / "val"
            / drive
            / "proj_depth"
            / "groundtruth"
            / cam
            / f"{frame_id}.png"
        )
        gt_path.unlink()
        ds = KITTIDataset(root=tmp_path)
        assert len(ds) == 2
        frame_ids = {s.metadata["frame_id"] for s in ds}
        assert frame_id not in frame_ids

    def test_manifest_cached_and_reused(self, tmp_path: Path) -> None:
        _write_fake_kitti(tmp_path, frames=2)
        KITTIDataset(root=tmp_path)
        manifest_dir = tmp_path / ".plumbline_manifest"
        assert manifest_dir.exists()
        # Re-opening should use the cached manifest (same record count).
        ds2 = KITTIDataset(root=tmp_path)
        assert len(ds2) == 2

    def test_sample_list_monodepth2_format(self, tmp_path: Path) -> None:
        entries = _write_fake_kitti(tmp_path, frames=3)
        list_path = tmp_path / "eigen_test.txt"
        with list_path.open("w") as f:
            f.write("# comment\n\n")
            # Monodepth2 format: "<date>/<drive>_sync <frame> l|r"
            for drive, frame_id, _cam in entries[:2]:
                # Strip leading zeros to test zero-padding canonicalization.
                bare = str(int(frame_id))
                f.write(f"2011_09_26/{drive} {bare} l\n")
        ds = KITTIDataset(root=tmp_path, sample_list=list_path)
        samples = list(ds)
        assert len(samples) == 2
        assert samples[0].metadata["frame_id"].startswith("000000")

    def test_sample_list_requires_matching_camera(self, tmp_path: Path) -> None:
        entries = _write_fake_kitti(tmp_path, frames=1)
        list_path = tmp_path / "list.txt"
        drive, frame_id, _cam = entries[0]
        list_path.write_text(f"2011_09_26/{drive} {frame_id} r\n")
        with pytest.raises(ValueError, match="does not match"):
            KITTIDataset(root=tmp_path, sample_list=list_path, camera="image_02")

    def test_sample_list_missing_file_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        _write_fake_kitti(tmp_path, frames=1)
        with pytest.raises(DatasetNotAvailable, match="sample_list not found"):
            KITTIDataset(root=tmp_path, sample_list=tmp_path / "missing.txt")

    def test_sample_list_entry_without_data_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        _write_fake_kitti(tmp_path, frames=1)
        list_path = tmp_path / "list.txt"
        # Point at a frame that doesn't exist on disk.
        list_path.write_text("2011_09_26/2011_09_26_drive_0002_sync 0000009999 l\n")
        with pytest.raises(DatasetNotAvailable, match="Missing data"):
            KITTIDataset(root=tmp_path, sample_list=list_path)

    def test_load_kitti_calib_image_02(self, tmp_path: Path) -> None:
        p = tmp_path / "calib_cam_to_cam.txt"
        p.write_text(
            "calib_time: x\n"
            "P_rect_02: 721.5377 0 609.5593 44.85728 0 721.5377 172.854 0.2163791 "
            "0 0 1 0.002745884\n"
        )
        K = load_kitti_calib(p, camera="image_02")
        assert K.shape == (3, 3) and K.dtype == np.float32
        assert abs(K[0, 0] - 721.5377) < 1e-3
        assert abs(K[0, 2] - 609.5593) < 1e-3

    def test_load_kitti_calib_missing_key(self, tmp_path: Path) -> None:
        p = tmp_path / "calib_cam_to_cam.txt"
        p.write_text("P_rect_01: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="P_rect_02"):
            load_kitti_calib(p, camera="image_02")

    def test_load_kitti_depth_png_roundtrip(self, tmp_path: Path) -> None:
        depth_m = np.array([[0.0, 1.5, 5.25], [10.0, 20.0, 0.0]], dtype=np.float32)
        depth_u16 = (depth_m * 256.0).astype(np.uint16)
        p = tmp_path / "d.png"
        Image.fromarray(depth_u16).save(p)
        loaded = load_kitti_depth_png_to_m(p)
        np.testing.assert_allclose(loaded, depth_m, atol=1e-3)
        assert loaded.dtype == np.float32

    def test_load_kitti_depth_rejects_wrong_dtype(self, tmp_path: Path) -> None:
        p = tmp_path / "d.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(p)
        with pytest.raises(ValueError, match="uint16"):
            load_kitti_depth_png_to_m(p)

    def test_parse_eigen_sample_list_monodepth2(self, tmp_path: Path) -> None:
        p = tmp_path / "list.txt"
        p.write_text(
            "# comment\n"
            "2011_09_26/2011_09_26_drive_0002_sync 69 l\n"
            "2011_09_26/2011_09_26_drive_0002_sync 54 r\n"
        )
        entries = parse_eigen_sample_list(p)
        assert entries == [
            ("2011_09_26_drive_0002_sync", "0000000069", "image_02"),
            ("2011_09_26_drive_0002_sync", "0000000054", "image_03"),
        ]

    def test_parse_eigen_sample_list_explicit_camera(self, tmp_path: Path) -> None:
        p = tmp_path / "list.txt"
        p.write_text("2011_09_26_drive_0002_sync 0000000069 image_02\n")
        entries = parse_eigen_sample_list(p)
        assert entries == [("2011_09_26_drive_0002_sync", "0000000069", "image_02")]

    def test_parse_eigen_sample_list_rejects_bad_line(self, tmp_path: Path) -> None:
        p = tmp_path / "list.txt"
        p.write_text("drive_sync 12345\n")  # only 2 fields
        with pytest.raises(ValueError, match="3 whitespace fields"):
            parse_eigen_sample_list(p)

    def test_parse_eigen_sample_list_rejects_bad_camera(self, tmp_path: Path) -> None:
        p = tmp_path / "list.txt"
        p.write_text("2011_09_26_drive_0002_sync 0000000069 middle\n")
        with pytest.raises(ValueError, match="camera token"):
            parse_eigen_sample_list(p)

    def test_parse_eigen_sample_list_rejects_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "list.txt"
        p.write_text("# just a comment\n\n")
        with pytest.raises(ValueError, match="0 sample entries"):
            parse_eigen_sample_list(p)

    def test_garg_crop_mask_shape_and_extent(self) -> None:
        mask = garg_crop_mask((375, 1242))
        assert mask.shape == (375, 1242) and mask.dtype == bool
        # Spec-checks: top row and bottom row should be outside the crop.
        assert not mask[0].any()
        assert not mask[-1].any()
        # A sizable middle region should be inside.
        assert mask[200, 600]

    def test_eigen_and_garg_crops_differ(self) -> None:
        # Eigen and Garg overlap substantially but neither contains the other:
        # Garg's bottom extends further than Eigen's (0.992 vs 0.914),
        # while Eigen's top extends further than Garg's (0.332 vs 0.408).
        eigen = eigen_crop_mask((375, 1242))
        garg = garg_crop_mask((375, 1242))
        assert eigen.shape == garg.shape == (375, 1242)
        assert int(eigen.sum()) > 0 and int(garg.sum()) > 0
        # Asymmetric containment: there are rows only in one mask and rows only
        # in the other.
        rows_only_in_eigen = (eigen & ~garg).any(axis=1)
        rows_only_in_garg = (garg & ~eigen).any(axis=1)
        assert rows_only_in_eigen.any()
        assert rows_only_in_garg.any()


# ---------------------------------------------------------------------------
# DIODE
# ---------------------------------------------------------------------------


def _write_fake_diode(
    root: Path,
    *,
    split: str = "val",
    domain: str = "indoors",
    scenes: int = 1,
    scans_per_scene: int = 1,
    frames: int = 2,
    H: int = 24,
    W: int = 32,
) -> list[str]:
    """Write a minimal DIODE-shaped tree and return sample_ids for reference."""
    sample_ids: list[str] = []
    for s in range(scenes):
        scene = f"scene_{s:05d}"
        for k in range(scans_per_scene):
            scan = f"scan_{k:05d}"
            scan_dir = root / split / domain / scene / scan
            scan_dir.mkdir(parents=True, exist_ok=True)
            for f_idx in range(frames):
                base = f"{s:05d}_{k:05d}_{domain}_{100 + f_idx}_000"
                # RGB
                Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
                    scan_dir / f"{base}.png"
                )
                # Depth: DIODE ships (H, W, 1) float32 — we keep that shape on
                # disk so load_diode_depth_m's squeeze path is exercised.
                depth = np.full((H, W, 1), 2.5, dtype=np.float32)
                # Poke an invalid pixel so the mask boolification is testable.
                depth[0, 0, 0] = 0.0
                np.save(scan_dir / f"{base}_depth.npy", depth)
                # Mask: uint8, 1=valid, 0=invalid.
                mask = np.ones((H, W), dtype=np.uint8)
                mask[0, 0] = 0
                np.save(scan_dir / f"{base}_depth_mask.npy", mask)
                sample_ids.append(f"{domain}/{scene}/{scan}/{base}")
    return sample_ids


class TestDIODE:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            DIODEDataset(root=tmp_path / "nope")

    def test_missing_split_subtree(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        (tmp_path / "val").mkdir()  # root exists but indoors/ under it doesn't
        with pytest.raises(DatasetNotAvailable):
            list(DIODEDataset(root=tmp_path, domain="indoors"))

    def test_invalid_split_errors(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path)
        with pytest.raises(ValueError, match="split"):
            DIODEDataset(root=tmp_path, split="test")

    def test_invalid_domain_errors(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path)
        with pytest.raises(ValueError, match="domain"):
            DIODEDataset(root=tmp_path, domain="mixed")

    def test_basic_load_indoors(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path, frames=3)
        ds = DIODEDataset(root=tmp_path, domain="indoors")
        samples = list(ds)
        assert len(samples) == 3
        s = samples[0]
        assert s.num_views == 1
        assert s.images.shape == (1, 24, 32, 3)
        assert s.depth_gt is not None and s.depth_gt.shape == (1, 24, 32)
        assert s.depth_gt.dtype == np.float32
        # Invalid pixel at (0, 0) should be marked via depth_valid (mask), not
        # by zeroing out depth.
        assert s.depth_valid is not None
        assert s.depth_valid.shape == (1, 24, 32) and s.depth_valid.dtype == bool
        assert not bool(s.depth_valid[0, 0, 0])
        assert bool(s.depth_valid[0, -1, -1])

    def test_default_intrinsic_is_diode_devkit_value(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path)
        ds = DIODEDataset(root=tmp_path)
        s = next(iter(ds))
        fx, fy, cx, cy = DIODE_INTRINSIC
        assert s.intrinsics[0, 0, 0] == fx
        assert s.intrinsics[0, 1, 1] == fy
        assert s.intrinsics[0, 0, 2] == cx
        assert s.intrinsics[0, 1, 2] == cy
        assert s.metadata["intrinsic_source"] == "diode_devkit_default"

    def test_intrinsic_override(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path)
        ds = DIODEDataset(root=tmp_path, intrinsic=(1000.0, 1000.0, 500.0, 375.0))
        s = next(iter(ds))
        assert s.intrinsics[0, 0, 0] == 1000.0
        assert s.metadata["intrinsic_source"] == "user-supplied"

    def test_domain_alias_indoor_maps_to_indoors(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path, domain="indoors")
        # Pass the English-preferred singular form; loader should canonicalise.
        ds = DIODEDataset(root=tmp_path, domain="indoor")
        assert len(ds) == 2
        assert next(iter(ds)).metadata["domain"] == "indoors"

    def test_domain_both_concatenates(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path, domain="indoors", scenes=1, frames=2)
        _write_fake_diode(tmp_path, domain="outdoor", scenes=1, frames=3)
        ds = DIODEDataset(root=tmp_path, domain="both")
        domains = [s.metadata["domain"] for s in ds]
        assert len(domains) == 5
        assert domains.count("indoors") == 2 and domains.count("outdoor") == 3

    def test_domain_both_skips_missing(self, tmp_path: Path) -> None:
        # Only indoors on disk; domain=both should yield indoor samples alone.
        _write_fake_diode(tmp_path, domain="indoors")
        ds = DIODEDataset(root=tmp_path, domain="both")
        assert len(ds) == 2
        assert {s.metadata["domain"] for s in ds} == {"indoors"}

    def test_scene_whitelist(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path, scenes=3, frames=1)
        ds = DIODEDataset(root=tmp_path, scenes=["scene_00001"])
        assert len(ds) == 1
        assert next(iter(ds)).metadata["scene"] == "scene_00001"

    def test_skips_samples_with_missing_rgb_or_mask(self, tmp_path: Path) -> None:
        ids = _write_fake_diode(tmp_path, frames=3)
        # Remove the RGB for the second sample; loader should silently drop it.
        base_parts = ids[1].split("/")  # .../<scan>/<base>
        scan_dir = tmp_path / "val" / base_parts[0] / base_parts[1] / base_parts[2]
        (scan_dir / f"{base_parts[3]}.png").unlink()
        ds = DIODEDataset(root=tmp_path)
        assert len(ds) == 2

    def test_manifest_cached(self, tmp_path: Path) -> None:
        _write_fake_diode(tmp_path)
        DIODEDataset(root=tmp_path)
        assert (tmp_path / ".plumbline_manifest").exists()

    def test_load_diode_depth_m_squeezes_trailing_axis(self, tmp_path: Path) -> None:
        depth = np.full((16, 8, 1), 3.25, dtype=np.float32)
        p = tmp_path / "d.npy"
        np.save(p, depth)
        out = load_diode_depth_m(p)
        assert out.shape == (16, 8) and out.dtype == np.float32
        assert out[0, 0] == 3.25

    def test_load_diode_depth_m_rejects_higher_dim(self, tmp_path: Path) -> None:
        p = tmp_path / "d.npy"
        np.save(p, np.zeros((4, 4, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected 2D depth"):
            load_diode_depth_m(p)

    def test_load_diode_depth_mask_returns_bool(self, tmp_path: Path) -> None:
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        p = tmp_path / "m.npy"
        np.save(p, mask)
        out = load_diode_depth_mask(p)
        assert out.dtype == bool
        np.testing.assert_array_equal(out, np.array([[True, False], [False, True]]))


# ---------------------------------------------------------------------------
# DTU
# ---------------------------------------------------------------------------


def _write_fake_dtu_cam(path: Path, *, tx: float = 0.0) -> None:
    """Write an MVSNet-style _cam.txt with identity rotation + x translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("extrinsic\n")
        f.write(f"1 0 0 {tx}\n")
        f.write("0 1 0 0\n")
        f.write("0 0 1 0\n")
        f.write("0 0 0 1\n")
        f.write("\n")
        f.write("intrinsic\n")
        f.write("100 0 16\n")
        f.write("0 100 8\n")
        f.write("0 0 1\n")
        f.write("\n")
        f.write("425 2.5\n")


def _write_minimal_ply(path: Path, points: np.ndarray) -> None:
    """Write an ASCII PLY with just (x, y, z) float properties."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def _write_fake_dtu(
    root: Path,
    *,
    scan_ids: tuple[int, ...] = (1, 4),
    views: int = 4,
    light: int = 3,
    write_gt: bool = True,
    H: int = 16,
    W: int = 32,
) -> None:
    # Shared Cameras_1/: one cam file per view index, 0-indexed.
    for v in range(views):
        _write_fake_dtu_cam(root / "Cameras_1" / f"{v:08d}_cam.txt", tx=float(v))
    # Per-scan Rectified/scanN_train/rect_<VVV>_<L>_r5000.png.
    for scan_id in scan_ids:
        scan_dir = root / "Rectified" / f"scan{scan_id}_train"
        scan_dir.mkdir(parents=True, exist_ok=True)
        for v in range(views):
            view_1based = v + 1
            # Also include a non-canonical light so we can test filtering.
            for L in (light, (light + 1) % 7):
                Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
                    scan_dir / f"rect_{view_1based:03d}_{L}_r5000.png"
                )
        if write_gt:
            pts = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [10.0, 0.0, 1.0],
                    [0.0, 10.0, 1.0],
                    [5.0, 5.0, 2.0],
                ],
                dtype=np.float32,
            )
            _write_minimal_ply(root / "Points" / "stl" / f"stl{scan_id:03d}_total.ply", pts)


class TestDTU:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises(DatasetNotAvailable):
            DTUDataset(root=tmp_path / "nope")

    def test_missing_cameras_dir(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        (tmp_path / "Rectified").mkdir()
        with pytest.raises(DatasetNotAvailable, match="Cameras_1"):
            DTUDataset(root=tmp_path, scans=[1])

    def test_invalid_light_range(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,))
        with pytest.raises(ValueError, match="light"):
            DTUDataset(root=tmp_path, scans=[1], light=7)

    def test_invalid_views_per_sample(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,))
        with pytest.raises(ValueError, match="views_per_sample"):
            DTUDataset(root=tmp_path, scans=[1], views_per_sample=0)

    def test_invalid_split(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,))
        with pytest.raises(ValueError, match="DTU split"):
            DTUDataset(root=tmp_path, split="nope")

    def test_basic_load_with_custom_scans(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,), views=4)
        ds = DTUDataset(root=tmp_path, scans=[1], views_per_sample=2)
        samples = list(ds)
        # Sliding window over 4 views at size 2 → 3 samples.
        assert len(samples) == 3
        s = samples[0]
        assert s.num_views == 2
        assert s.images.shape == (2, 16, 32, 3)
        # First camera is identity after rebase.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        # GT point cloud attached.
        assert s.point_cloud_gt is not None and s.point_cloud_gt.shape[1] == 3
        # Metadata carries scan id + view indices.
        assert s.metadata["scan_id"] == 1
        assert s.metadata["view_indices"] == [0, 1]
        assert s.metadata["units"] == "mm"

    def test_test_split_uses_canonical_22_scans(self) -> None:
        assert len(DTU_MVS_TEST_SCANS) == 22
        # Spot-check a few MVSNet benchmark scans.
        assert 1 in DTU_MVS_TEST_SCANS
        assert 118 in DTU_MVS_TEST_SCANS

    def test_test_split_missing_scan_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable

        # Write Cameras_1 + Rectified but no scan1_train.
        for v in range(2):
            _write_fake_dtu_cam(tmp_path / "Cameras_1" / f"{v:08d}_cam.txt")
        (tmp_path / "Rectified").mkdir()
        with pytest.raises(DatasetNotAvailable, match="test-split scan"):
            DTUDataset(root=tmp_path, split="test", views_per_sample=1)

    def test_custom_scans_skip_missing_silently(self, tmp_path: Path) -> None:
        # scan1 exists on disk; scan99 doesn't. Custom list should use whatever's there.
        _write_fake_dtu(tmp_path, scan_ids=(1,))
        ds = DTUDataset(root=tmp_path, scans=[1, 99], views_per_sample=4)
        samples = list(ds)
        assert len(samples) == 1
        assert samples[0].metadata["scan_id"] == 1

    def test_light_selection(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,), views=3, light=3)
        # Light 3 is the default; picking 4 should still work because
        # _write_fake_dtu also writes (light + 1) % 7 = 4 for each view.
        ds3 = DTUDataset(root=tmp_path, scans=[1], views_per_sample=3, light=3)
        ds4 = DTUDataset(root=tmp_path, scans=[1], views_per_sample=3, light=4)
        s3 = next(iter(ds3))
        s4 = next(iter(ds4))
        assert s3.metadata["light"] == 3
        assert s4.metadata["light"] == 4
        # Different light → different pixel values (random images).
        assert not np.array_equal(s3.images, s4.images)

    def test_multiple_scans_produce_separate_samples(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1, 4), views=2)
        ds = DTUDataset(root=tmp_path, scans=[1, 4], views_per_sample=2)
        samples = list(ds)
        assert len(samples) == 2
        assert {s.metadata["scan_id"] for s in samples} == {1, 4}

    def test_max_gt_points_subsamples_deterministically(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,), views=2)
        ds_a = DTUDataset(root=tmp_path, scans=[1], views_per_sample=2, max_gt_points=2)
        ds_b = DTUDataset(root=tmp_path, scans=[1], views_per_sample=2, max_gt_points=2)
        a = next(iter(ds_a)).point_cloud_gt
        b = next(iter(ds_b)).point_cloud_gt
        assert a is not None and b is not None
        assert a.shape == (2, 3)
        np.testing.assert_array_equal(a, b)  # same seed → identical subset

    def test_extrinsics_rebased_to_first_camera(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,), views=3)
        ds = DTUDataset(root=tmp_path, scans=[1], views_per_sample=3)
        s = next(iter(ds))
        # First view is identity after rebase_to_first_camera.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        # Subsequent cameras remain distinct (our fake cams offset in x).
        assert not np.allclose(s.extrinsics_gt[1], np.eye(4))

    def test_manifest_cached(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,))
        DTUDataset(root=tmp_path, scans=[1], views_per_sample=2)
        assert (tmp_path / ".plumbline_manifest").exists()

    def test_no_gt_ply_leaves_point_cloud_none(self, tmp_path: Path) -> None:
        _write_fake_dtu(tmp_path, scan_ids=(1,), views=2, write_gt=False)
        ds = DTUDataset(root=tmp_path, scans=[1], views_per_sample=2)
        s = next(iter(ds))
        assert s.point_cloud_gt is None

    def test_load_dtu_cam_basic(self, tmp_path: Path) -> None:
        _write_fake_dtu_cam(tmp_path / "cam.txt", tx=5.0)
        K, E = load_dtu_cam(tmp_path / "cam.txt")
        assert K.shape == (3, 3) and E.shape == (4, 4)
        assert K[0, 0] == 100 and K[1, 2] == 8
        np.testing.assert_allclose(E[:3, :3], np.eye(3))
        assert E[0, 3] == 5.0

    def test_load_dtu_cam_rejects_missing_markers(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        p.write_text("1 2 3 4\n")
        with pytest.raises(ValueError, match="extrinsic"):
            load_dtu_cam(p)

    def test_load_dtu_cam_rejects_short_extrinsic(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        p.write_text("extrinsic\n1 2 3\nintrinsic\n1 0 0 0 1 0 0 0 1\n")
        with pytest.raises(ValueError, match="16 extrinsic"):
            load_dtu_cam(p)

    def test_load_dtu_cam_rejects_short_intrinsic(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        p.write_text("extrinsic\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\nintrinsic\n1 0 0\n")
        with pytest.raises(ValueError, match="9 intrinsic"):
            load_dtu_cam(p)


# ---------------------------------------------------------------------------
# ScanNet-1500 (two-view pose benchmark)
# ---------------------------------------------------------------------------


def _write_fake_scannet_1500(
    root: Path,
    *,
    n_pairs: int = 3,
    H: int = 480,
    W: int = 640,
) -> Path:
    """Lay out a minimal ScanNet-test tree + pairs file for loader tests."""
    scene = "scene0707_00"
    sens_dir = root / "scans_test" / scene / "sens"
    sens_dir.mkdir(parents=True, exist_ok=True)
    # K (1165.72, 1165.74, 649.095, 484.765) from the real SuperGlue pairs.
    K = np.array([[1165.72, 0, 649.095], [0, 1165.74, 484.765], [0, 0, 1]])
    K_flat = " ".join(f"{v}" for v in K.flatten())
    lines = []
    for i in range(n_pairs):
        f0 = f"frame-{i * 10:06d}"
        f1 = f"frame-{i * 10 + 60:06d}"
        Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
            sens_dir / f"{f0}.color.jpg", quality=85
        )
        Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
            sens_dir / f"{f1}.color.jpg", quality=85
        )
        # Identity relative pose (same camera).
        T = np.eye(4).flatten()
        T_flat = " ".join(f"{v}" for v in T)
        lines.append(
            f"scans_test/{scene}/sens/{f0}.color.jpg "
            f"scans_test/{scene}/sens/{f1}.color.jpg 0 0 "
            f"{K_flat} {K_flat} {T_flat}"
        )
    pairs_path = root / "scannet_test_pairs_with_gt.txt"
    pairs_path.write_text("\n".join(lines) + "\n")
    return pairs_path


class TestScanNet1500:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.scannet_1500 import ScanNet1500Dataset

        with pytest.raises(DatasetNotAvailable):
            ScanNet1500Dataset(root=tmp_path / "nope", pairs_file=tmp_path / "nope.txt")

    def test_missing_pairs_file(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.scannet_1500 import ScanNet1500Dataset

        with pytest.raises(DatasetNotAvailable, match="pairs_file"):
            ScanNet1500Dataset(root=tmp_path, pairs_file=tmp_path / "nope.txt")

    def test_loads_pairs(self, tmp_path: Path) -> None:
        from plumbline.datasets.scannet_1500 import ScanNet1500Dataset

        pairs = _write_fake_scannet_1500(tmp_path, n_pairs=3)
        ds = ScanNet1500Dataset(root=tmp_path, pairs_file=pairs)
        assert len(ds) == 3
        samples = list(ds)
        s = samples[0]
        assert s.num_views == 2
        assert s.images.shape == (2, 480, 640, 3)
        # First camera is world origin after rebase.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        # Identity relative pose in the test data → cam1 also identity.
        np.testing.assert_allclose(s.extrinsics_gt[1], np.eye(4), atol=1e-5)
        # Intrinsics 2x3x3 from pair file.
        assert s.intrinsics.shape == (2, 3, 3)
        assert s.intrinsics[0, 0, 0] == pytest.approx(1165.72)

    def test_missing_image_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.scannet_1500 import ScanNet1500Dataset

        pairs = _write_fake_scannet_1500(tmp_path, n_pairs=1)
        # Delete one image after loader construction succeeds — iterator
        # should then raise cleanly.
        (tmp_path / "scans_test" / "scene0707_00" / "sens").glob("*.jpg")
        img = next((tmp_path / "scans_test" / "scene0707_00" / "sens").glob("*.jpg"))
        img.unlink()
        ds = ScanNet1500Dataset(root=tmp_path, pairs_file=pairs)
        with pytest.raises(DatasetNotAvailable, match="Missing ScanNet"):
            list(ds)

    def test_parse_rejects_short_line(self, tmp_path: Path) -> None:
        from plumbline.datasets.scannet_1500 import parse_scannet_1500_pairs

        p = tmp_path / "bad.txt"
        p.write_text("foo.jpg bar.jpg 0 0 1 2 3\n")  # way too few tokens
        with pytest.raises(ValueError, match="38 tokens"):
            list(parse_scannet_1500_pairs(p))

    def test_parse_yields_scene_id(self, tmp_path: Path) -> None:
        from plumbline.datasets.scannet_1500 import parse_scannet_1500_pairs

        pairs = _write_fake_scannet_1500(tmp_path, n_pairs=2)
        recs = list(parse_scannet_1500_pairs(pairs))
        assert len(recs) == 2
        assert recs[0]["scene"] == "scene0707_00"
        assert recs[0]["pair_id"] == "pair_00001_scene0707_00"


# ---------------------------------------------------------------------------
# Co3Dv2
# ---------------------------------------------------------------------------


def _write_fake_co3dv2(
    root: Path,
    *,
    category: str = "hydrant",
    sequences: int = 2,
    frames_per_sequence: int = 6,
    H: int = 64,
    W: int = 96,
) -> None:
    """Write a minimal Co3Dv2-shaped tree for loader smoke tests."""
    import gzip
    import json

    cat_dir = root / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    for s in range(sequences):
        seq_name = f"{category}_seq_{s:04d}"
        images_dir = cat_dir / seq_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for f in range(frames_per_sequence):
            img_name = f"frame{f + 1:06d}.jpg"
            img_path = images_dir / img_name
            Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8)).save(
                img_path, quality=85
            )
            # Identity-ish extrinsics with small translation in world frame.
            R = np.eye(3).tolist()
            T = [0.0, 0.0, 0.5 + f * 0.1]
            annotations.append(
                {
                    "sequence_name": seq_name,
                    "frame_number": f + 1,
                    "frame_timestamp": float(f),
                    "image": {
                        "path": f"{category}/{seq_name}/images/{img_name}",
                        "size": [H, W],
                    },
                    "viewpoint": {
                        "R": R,
                        "T": T,
                        "focal_length": [2.0, 2.0 * H / W],  # ndc_norm_image_bounds
                        "principal_point": [0.0, 0.0],
                        "intrinsics_format": "ndc_norm_image_bounds",
                    },
                }
            )
    # Co3Dv2 stores frame annotations as a gzipped JSON list.
    anno_path = cat_dir / "frame_annotations.jgz"
    with gzip.open(anno_path, "wt", encoding="utf-8") as f:
        json.dump(annotations, f)


class TestCo3Dv2:
    def test_missing_root(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        with pytest.raises(DatasetNotAvailable):
            Co3Dv2Dataset(root=tmp_path / "nope")

    def test_empty_categories(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        # Root exists but no category has frame_annotations.jgz.
        (tmp_path / "fake_dir").mkdir()
        with pytest.raises(DatasetNotAvailable, match="categories"):
            Co3Dv2Dataset(root=tmp_path)

    def test_invalid_views_per_sample(self, tmp_path: Path) -> None:
        _write_fake_co3dv2(tmp_path)
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        with pytest.raises(ValueError, match="views_per_sample"):
            Co3Dv2Dataset(root=tmp_path, views_per_sample=0)

    def test_basic_load(self, tmp_path: Path) -> None:
        _write_fake_co3dv2(tmp_path, sequences=2, frames_per_sequence=6)
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        ds = Co3Dv2Dataset(root=tmp_path, views_per_sample=4)
        samples = list(ds)
        # 2 sequences × (6 frames - 4 + 1) = 2 × 3 = 6 samples
        assert len(samples) == 6
        s = samples[0]
        assert s.num_views == 4
        assert s.images.shape == (4, 64, 96, 3)
        # First camera is identity after rebase.
        np.testing.assert_allclose(s.extrinsics_gt[0], np.eye(4), atol=1e-5)
        # Intrinsics: pixel-space with principal point at image centre.
        assert s.intrinsics[0, 0, 2] == pytest.approx(48.0)  # cx = W/2
        assert s.intrinsics[0, 1, 2] == pytest.approx(32.0)  # cy = H/2
        # fx / fy recovered from fx_ndc=2.0 on W/2: fx_px = 2 * 48 = 96
        assert s.intrinsics[0, 0, 0] == pytest.approx(96.0)

    def test_category_whitelist(self, tmp_path: Path) -> None:
        _write_fake_co3dv2(tmp_path, category="hydrant")
        _write_fake_co3dv2(tmp_path, category="teddybear")
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        ds = Co3Dv2Dataset(root=tmp_path, categories=["hydrant"], views_per_sample=4)
        cats = {s.metadata["category"] for s in ds}
        assert cats == {"hydrant"}

    def test_missing_category_errors(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import DatasetNotAvailable
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        _write_fake_co3dv2(tmp_path, category="hydrant")
        with pytest.raises(DatasetNotAvailable, match="not found"):
            Co3Dv2Dataset(root=tmp_path, categories=["apple"], views_per_sample=4)

    def test_sequence_whitelist_prunes(self, tmp_path: Path) -> None:
        _write_fake_co3dv2(tmp_path, sequences=3, frames_per_sequence=4)
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        ds = Co3Dv2Dataset(
            root=tmp_path,
            sequences=["hydrant_seq_0001"],
            views_per_sample=4,
        )
        seqs = {s.metadata["sequence"] for s in ds}
        assert seqs == {"hydrant_seq_0001"}

    def test_max_sequences_per_category(self, tmp_path: Path) -> None:
        _write_fake_co3dv2(tmp_path, sequences=3, frames_per_sequence=4)
        from plumbline.datasets.co3dv2 import Co3Dv2Dataset

        ds = Co3Dv2Dataset(
            root=tmp_path, views_per_sample=4, max_sequences_per_category=2
        )
        assert len(ds) == 2  # 2 sequences × 1 sliding window each at full-frame count

    def test_pytorch3d_to_opencv_identity(self) -> None:
        """PyTorch3D identity → world_from_cam that's the axis-flip itself.

        When R = I, T = 0 in PyTorch3D's right-multiply form, the world
        frame is the PyTorch3D camera frame. plumbline asks for the
        OpenCV world_from_camera; that's a pure axis-flip (x → -x, y → -y).
        """
        from plumbline.datasets.co3dv2 import co3d_pytorch3d_to_opencv

        R = np.eye(3)
        T = np.zeros(3)
        E = co3d_pytorch3d_to_opencv(R, T)
        # World origin = PyTorch3D origin = OpenCV origin (at cam0). Translation = 0.
        np.testing.assert_allclose(E[:3, 3], 0.0, atol=1e-10)
        # Rotation: OpenCV cam axes negate PyTorch3D X and Y.
        expected_R = np.diag([-1.0, -1.0, 1.0])
        # world_from_cam = invert(cam_from_world = flip) = flip (self-inverse).
        np.testing.assert_allclose(E[:3, :3], expected_R, atol=1e-10)

    def test_ndc_to_pixel_isotropic(self) -> None:
        from plumbline.datasets.co3dv2 import co3d_ndc_intrinsics_to_pixel

        # Isotropic NDC with focal=1 and principal point at image centre.
        # Larger side spans [-s, s] where s = W/H; shorter is [-1, 1].
        H, W = 400, 600
        K = co3d_ndc_intrinsics_to_pixel(
            focal_length=(1.0, 1.0),
            principal_point=(0.0, 0.0),
            size_hw=(H, W),
            intrinsics_format="ndc_isotropic",
        )
        # fx = 1 * max(H, W) / 2 = 300
        assert K[0, 0] == pytest.approx(300.0)
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(300.0)  # cx = W/2
        assert K[1, 2] == pytest.approx(200.0)  # cy = H/2

    def test_ndc_unknown_format_errors(self) -> None:
        from plumbline.datasets.co3dv2 import co3d_ndc_intrinsics_to_pixel

        with pytest.raises(ValueError, match="unknown Co3D intrinsics_format"):
            co3d_ndc_intrinsics_to_pixel(
                focal_length=(1.0, 1.0),
                principal_point=(0.0, 0.0),
                size_hw=(100, 100),
                intrinsics_format="nope",
            )
