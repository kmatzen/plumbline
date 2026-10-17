import hashlib
import json

import cv2
import numpy as np
import pytest
from paper.rgbd_track_audit import audit, observed_spread, world_observation
from paper.stage_rgbd_validation import sha


def test_audit_keeps_missing_origin_and_checks_depth_provenance(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 0.4
    k = np.repeat(np.array([[10.0, 0, 5], [0, 10, 5], [0, 0, 1]])[None], 2, axis=0)
    np.savez(run / "reference.npz", extrinsics=poses, intrinsics=k)
    np.savez(run / "prediction.npz", extrinsics=poses)
    frames = [
        {
            "path": f"rgb/{i}.png",
            "timestamp": str(i),
            "reference_timestamp": str(i + 0.005),
            "file_sha256": f"synthetic-rgb-{i}",
        }
        for i in range(2)
    ]
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sequence": "synthetic",
                "frames": frames,
                "reference_sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
                "prediction_file_sha256": sha(run / "prediction.npz"),
            }
        )
    )
    tracks = tmp_path / "tracks.json"
    tracks.write_text(
        json.dumps(
            {
                "source_manifest_sha256": sha(run / "manifest.json"),
                "reference_file_sha256": sha(run / "reference.npz"),
                "tracks": [[{"xy": [5, 5]}, {"xy": [3, 5]}], [{"xy": None}, {"xy": [3, 5]}]],
            }
        )
    )
    depth_root = tmp_path / "depth"
    depth_root.mkdir()
    depth_file = depth_root / "depth.png"
    cv2.imwrite(str(depth_file), np.full((480, 640), 10000, dtype=np.uint16))
    depth_record = {
        "source_manifest_sha256": sha(run / "manifest.json"),
        "depth_scale_divisor": 5000,
        "zero_depth": "missing",
        "rows": [
            {
                "frame": i,
                "rgb_path": f["path"],
                "rgb_timestamp": f["timestamp"],
                "rgb_sha256": f["file_sha256"],
                "status": "staged",
                "depth_sha256": sha(depth_file),
                "nearest_depth": {
                    "path": "depth.png",
                    "within_tolerance": True,
                    "delta_seconds": 0.003,
                },
            }
            for i, f in enumerate(frames)
        ],
    }
    (depth_root / "depth-manifest.json").write_text(json.dumps(depth_record))
    result = audit(run, tracks, depth_root, tmp_path / "result.json")
    assert result["tracks"][0]["max_later_distance_m"] == pytest.approx(0)
    assert result["tracks"][1]["later_comparable_count"] == 0
    assert result["tracks"][1]["rows"][1]["status"] == "measured"
    assert result["tracks"][1]["rows"][1]["distance_from_first_frame_m"] is None
    assert result["frames"][0]["reference_minus_rgb_seconds"] == pytest.approx(0.005)
    depth_file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(run, tracks, depth_root, tmp_path / "bad.json")


def test_static_point_is_fixed_when_camera_translates():
    k = np.array([[10.0, 0, 5], [0, 10, 5], [0, 0, 1]])
    depth = np.full((11, 11), 10000, dtype=np.uint16)
    start = world_observation([5, 5], depth, k, np.eye(4))
    pose = np.eye(4)
    pose[0, 3] = 0.4
    later = world_observation([3, 5], depth, k, pose)
    np.testing.assert_allclose(start["world_point_m"], later["world_point_m"])
    # A track staying at the old pixel does not describe the same world point.
    wrong = world_observation([5, 5], depth, k, pose)
    np.testing.assert_allclose(
        np.array(wrong["world_point_m"]) - start["world_point_m"], [0.4, 0, 0]
    )


def test_spread_retains_missing_origin_and_never_bridges_gaps():
    rows = [
        {"frame": 0},
        {"frame": 1, "world_point_m": [0, 0, 0]},
        {"frame": 2, "world_point_m": [0, 0, 2]},
        {"frame": 3},
        {"frame": 4, "world_point_m": [0, 0, 5]},
    ]
    result = observed_spread(rows)
    assert result["measured_count"] == 3
    assert result["available_pair_count"] == 3
    assert result["diameter"] == {"frames": [1, 4], "distance_m": 5}
    assert result["adjacent_pairs"] == [{"frames": [1, 2], "distance_m": 2}]
    assert result["planned_adjacent_pair_count"] == 4
    assert result["median_adjacent_distance_m"] == 2
    assert "world_point_m" not in rows[0]
    for subset in ([], rows[:1], rows[:2]):
        result = observed_spread(subset)
        assert result["diameter"] is None
        assert result["median_adjacent_distance_m"] is None
    with pytest.raises(ValueError, match="consecutive"):
        observed_spread([rows[1], rows[4]])


def test_axial_depth_is_not_euclidean_range_and_holes_are_missing():
    depth = np.full((3, 3), 5000, dtype=np.uint16)
    result = world_observation([1, 1], depth, np.eye(3), np.eye(4))
    assert result["camera_point_m"] == [1.0, 1.0, 1.0]
    depth[1, 1] = 0
    result = world_observation([1, 1], depth, np.eye(3), np.eye(4))
    assert result["status"] == "missing_depth"
    assert result["patch_valid_pixels"] == 8
    assert world_observation(None, depth, np.eye(3), np.eye(4))["status"] == "missing_track"
    assert world_observation([9, 9], depth, np.eye(3), np.eye(4))["status"] == "outside_depth_image"
    with pytest.raises(ValueError, match="uint16"):
        world_observation([1, 1], depth.astype(float), np.eye(3), np.eye(4))
