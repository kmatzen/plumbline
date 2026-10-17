import numpy as np
import pytest
from paper.probe_tum_pose_timing import interpolate
from scipy.spatial.transform import Rotation


def test_interpolation_translation_rotation_and_exact_endpoints():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 2
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result, record = interpolate([0.0, 0.02], poses, 0.01)
    assert record["status"] == "interpolated"
    np.testing.assert_allclose(result[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(
        result[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-15
    )
    for t, pose in zip([0.0, 0.02], poses, strict=True):
        result, record = interpolate([0.0, 0.02], poses, t)
        assert record["status"] == "exact"
        np.testing.assert_array_equal(result, pose)


def test_no_extrapolation_or_interpolation_across_large_gaps():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    assert interpolate([0.0, 1.0], poses, 0.5)[1]["status"] == "reference_gap"
    assert interpolate([0.0, 1.0], poses, -1)[1]["status"] == "outside_reference_span"
    assert interpolate([0.0, 1.0], poses, 2)[1]["status"] == "outside_reference_span"
    with pytest.raises(ValueError):
        interpolate([1.0, 0.0], poses, 0.0)


def test_duplicate_timestamp_is_ambiguous_only_when_needed():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    times = [0.0, 0.01, 0.01, 0.02, 0.03]
    poses[2, 0, 3] = 1
    assert interpolate(times, poses, 0.01)[1]["status"] == "ambiguous_duplicate_timestamp"
    assert interpolate(times, poses, 0.005)[1]["status"] == "ambiguous_duplicate_bracket"
    assert interpolate(times, poses, 0.015)[1]["status"] == "ambiguous_duplicate_bracket"
    assert interpolate(times, poses, 0.025)[1]["status"] == "interpolated"
    assert interpolate(times, poses, 0.02)[1]["status"] == "exact"
