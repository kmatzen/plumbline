import numpy as np
import pytest
from paper.rgbd_track_audit import world_observation


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
