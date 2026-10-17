import numpy as np
import pytest

from plumbline.object_reprojection import score_static_track


def fixture():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, 1, 2]
    k = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]])
    # A static point at (0, 0, 5) moves left as the camera moves right.
    boxes = np.array([[100, 100, 100, 100], [80, 100, 80, 100], [60, 100, 60, 100]])
    return poses, k, boxes


def score(poses, k, boxes, depths=(1, 5, 10), verified=True):
    return score_static_track(poses, k, boxes, (200, 200), depths, static_verified=verified)


def test_known_projection_and_depth():
    result = score(*fixture())
    assert result["score"] == pytest.approx(0)
    assert result["depth"] == 5
    assert not result["depth_at_grid_boundary"]
    assert np.allclose(result["projected_xy"], [[100, 100], [80, 100], [60, 100]])


def test_wrong_motion_has_nonzero_error():
    poses, k, boxes = fixture()
    poses[:, 0, 3] *= -1
    assert score(poses, k, boxes)["score"] > 0


def test_gauge_and_scale_invariance_with_scaled_depth_grid():
    poses, k, boxes = fixture()
    original = score(poses, k, boxes)
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [9, -2, 3]
    poses[:, :3, 3] *= 3
    assert score(transform @ poses, k, boxes, (3, 15, 30))["score"] == original["score"]


def test_box_interior_is_not_point_error():
    poses, k, boxes = fixture()
    boxes = boxes + np.array([-10, -10, 10, 10])
    assert score(poses, k, boxes)["score"] == 0


def test_all_depths_behind_camera_are_unavailable():
    poses, k, boxes = fixture()
    poses[1, 2, 3] = 20
    assert score(poses, k, boxes)["status"] == "unavailable"


def test_boundary_is_reported():
    assert score(*fixture(), depths=(1, 5))["depth_at_grid_boundary"]


@pytest.mark.parametrize("depths", [(0, 1), (5, 1), (1, float("nan")), (1,)])
def test_bad_depth_grid_rejected(depths):
    with pytest.raises(ValueError):
        score(*fixture(), depths=depths)


def test_static_verification_required():
    with pytest.raises(ValueError, match="static-object"):
        score(*fixture(), verified=False)


def test_invalid_pose_rejected():
    poses, k, boxes = fixture()
    poses[1, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        score(poses, k, boxes)


def test_invalid_intrinsics_rejected():
    poses, k, boxes = fixture()
    k[0, 0] = 0
    with pytest.raises(ValueError, match="intrinsics"):
        score(poses, k, boxes)
