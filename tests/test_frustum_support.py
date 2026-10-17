import numpy as np
import pytest

from plumbline.frustum_support import box_support


def fixture():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, -1, -2]
    k = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]])
    boxes = np.array([[98, 98, 102, 102], [78, 98, 82, 102], [58, 98, 62, 102]])
    return poses, k, boxes


def test_known_static_point_has_positive_witness():
    poses, k, boxes = fixture()
    result = box_support(poses, k, boxes)
    assert result["status"] == "positive_witness"
    point = np.r_[result["point_world"], 1]
    projected = (k @ (poses @ point)[:, :3].T).T
    xy = projected[:, :2] / projected[:, 2:]
    assert (xy >= boxes[:, :2] - 1e-6).all() and (xy <= boxes[:, 2:] + 1e-6).all()


def test_anchor_relaxation_is_not_full_track_consistency():
    poses, k, boxes = fixture()
    poses[:] = np.eye(4)
    assert box_support(poses, k, boxes, anchor_only=True)["status"] == "positive_witness"
    assert box_support(poses, k, boxes)["status"] != "positive_witness"


def test_scale_and_origin_do_not_change_support():
    poses, k, boxes = fixture()
    original = box_support(poses, k, boxes)
    poses[:, :3, 3] = poses[:, :3, 3] * 7 - np.array([100, -20, 4])
    changed = box_support(poses, k, boxes)
    assert original["status"] == changed["status"] == "positive_witness"


def test_opposing_cameras_do_not_accept_behind_camera_points():
    poses, k, boxes = fixture()
    poses[:] = np.eye(4)
    poses[1, :3, :3] = np.diag([-1, 1, -1])
    assert box_support(poses, k, boxes, anchor_only=True)["status"] != "positive_witness"


def test_reversed_box_corners_are_not_silently_reordered():
    poses, k, boxes = fixture()
    boxes[1] = [82, 98, 78, 102]
    with pytest.raises(ValueError, match="ordered boxes"):
        box_support(poses, k, boxes)


def test_rotated_world_coordinates_preserve_support():
    poses, k, boxes = fixture()
    gauge = np.eye(4)
    gauge[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    gauge[:3, 3] = [4, 7, 8]
    transformed = poses @ np.linalg.inv(gauge)
    original = box_support(poses, k, boxes)
    changed = box_support(transformed, k, boxes)
    assert original["status"] == changed["status"] == "positive_witness"
