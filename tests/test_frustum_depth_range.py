import numpy as np

from plumbline.frustum_depth_range import depth_range


def test_depth_bounds_from_known_disparity_and_gauge():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = -1
    k = np.diag([100., 100., 1.])
    boxes = np.array([[-1, -1, 1, 1], [-21, -1, -19, 1.]])
    result = depth_range(poses, k, boxes)
    assert result["status"] == "checked"
    np.testing.assert_allclose(result["lower"]["depth"], 100 / 22)
    np.testing.assert_allclose(result["upper"]["depth"], 100 / 18)
    gauge = np.eye(4)
    gauge[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    gauge[:3, 3] = [4, 8, -3]
    rotated = depth_range(poses @ np.linalg.inv(gauge), k, boxes)
    np.testing.assert_allclose(rotated["lower"]["depth"], result["lower"]["depth"])
    np.testing.assert_allclose(rotated["upper"]["depth"], result["upper"]["depth"])
    poses[:, :3, 3] = poses[:, :3, 3] * 7 - np.array([10, 20, 30])
    scaled = depth_range(poses, k, boxes)
    np.testing.assert_allclose(scaled["lower"]["depth"], 7 * 100 / 22)
    np.testing.assert_allclose(scaled["upper"]["depth"], 7 * 100 / 18)


def test_no_translation_cannot_bound_depth_and_contradictions_are_retained():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    k = np.diag([100., 100., 1.])
    boxes = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1.]])
    result = depth_range(poses, k, boxes)
    assert result["lower"]["depth"] == 0
    assert result["upper"]["status"] == "unbounded"
    boxes[1, [0, 2]] += 20
    assert depth_range(poses, k, boxes)["status"] != "checked"
