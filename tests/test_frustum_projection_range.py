import numpy as np

from plumbline.frustum_projection_range import projection_range


def test_unbounded_depth_can_have_bounded_image_projection():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    k = np.diag([100., 100., 1.])
    boxes = np.array([[-1, -2, 1, 2.]] * 2)
    result = projection_range(poses, k, boxes, np.eye(4), k)
    assert result["status"] == "checked"
    for key, expected in zip(("x_lower", "x_upper", "y_lower", "y_upper"), (-1, 1, -2, 2)):
        np.testing.assert_allclose(result["bounds"][key]["pixel"], expected)


def test_translation_exposes_unbounded_prediction_and_reverse_camera_is_infeasible():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    k = np.diag([100., 100., 1.])
    boxes = np.array([[-1, -1, 1, 1.]] * 2)
    query = np.eye(4)
    query[0, 3] = -1
    result = projection_range(poses, k, boxes, query, k)
    assert result["bounds"]["x_lower"]["status"] == "unbounded"
    np.testing.assert_allclose(result["bounds"]["x_upper"]["pixel"], 1)
    query = np.diag([-1., 1, -1, 1])
    result = projection_range(poses, k, boxes, query, k)
    assert all(r["status"] == "infeasible" for r in result["bounds"].values())


def test_known_disparity_predicts_query_interval_and_is_gauge_invariant():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = -1
    query = np.eye(4)
    query[0, 3] = -2
    k = np.diag([100., 100., 1.])
    boxes = np.array([[-1, -1, 1, 1], [-21, -1, -19, 1.]])
    result = projection_range(poses, k, boxes, query, k)
    np.testing.assert_allclose(result["bounds"]["x_lower"]["pixel"], -43)
    np.testing.assert_allclose(result["bounds"]["x_upper"]["pixel"], -37)
    gauge = np.eye(4)
    gauge[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    gauge[:3, 3] = [10, 20, 30]
    other = projection_range(poses @ np.linalg.inv(gauge), k, boxes, query @ np.linalg.inv(gauge), k)
    for key in result["bounds"]:
        np.testing.assert_allclose(other["bounds"][key]["pixel"], result["bounds"][key]["pixel"])
