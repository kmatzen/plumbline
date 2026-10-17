import numpy as np

from paper.probe_layered_minimum_error import minimum_box_error
from paper.probe_real_minimum_error import minimum_error
from paper.render_layered_camera_control import K, anchors, camera_pose, project_anchors


def test_real_solver_matches_frozen_control_solver():
    points, ids = anchors()
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    xy = np.stack([project_anchors(points, ids, t)[0][4] for t in range(24)])
    original = minimum_box_error(cameras, xy)
    result = minimum_error(cameras, K, xy)
    assert result["status"] == original["status"] == "numerical_bracket"
    assert (result["lower"], result["upper"]) == (original["lower"], original["upper"])
    # Changing focal length and principal point must use the supplied K.
    changed = K.copy()
    changed[0, 0] *= 2
    changed[0, 2] += 30
    new_xy = xy.copy()
    new_xy[:, 0] = (xy[:, 0]-K[0, 2])*2+changed[0, 2]
    assert minimum_error(cameras, changed, new_xy)["upper"] <= .01
