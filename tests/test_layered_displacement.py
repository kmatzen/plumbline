import numpy as np

from paper.analyze_layered_displacement import project
from paper.probe_evidence_sensitivity import alternatives
from paper.render_layered_camera_control import anchors, camera_pose, project_anchors


def test_projection_and_global_gauge_preserve_reference_pixels():
    points, ids = anchors()
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    baseline = project(cameras, points)
    for t in range(24):
        np.testing.assert_allclose(baseline[t], project_anchors(points, ids, t)[0], atol=1e-10)
    changed = dict(alternatives(cameras))["global_rigid_gauge"]
    gauge = np.linalg.inv(changed[0]) @ cameras[0]
    transformed = (np.c_[points, np.ones(len(points))] @ gauge.T)[:, :3]
    np.testing.assert_allclose(project(changed, transformed), baseline, atol=1e-10)


def test_local_change_leaves_prefix_but_moves_query_points():
    points, _ = anchors()
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    changed = dict(alternatives(cameras))["rotation_x_+5deg"]
    delta = project(changed, points)-project(cameras, points)
    np.testing.assert_allclose(delta[:12], 0, atol=1e-10)
    assert np.max(np.abs(delta[12:])) > 20
