import numpy as np
from paper.build_anchor_preview import fit_point, project


def test_fit_and_unseen_projection_recover_known_static_point():
    poses = np.tile(np.eye(4), (20, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 1, 20)
    k = np.array([[0.8, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    size = np.array([640, 480])
    point = np.array([0.2, 0.3, 4])
    pixels = np.array([project(point, p, k, size) for p in poses])
    fitted = fit_point(pixels[:11], poses[:11], np.tile(k, (11, 1, 1)), size)
    assert np.allclose(fitted, point)
    assert np.allclose(project(fitted, poses[-1], k, size), pixels[-1])
    changed = poses[-1].copy()
    changed[0, 3] += 0.5
    assert np.linalg.norm(project(fitted, changed, k, size) - pixels[-1]) > 50


def test_stationary_fit_and_behind_camera_projection_unavailable():
    poses = np.tile(np.eye(4), (11, 1, 1))
    assert (
        fit_point(np.tile([10, 20], (11, 1)), poses, np.tile(np.eye(3), (11, 1, 1)), [100, 100])
        is None
    )
    assert project(np.array([0, 0, -1]), poses[0], np.eye(3), np.array([100, 100])) is None
