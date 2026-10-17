import numpy as np

from plumbline.cheirality import triangulation_diagnostics
from plumbline.image_evidence import sampson_residuals


def test_rotated_camera_recovers_known_ray_lengths_and_view_swap():
    p, q = np.eye(4), np.eye(4)
    angle = 0.2
    c, s = np.cos(angle), np.sin(angle)
    q[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    q[:3, 3] = [0.7, 0.2, 0.1]
    points = np.array([[0.2, 0.1, 4], [1, -0.5, 8]])
    target = (points - q[:3, 3]) @ q[:3, :3]
    a, b = points[:, :2] / points[:, 2:], target[:, :2] / target[:, 2:]
    result = triangulation_diagnostics(a, b, p, q)
    reverse = triangulation_diagnostics(b, a, q, p)
    expected = np.c_[np.linalg.norm(points, axis=1), np.linalg.norm(target, axis=1)]
    expected /= np.linalg.norm(q[:3, 3])
    assert np.allclose(result["depths_baseline_units"], expected)
    assert np.allclose(reverse["depths_baseline_units"], expected[:, ::-1])
    assert result["positive_both"].all()


def test_sign_reversal_is_invisible_to_epipolar_error_but_flips_depth():
    p, q = np.eye(4), np.eye(4)
    q[0, 3] = 1
    points = np.array([[0.2, 0.1, 4], [1, -0.5, 8], [-0.3, 1, 3]])
    a = points[:, :2] / points[:, 2:]
    shifted = points - q[:3, 3]
    b = shifted[:, :2] / shifted[:, 2:]
    native = triangulation_diagnostics(a, b, p, q)
    q[0, 3] = -1
    reversed_result = triangulation_diagnostics(a, b, p, q)
    assert native["positive_both"].all()
    assert reversed_result["negative_both"].all()
    assert np.allclose(native["ray_gap_baseline_units"], 0, atol=1e-12)
    assert np.allclose(native["depths_baseline_units"], -reversed_result["depths_baseline_units"])
    assert np.allclose(sampson_residuals(a, b, p, q), 0)


def test_zero_baseline_and_parallel_rays_do_not_pass():
    p, q = np.eye(4), np.eye(4)
    a = np.array([[0.1, 0.2], [-0.3, 0.2]])
    assert triangulation_diagnostics(a, a, p, q) is None
    q[0, 3] = 1
    result = triangulation_diagnostics(a, a, p, q)
    assert not result["resolved"].any()
    assert not result["positive_both"].any()
    assert np.isnan(result["depths_baseline_units"]).all()
    # A fabricated translation still has zero epipolar residual on these rays.
    assert np.allclose(sampson_residuals(a, a, p, q), 0)


def test_scale_and_coordinate_gauge_do_not_change_diagnostics():
    p, q = np.eye(4), np.eye(4)
    q[:3, 3] = [1, 0.2, 0]
    a, b = np.array([[0.2, 0.3]]), np.array([[0.1, 0.25]])
    original = triangulation_diagnostics(a, b, p, q)
    q[:3, 3] *= 7
    gauge = np.eye(4)
    gauge[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    gauge[:3, 3] = [5, 3, -1]
    changed = triangulation_diagnostics(a, b, gauge @ p, gauge @ q)
    for key in original:
        assert np.allclose(original[key], changed[key], equal_nan=True)
