import numpy as np
import pytest

from plumbline.image_evidence import sampson_residuals


def test_known_geometry_and_wrong_rotation():
    rng = np.random.default_rng(41)
    points = rng.uniform([-1, -1, 3], [1, 1, 8], size=(100, 3))
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.5
    a = points[:, :2] / points[:, 2:]
    points_b = points - second[:3, 3]
    b = points_b[:, :2] / points_b[:, 2:]
    errors = sampson_residuals(a, b, first, second)
    assert errors is not None
    np.testing.assert_allclose(errors, 0, atol=1e-12)
    wrong = second.copy()
    angle = 0.2
    wrong[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    wrong_errors = sampson_residuals(a, b, first, wrong)
    assert np.median(wrong_errors) > 0.005
    # Epipolar geometry cannot determine baseline length.
    second[0, 3] *= 1e-8
    np.testing.assert_allclose(sampson_residuals(a, b, first, second), errors, atol=1e-12)


def test_zero_baseline_abstains_and_nonfinite_rejected():
    assert sampson_residuals(np.zeros((5, 2)), np.zeros((5, 2)), np.eye(4), np.eye(4)) is None
    with pytest.raises(ValueError, match="finite"):
        sampson_residuals(np.full((5, 2), np.nan), np.zeros((5, 2)), np.eye(4), np.eye(4))
