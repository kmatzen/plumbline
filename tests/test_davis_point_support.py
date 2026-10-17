import numpy as np
from paper.probe_davis_point_support import pixel_calibration


def test_pixel_calibration_preserves_overlay_projection_convention():
    k = np.array([[[0.8, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]])
    original = k.copy()
    camera_point = np.array([0.2, -0.4, 3.0])
    uv = k[0] @ camera_point
    expected = uv[:2] / uv[2] * [854, 480] - 0.5
    converted = pixel_calibration(k, [60, 480, 854, 3])[0] @ camera_point
    np.testing.assert_allclose(converted[:2] / converted[2], expected)
    np.testing.assert_array_equal(k, original)
