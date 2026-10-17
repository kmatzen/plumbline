import numpy as np
from paper.compare_davis_geometry import symmetric_distances


def test_symmetric_pixel_distance_has_known_units():
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = 1
    first = np.array([[20, 30], [40, 50]])
    second = np.array([[10, 32], [30, 50]])
    result = symmetric_distances(first, second, np.eye(3), np.eye(3), (100, 100), (100, 100), a, b)
    assert np.allclose(result, [2, 0])


def test_no_baseline_is_unavailable():
    result = symmetric_distances(
        np.array([[20, 30]]),
        np.array([[20, 30]]),
        np.eye(3),
        np.eye(3),
        (100, 100),
        (100, 100),
        np.eye(4),
        np.eye(4),
    )
    assert result is None


def test_view_swap_preserves_symmetric_distance():
    a, b = np.eye(4), np.eye(4)
    b[:3, 3] = [1, 0.2, 0.1]
    first, second = np.array([[20, 30]]), np.array([[10, 32]])
    forward = symmetric_distances(first, second, np.eye(3), np.eye(3), (100, 100), (200, 100), a, b)
    reverse = symmetric_distances(second, first, np.eye(3), np.eye(3), (200, 100), (100, 100), b, a)
    assert np.allclose(forward, reverse)
