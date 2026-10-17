import numpy as np
import pytest
from paper.probe_davis_rotation import perturb
from scipy.spatial.transform import Rotation

from plumbline.image_evidence import target_epipolar_lines


def test_projected_static_points_lie_on_lines_with_nontrivial_rotation():
    k = np.array([[0.8, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1]])
    first, second = np.eye(4), np.eye(4)
    second[:3, :3] = Rotation.from_euler("xyz", [5, -10, 3], degrees=True).as_matrix()
    second[:3, 3] = [0.8, 0.2, 0.1]
    points = np.array([[0.1, 0.3, 5], [-0.4, 0.8, 6], [1, -0.5, 4]])

    def project(pose, size):
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        uv = local @ k.T
        return uv[:, :2] / uv[:, 2:] * size - 0.5

    a = project(first, [640, 480])
    b = project(second, [800, 600])
    lines = target_epipolar_lines(a, k, k, (640, 480), (800, 600), first, second)
    assert np.allclose(np.sum(lines * np.c_[b, np.ones(3)], axis=1), 0, atol=1e-10)
    assert np.allclose(np.linalg.norm(lines[:, :2], axis=1), 1)
    wrong = target_epipolar_lines(a, k, k, (640, 480), (800, 600), first, perturb(second, 15, 1))
    assert np.median(np.abs(np.sum(wrong * np.c_[b, np.ones(3)], axis=1))) > 1


def test_zero_translation_has_no_lines():
    assert (
        target_epipolar_lines(
            np.array([[20, 30]]), np.eye(3), np.eye(3), (100, 100), (100, 100), np.eye(4), np.eye(4)
        )
        is None
    )


def test_invalid_rotation_rejected():
    wrong = np.eye(4)
    wrong[0, 0] = 2
    with pytest.raises(AssertionError):
        target_epipolar_lines(
            np.array([[20, 30]]), np.eye(3), np.eye(3), (100, 100), (100, 100), np.eye(4), wrong
        )


def test_perturb_preserves_position_and_original():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    changed = perturb(pose, 15, 2)
    assert np.array_equal(changed[:3, 3], pose[:3, 3])
    assert np.array_equal(pose[:3, :3], np.eye(3))
    assert np.linalg.det(changed[:3, :3]) == pytest.approx(1)


def test_world_gauge_does_not_change_pixel_lines():
    p, q = np.eye(4), np.eye(4)
    q[:3, 3] = [1, 0.2, 0.4]
    pixels = np.array([[20, 30], [40, 50]])
    k = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    before = target_epipolar_lines(pixels, k, k, (100, 100), (100, 100), p, q)
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    gauge[:3, 3] = [4, 5, 6]
    after = target_epipolar_lines(pixels, k, k, (100, 100), (100, 100), gauge @ p, gauge @ q)
    assert np.allclose(before, after)


def test_point_at_epipole_is_undefined_not_zero_error():
    p, q = np.eye(4), np.eye(4)
    q[2, 3] = 1
    k = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    lines = target_epipolar_lines(np.array([[49.5, 49.5]]), k, k, (100, 100), (100, 100), p, q)
    assert np.isnan(lines).all()
