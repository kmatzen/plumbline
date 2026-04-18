"""Tests for canonical-convention assertion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    assert_valid_point_map,
    camera_from_world,
    depth_is_valid,
    invert_pose,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)


def _make_K(fx: float = 500, fy: float = 500, cx: float = 320, cy: float = 240) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float64)


def _make_E(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    R = _random_rotation() if R is None else R
    t = np.zeros(3) if t is None else t
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


class TestImage:
    def test_valid_hwc(self) -> None:
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        assert_valid_image(img)

    def test_valid_nhwc(self) -> None:
        img = np.zeros((3, 10, 20, 3), dtype=np.uint8)
        assert_valid_image(img)

    def test_wrong_dtype(self) -> None:
        img = np.zeros((10, 20, 3), dtype=np.float32)
        with pytest.raises(AssertionError, match="uint8"):
            assert_valid_image(img)

    def test_wrong_channels(self) -> None:
        img = np.zeros((10, 20, 4), dtype=np.uint8)
        with pytest.raises(AssertionError, match="3 channels"):
            assert_valid_image(img)

    def test_wrong_ndim(self) -> None:
        img = np.zeros((10, 20), dtype=np.uint8)
        with pytest.raises(AssertionError, match="ndim"):
            assert_valid_image(img)

    def test_not_ndarray(self) -> None:
        with pytest.raises(AssertionError, match=r"np\.ndarray"):
            assert_valid_image([[0, 0, 0]] * 10)  # type: ignore[arg-type]


class TestIntrinsics:
    def test_valid_single(self) -> None:
        assert_valid_intrinsics(_make_K())

    def test_valid_batch(self) -> None:
        K = np.stack([_make_K(), _make_K(600, 600, 320, 240)])
        assert_valid_intrinsics(K)

    def test_bad_bottom_row(self) -> None:
        K = _make_K()
        K[2, 0] = 0.1
        with pytest.raises(AssertionError, match=r"\[0, 0, 1\]"):
            assert_valid_intrinsics(K)

    def test_negative_focal(self) -> None:
        K = _make_K(fx=-500)
        with pytest.raises(AssertionError, match="focal"):
            assert_valid_intrinsics(K)

    def test_nan_rejected(self) -> None:
        K = _make_K()
        K[0, 2] = np.nan
        with pytest.raises(AssertionError, match="non-finite"):
            assert_valid_intrinsics(K)


class TestExtrinsics:
    def test_valid_identity(self) -> None:
        assert_valid_extrinsics(np.eye(4))

    def test_valid_random(self) -> None:
        assert_valid_extrinsics(_make_E(_random_rotation(42), np.array([1.0, 2.0, 3.0])))

    def test_valid_batch(self) -> None:
        E = np.stack([_make_E(_random_rotation(i)) for i in range(4)])
        assert_valid_extrinsics(E)

    def test_non_orthonormal(self) -> None:
        E = np.eye(4)
        E[:3, :3] = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
        with pytest.raises(AssertionError, match="orthonormal"):
            assert_valid_extrinsics(E)

    def test_left_handed(self) -> None:
        R = _random_rotation()
        R[:, 0] *= -1  # flip determinant sign
        E = _make_E(R=R)
        with pytest.raises(AssertionError, match="determinant"):
            assert_valid_extrinsics(E)

    def test_bad_bottom_row(self) -> None:
        E = np.eye(4)
        E[3, 0] = 0.1
        with pytest.raises(AssertionError, match=r"\[0, 0, 0, 1\]"):
            assert_valid_extrinsics(E)


class TestDepth:
    def test_valid(self) -> None:
        d = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
        assert_valid_depth(d)

    def test_rejects_negative(self) -> None:
        d = np.array([[1.0, -2.0]], dtype=np.float32)
        with pytest.raises(AssertionError, match="negative"):
            assert_valid_depth(d)

    def test_rejects_inf(self) -> None:
        d = np.array([[np.inf, 2.0]], dtype=np.float32)
        with pytest.raises(AssertionError, match=r"\+/-inf|non-finite"):
            assert_valid_depth(d)

    def test_integer_dtype_rejected(self) -> None:
        d = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(AssertionError, match="floating"):
            assert_valid_depth(d)

    def test_depth_is_valid_mask(self) -> None:
        d = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
        mask = depth_is_valid(d)
        assert mask.tolist() == [[True, False], [False, True]]


class TestPointMap:
    def test_valid(self) -> None:
        pmap = np.zeros((4, 4, 3), dtype=np.float32)
        assert_valid_point_map(pmap)

    def test_nan_allowed(self) -> None:
        pmap = np.zeros((4, 4, 3), dtype=np.float32)
        pmap[0, 0] = np.nan
        assert_valid_point_map(pmap)

    def test_wrong_last_dim(self) -> None:
        pmap = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(AssertionError, match="last dim must be 3"):
            assert_valid_point_map(pmap)


class TestInvertPose:
    def test_identity_is_identity(self) -> None:
        np.testing.assert_allclose(invert_pose(np.eye(4)), np.eye(4))

    def test_inverse_composes_to_identity(self) -> None:
        E = _make_E(_random_rotation(7), np.array([0.5, -1.0, 2.0]))
        result = invert_pose(E) @ E
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_batch(self) -> None:
        E = np.stack([_make_E(_random_rotation(i), np.array([i, 0.0, 0.0])) for i in range(3)])
        inv = invert_pose(E)
        result = inv @ E
        np.testing.assert_allclose(result, np.broadcast_to(np.eye(4), (3, 4, 4)), atol=1e-10)

    def test_camera_from_world_alias(self) -> None:
        E = _make_E(_random_rotation(11), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(camera_from_world(E), invert_pose(E))


class TestWorldFrame:
    def test_identity_first_camera(self) -> None:
        E = np.stack([np.eye(4), _make_E(_random_rotation(1))])
        assert world_from_camera_is_identity(E)

    def test_non_identity_first_camera(self) -> None:
        E = np.stack([_make_E(_random_rotation(1)), np.eye(4)])
        assert not world_from_camera_is_identity(E)

    def test_rebase(self) -> None:
        E0 = _make_E(_random_rotation(3), np.array([2.0, -1.0, 0.5]))
        E1 = _make_E(_random_rotation(4), np.array([0.0, 0.0, 1.0]))
        E = np.stack([E0, E1])
        rebased = rebase_to_first_camera(E)
        np.testing.assert_allclose(rebased[0], np.eye(4), atol=1e-10)
        # Relative transform (inverse of E0 applied to E1) should be preserved.
        expected_rel = invert_pose(E0) @ E1
        np.testing.assert_allclose(rebased[1], expected_rel, atol=1e-10)
