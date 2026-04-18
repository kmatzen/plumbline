"""Geometry round-trip tests for the canonical conventions.

``plan.md § 9`` calls out coordinate-system bugs as the #1 failure mode for
3D foundation model harnesses. These tests build synthetic scenes with known
cameras + known 3D points + known depth, exercise every convention-relevant
function in ``plumbline``, and verify:

- A 3D point projects to the expected pixel and back to the original 3D.
- Two cameras looking at the same world-frame point see consistent
  (pixel, depth) pairs.
- ``rebase_to_first_camera`` preserves pairwise relative transforms.
- The COLMAP conversion used in ETH3D (``camera_from_world`` → our
  ``world_from_camera``) round-trips for a known pose.
- The Sintel cam-file convention (3x4 ``P = [R|t]`` in ``camera_from_world``
  form per the SDK) round-trips similarly.
- Reprojection between two views produces the expected pixel shift for a
  pure translation baseline.

Any convention drift in the loaders or the runner will surface here before
we touch a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.conventions import (
    assert_valid_extrinsics,
    assert_valid_intrinsics,
    camera_from_world,
    invert_pose,
    rebase_to_first_camera,
)
from plumbline.datasets.eth3d import quat_to_rot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rand_rot(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


def _project(K: np.ndarray, E_camera_from_world: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """Project world-frame 3D points via the OpenCV pinhole model.

    Returns ``(N, 3)`` with (u, v, depth-in-camera-frame).
    """
    X_h = np.concatenate([X_world, np.ones((X_world.shape[0], 1))], axis=1)
    X_cam = (E_camera_from_world @ X_h.T).T[:, :3]
    depth = X_cam[:, 2]
    uv = (K @ (X_cam / depth[:, None]).T).T[:, :2]
    return np.concatenate([uv, depth[:, None]], axis=1)


def _unproject(K: np.ndarray, E_world_from_camera: np.ndarray, uvd: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_project`: (u, v, depth) in pixels → world-frame XYZ."""
    Kinv = np.linalg.inv(K)
    uvw = np.concatenate([uvd[:, :2], np.ones((uvd.shape[0], 1))], axis=1)
    X_cam = (Kinv @ uvw.T).T * uvd[:, 2:3]
    X_h = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
    return (E_world_from_camera @ X_h.T).T[:, :3]


# ---------------------------------------------------------------------------
# Pose algebra
# ---------------------------------------------------------------------------


class TestPoseAlgebra:
    def test_invert_is_involutive(self) -> None:
        E = _pose(_rand_rot(7), np.array([1.0, -2.0, 0.5]))
        np.testing.assert_allclose(invert_pose(invert_pose(E)), E, atol=1e-12)

    def test_camera_from_world_is_inverse(self) -> None:
        E_wfc = _pose(_rand_rot(11), np.array([3.0, 1.5, -0.2]))
        E_cfw = camera_from_world(E_wfc)
        # Composition must be identity (within fp).
        np.testing.assert_allclose(E_cfw @ E_wfc, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(E_wfc @ E_cfw, np.eye(4), atol=1e-12)

    def test_rebase_preserves_relative_transforms(self) -> None:
        """After rebasing to camera 0, pairwise relative poses must be unchanged."""
        poses = np.stack([_pose(_rand_rot(i), np.array([i * 0.5, 0, 0])) for i in range(4)])
        rebased = rebase_to_first_camera(poses)
        # rebased[0] must be identity.
        np.testing.assert_allclose(rebased[0], np.eye(4), atol=1e-10)
        # rel(i→j) must be invariant under global left-multiplication.
        for i in range(poses.shape[0]):
            for j in range(poses.shape[0]):
                orig_rel = invert_pose(poses[i]) @ poses[j]
                new_rel = invert_pose(rebased[i]) @ rebased[j]
                np.testing.assert_allclose(orig_rel, new_rel, atol=1e-10)


# ---------------------------------------------------------------------------
# Single-camera projection round-trip
# ---------------------------------------------------------------------------


class TestProjectionRoundTrip:
    def test_project_unproject_monocular(self) -> None:
        """3D → (u, v, d) → 3D round-trips exactly."""
        K = _make_K(fx=500, fy=500, cx=320, cy=240)
        # World frame == camera frame (camera at origin).
        E_wfc = np.eye(4, dtype=np.float64)
        E_cfw = camera_from_world(E_wfc)
        rng = np.random.default_rng(42)
        # Points in front of the camera (+Z).
        X_world = rng.uniform(-1, 1, size=(50, 3))
        X_world[:, 2] = rng.uniform(0.5, 10.0, size=50)
        uvd = _project(K, E_cfw, X_world)
        X_back = _unproject(K, E_wfc, uvd)
        np.testing.assert_allclose(X_back, X_world, atol=1e-10)

    def test_project_unproject_non_identity_pose(self) -> None:
        """Same round-trip with a non-trivial world-from-camera pose."""
        K = _make_K(fx=600, fy=600, cx=256, cy=256)
        E_wfc = _pose(_rand_rot(3), np.array([2.0, 1.0, -0.5]))
        E_cfw = camera_from_world(E_wfc)

        # Construct points in camera frame, then transform to world.
        rng = np.random.default_rng(99)
        X_cam = rng.uniform(-0.5, 0.5, size=(30, 3))
        X_cam[:, 2] = rng.uniform(1.0, 5.0, size=30)
        X_world = (E_wfc @ np.concatenate([X_cam, np.ones((30, 1))], axis=1).T).T[:, :3]
        uvd = _project(K, E_cfw, X_world)
        X_back = _unproject(K, E_wfc, uvd)
        np.testing.assert_allclose(X_back, X_world, atol=1e-9)

    def test_depth_in_uvd_matches_camera_z(self) -> None:
        """The depth component of projection must equal camera-frame Z, not range."""
        K = _make_K(500, 500, 320, 240)
        E_cfw = np.eye(4)
        # A point at (2, 3, 5) in the camera frame: depth = 5 (Z), range = sqrt(4+9+25).
        X = np.array([[2.0, 3.0, 5.0]])
        uvd = _project(K, E_cfw, X)
        assert uvd[0, 2] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Multi-view consistency
# ---------------------------------------------------------------------------


class TestMultiViewConsistency:
    def test_stereo_translation_baseline(self) -> None:
        """Pure-X baseline: disparity = fx * b / depth (standard stereo formula)."""
        K = _make_K(fx=500, fy=500, cx=320, cy=240)
        baseline = 0.1  # 10 cm
        E0_wfc = np.eye(4)
        # Second camera offset in +X in the world frame.
        E1_wfc = np.eye(4)
        E1_wfc[0, 3] = baseline

        E0_cfw = camera_from_world(E0_wfc)
        E1_cfw = camera_from_world(E1_wfc)

        # A point at depth 2 m straight ahead.
        X_w = np.array([[0.0, 0.0, 2.0]])
        uvd0 = _project(K, E0_cfw, X_w)
        uvd1 = _project(K, E1_cfw, X_w)

        # Camera 1 is offset +X; the projected pixel moves to the LEFT (smaller u).
        disparity = uvd0[0, 0] - uvd1[0, 0]
        expected = K[0, 0] * baseline / X_w[0, 2]
        assert disparity == pytest.approx(expected, rel=1e-10)

    def test_point_triangulation_across_views(self) -> None:
        """A world point unprojected from each of two views gives the same 3D."""
        K = _make_K(fx=500, fy=500, cx=320, cy=240)

        # Two cameras with distinct poses (both looking roughly toward the world origin).
        R0 = _rand_rot(1)
        R1 = _rand_rot(2)
        t0 = np.array([0.0, 0.0, -5.0])
        t1 = np.array([0.5, -0.2, -5.0])
        E0_cfw = _pose(R0, t0)
        E1_cfw = _pose(R1, t1)
        E0_wfc = invert_pose(E0_cfw)
        E1_wfc = invert_pose(E1_cfw)

        X_w = np.array([[0.1, 0.2, 0.3]])
        uvd0 = _project(K, E0_cfw, X_w)
        uvd1 = _project(K, E1_cfw, X_w)
        X_back_0 = _unproject(K, E0_wfc, uvd0)
        X_back_1 = _unproject(K, E1_wfc, uvd1)
        np.testing.assert_allclose(X_back_0, X_w, atol=1e-10)
        np.testing.assert_allclose(X_back_1, X_w, atol=1e-10)
        np.testing.assert_allclose(X_back_0, X_back_1, atol=1e-10)


# ---------------------------------------------------------------------------
# Loader-shaped conversions: COLMAP quaternion (ETH3D) and Sintel .cam
# ---------------------------------------------------------------------------


class TestLoaderConventionRoundTrips:
    def test_colmap_quaternion_identity(self) -> None:
        """COLMAP's (qw,qx,qy,qz) = (1,0,0,0) → identity rotation."""
        R = quat_to_rot(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_colmap_round_trip(self) -> None:
        """A known rotation → quaternion → rotation is preserved."""
        # Use a 90° rotation about +Y (easy to reason about).
        theta = np.pi / 2
        expected_R = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ]
        )
        # COLMAP quaternion for this rotation: qw = cos(theta/2), qy = sin(theta/2).
        q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
        got_R = quat_to_rot(q)
        np.testing.assert_allclose(got_R, expected_R, atol=1e-12)

    def test_colmap_pose_becomes_world_from_camera_after_invert(self) -> None:
        """ETH3D loader flow: COLMAP cfw → invert_pose → our wfc."""
        # Place a camera at world position (2, 0, 0) looking along -X (toward origin).
        camera_position_world = np.array([2.0, 0.0, 0.0])
        # Rotation of world_from_camera: camera's +Z points toward -X of world.
        # Start from identity and rotate so camera's -Z (forward in OpenCV) aligns with -X of world.
        # We'll just construct wfc directly and work out cfw.
        R_wfc = _rand_rot(5)
        t_wfc = camera_position_world
        E_wfc = _pose(R_wfc, t_wfc)

        # A COLMAP-style cfw = inverse of wfc.
        E_cfw = invert_pose(E_wfc)

        # Now exercise the loader's round-trip: cfw → invert_pose → wfc.
        recovered_wfc = invert_pose(E_cfw)
        np.testing.assert_allclose(recovered_wfc, E_wfc, atol=1e-12)

        # And camera_from_world(recovered_wfc) gives back the cfw we started with.
        np.testing.assert_allclose(camera_from_world(recovered_wfc), E_cfw, atol=1e-12)

    def test_sintel_cam_3x4_to_4x4_then_invert(self) -> None:
        """Sintel's .cam stores 3x4 [R|t] in camera_from_world form.

        The loader homogenizes to 4x4 and inverts to get our wfc. Verify a
        constructed example round-trips.
        """
        R_cfw = _rand_rot(13)
        t_cfw = np.array([0.3, -0.2, 2.0])
        RT_3x4 = np.concatenate([R_cfw, t_cfw[:, None]], axis=1)

        # Emulate loader: 4x4 homogenize.
        E_cfw = np.eye(4)
        E_cfw[:3, :4] = RT_3x4
        assert_valid_extrinsics(E_cfw)

        # Loader then inverts to get wfc.
        E_wfc = invert_pose(E_cfw)
        assert_valid_extrinsics(E_wfc)

        # Round-trip check.
        np.testing.assert_allclose(camera_from_world(E_wfc), E_cfw, atol=1e-12)


# ---------------------------------------------------------------------------
# Intrinsics: resize unscaling
# ---------------------------------------------------------------------------


class TestIntrinsicsUnscaling:
    def test_resize_scaling_is_linear(self) -> None:
        """K for a resized image has fx, fy, cx, cy scaled by the resize factors.

        Plan § 9: "Adapters must unscale predicted intrinsics back to
        input-image pixels before returning." This test documents the math
        so an adapter author knows the right formula.
        """
        K_native = _make_K(fx=1000, fy=1000, cx=512, cy=384)
        assert_valid_intrinsics(K_native.astype(np.float32))

        native_hw = (768, 1024)
        resized_hw = (384, 512)
        sx = resized_hw[1] / native_hw[1]
        sy = resized_hw[0] / native_hw[0]
        assert sx == sy == 0.5  # isotropic resize for simplicity

        K_resized = K_native.copy()
        K_resized[0, 0] *= sx
        K_resized[0, 2] *= sx
        K_resized[1, 1] *= sy
        K_resized[1, 2] *= sy
        assert_valid_intrinsics(K_resized.astype(np.float32))

        # A world point projected with (K_native, native_hw) and with
        # (K_resized, resized_hw) must map to the same normalized image
        # coordinate.
        X = np.array([[0.3, -0.4, 2.0]])
        uv_native = _project(K_native, np.eye(4), X)[0, :2]
        uv_resized = _project(K_resized, np.eye(4), X)[0, :2]
        np.testing.assert_allclose(uv_resized, uv_native * sx, atol=1e-10)
