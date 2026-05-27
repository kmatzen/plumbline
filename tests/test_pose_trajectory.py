"""Tests for trajectory-pose metrics (TUM-RGBD-style ATE / RPE).

These verify plumbline's ``trajectory_ate_rmse_sim3`` and
``trajectory_rpe_rmse_sim3`` produce sensible numbers on a synthetic
trajectory — and, importantly, that they match the convention used by
MonST3R / DUSt3R / SLAM papers (which all wrap ``evo`` under the hood).

The metrics live in ``src/plumbline/metrics/pose.py`` and depend on the
optional ``evo`` extra; tests skip cleanly if it's not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "evo", reason="trajectory-pose metrics need `pip install evo` (optional extra)"
)

from plumbline.metrics.pose import (
    trajectory_ate_rmse_sim3,
    trajectory_rpe_rmse_sim3,
)


def _curved_trajectory(n: int = 30, rng_seed: int = 0) -> np.ndarray:
    """Build a non-collinear, non-stationary trajectory.

    Umeyama alignment needs N ≥ 3 non-collinear correspondences — so a
    helix-like path (x=cos t, y=sin t, z=0.3 t) plus per-frame rotation.
    """
    ts = np.linspace(0.0, 2.0 * np.pi, n)
    E = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    E[:, 0, 3] = np.cos(ts)
    E[:, 1, 3] = np.sin(ts)
    E[:, 2, 3] = 0.3 * ts
    for i, t in enumerate(ts):
        c, s = np.cos(t), np.sin(t)
        E[i, :3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return E


class TestTrajectoryATE:
    """ATE-RMSE under Sim(3) Umeyama alignment, ``correct_scale=True``."""

    def test_identical_trajectory_zero_error(self) -> None:
        E = _curved_trajectory()
        ate = trajectory_ate_rmse_sim3(E, E)
        assert ate < 1e-5, f"identical trajectory should produce ~0 ATE, got {ate}"

    def test_sim3_equivalent_pred_absorbed(self) -> None:
        """Pred = scale × R × gt + t with arbitrary (s, R, t) → ATE ≈ 0."""
        E_gt = _curved_trajectory()
        # Pick a non-identity Sim(3): scale 3.7, rotation about z by 0.7 rad, translate (1.5, -2, 0.5).
        theta = 0.7
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        t_off = np.array([1.5, -2.0, 0.5])
        s = 3.7
        E_pred = E_gt.copy()
        for i in range(E_gt.shape[0]):
            E_pred[i, :3, 3] = s * R @ E_gt[i, :3, 3] + t_off
            E_pred[i, :3, :3] = R @ E_gt[i, :3, :3]
        ate = trajectory_ate_rmse_sim3(E_pred, E_gt)
        assert ate < 1e-4, f"Sim(3)-equivalent pred should give ~0 ATE, got {ate}"

    def test_gaussian_translation_noise(self) -> None:
        """ATE scales with noise magnitude."""
        E_gt = _curved_trajectory(n=50)
        rng = np.random.default_rng(42)
        E_noisy = E_gt.copy()
        E_noisy[:, :3, 3] += rng.normal(scale=0.05, size=(50, 3))
        ate = trajectory_ate_rmse_sim3(E_noisy, E_gt)
        # 0.05-stddev iso noise on translation → ATE-RMSE in same order.
        # Sim(3) absorbs a tiny constant offset; we expect ~0.05.
        assert 0.02 < ate < 0.1, f"noisy ATE out of expected range: {ate}"


class TestTrajectoryRPE:
    """RPE-RMSE at delta=1, ``all_pairs=True``, ``correct_scale=True``."""

    def test_identical_zero(self) -> None:
        E = _curved_trajectory()
        rpe_t, rpe_r = trajectory_rpe_rmse_sim3(E, E)
        assert rpe_t < 1e-5
        assert rpe_r < 1e-3  # rotation in degrees, still very small

    def test_constant_scale_pred_absorbed(self) -> None:
        """RPE under Sim(3) alignment absorbs a constant scale on pred."""
        E_gt = _curved_trajectory()
        E_pred = E_gt.copy()
        E_pred[:, :3, 3] *= 2.5
        rpe_t, rpe_r = trajectory_rpe_rmse_sim3(E_pred, E_gt)
        assert rpe_t < 1e-4, f"scale-only pred should give ~0 RPE-trans, got {rpe_t}"
        # Rotations are scale-invariant in any case.
        assert rpe_r < 1e-3

    def test_per_frame_rotation_perturbation(self) -> None:
        """A small constant rotation perturbation between consecutive frames
        produces a measurable RPE-rot (in degrees)."""
        E_gt = _curved_trajectory(n=20)
        # Add a 1° rotation about z per frame to pred.
        delta_theta = np.deg2rad(1.0)
        dR = np.array(
            [
                [np.cos(delta_theta), -np.sin(delta_theta), 0],
                [np.sin(delta_theta), np.cos(delta_theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        E_pred = E_gt.copy()
        for i in range(E_gt.shape[0]):
            # Stack i rotations of dR
            R_extra = np.linalg.matrix_power(dR, i)
            E_pred[i, :3, :3] = R_extra @ E_gt[i, :3, :3]
        _, rpe_r = trajectory_rpe_rmse_sim3(E_pred, E_gt)
        # Each consecutive pair has a +1° relative rotation drift in pred → RPE-rot ≈ 1°.
        # (Sim(3) alignment can null out one global rotation but not the
        # per-frame drift.)
        assert 0.5 < rpe_r < 2.0, (
            f"~1°/frame drift should yield RPE-rot ~1°; got {rpe_r}"
        )


def test_too_few_frames_raises_or_nans() -> None:
    """ATE/RPE require N ≥ 3 non-collinear frames. Smaller inputs should
    fail loudly (not silently produce 0)."""
    E = np.tile(np.eye(4), (2, 1, 1)).astype(np.float64)
    E[1, 0, 3] = 1.0
    with pytest.raises(Exception):  # noqa: B017 — any failure mode is fine
        # 2 frames is well below the 3 Umeyama needs.
        trajectory_ate_rmse_sim3(E, E)
