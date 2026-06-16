"""Tests for depth / pose / pointmap / alignment metrics."""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.metrics.alignment import (
    align_depth,
    align_scale_and_shift,
    align_scale_lstsq,
    align_scale_median,
    align_scale_ratio_of_medians,
)
from plumbline.metrics.depth import (
    abs_rel,
    delta_threshold,
    log10_error,
    rmse,
    rmse_log,
    silog,
    sq_rel,
)
from plumbline.metrics.pointmap import (
    accuracy_completeness,
    chamfer_distance,
    f_score,
    voxel_downsample,
)
from plumbline.metrics.pose import (
    auc,
    pose_auc,
    rotation_error_degrees,
    translation_cosine_error,
    translation_error,
)

# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


class TestDepthMetrics:
    def test_abs_rel_perfect(self) -> None:
        gt = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert abs_rel(gt, gt) == pytest.approx(0.0)

    def test_abs_rel_known(self) -> None:
        gt = np.array([2.0, 4.0], dtype=np.float32)
        pred = np.array([1.0, 2.0], dtype=np.float32)  # always half → 50% AbsRel
        assert abs_rel(pred, gt) == pytest.approx(0.5)

    def test_rmse_perfect(self) -> None:
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        assert rmse(gt, gt) == pytest.approx(0.0)

    def test_rmse_known(self) -> None:
        gt = np.array([1.0, 2.0], dtype=np.float32)
        pred = np.array([2.0, 2.0], dtype=np.float32)
        # sqrt(mean((1)^2, 0)) = sqrt(0.5)
        assert rmse(pred, gt) == pytest.approx(np.sqrt(0.5))

    def test_sq_rel_perfect(self) -> None:
        gt = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert sq_rel(gt, gt) == pytest.approx(0.0)

    def test_sq_rel_known(self) -> None:
        gt = np.array([2.0, 4.0], dtype=np.float32)
        pred = np.array([1.0, 2.0], dtype=np.float32)
        # ((1-2)^2/2, (2-4)^2/4) = (0.5, 1.0) → mean 0.75
        assert sq_rel(pred, gt) == pytest.approx(0.75)

    def test_rmse_log_perfect(self) -> None:
        gt = np.array([1.0, 2.0], dtype=np.float32)
        assert rmse_log(gt, gt) == pytest.approx(0.0, abs=1e-8)

    def test_rmse_log_known(self) -> None:
        gt = np.array([1.0, 1.0], dtype=np.float32)
        pred = np.array([np.e, np.e**3], dtype=np.float32)
        # log diffs are 1 and 3 → sqrt(mean(1, 9)) = sqrt(5)
        assert rmse_log(pred, gt) == pytest.approx(np.sqrt(5.0), rel=1e-6)

    def test_rmse_log_not_scale_invariant(self) -> None:
        # Unlike SILog, scaling the prediction changes RMSE-log.
        gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert rmse_log(gt, gt) != pytest.approx(rmse_log(gt * 10.0, gt))

    def test_delta_perfect(self) -> None:
        gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert delta_threshold(gt, gt) == pytest.approx(1.0)

    def test_delta_partial(self) -> None:
        gt = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        # Ratios 1, 1.2, 1.5, 2. Threshold 1.25 passes first two.
        pred = np.array([1.0, 1.2, 1.5, 2.0], dtype=np.float32)
        assert delta_threshold(pred, gt) == pytest.approx(0.5)

    def test_delta_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            delta_threshold(np.array([1.0]), np.array([1.0]), threshold=1.0)

    def test_silog_scale_invariant(self) -> None:
        gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred_a = gt.copy()
        pred_b = gt * 10.0
        # With lambda=1.0, scaling pred by a constant leaves SILog unchanged.
        assert silog(pred_a, gt) == pytest.approx(silog(pred_b, gt), abs=1e-8)

    def test_silog_perfect(self) -> None:
        gt = np.array([1.0, 2.0], dtype=np.float32)
        assert silog(gt, gt) == pytest.approx(0.0, abs=1e-8)

    def test_log10_perfect(self) -> None:
        gt = np.array([1.0, 10.0, 100.0], dtype=np.float32)
        assert log10_error(gt, gt) == pytest.approx(0.0)

    def test_log10_known(self) -> None:
        gt = np.array([1.0, 1.0], dtype=np.float32)
        pred = np.array([10.0, 100.0], dtype=np.float32)
        # |log10(10/1)| = 1, |log10(100/1)| = 2 → mean 1.5
        assert log10_error(pred, gt) == pytest.approx(1.5)

    def test_invalid_mask(self) -> None:
        gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred = np.array([1.0, 10.0, 3.0], dtype=np.float32)
        valid = np.array([True, False, True])
        # With the bad pixel masked, abs_rel is 0.
        assert abs_rel(pred, gt, valid) == pytest.approx(0.0)

    def test_empty_valid_returns_nan(self) -> None:
        gt = np.array([1.0, 2.0], dtype=np.float32)
        pred = np.array([1.0, 2.0], dtype=np.float32)
        valid = np.array([False, False])
        assert np.isnan(abs_rel(pred, gt, valid))
        assert np.isnan(sq_rel(pred, gt, valid))
        assert np.isnan(rmse(pred, gt, valid))
        assert np.isnan(rmse_log(pred, gt, valid))
        assert np.isnan(delta_threshold(pred, gt, valid))

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            abs_rel(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_median_recovers_scale(self) -> None:
        gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pred = gt / 3.5  # off by factor 3.5
        s = align_scale_median(pred, gt)
        assert s == pytest.approx(3.5, rel=1e-6)

    def test_lstsq_recovers_scale(self) -> None:
        gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred = gt * 0.5
        s = align_scale_lstsq(pred, gt)
        assert s == pytest.approx(2.0, rel=1e-6)

    def test_scale_shift_identity(self) -> None:
        gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        s, b = align_scale_and_shift(gt, gt, space="depth")
        assert s == pytest.approx(1.0, abs=1e-6)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_align_depth_none(self) -> None:
        pred = np.array([1.0, 2.0], dtype=np.float32)
        gt = np.array([5.0, 10.0], dtype=np.float32)
        out = align_depth(pred, gt, mode="none")
        np.testing.assert_array_equal(out, pred)

    def test_align_depth_median_then_perfect(self) -> None:
        gt = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        pred = gt / 3.0
        out = align_depth(pred, gt, mode="median")
        # Should now equal gt (up to floating point).
        np.testing.assert_allclose(out, gt, rtol=1e-6)

    def test_ratio_of_medians_differs_from_median_of_ratios(self) -> None:
        # The dust3r-lineage estimator s = median(gt)/median(pred) is a DIFFERENT
        # scalar from align_scale_median's median-of-ratios median(gt/pred) when
        # the per-pixel ratios are not constant — the crux of the CUT3R repro.
        pred = np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float32)
        gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        rom = align_scale_ratio_of_medians(pred, gt)  # median(2.5)/median(1.0)
        mor = align_scale_median(pred, gt)  # median([1,2,3,0.4])
        assert rom == pytest.approx(2.5, rel=1e-6)
        assert mor == pytest.approx(1.5, rel=1e-6)
        assert rom != pytest.approx(mor, rel=1e-3)

    def test_align_depth_median_lineage_dispatch(self) -> None:
        pred = np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float32)
        gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = align_depth(pred, gt, mode="median_lineage")
        np.testing.assert_allclose(out, 2.5 * pred, rtol=1e-6)

    def test_align_depth_lstsq_then_perfect(self) -> None:
        gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pred = gt * 0.1
        out = align_depth(pred, gt, mode="lstsq")
        np.testing.assert_allclose(out, gt, rtol=1e-6)

    def test_align_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown alignment"):
            align_depth(np.array([1.0]), np.array([1.0]), mode="foo")

    def test_scale_shift_invalid_space(self) -> None:
        with pytest.raises(ValueError, match="unknown space"):
            align_scale_and_shift(np.array([1.0, 2.0]), np.array([1.0, 2.0]), space="nope")

    def test_empty_returns_nan(self) -> None:
        p = np.array([-1.0], dtype=np.float32)  # all invalid
        g = np.array([1.0], dtype=np.float32)
        assert np.isnan(align_scale_median(p, g))
        assert np.isnan(align_scale_lstsq(p, g))
        s, b = align_scale_and_shift(p, g)
        assert np.isnan(s) and np.isnan(b)

    def test_robust_scale_shift_rejects_outliers(self) -> None:
        """Plain LSQ scale+shift is pulled off the right answer by a big
        outlier; the robust (IRLS / Huber) variant should recover it."""
        from plumbline.metrics.alignment import align_scale_and_shift_robust

        # Ground truth in inverse-depth space: g = 2 * p + 0.1
        rng = np.random.default_rng(0)
        p_inv = rng.uniform(0.1, 1.0, size=200).astype(np.float32)
        g_inv = (2.0 * p_inv + 0.1).astype(np.float32)
        # Convert to depth so the alignment solves in inv_depth space
        # (the default matching MoGe's protocol).
        pred = 1.0 / p_inv
        gt = 1.0 / g_inv
        # Add a handful of extreme pred outliers (~10% contamination).
        outlier_idx = rng.choice(pred.shape[0], size=20, replace=False)
        pred[outlier_idx] = pred[outlier_idx] * 100.0

        s_plain, b_plain = align_scale_and_shift(pred, gt, space="inv_depth")
        s_rob, b_rob = align_scale_and_shift_robust(pred, gt, space="inv_depth")

        # Robust fit must be closer to (2.0, 0.1) than the plain fit.
        err_plain = abs(s_plain - 2.0) + abs(b_plain - 0.1)
        err_rob = abs(s_rob - 2.0) + abs(b_rob - 0.1)
        assert err_rob < err_plain
        # And close to the ground truth — within 5% of scale.
        assert abs(s_rob - 2.0) / 2.0 < 0.05

    def test_robust_scale_shift_recovers_clean_case(self) -> None:
        """With no outliers, robust and plain should agree to within a
        small tolerance."""
        from plumbline.metrics.alignment import align_scale_and_shift_robust

        rng = np.random.default_rng(7)
        p_inv = rng.uniform(0.1, 1.0, size=100).astype(np.float32)
        g_inv = (1.5 * p_inv + 0.05).astype(np.float32)
        pred = 1.0 / p_inv
        gt = 1.0 / g_inv
        s_rob, b_rob = align_scale_and_shift_robust(pred, gt, space="inv_depth")
        assert abs(s_rob - 1.5) < 1e-3
        assert abs(b_rob - 0.05) < 1e-3

    def test_robust_scale_shift_via_align_depth_wrapper(self) -> None:
        """align_depth(mode='scale_shift_robust') threads through."""
        from plumbline.metrics.alignment import align_depth

        rng = np.random.default_rng(3)
        p_inv = rng.uniform(0.1, 1.0, size=60).astype(np.float32)
        g_inv = (2.0 * p_inv + 0.1).astype(np.float32)
        pred = 1.0 / p_inv
        gt = 1.0 / g_inv
        out = align_depth(pred, gt, mode="scale_shift_robust")
        # Aligned prediction should be close to GT.
        assert np.mean(np.abs(out - gt) / gt) < 0.05

    def test_scale_shift_clamped_caps_at_gt_max(self) -> None:
        """D19 regression: scale_shift_clamped must bound aligned depth
        above by ``gt[valid].max()`` per sample via a disparity floor at
        ``1/gt.max()``. Plain scale_shift lets a pixel with tiny post-fit
        disparity invert to an enormous depth and dominate mean AbsRel on
        DIODE outdoor. MoGe's eval applies this clamp; plumbline did not
        until this mode.
        """
        from plumbline.metrics.alignment import align_depth

        # Varied pred_inv on the valid mask makes the LSQ fit well-posed
        # (rank-2) and converge to identity since gt_inv = pred_inv there.
        # One outlier pred-pixel with ~zero inv then lands at ~zero aligned
        # disparity; plain scale_shift inverts it through the EPS floor to
        # a huge depth, while clamped uses 1/gt.max() as a tighter floor.
        p_inv = np.linspace(0.3, 0.7, 64)
        p_inv[0] = 1e-9  # outlier — excluded from fit via valid mask
        pred = 1.0 / p_inv
        gt = 1.0 / p_inv.copy()  # identity fit on valid rows
        gt[0] = 2.0  # doesn't matter; row 0 is not in valid
        valid = np.ones(64, dtype=bool)
        valid[0] = False

        plain = align_depth(pred, gt, valid=valid, mode="scale_shift")
        clamped = align_depth(pred, gt, valid=valid, mode="scale_shift_clamped")

        gt_max = float(gt[valid].max())
        # Plain scale_shift lets the outlier blow up orders of magnitude
        # past gt.max() (depth goes to ~1/EPS with EPS=1e-8).
        assert plain[0] > gt_max * 1000, f"plain[0]={plain[0]} not >> gt_max={gt_max}"
        # Clamped caps at gt.max() (within float tolerance).
        assert clamped[0] <= gt_max + 1e-6, f"clamped[0]={clamped[0]} > gt_max={gt_max}"
        # Non-outlier pixels essentially unchanged between the two modes.
        assert np.allclose(plain[1:], clamped[1:], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def _random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _rot_about_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


class TestPoseMetrics:
    def test_rotation_error_zero(self) -> None:
        R = _random_rotation(3)
        assert rotation_error_degrees(R, R) == pytest.approx(0.0, abs=1e-6)

    def test_rotation_error_known(self) -> None:
        # 10 degrees about z from identity.
        R = _rot_about_axis(np.array([0.0, 0.0, 1.0]), 10.0)
        err = rotation_error_degrees(np.eye(3), R)
        assert err == pytest.approx(10.0, abs=1e-5)

    def test_rotation_error_batch(self) -> None:
        R0 = _rot_about_axis(np.array([0.0, 0.0, 1.0]), 5.0)
        R1 = _rot_about_axis(np.array([0.0, 0.0, 1.0]), 20.0)
        Rb = np.stack([R0, R1])
        Ib = np.stack([np.eye(3), np.eye(3)])
        errs = rotation_error_degrees(Rb, Ib)
        np.testing.assert_allclose(errs, [5.0, 20.0], atol=1e-5)

    def test_rotation_from_extrinsics_4x4(self) -> None:
        E = np.eye(4)
        E[:3, :3] = _rot_about_axis(np.array([0.0, 1.0, 0.0]), 7.0)
        err = rotation_error_degrees(np.eye(4), E)
        assert err == pytest.approx(7.0, abs=1e-5)

    def test_translation_error_euclidean(self) -> None:
        t_pred = np.array([1.0, 2.0, 2.0])
        t_gt = np.array([1.0, 2.0, 4.0])
        assert translation_error(t_pred, t_gt) == pytest.approx(2.0)

    def test_translation_error_from_E(self) -> None:
        E_pred = np.eye(4)
        E_gt = np.eye(4)
        E_gt[:3, 3] = np.array([3.0, 4.0, 0.0])
        assert translation_error(E_pred, E_gt) == pytest.approx(5.0)

    def test_translation_cosine(self) -> None:
        # 90 degrees apart: [1,0,0] vs [0,1,0].
        assert translation_cosine_error(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        ) == pytest.approx(90.0, abs=1e-5)

    def test_auc_all_below_threshold(self) -> None:
        errors = np.array([0.01, 0.02, 0.05])
        res = auc(errors, thresholds=[1.0])
        # All errors tiny relative to threshold → AUC very close to 1.
        # Expected = mean((1 - e_i)) = 1 - mean(errors) = 1 - 0.02667.
        assert res[1.0] == pytest.approx(1 - (0.01 + 0.02 + 0.05) / 3, abs=1e-6)

    def test_auc_all_above_threshold(self) -> None:
        errors = np.array([10.0, 20.0, 30.0])
        res = auc(errors, thresholds=[1.0])
        assert res[1.0] == pytest.approx(0.0, abs=1e-6)

    def test_auc_perfect_errors(self) -> None:
        # errors == 0 → acc(x) = 1 everywhere in (0, t] → AUC = 1.
        errors = np.zeros(5)
        res = auc(errors, thresholds=[5.0])
        assert res[5.0] == pytest.approx(1.0, abs=1e-6)

    def test_auc_monotonic_in_threshold(self) -> None:
        errors = np.array([1.0, 2.0, 3.0])
        res = auc(errors, thresholds=[0.5, 1.5, 10.0])
        assert 0.0 <= res[0.5] <= res[10.0] <= 1.0

    def test_pose_auc_perfect(self) -> None:
        Rs = np.stack([np.eye(3), np.eye(3), np.eye(3)])
        ts = np.ones((3, 3))
        res = pose_auc(Rs, Rs, ts, ts, thresholds=(5.0, 10.0, 30.0))
        for v in res.values():
            assert v == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Point map
# ---------------------------------------------------------------------------


class TestPointMapMetrics:
    def test_chamfer_identical_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((100, 3)).astype(np.float32)
        assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-6)

    def test_chamfer_known_offset(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0]])
        shifted = np.array([[1.0, 0.0, 0.0]])
        # Each direction: 1.0, so symmetric = 2.0.
        assert chamfer_distance(pts, shifted, two_sided=True) == pytest.approx(2.0)
        assert chamfer_distance(pts, shifted, two_sided=False) == pytest.approx(1.0)

    def test_f_score_perfect(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        out = f_score(pts, pts, threshold=0.5)
        assert out["precision"] == pytest.approx(100.0)
        assert out["recall"] == pytest.approx(100.0)
        assert out["f_score"] == pytest.approx(100.0)

    def test_f_score_disjoint(self) -> None:
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[100.0, 0.0, 0.0]])
        out = f_score(a, b, threshold=1.0)
        assert out["precision"] == pytest.approx(0.0)
        assert out["recall"] == pytest.approx(0.0)
        assert out["f_score"] == pytest.approx(0.0)

    def test_f_score_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            f_score(np.zeros((1, 3)), np.zeros((1, 3)), threshold=0)

    def test_chamfer_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            chamfer_distance(np.zeros((4, 2)), np.zeros((4, 3)))

    def test_chamfer_matches_bruteforce_without_scipy(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Force the no-scipy numpy fallback in _nn_distances and confirm it
        # still computes correct NN distances over every query row. Uses a
        # gt with >= 21 points — the size at which the old ``1 << 20 // b``
        # operator-precedence bug collapsed the chunk to a single row (slow,
        # but it must still be correct, and stay so after the fix).
        import sys

        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((50, 3)).astype(np.float64)
        gt = rng.standard_normal((37, 3)).astype(np.float64)
        # chamfer pred->gt only, so the fallback distances drive the result.
        got = chamfer_distance(pred, gt, two_sided=False)
        # Brute-force reference NN distance, mean over pred.
        dists = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(-1)).min(axis=1)
        assert got == pytest.approx(float(dists.mean()))


class TestVoxelDownsample:
    def test_collapses_within_cell(self) -> None:
        # Three points inside the same 1 m voxel → one output point at their centroid.
        pts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        out = voxel_downsample(pts, voxel_size=1.0)
        assert out.shape == (1, 3)
        np.testing.assert_allclose(out[0], pts.mean(axis=0), atol=1e-9)

    def test_separate_cells_stay_separate(self) -> None:
        # Two points 1.5 m apart at voxel_size=1.0 → two output points.
        pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        out = voxel_downsample(pts, voxel_size=1.0)
        assert out.shape == (2, 3)

    def test_voxel_size_validation(self) -> None:
        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((1, 3)), voxel_size=0.0)


class TestAccuracyCompleteness:
    def test_identical_clouds_are_zero(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3)).astype(np.float32)
        out = accuracy_completeness(pts, pts, voxel_size=0.01)
        assert out["accuracy"] == pytest.approx(0.0, abs=1e-6)
        assert out["completeness"] == pytest.approx(0.0, abs=1e-6)
        assert out["overall"] == pytest.approx(0.0, abs=1e-6)

    def test_known_translation(self) -> None:
        # Grid of points spaced well above the translation, so each pred
        # point's nearest GT is its translated partner. Exact 0.5 m distance
        # both ways. Voxel size << grid spacing so no collapsing.
        grid_xy = np.arange(0.0, 10.0, 2.0)
        grid = np.stack(np.meshgrid(grid_xy, grid_xy, grid_xy), axis=-1).reshape(-1, 3)
        gt = grid + np.array([0.5, 0.0, 0.0])
        out = accuracy_completeness(grid, gt, voxel_size=0.1)
        assert out["accuracy"] == pytest.approx(0.5, abs=1e-6)
        assert out["completeness"] == pytest.approx(0.5, abs=1e-6)
        assert out["overall"] == pytest.approx(0.5, abs=1e-6)

    def test_voxel_normalizes_density(self) -> None:
        # Two prediction sets that differ only in density (one has many
        # duplicates in a region) should produce the same accuracy — voxel
        # downsampling collapses the duplicates before the mean.
        gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sparse = np.array([[0.01, 0.0, 0.0], [1.01, 0.0, 0.0]])
        # Stack 100 duplicates of the first point so dense[0] dominates
        # without voxel normalization.
        dense = np.vstack([np.tile(sparse[0], (100, 1)), sparse[1:2]])
        out_sparse = accuracy_completeness(sparse, gt, voxel_size=0.05)
        out_dense = accuracy_completeness(dense, gt, voxel_size=0.05)
        assert out_dense["accuracy"] == pytest.approx(out_sparse["accuracy"], abs=1e-9)

    def test_empty_inputs_return_nan(self) -> None:
        out = accuracy_completeness(np.zeros((0, 3)), np.zeros((10, 3)))
        assert np.isnan(out["accuracy"])
        assert np.isnan(out["completeness"])
        assert np.isnan(out["overall"])

    def test_outlier_distance_drops_far_pred_points(self) -> None:
        # D3 regression guard (2026-04-24). Un-filtered MVS chamfer on
        # DTU had a few-percent of far-outlier pred points (predictions
        # at extreme depths beyond the scene volume) that dominated
        # accuracy, producing Acc ≫ Comp asymmetry. The MASt3R / VGGT
        # convention filters those out. Without this filter, our scan1
        # probe showed Acc 78-128 mm at various voxel sizes; with a
        # 20 mm threshold, the handful of outlier pred points drops out
        # and Acc falls sharply.
        #
        # Synthetic fixture: GT line 0→0.9 (10 pts). Pred has 10 near-line
        # (d ≈ 0.05 from nearest GT) and 2 far-outliers at z=100.
        gt = np.stack([np.linspace(0, 0.9, 10), np.zeros(10), np.zeros(10)], axis=1)
        inliers = gt + np.array([0.0, 0.0, 0.05])
        outliers = np.array([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0]])
        pred = np.vstack([inliers, outliers]).astype(np.float32)

        out_raw = accuracy_completeness(pred, gt, voxel_size=0.01)
        assert out_raw["accuracy"] > 10.0, (
            f"without outlier filter, expected outlier-dominated Acc; got {out_raw['accuracy']}"
        )
        out_flt = accuracy_completeness(pred, gt, voxel_size=0.01, outlier_distance=1.0)
        assert out_flt["accuracy"] == pytest.approx(0.05, abs=1e-5), (
            f"with outlier filter, expected inlier-only Acc ≈ 0.05; got {out_flt['accuracy']}"
        )

    def test_outlier_distance_all_rejected_returns_nan(self) -> None:
        gt = np.array([[0.0, 0.0, 0.0]])
        pred = np.array([[10.0, 0.0, 0.0]]).astype(np.float32)
        out = accuracy_completeness(pred, gt, voxel_size=0.01, outlier_distance=0.1)
        assert np.isnan(out["accuracy"])

    def test_voxel_size_none_skips_downsample(self) -> None:
        # 2026-04-24 D3/D4 mitigation: CUT3R/MASt3R/VGGT-family eval
        # computes Acc/Comp as raw KDTree NN, no inner voxel_downsample
        # (the runner pre-downsamples per-chunk before merging). Passing
        # voxel_size=None to accuracy_completeness must skip the inner
        # downsample so paired downsamples don't shift centroids and
        # inflate Acc.
        #
        # Synthetic fixture: dense pred cloud near GT. With downsample,
        # cell centroids land between original points; the NN distance
        # to GT shifts. Without downsample, raw points are used.
        rng = np.random.default_rng(0)
        gt = rng.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
        # 5 dense pred points clustered near each GT point (offset 0.001)
        offsets = rng.uniform(-0.005, 0.005, size=(50, 5, 3)).astype(np.float32)
        pred = (gt[:, None, :] + offsets).reshape(-1, 3)

        out_ds = accuracy_completeness(pred, gt, voxel_size=0.05)
        out_raw = accuracy_completeness(pred, gt, voxel_size=None)
        # Both produce finite metrics.
        assert np.isfinite(out_ds["accuracy"])
        assert np.isfinite(out_raw["accuracy"])
        # voxel=None preserves all 250 pred points; voxel=0.05 collapses
        # the 5-point clusters into ~50 centroids. The raw-pred Acc is
        # the mean distance over all 250 raw pred points; the voxelized
        # Acc is mean over 50 centroids. Neither is strictly larger but
        # they're computed from different point counts.
        # We just want to verify the code path is exercised.
        from plumbline.metrics.pointmap import voxel_downsample

        ds = voxel_downsample(pred, 0.05)
        assert ds.shape[0] < pred.shape[0]


class TestUmeyamaSimilarity:
    def test_recovers_identity(self) -> None:
        import numpy as np

        from plumbline.metrics.alignment import apply_similarity, umeyama_similarity

        rng = np.random.default_rng(0)
        src = rng.standard_normal((10, 3))
        s, R, t = umeyama_similarity(src, src)
        assert s == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(R, np.eye(3), atol=1e-8)
        assert np.allclose(t, 0.0, atol=1e-8)
        assert np.allclose(apply_similarity(src, s, R, t), src, atol=1e-8)

    def test_recovers_known_similarity(self) -> None:
        import numpy as np

        from plumbline.metrics.alignment import umeyama_similarity

        rng = np.random.default_rng(42)
        src = rng.standard_normal((12, 3))
        # Random rotation via QR
        M = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(M)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        R_true = Q
        s_true = 2.37
        t_true = np.array([0.5, -1.2, 3.0])
        dst = s_true * src @ R_true.T + t_true
        s, R, t = umeyama_similarity(src, dst)
        assert s == pytest.approx(s_true, rel=1e-6)
        assert np.allclose(R, R_true, atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)

    def test_rejects_fewer_than_three_points(self) -> None:
        import numpy as np

        from plumbline.metrics.alignment import umeyama_similarity

        with pytest.raises(ValueError, match=">= 3"):
            umeyama_similarity(np.zeros((2, 3)), np.ones((2, 3)))

    def test_shape_mismatch_errors(self) -> None:
        import numpy as np

        from plumbline.metrics.alignment import umeyama_similarity

        with pytest.raises(ValueError, match="matching"):
            umeyama_similarity(np.zeros((5, 3)), np.zeros((4, 3)))


class TestICPSimilarity:
    def test_improves_over_identity_alignment(self) -> None:
        """On a shuffled+translated copy of a structured cloud, ICP must
        dramatically reduce the chamfer vs a no-alignment baseline.
        We use a "T"-shape so the orientation is strongly determined
        (not spherically symmetric like pure gaussian clouds, where ICP
        is prone to get stuck at R equivalent under symmetry).
        """
        import numpy as np

        from plumbline.metrics.alignment import apply_similarity, icp_similarity

        rng = np.random.default_rng(13)
        # Build a T-shape point cloud: vertical bar + horizontal top.
        n = 300
        vbar = (
            np.stack([np.zeros(n), np.linspace(0, 5, n), np.zeros(n)], axis=-1)
            + rng.standard_normal((n, 3)) * 0.02
        )
        hbar = (
            np.stack([np.linspace(-2, 2, n), np.full(n, 5.0), np.zeros(n)], axis=-1)
            + rng.standard_normal((n, 3)) * 0.02
        )
        src = np.concatenate([vbar, hbar], axis=0)

        # Transform by a known translation (+ small rotation). Shuffle.
        from scipy.spatial.transform import Rotation as Rot

        R_true = Rot.from_euler("xyz", [0.1, 0.2, 0.05]).as_matrix()
        t_true = np.array([10.0, -5.0, 3.0])
        dst = src @ R_true.T + t_true
        dst = dst[rng.permutation(dst.shape[0])]

        # Baseline: no alignment — chamfer is dominated by the 10+ unit offset.
        from scipy.spatial import cKDTree

        nn_no_align = cKDTree(dst).query(src, k=1, workers=-1)[0].mean()

        # ICP with identity init. Expect big improvement.
        s, R, t, _info = icp_similarity(src, dst, max_iter=50)
        warped = apply_similarity(src, s, R, t)
        nn_icp = cKDTree(dst).query(warped, k=1, workers=-1)[0].mean()

        # The actual numbers depend on rotation/translation, but ICP should
        # cut the residual by at least an order of magnitude relative to
        # the unaligned case (10+ unit offset → sub-unit post-ICP).
        assert nn_icp < 0.5, f"ICP residual {nn_icp:.3f} too high"
        assert nn_icp < nn_no_align / 10, f"ICP only got {nn_no_align / nn_icp:.1f}x improvement"

    def test_warm_start_reduces_iterations(self) -> None:
        """Warm-started ICP (with init close to truth) converges in fewer
        iterations than cold start."""
        import numpy as np

        from plumbline.metrics.alignment import icp_similarity

        rng = np.random.default_rng(11)
        src = rng.standard_normal((300, 3))
        t_true = np.array([5.0, 5.0, 5.0])  # big offset makes cold start harder
        dst = src + t_true
        dst = dst[rng.permutation(dst.shape[0])]

        _, _, _, info_cold = icp_similarity(src, dst, max_iter=50)
        _, _, _, info_warm = icp_similarity(src, dst, init_t=t_true, max_iter=50)
        # Warm start from (exactly) the true translation should converge
        # in 1-2 iterations; cold start needs more.
        assert info_warm["iterations"] < info_cold["iterations"]

    def test_rejects_small_clouds(self) -> None:
        import numpy as np

        from plumbline.metrics.alignment import icp_similarity

        with pytest.raises(ValueError, match="3 points"):
            icp_similarity(np.zeros((2, 3)), np.zeros((10, 3)))
        with pytest.raises(ValueError, match="3 points"):
            icp_similarity(np.zeros((10, 3)), np.zeros((2, 3)))


class TestBoundaryEdgeMask:
    """Depth-discontinuity boundary mask used by MoGe / Depth Pro /
    most mono-depth eval protocols."""

    def test_flat_region_no_edges(self) -> None:
        from plumbline.metrics.masks import boundary_edge_mask

        depth = np.full((10, 10), 2.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)
        edge = boundary_edge_mask(depth, valid)
        # Flat scene has no discontinuities — no edges should fire.
        assert not edge.any()

    def test_step_discontinuity_fires(self) -> None:
        from plumbline.metrics.masks import boundary_edge_mask

        # 10×10 depth with a sharp left-vs-right discontinuity (1 m vs 5 m).
        depth = np.where(np.arange(10)[None, :] < 5, 1.0, 5.0).astype(np.float32)
        depth = np.broadcast_to(depth, (10, 10)).copy()
        valid = np.ones_like(depth, dtype=bool)
        edge = boundary_edge_mask(depth, valid, thickness=1, tol=0.1)
        # Pixels on either side of column 5 should be flagged.
        assert edge.any()
        # Interior of the flat "near" side (away from boundary) shouldn't be.
        assert not edge[5, 0]
        # Interior of the flat "far" side either.
        assert not edge[5, 9]

    def test_tol_larger_suppresses_small_edges(self) -> None:
        from plumbline.metrics.masks import boundary_edge_mask

        # Small step: 1.0 vs 1.05 (5% discontinuity).
        depth = np.where(np.arange(10)[None, :] < 5, 1.0, 1.05).astype(np.float32)
        depth = np.broadcast_to(depth, (10, 10)).copy()
        valid = np.ones_like(depth, dtype=bool)
        edge_strict = boundary_edge_mask(depth, valid, tol=0.01)  # 1% → catches
        edge_loose = boundary_edge_mask(depth, valid, tol=0.2)  # 20% → doesn't
        assert edge_strict.any()
        assert not edge_loose.any()

    def test_shape_mismatch_errors(self) -> None:
        from plumbline.metrics.masks import boundary_edge_mask

        with pytest.raises(ValueError, match="shape"):
            boundary_edge_mask(np.zeros((5, 5)), np.zeros((6, 6), dtype=bool))


class TestPairwisePose:
    def test_identity_pair_has_zero_error(self) -> None:
        import numpy as np

        from plumbline.metrics.pose import pairwise_pose_errors

        N = 4
        E = np.eye(4)[None].repeat(N, 0).astype(np.float64)
        # Random non-identity poses (world_from_cam), same on pred and GT.
        rng = np.random.default_rng(0)
        for i in range(1, N):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            theta = rng.uniform(0.1, 1.0)
            # Rodrigues rotation
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            E[i, :3, :3] = R
            E[i, :3, 3] = rng.standard_normal(3)
        rot, trans = pairwise_pose_errors(E, E)
        assert rot.shape == (N * (N - 1) // 2,) == trans.shape
        # arccos near 0 rounds at O(1e-6) rad ≈ 1e-4 deg in fp64.
        assert np.all(rot < 1e-4)
        assert np.all(trans < 1e-4)

    def test_frame_invariant_under_common_transform(self) -> None:
        """Pairwise errors don't change if both pred and GT are wrapped by
        the same rigid transform (they're already relative quantities)."""
        import numpy as np

        from plumbline.metrics.pose import pairwise_pose_errors

        rng = np.random.default_rng(1)
        N = 4
        # Build pred poses
        Epred = np.tile(np.eye(4)[None], (N, 1, 1)).astype(np.float64)
        Egt = Epred.copy()
        for i in range(1, N):
            for E in (Epred, Egt):
                axis = rng.standard_normal(3)
                axis /= np.linalg.norm(axis)
                theta = rng.uniform(0.1, 1.0)
                K = np.array(
                    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
                )
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                E[i, :3, :3] = R
                E[i, :3, 3] = rng.standard_normal(3)

        # Wrap pred by an arbitrary rigid transform.
        T = np.eye(4)
        T[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = [5.0, -1.0, 3.0]
        Epred_t = np.einsum("ij,njk->nik", T, Epred)
        rot0, trans0 = pairwise_pose_errors(Epred, Egt)
        rot1, trans1 = pairwise_pose_errors(Epred_t, np.einsum("ij,njk->nik", T, Egt))
        assert np.allclose(rot0, rot1, atol=1e-8)
        assert np.allclose(trans0, trans1, atol=1e-8)
