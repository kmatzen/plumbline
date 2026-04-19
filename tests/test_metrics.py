"""Tests for depth / pose / pointmap / alignment metrics."""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.metrics.alignment import (
    align_depth,
    align_scale_and_shift,
    align_scale_lstsq,
    align_scale_median,
)
from plumbline.metrics.depth import abs_rel, delta_threshold, log10_error, rmse, silog
from plumbline.metrics.pointmap import chamfer_distance, f_score
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
        assert np.isnan(rmse(pred, gt, valid))
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


class TestUmeyamaSimilarity:
    def test_recovers_identity(self) -> None:
        import numpy as np
        from plumbline.metrics.alignment import umeyama_similarity, apply_similarity

        rng = np.random.default_rng(0)
        src = rng.standard_normal((10, 3))
        s, R, t = umeyama_similarity(src, src)
        assert s == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(R, np.eye(3), atol=1e-8)
        assert np.allclose(t, 0.0, atol=1e-8)
        assert np.allclose(apply_similarity(src, s, R, t), src, atol=1e-8)

    def test_recovers_known_similarity(self) -> None:
        import numpy as np
        from plumbline.metrics.alignment import apply_similarity, umeyama_similarity

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
