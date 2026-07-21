"""Coverage-focused tests for ``plumbline.metrics.pose``.

``tests/test_pose_trajectory.py`` already exercises the evo-backed ATE/RPE
happy paths. This file targets what's left uncovered: shape-validation
guards, the antipodal translation convention, the ``vggt_co3d_histogram``
AUC mode, empty-input edge cases, and the degenerate-view branches of
``pairwise_relative_poses``/``pairwise_pose_errors`` — none of which need a
model or a dataset, just direct calls into the pure-numpy functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.metrics.pose import (
    accuracy_at_threshold,
    auc,
    pairwise_pose_errors,
    pairwise_relative_poses,
    pose_auc,
    rotation_error_degrees,
    translation_cosine_error,
    translation_error,
)


class TestShapeValidation:
    """Every public function must fail loudly on mismatched pred/gt shapes
    rather than silently broadcasting or truncating."""

    def test_rotation_error_degrees_shape_mismatch_raises(self) -> None:
        R_pred = np.tile(np.eye(3), (2, 1, 1))
        R_gt = np.eye(3)
        with pytest.raises(ValueError, match="rotation shape mismatch"):
            rotation_error_degrees(R_pred, R_gt)

    def test_translation_error_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="translation shape mismatch"):
            translation_error(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_translation_cosine_error_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="translation shape mismatch"):
            translation_cosine_error(np.zeros((2, 3)), np.zeros(3))

    def test_pairwise_pose_errors_shape_mismatch_raises(self) -> None:
        E_pred = np.tile(np.eye(4), (3, 1, 1))
        E_gt = np.tile(np.eye(4), (2, 1, 1))
        with pytest.raises(ValueError, match="pred/gt extrinsics shape mismatch"):
            pairwise_pose_errors(E_pred, E_gt)

    def test_pairwise_relative_poses_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"extrinsics must be \(N, 4, 4\)"):
            pairwise_relative_poses(np.eye(4))

    def test_rotation_error_degrees_rejects_non_rotation_shape(self) -> None:
        """``_extract_rot`` only accepts trailing (3,3) or (4,4); anything
        else (e.g. a bare translation vector) must fail loudly."""
        with pytest.raises(ValueError, match=r"expected \(\.\.\., 3, 3\) or"):
            rotation_error_degrees(np.zeros(3), np.zeros(3))

    def test_translation_error_rejects_non_translation_shape(self) -> None:
        """``_extract_trans`` only accepts trailing (3,) or (4,4); a raw
        rotation matrix must fail loudly rather than silently misreading it."""
        with pytest.raises(ValueError, match=r"expected \(\.\.\., 3\) or"):
            translation_error(np.zeros(5), np.zeros(5))


class TestTranslationCosineAntipodal:
    """A sign-flipped translation should read as 0deg under the antipodal
    convention (VGGT/PoseDiffusion) but ~180deg without it."""

    def test_antipodal_folds_sign_flip_to_near_zero(self) -> None:
        t_pred = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([-1.0, 0.0, 0.0])
        raw = translation_cosine_error(t_pred, t_gt, antipodal=False)
        folded = translation_cosine_error(t_pred, t_gt, antipodal=True)
        assert raw == pytest.approx(180.0, abs=1e-6)
        assert folded == pytest.approx(0.0, abs=1e-6)


class TestAucModes:
    def test_vggt_co3d_histogram_matches_bounds(self) -> None:
        errors = np.array([1.0, 2.0, 3.0, 100.0])
        out = auc(errors, [5.0], mode="vggt_co3d_histogram")
        assert 0.0 <= out[5.0] <= 1.0

    def test_vggt_co3d_histogram_empty_errors_is_nan(self) -> None:
        out = auc(np.array([np.nan, np.nan]), [5.0], mode="vggt_co3d_histogram")
        assert np.isnan(out[5.0])

    def test_vggt_co3d_histogram_rejects_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="requires t > 0"):
            auc(np.array([1.0]), [0.0], mode="vggt_co3d_histogram")

    def test_unknown_auc_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown auc mode"):
            auc(np.array([1.0]), [5.0], mode="bogus")


class TestAccuracyAtThreshold:
    def test_empty_errors_yields_nan(self) -> None:
        out = accuracy_at_threshold(np.array([np.nan, np.nan]), [15.0])
        assert np.isnan(out[15.0])


class TestPoseAuc:
    def test_metric_translation_mode(self) -> None:
        R_pred = np.tile(np.eye(3), (4, 1, 1))
        R_gt = np.tile(np.eye(3), (4, 1, 1))
        t_pred = np.zeros((4, 3))
        t_gt = np.full((4, 3), 0.01)
        out = pose_auc(R_pred, R_gt, t_pred, t_gt, thresholds=(5.0,), translation_mode="metric")
        assert 0.0 <= out[5.0] <= 1.0

    def test_unknown_translation_mode_raises(self) -> None:
        R_pred = np.eye(3)[None]
        R_gt = np.eye(3)[None]
        t = np.zeros((1, 3))
        with pytest.raises(ValueError, match="unknown translation_mode"):
            pose_auc(R_pred, R_gt, t, t, translation_mode="bogus")


class TestPairwiseDegenerateViewCounts:
    """A single-view input has no pairs; the runner still calls through
    this path when a model's max_views caps a scene down to one frame."""

    def test_single_view_returns_empty_arrays(self) -> None:
        E = np.eye(4, dtype=np.float64)[None]
        R_rel, t_rel = pairwise_relative_poses(E)
        assert R_rel.shape == (0, 3, 3)
        assert t_rel.shape == (0, 3)

    def test_pairwise_pose_errors_single_view_is_empty(self) -> None:
        E = np.eye(4, dtype=np.float64)[None]
        rot, trans = pairwise_pose_errors(E, E)
        assert rot.shape == (0,)
        assert trans.shape == (0,)
