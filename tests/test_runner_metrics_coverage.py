"""Coverage-focused tests for ``plumbline.runner_metrics``.

These call ``_depth_metrics`` / ``_point_cloud_metrics`` / ``_pose_metrics``
directly rather than going through ``runner.evaluate`` — the branches under
test (resize-to-GT resolution, degenerate empty inputs, malformed point
maps, single-view pose inputs, the trajectory-metrics try/except ladder)
are unit-level concerns of these three helpers, and driving them through a
full evaluate() call would need a lot of dataset/model boilerplate to reach
the same lines. ``tests/test_mvs_pipeline.py`` and ``tests/test_runner.py``
already cover the integration path.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.runner_metrics import (
    _depth_metrics,
    _point_cloud_metrics,
    _pose_metrics,
)


class TestDepthMetricsEdgeCases:
    def test_empty_arrays_return_nan_for_every_key(self) -> None:
        """``pred_aligned.size == 0`` (e.g. a zero-pixel sample) must not
        raise or divide-by-zero; every metric key comes back NaN."""
        empty = np.zeros((0, 0), dtype=np.float32)
        out = _depth_metrics(
            pred=empty,
            gt=empty,
            valid=None,
            scale_alignment="none",
            delta_thresholds=(1.25, 1.25**2, 1.25**3),
        )
        assert set(out) == {
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "log10",
            "silog",
            "delta_1",
            "delta_2",
            "delta_3",
        }
        assert all(np.isnan(v) for v in out.values())

    def test_depth_clip_bounds_the_aligned_prediction(self) -> None:
        """A post-alignment clip caps aligned depth to a physical range so a
        handful of alignment-induced outliers can't blow up AbsRel."""
        gt = np.full((4, 4), 2.0, dtype=np.float32)
        pred = gt.copy()
        # One wild outlier pixel that would otherwise dominate abs_rel.
        pred[0, 0] = 1e6
        unclipped = _depth_metrics(
            pred=pred,
            gt=gt,
            valid=None,
            scale_alignment="none",
            delta_thresholds=(1.25,),
        )
        clipped = _depth_metrics(
            pred=pred,
            gt=gt,
            valid=None,
            scale_alignment="none",
            delta_thresholds=(1.25,),
            depth_clip=(0.0, 3.0),
        )
        assert clipped["abs_rel"] < unclipped["abs_rel"]

    def test_resize_prediction_to_gt_resolution(self) -> None:
        """Prediction at model-native resolution must be resized (bilinear)
        to GT resolution before metrics — GT itself is never touched."""
        gt = np.ones((1, 8, 8), dtype=np.float32)
        pred = np.ones((1, 4, 4), dtype=np.float32)  # half-resolution
        out = _depth_metrics(
            pred=pred,
            gt=gt,
            valid=None,
            scale_alignment="none",
            delta_thresholds=(1.25,),
        )
        # Both are uniformly 1.0, so a correctly-resized prediction gives a
        # perfect score regardless of the resolution mismatch.
        assert out["abs_rel"] == pytest.approx(0.0, abs=1e-6)
        assert out["delta_1"] == pytest.approx(1.0)

    def test_resize_noop_when_spatial_dims_already_match(self) -> None:
        """``_resize_depth_to_gt`` is a no-op (besides a dtype cast) when the
        trailing (H, W) already matches GT, even if the leading batch
        dimension differs — the spatial-only shortcut avoids a pointless
        PIL round-trip."""
        from plumbline.runner_metrics import _resize_depth_to_gt

        gt = np.ones((1, 4, 4), dtype=np.float32)
        pred = np.full((2, 4, 4), 3.0, dtype=np.float32)
        out = _resize_depth_to_gt(pred, gt)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, pred.astype(np.float64))

    def test_resize_raises_on_ndim_mismatch(self) -> None:
        gt = np.ones((2, 8, 8), dtype=np.float32)
        pred = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="ndim mismatch"):
            _depth_metrics(
                pred=pred,
                gt=gt,
                valid=None,
                scale_alignment="none",
                delta_thresholds=(1.25,),
            )


class TestPointCloudMetricsEdgeCases:
    def test_bad_point_map_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="point_map must end in 3"):
            _point_cloud_metrics(
                point_map=np.zeros((4, 4)),
                point_cloud_gt=np.zeros((10, 3)),
                f_score_threshold=0.05,
            )

    def test_all_nan_point_map_yields_nan_metrics(self) -> None:
        """Every predicted point invalid (NaN) — chamfer/F-score must not
        crash on an empty post-filter cloud."""
        pmap = np.full((1, 3, 3, 3), np.nan, dtype=np.float32)
        out = _point_cloud_metrics(
            point_map=pmap,
            point_cloud_gt=np.zeros((5, 3), dtype=np.float32),
            f_score_threshold=0.05,
        )
        assert set(out) == {"chamfer", "precision", "recall", "f_score"}
        assert all(np.isnan(v) for v in out.values())


class TestPoseMetricsEdgeCases:
    def test_shape_mismatch_raises(self) -> None:
        E_pred = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        E_gt = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        with pytest.raises(ValueError, match="pose shape mismatch"):
            _pose_metrics(E_pred, E_gt, (5.0,), (15.0,), "analytic", False, False)

    def test_single_view_skips_camera_zero_special_case(self) -> None:
        """When there's only one view, ``E_pred[1:]`` would be empty — the
        function instead falls back to using the (single) view directly."""
        E_pred = np.eye(4, dtype=np.float32)[None]
        E_gt = np.eye(4, dtype=np.float32)[None]
        out = _pose_metrics(E_pred, E_gt, (5.0,), (15.0,), "analytic", False, False)
        assert out["rotation_error_deg_mean"] == pytest.approx(0.0, abs=1e-6)
        # No pairwise metrics: fewer than 2 views.
        assert "pairwise_rot_err_deg_mean" not in out

    def test_trajectory_metrics_populate_ate_rpe_when_evo_available(self) -> None:
        """With >=3 views and ``trajectory_metrics=True``, ATE/RPE keys are
        present (requires the optional ``evo`` extra; see
        ``tests/test_pose_trajectory.py`` for the dedicated evo tests)."""
        pytest.importorskip("evo", reason="trajectory metrics need `pip install evo`")
        n = 5
        ts = np.linspace(0.0, 2.0 * np.pi, n)
        E = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
        E[:, 0, 3] = np.cos(ts)
        E[:, 1, 3] = np.sin(ts)
        out = _pose_metrics(E, E, (5.0,), (15.0,), "analytic", False, True)
        assert out["trajectory_ate_rmse"] == pytest.approx(0.0, abs=1e-4)
        assert "trajectory_rpe_trans_rmse" in out
        assert "trajectory_rpe_rot_deg_rmse" in out

    def test_trajectory_metrics_import_error_is_silently_skipped(self, monkeypatch) -> None:
        """If `evo` isn't importable, the trajectory keys are simply absent
        rather than crashing the whole per-sample metric computation."""
        import plumbline.runner_metrics as rm

        def _raise_import_error(*args: object, **kwargs: object) -> float:
            raise ImportError("no evo")

        monkeypatch.setattr(rm, "trajectory_ate_rmse_sim3", _raise_import_error)
        n = 5
        E = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        out = rm._pose_metrics(E, E, (5.0,), (15.0,), "analytic", False, True)
        assert "trajectory_ate_rmse" not in out

    def test_trajectory_metrics_degenerate_trajectory_yields_nan(self, monkeypatch) -> None:
        """Umeyama can fail on degenerate (collinear/stationary) trajectories
        — the failure must be recorded as NaN, not crash the sample."""
        import plumbline.runner_metrics as rm

        def _raise(*args: object, **kwargs: object) -> float:
            raise np.linalg.LinAlgError("degenerate")

        monkeypatch.setattr(rm, "trajectory_ate_rmse_sim3", _raise)
        n = 5
        E = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        out = rm._pose_metrics(E, E, (5.0,), (15.0,), "analytic", False, True)
        assert np.isnan(out["trajectory_ate_rmse"])
        assert np.isnan(out["trajectory_rpe_trans_rmse"])
        assert np.isnan(out["trajectory_rpe_rot_deg_rmse"])
