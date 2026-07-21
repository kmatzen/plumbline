"""Coverage-focused tests for ``plumbline.runner``.

``tests/test_runner.py`` and ``tests/test_mvs_pipeline.py`` already cover the
mono-depth and standard multi-view happy paths. This file targets the large
remaining gaps: input validation, the cache-corruption and OOM/exception
skip-not-crash paths, the boundary-mask and pointcloud-alignment branches of
``_compute_metrics``, the per-view-masked scene-aggregation pipeline (CUT3R/
ETH3D-DTU lineage), and the small orchestration helpers (environment/seed
capture, view trimming, safe-len) that ``evaluate`` leans on.

Several tests call ``runner``'s internal (underscore-prefixed) helpers
directly rather than only through ``evaluate()`` — this mirrors the existing
pattern in ``test_mvs_pipeline.py`` (which monkeypatches
``runner_mod.icp_similarity`` / ``runner_mod.voxel_downsample``) and is the
most direct way to exercise pure-function edge cases (empty inputs, shape
mismatches, warning branches) without reconstructing a full evaluate() call
for each one.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import plumbline.runner as runner_mod
from plumbline.cache import PredictionCache
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.runner import evaluate


def _eye_stack(n: int) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1))


def _spread_extrinsics(n: int) -> np.ndarray:
    """Cameras placed along +X at unit spacing — non-collinear enough for
    Umeyama warm starts once n >= 3."""
    ext = _eye_stack(n)
    for i in range(n):
        ext[i, 0, 3] = float(i)
    return ext


class _OneViewDataset(Dataset):
    name = "one-view"
    split = "test"

    def __init__(self, n_samples: int = 1) -> None:
        self.n = n_samples

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(
                sample_id=f"s{i}",
                images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                intrinsics=np.eye(3, dtype=np.float32)[None],
                extrinsics_gt=np.eye(4, dtype=np.float32)[None],
                depth_gt=np.ones((1, 4, 4), dtype=np.float32),
            )

    def __len__(self) -> int:
        return self.n


class _ConstantDepthModel(Model):
    name = "const-depth"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}), is_metric=False, min_views=1, max_views=math.inf
    )

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.n_calls = 0

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        self.n_calls += 1
        n, h, w, _ = images.shape
        return Prediction(depth=np.ones((n, h, w), dtype=np.float32))


class TestInputValidation:
    def test_unknown_scale_alignment_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown scale_alignment"):
            evaluate(
                model=_ConstantDepthModel(),
                dataset=_OneViewDataset(),
                tasks=["mono_depth"],
                scale_alignment="bogus",
                cache=PredictionCache(tmp_path),
            )

    def test_unknown_aggregation_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown aggregation"):
            evaluate(
                model=_ConstantDepthModel(),
                dataset=_OneViewDataset(),
                tasks=["mono_depth"],
                aggregation="bogus",
                cache=PredictionCache(tmp_path),
            )


class TestCacheCorruptionAndOOM:
    def test_corrupt_cache_entry_is_evicted_and_reinferred(self, tmp_path: Path) -> None:
        """A truncated/corrupt cache archive must not wedge the run into
        forever-serving a load failure — it's evicted and re-inferred."""
        ds = _OneViewDataset(n_samples=2)
        model = _ConstantDepthModel()
        cache = PredictionCache(tmp_path)
        evaluate(model=model, dataset=ds, tasks=["mono_depth"], cache=cache)
        assert model.n_calls == 2

        original_load = cache.load
        evicted: list[str] = []
        original_evict = cache.evict

        def failing_load(*args, **kwargs):
            raise OSError("simulated corrupt archive")

        def tracking_evict(*args, **kwargs):
            evicted.append(kwargs.get("input_fingerprint", ""))
            return original_evict(*args, **kwargs)

        cache.load = failing_load  # type: ignore[method-assign]
        cache.evict = tracking_evict  # type: ignore[method-assign]
        try:
            report = evaluate(model=model, dataset=ds, tasks=["mono_depth"], cache=cache)
        finally:
            cache.load = original_load  # type: ignore[method-assign]

        assert model.n_calls == 4, "both samples must be re-inferred after the load failure"
        assert len(evicted) == 2
        assert report.n_evaluated == 2

    def test_oom_error_is_skipped_not_crashed(self, tmp_path: Path) -> None:
        """MemoryError (a CPU-reachable member of ``_OOM_TYPES``, unlike
        ``torch.cuda.OutOfMemoryError`` which needs a GPU) on one sample must
        be recorded as a skip and the run must continue to the next sample."""

        class _FlakyModel(_ConstantDepthModel):
            name = "flaky"

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                self.n_calls += 1
                if self.n_calls == 1:
                    raise MemoryError("simulated OOM")
                return super().predict(images, intrinsics)

        report = evaluate(
            model=_FlakyModel(),
            dataset=_OneViewDataset(n_samples=2),
            tasks=["mono_depth"],
            cache=PredictionCache(tmp_path),
        )
        assert report.n_skipped == 1
        assert report.n_evaluated == 1
        skipped = [r for r in report.per_sample if r.skipped]
        assert skipped[0].skip_reason == "OOM or adapter error (see logs)"

    def test_adapter_exception_is_skipped_not_crashed(self, tmp_path: Path) -> None:
        """A non-OOM exception from the adapter (a bug, a malformed input)
        must also be a per-sample skip, not a run-ending crash."""

        class _BrokenModel(_ConstantDepthModel):
            name = "broken"

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                raise RuntimeError("adapter bug")

        report = evaluate(
            model=_BrokenModel(),
            dataset=_OneViewDataset(n_samples=2),
            tasks=["mono_depth"],
            cache=PredictionCache(tmp_path),
        )
        assert report.n_skipped == 2
        assert report.n_evaluated == 0

    def test_cache_save_failure_is_a_warning_not_a_crash(self, tmp_path: Path) -> None:
        """A full disk / permissions error while writing the cache entry must
        not prevent the prediction from being scored this run."""
        cache = PredictionCache(tmp_path)

        def failing_save(*args, **kwargs):
            raise OSError("disk full")

        cache.save = failing_save  # type: ignore[method-assign]
        report = evaluate(
            model=_ConstantDepthModel(),
            dataset=_OneViewDataset(),
            tasks=["mono_depth"],
            cache=cache,
        )
        assert report.n_evaluated == 1

    def test_below_min_views_is_skipped(self, tmp_path: Path) -> None:
        """A model that declares ``min_views`` higher than what the dataset
        provides must skip rather than call ``predict`` with too few views."""

        class _NeedsFiveViews(_ConstantDepthModel):
            name = "needs-5-views"
            capabilities = ModelCapabilities(
                tasks=frozenset({"mono_depth"}), is_metric=False, min_views=5, max_views=math.inf
            )

        model = _NeedsFiveViews()
        report = evaluate(
            model=model,
            dataset=_OneViewDataset(),
            tasks=["mono_depth"],
            cache=PredictionCache(tmp_path),
        )
        assert model.n_calls == 0
        assert report.n_skipped == 1
        assert report.n_evaluated == 0


class TestBoundaryMaskIntegration:
    def test_mask_boundaries_runs_end_to_end(self, tmp_path: Path) -> None:
        """``mask_boundaries=True`` must AND the MoGe-style edge mask into
        the valid mask before computing depth metrics, without crashing on
        a GT depth map that actually has discontinuities."""
        rng = np.random.default_rng(0)
        depth = np.ones((2, 12, 12), dtype=np.float32)
        depth[:, :, 6:] = 3.0  # a hard depth step down the middle

        class _StepDataset(Dataset):
            name = "step-depth"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield Sample(
                    sample_id="s0",
                    images=np.zeros((2, 12, 12, 3), dtype=np.uint8),
                    intrinsics=np.eye(3, dtype=np.float32)[None].repeat(2, axis=0),
                    extrinsics_gt=_eye_stack(2),
                    depth_gt=depth,
                )

            def __len__(self) -> int:
                return 1

        class _NoisyModel(_ConstantDepthModel):
            name = "noisy"

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                n, h, w, _ = images.shape
                noise = rng.normal(scale=0.01, size=(n, h, w)).astype(np.float32)
                return Prediction(depth=depth + noise)

        report = evaluate(
            model=_NoisyModel(),
            dataset=_StepDataset(),
            tasks=["mono_depth"],
            scale_alignment="none",
            mask_boundaries=True,
            boundary_thickness=1,
            boundary_tol=0.1,
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 1
        assert np.isfinite(report.aggregate_metrics["abs_rel"])


class TestSceneAggregationSkipAndWarnPaths:
    def test_scene_mode_skips_sample_with_no_recoverable_point_map(self, tmp_path: Path) -> None:
        """``aggregation='scene'`` with a GT point cloud but a prediction
        that offers neither ``point_map`` nor depth+K+E must be recorded as
        a skip, not silently dropped from the scene accumulation."""

        class _EmptyPointCloudDataset(Dataset):
            name = "empty-pc"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield Sample(
                    sample_id="scene0/s0",
                    images=np.zeros((2, 4, 4, 3), dtype=np.uint8),
                    intrinsics=np.eye(3, dtype=np.float32)[None].repeat(2, axis=0),
                    extrinsics_gt=_spread_extrinsics(2),
                    point_cloud_gt=np.array(
                        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
                    ),
                )

            def __len__(self) -> int:
                return 1

        class _NothingModel(Model):
            name = "nothing"
            version = "1.0"
            capabilities = ModelCapabilities(
                tasks=frozenset({"mvs_depth"}), is_metric=True, min_views=1, max_views=math.inf
            )

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                return Prediction()

        report = evaluate(
            model=_NothingModel(),
            dataset=_EmptyPointCloudDataset(),
            tasks=["mvs_depth"],
            aggregation="scene",
            cache=PredictionCache(tmp_path),
        )
        assert report.n_skipped == 1
        assert report.n_evaluated == 0
        assert report.per_sample[0].skip_reason == "no point map (missing depth or K/E)"

    def test_scene_mode_drops_nonfinite_scene_before_aggregating(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A scene whose GT point cloud is empty produces a NaN 'overall'
        for that scene (matches ``accuracy_completeness``'s own empty-input
        contract). The cross-scene aggregate must skip it via nanmean and
        log which scene(s) were dropped — one divergent scan must not
        poison the whole benchmark number (D-series regression class)."""

        class _TwoSceneDataset(Dataset):
            name = "two-scene"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                good_pcd = np.array(
                    [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32
                )
                yield Sample(
                    sample_id="good_scene/s0",
                    images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                    intrinsics=np.eye(3, dtype=np.float32)[None],
                    extrinsics_gt=_eye_stack(1),
                    point_cloud_gt=good_pcd,
                )
                yield Sample(
                    sample_id="bad_scene/s0",
                    images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                    intrinsics=np.eye(3, dtype=np.float32)[None],
                    extrinsics_gt=_eye_stack(1),
                    point_cloud_gt=np.zeros((0, 3), dtype=np.float32),
                )

            def __len__(self) -> int:
                return 2

        class _ZeroPointMapModel(Model):
            name = "zero-pointmap"
            version = "1.0"
            capabilities = ModelCapabilities(
                tasks=frozenset({"mvs_depth"}), is_metric=True, min_views=1, max_views=math.inf
            )

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                n, h, w, _ = images.shape
                return Prediction(point_map=np.zeros((n, h, w, 3), dtype=np.float32))

        with caplog.at_level("WARNING"):
            report = evaluate(
                model=_ZeroPointMapModel(),
                dataset=_TwoSceneDataset(),
                tasks=["mvs_depth"],
                aggregation="scene",
                scene_voxel_size=0.1,
                cache=PredictionCache(tmp_path),
            )

        assert set(report.per_scene_metrics) == {"good_scene", "bad_scene"}
        assert np.isnan(report.per_scene_metrics["bad_scene"]["overall"])
        assert np.isfinite(report.per_scene_metrics["good_scene"]["overall"])
        assert np.isfinite(report.aggregate_metrics["overall"])
        assert report.aggregate_metrics["overall"] == pytest.approx(
            report.per_scene_metrics["good_scene"]["overall"]
        )
        assert any("dropped" in r.getMessage() for r in caplog.records)


def _per_view_masked_sample(
    sample_id: str, *, n_views: int = 3, size: int = 6, depth: float = 2.0, valid: bool = True
) -> Sample:
    return Sample(
        sample_id=sample_id,
        images=np.zeros((n_views, size, size, 3), dtype=np.uint8),
        intrinsics=np.tile(
            np.array([[10.0, 0, size / 2], [0, 10.0, size / 2], [0, 0, 1]], dtype=np.float32),
            (n_views, 1, 1),
        ),
        extrinsics_gt=_eye_stack(n_views),
        depth_gt=np.full((n_views, size, size), depth, dtype=np.float32),
        depth_valid=np.full((n_views, size, size), valid, dtype=bool),
    )


class _PointMapMVSModel(Model):
    """Returns a spread point_map (not all-zero) + identity extrinsics."""

    name = "pointmap-mvs"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mvs_depth"}), is_metric=True, min_views=1, max_views=math.inf
    )

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        n, h, w, _ = images.shape
        rng = np.random.default_rng(7)
        pmap = rng.uniform(-1.0, 1.0, size=(n, h, w, 3)).astype(np.float32)
        return Prediction(point_map=pmap, extrinsics=_eye_stack(n))


class _DepthKEModel(Model):
    """Returns depth + K + E but no point_map — forces back-projection."""

    name = "depth-ke"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mvs_depth"}), is_metric=True, min_views=1, max_views=math.inf
    )

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        n, h, w, _ = images.shape
        K = np.tile(
            np.array([[10.0, 0, w / 2], [0, 10.0, h / 2], [0, 0, 1]], dtype=np.float32),
            (n, 1, 1),
        )
        return Prediction(
            depth=np.full((n, h, w), 2.0, dtype=np.float32),
            intrinsics=K,
            extrinsics=_eye_stack(n),
        )


class TestPerViewMaskedSceneAggregation:
    """CUT3R/MASt3R/VGGT-family DTU-eval lineage: per-view masked pred+GT
    clouds accumulated per scene, then ICP + chamfer at scene-merge time.
    ``test_mvs_pipeline.py`` covers the non-per-view-masked scene path;
    this class covers the ``per_view_masked=True`` branch, which was
    entirely unexercised."""

    def test_accumulates_and_merges_without_icp(self, tmp_path: Path) -> None:
        class _DS(Dataset):
            name = "pvm-ds"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield _per_view_masked_sample("scene0/w0")
                yield _per_view_masked_sample("scene0/w1")

            def __len__(self) -> int:
                return 2

        report = evaluate(
            model=_PointMapMVSModel(),
            dataset=_DS(),
            tasks=["mvs_depth"],
            aggregation="scene",
            per_view_masked=True,
            per_view_crop=4,
            scene_voxel_size=0.2,
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 2
        assert report.n_skipped == 0
        assert "scene0" in report.per_scene_metrics
        assert np.isfinite(report.per_scene_metrics["scene0"]["overall"])

    def test_scene_merge_with_icp_runs_bbox_warm_start(self, tmp_path: Path) -> None:
        class _DS(Dataset):
            name = "pvm-icp-ds"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield _per_view_masked_sample("scene0/w0")
                yield _per_view_masked_sample("scene0/w1")

            def __len__(self) -> int:
                return 2

        report = evaluate(
            model=_PointMapMVSModel(),
            dataset=_DS(),
            tasks=["mvs_depth"],
            aggregation="scene",
            per_view_masked=True,
            per_view_crop=4,
            pointcloud_alignment="icp",
            scene_voxel_size=0.2,
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 2
        assert np.isfinite(report.per_scene_metrics["scene0"]["overall"])

    def test_geometric_consistency_mask_runs_with_backprojected_depth(self, tmp_path: Path) -> None:
        """Exercises ``_geometric_consistency_mask`` (PatchmatchNet-style
        reprojection filtering) alongside the depth->point_map
        back-projection fallback for adapters that don't emit a dense
        point_map directly (DA3, DA-V2 metric)."""

        class _DS(Dataset):
            name = "pvm-geo-ds"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield _per_view_masked_sample("scene0/w0")

            def __len__(self) -> int:
                return 1

        report = evaluate(
            model=_DepthKEModel(),
            dataset=_DS(),
            tasks=["mvs_depth"],
            aggregation="scene",
            per_view_masked=True,
            per_view_crop=4,
            geometric_consistency=True,
            geo_num_src_views=2,
            geo_mask_thres=1,
            scene_voxel_size=0.2,
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 1

    def test_skips_when_gt_validity_mask_is_entirely_false(self, tmp_path: Path) -> None:
        class _DS(Dataset):
            name = "pvm-novalid-ds"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                yield _per_view_masked_sample("scene0/w0", valid=False)

            def __len__(self) -> int:
                return 1

        report = evaluate(
            model=_PointMapMVSModel(),
            dataset=_DS(),
            tasks=["mvs_depth"],
            aggregation="scene",
            per_view_masked=True,
            cache=PredictionCache(tmp_path),
        )
        assert report.n_skipped == 1
        assert report.n_evaluated == 0
        assert "per-view-masked" in report.per_sample[0].skip_reason


class TestAlignedPointMapInternals:
    """Direct calls into ``runner._aligned_point_map`` for branches that
    ``evaluate()`` only reaches when ``sample.point_cloud_gt`` is already
    set (so the ICP-without-GT and camera_centers-too-few-views warnings
    can't be reached end-to-end without also satisfying preconditions that
    would skip past them first)."""

    def test_camera_centers_warns_below_three_views(self, caplog: pytest.LogCaptureFixture) -> None:
        sample = _per_view_masked_sample("s", n_views=2)
        prediction = Prediction(
            point_map=np.zeros((2, 6, 6, 3), dtype=np.float32), extrinsics=_eye_stack(2)
        )
        with caplog.at_level("WARNING"):
            out = runner_mod._aligned_point_map(
                prediction=prediction, sample=sample, pointcloud_alignment="camera_centers"
            )
        assert out is not None
        assert "needs >= 3 views" in caplog.text

    def test_icp_without_point_cloud_gt_warns_and_returns_unaligned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sample = _per_view_masked_sample("s", n_views=3)
        pmap = np.ones((3, 6, 6, 3), dtype=np.float32)
        prediction = Prediction(point_map=pmap, extrinsics=_spread_extrinsics(3))
        with caplog.at_level("WARNING"):
            out = runner_mod._aligned_point_map(
                prediction=prediction, sample=sample, pointcloud_alignment="icp"
            )
        assert out is not None
        np.testing.assert_array_equal(out, pmap)
        assert "needs sample.point_cloud_gt" in caplog.text

    def test_icp_with_point_cloud_gt_aligns(self) -> None:
        from dataclasses import replace

        sample = _per_view_masked_sample("s", n_views=3)
        sample = replace(
            sample,
            extrinsics_gt=_spread_extrinsics(3),
            point_cloud_gt=np.array(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
        )
        pmap = np.zeros((3, 6, 6, 3), dtype=np.float32)
        pmap[..., 2] = 1.0
        prediction = Prediction(point_map=pmap, extrinsics=_spread_extrinsics(3))
        out = runner_mod._aligned_point_map(
            prediction=prediction, sample=sample, pointcloud_alignment="icp"
        )
        assert out is not None
        assert np.all(np.isfinite(out))

    def test_unknown_alignment_raises(self) -> None:
        sample = _per_view_masked_sample("s", n_views=1)
        prediction = Prediction(point_map=np.zeros((1, 6, 6, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="unknown pointcloud_alignment"):
            runner_mod._aligned_point_map(
                prediction=prediction, sample=sample, pointcloud_alignment="bogus"
            )


class TestBboxSimilarityWarmStart:
    def test_empty_inputs_return_identity(self) -> None:
        s, R, t = runner_mod._bbox_similarity_warm_start(np.zeros((0, 3)), np.zeros((0, 3)))
        assert s == 1.0
        np.testing.assert_array_equal(R, np.eye(3))
        np.testing.assert_array_equal(t, np.zeros(3))

    def test_scale_matches_bbox_diagonal_ratio(self) -> None:
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dst = src * 2.0
        s, R, _t = runner_mod._bbox_similarity_warm_start(src, dst)
        assert s == pytest.approx(2.0, rel=1e-6)
        np.testing.assert_array_equal(R, np.eye(3))


class TestGeometricConsistencyMaskInternals:
    def test_returns_none_when_prediction_fields_missing(self) -> None:
        prediction = Prediction(depth=None)
        out = runner_mod._geometric_consistency_mask(
            prediction=prediction,
            num_src_views=2,
            geo_pixel_thres=1.0,
            geo_depth_thres=0.01,
            geo_mask_thres=1,
        )
        assert out is None

    def test_single_view_is_trivially_all_true(self) -> None:
        prediction = Prediction(
            depth=np.full((1, 4, 4), 2.0, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32)[None],
            extrinsics=_eye_stack(1),
        )
        out = runner_mod._geometric_consistency_mask(
            prediction=prediction,
            num_src_views=4,
            geo_pixel_thres=1.0,
            geo_depth_thres=0.01,
            geo_mask_thres=1,
        )
        assert out is not None
        assert out.shape == (1, 4, 4)
        assert out.all()


class TestTrimSampleToViews:
    def test_noop_when_already_within_bound(self) -> None:
        sample = _per_view_masked_sample("s", n_views=2)
        assert runner_mod._trim_sample_to_views(sample, 4) is sample

    def test_trims_every_per_view_field(self) -> None:
        sample = _per_view_masked_sample("s", n_views=4)
        trimmed = runner_mod._trim_sample_to_views(sample, 2)
        assert trimmed.images.shape[0] == 2
        assert trimmed.intrinsics.shape[0] == 2
        assert trimmed.extrinsics_gt.shape[0] == 2
        assert trimmed.depth_gt.shape[0] == 2
        assert trimmed.depth_valid.shape[0] == 2


class TestSafeLen:
    def test_returns_none_when_len_raises_typeerror(self) -> None:
        class _NoLenDataset(Dataset):
            name = "no-len"
            split = "test"

            def __iter__(self) -> Iterator[Sample]:
                return iter(())

            def __len__(self) -> int:
                raise TypeError("length not known ahead of time")

        assert runner_mod._safe_len(_NoLenDataset()) is None


class TestEnvironmentAndSeedCapture:
    """These don't need real CUDA — the branches under test are reached by
    monkeypatching ``torch.cuda`` so the runner's environment-capture and
    seeding logic can be verified on CPU-only CI."""

    def test_detect_environment_records_cuda_details_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "fake-gpu")
        env = runner_mod._detect_environment()
        assert env.cuda_version == "12.4"
        assert env.gpu_name == "fake-gpu"

    def test_detect_environment_swallows_cuda_probe_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        def _boom() -> bool:
            raise RuntimeError("driver mismatch")

        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        env = runner_mod._detect_environment()
        assert env.torch_version == torch.__version__
        assert env.cuda_version is None
        assert env.gpu_name is None

    def test_seed_everything_seeds_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        calls: list[int] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
        runner_mod._seed_everything(123)
        # torch.manual_seed() itself unconditionally calls
        # cuda.manual_seed_all when CUDA is compiled in, on top of the
        # runner's own explicit call guarded by is_available() — assert the
        # runner's call happened at all, not an exact count tied to torch
        # internals.
        assert 123 in calls


class TestTryEmptyCudaCache:
    def test_calls_torch_empty_cache_when_cuda_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        calls = {"n": 0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1)
        )
        runner_mod._try_empty_cuda_cache()
        assert calls["n"] == 1

    def test_noop_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        runner_mod._try_empty_cuda_cache()  # must not raise


class TestApplyBoundaryMaskInternals:
    def test_batched_multiview_shape_preserved(self) -> None:
        depth = np.ones((2, 8, 8), dtype=np.float32)
        depth[:, :, 4:] = 3.0
        out = runner_mod._apply_boundary_mask(gt=depth, valid=None, thickness=1, tol=0.1)
        assert out.shape == depth.shape
        assert out.dtype == np.bool_

    def test_single_view_2d_input(self) -> None:
        depth = np.ones((8, 8), dtype=np.float32)
        depth[:, 4:] = 3.0
        out = runner_mod._apply_boundary_mask(gt=depth, valid=None, thickness=1, tol=0.1)
        assert out.shape == depth.shape
