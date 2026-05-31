"""Tests for runner.evaluate() using fake models and fake datasets."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from plumbline.cache import PredictionCache
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.runner import evaluate


class _PerfectDepthModel(Model):
    """Model that predicts exactly the GT depth; metrics should be ideal."""

    name = "perfect-depth"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
    )

    def __init__(self, gt_store: dict[str, np.ndarray], device: str = "cpu") -> None:
        # Hack: the model needs per-call GT to return. Runner drives the
        # protocol via the dataset's sample, so we pass a shared store.
        self._gt_store = gt_store
        self._current_id: str | None = None

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        # We set _current_id from the outside test harness.
        depth = self._gt_store[self._current_id].copy()
        return Prediction(depth=depth)


class _ConstantDepthModel(Model):
    """Model that predicts a uniform value of 1.0 for all pixels."""

    name = "constant-depth"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
    )

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        n, h, w, _ = images.shape
        return Prediction(depth=np.ones((n, h, w), dtype=np.float32))


class _EmptyPredictionModel(Model):
    """Declares mono_depth support but returns a prediction with no depth.

    Exercises the footgun where an adapter silently emits nothing for the
    requested task — the runner must treat this as a skip, not an
    "evaluated" sample (D28).
    """

    name = "empty-prediction"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
    )

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        return Prediction(depth=None)


class _SyntheticDataset(Dataset):
    name = "synthetic"
    split = "test"

    def __init__(self, n_samples: int = 4) -> None:
        self.n = n_samples
        rng = np.random.default_rng(0)
        self._samples: list[Sample] = []
        for i in range(n_samples):
            depth = (rng.random((1, 8, 8), dtype=np.float32) + 0.5).astype(np.float32)
            self._samples.append(
                Sample(
                    sample_id=f"s{i}",
                    images=np.zeros((1, 8, 8, 3), dtype=np.uint8),
                    intrinsics=np.eye(3, dtype=np.float32)[None],
                    extrinsics_gt=np.eye(4, dtype=np.float32)[None],
                    depth_gt=depth,
                )
            )

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return self.n


class TestEvaluate:
    def test_perfect_depth_gives_ideal_metrics(self, tmp_path: Path) -> None:
        ds = _SyntheticDataset(n_samples=3)
        store = {s.sample_id: s.depth_gt for s in ds._samples}
        model = _PerfectDepthModel(store, device="cpu")

        # Wrap so we can track current_id. Monkey-patch via a closure-dataset.
        class _IDTaggedDataset(_SyntheticDataset):
            def __iter__(self) -> Iterator[Sample]:
                for sample in self._samples:
                    model._current_id = sample.sample_id
                    yield sample

        tagged = _IDTaggedDataset(n_samples=3)
        tagged._samples = ds._samples

        report = evaluate(
            model=model,
            dataset=tagged,
            tasks=["mono_depth"],
            scale_alignment="none",
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 3
        assert report.n_skipped == 0
        assert report.aggregate_metrics["abs_rel"] == 0.0
        assert report.aggregate_metrics["delta_1"] == 1.0
        assert report.model == "perfect-depth"
        assert report.scale_alignment == "none"

    def test_constant_model_with_median_alignment(self, tmp_path: Path) -> None:
        ds = _SyntheticDataset(n_samples=3)
        model = _ConstantDepthModel(device="cpu")
        report = evaluate(
            model=model,
            dataset=ds,
            tasks=["mono_depth"],
            scale_alignment="median",
            cache=PredictionCache(tmp_path),
        )
        # With median alignment, a constant prediction still has some AbsRel,
        # but it should be bounded; sanity-check finite values for each key.
        for key in ("abs_rel", "rmse", "delta_1", "delta_2", "delta_3"):
            assert np.isfinite(report.aggregate_metrics[key])

    def test_cache_hits_on_second_run(self, tmp_path: Path) -> None:
        ds = _SyntheticDataset(n_samples=2)
        model = _ConstantDepthModel(device="cpu")
        cache = PredictionCache(tmp_path)
        evaluate(model=model, dataset=ds, tasks=["mono_depth"], cache=cache)
        # Count calls to predict on the second run.
        calls = {"n": 0}
        original_predict = model.predict

        def counted(images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
            calls["n"] += 1
            return original_predict(images, intrinsics)

        model.predict = counted  # type: ignore[method-assign]
        evaluate(model=model, dataset=ds, tasks=["mono_depth"], cache=cache)
        assert calls["n"] == 0

    def test_invalid_task_raises(self, tmp_path: Path) -> None:
        ds = _SyntheticDataset(n_samples=1)
        model = _ConstantDepthModel(device="cpu")
        import pytest

        with pytest.raises(ValueError, match="does not support"):
            evaluate(model=model, dataset=ds, tasks=["pose"], cache=PredictionCache(tmp_path))

    def test_metricless_prediction_counts_as_skipped(self, tmp_path: Path) -> None:
        """A prediction that yields no metric must not inflate n_evaluated."""
        ds = _SyntheticDataset(n_samples=3)
        model = _EmptyPredictionModel(device="cpu")
        report = evaluate(
            model=model,
            dataset=ds,
            tasks=["mono_depth"],
            scale_alignment="none",
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 0
        assert report.n_skipped == 3
        # Every per-sample row is flagged skipped with a reason.
        assert all(r.skipped for r in report.per_sample)
        assert all(r.skip_reason for r in report.per_sample)

    def test_alignment_change_does_not_reinference(self, tmp_path: Path) -> None:
        """Verify cache invariance: changing alignment uses cached predictions."""
        ds = _SyntheticDataset(n_samples=2)
        model = _ConstantDepthModel(device="cpu")
        cache = PredictionCache(tmp_path)

        r1 = evaluate(
            model=model,
            dataset=ds,
            tasks=["mono_depth"],
            scale_alignment="median",
            cache=cache,
        )
        calls = {"n": 0}
        original_predict = model.predict

        def counted(images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
            calls["n"] += 1
            return original_predict(images, intrinsics)

        model.predict = counted  # type: ignore[method-assign]
        r2 = evaluate(
            model=model,
            dataset=ds,
            tasks=["mono_depth"],
            scale_alignment="lstsq",
            cache=cache,
        )
        assert calls["n"] == 0
        assert r1.aggregate_metrics["abs_rel"] != r2.aggregate_metrics["abs_rel"]
