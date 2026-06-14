"""A metric model scored under a rescaling alignment must warn.

`evaluate`'s `scale_alignment` defaults to "median" regardless of `is_metric`, so
a metric reproduction that forgets `scale_alignment: none` would be silently
scale-fit to GT — hiding the model's true metric error and over-reporting it.
The runner warns in that case; here we pin the warning's presence/absence.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from plumbline.cache import PredictionCache
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.runner import evaluate

_WARN_SIG = "is metric (is_metric=True)"


class _MetricModel(Model):
    name = "warn-metric"
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}), is_metric=True, min_views=1, max_views=math.inf
    )

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None) -> Prediction:
        n, h, w, _ = images.shape
        return Prediction(depth=np.ones((n, h, w), dtype=np.float32))


class _AffineModel(_MetricModel):
    name = "warn-affine"
    capabilities = ModelCapabilities(tasks=frozenset({"mono_depth"}), is_metric=False)


class _OneSampleDataset(Dataset):
    name = "warn-synthetic"
    split = "test"

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        depth = (rng.random((1, 8, 8), dtype=np.float32) + 0.5).astype(np.float32)
        self._s = Sample(
            sample_id="s0",
            images=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            intrinsics=np.eye(3, dtype=np.float32)[None],
            extrinsics_gt=np.eye(4, dtype=np.float32)[None],
            depth_gt=depth,
        )

    def __iter__(self) -> Iterator[Sample]:
        return iter([self._s])

    def __len__(self) -> int:
        return 1


def _warned(caplog) -> bool:
    return any(_WARN_SIG in r.getMessage() for r in caplog.records)


def _evaluate(model, alignment: str, tmp_path: Path) -> None:
    evaluate(
        model=model,
        dataset=_OneSampleDataset(),
        tasks=["mono_depth"],
        scale_alignment=alignment,
        cache=PredictionCache(tmp_path),
    )


def test_metric_model_with_median_alignment_warns(caplog, tmp_path: Path) -> None:
    with caplog.at_level("WARNING"):
        _evaluate(_MetricModel(), "median", tmp_path)
    assert _warned(caplog)


def test_metric_model_with_none_alignment_is_silent(caplog, tmp_path: Path) -> None:
    with caplog.at_level("WARNING"):
        _evaluate(_MetricModel(), "none", tmp_path)
    assert not _warned(caplog)


def test_affine_model_never_warns(caplog, tmp_path: Path) -> None:
    with caplog.at_level("WARNING"):
        _evaluate(_AffineModel(), "median", tmp_path)
    assert not _warned(caplog)
