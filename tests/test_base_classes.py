"""Tests for Model/Dataset base classes and registries."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from plumbline.datasets import DATASET_REGISTRY, Dataset, Sample, register_dataset
from plumbline.models import MODEL_REGISTRY, Model, ModelCapabilities, Prediction, register_model


class _FakeMonoModel(Model):
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=math.inf,
    )
    version = "fake-1.0"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> Prediction:
        n, h, w, _ = images.shape
        depth = np.ones((n, h, w), dtype=np.float32)
        return Prediction(depth=depth)


class _FakeDataset(Dataset):
    name = "fake"
    split = "test"

    def __init__(self, n_samples: int = 5) -> None:
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


class TestPrediction:
    def test_default_all_none(self) -> None:
        p = Prediction()
        assert p.depth is None
        assert not p.has("depth")
        assert not p.has("extrinsics")

    def test_has(self) -> None:
        p = Prediction(depth=np.zeros((1, 4, 4), dtype=np.float32))
        assert p.has("depth")
        assert not p.has("point_map")

    def test_metadata_default_separate(self) -> None:
        a = Prediction()
        b = Prediction()
        a.metadata["x"] = 1
        assert "x" not in b.metadata


class TestModelCapabilities:
    def test_frozen(self) -> None:
        import dataclasses

        caps = ModelCapabilities(tasks=frozenset({"mono_depth"}), is_metric=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.is_metric = True  # type: ignore[misc]

    def test_supports_task(self) -> None:
        caps = ModelCapabilities(tasks=frozenset({"mono_depth", "pose"}), is_metric=True)
        assert caps.supports_task("mono_depth")
        assert caps.supports_task("pose")
        assert not caps.supports_task("mvs_depth")


class TestModelRegistry:
    def setup_method(self) -> None:
        self._before = dict(MODEL_REGISTRY)

    def teardown_method(self) -> None:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(self._before)

    def test_register_and_from_hub(self) -> None:
        @register_model("fake-mono")
        class FakeMono(_FakeMonoModel):
            pass

        assert "fake-mono" in MODEL_REGISTRY
        instance = Model.from_hub("fake-mono", device="cpu")
        assert instance.name == "fake-mono"

    def test_duplicate_registration_errors(self) -> None:
        @register_model("dupe-model")
        class A(_FakeMonoModel):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_model("dupe-model")
            class B(_FakeMonoModel):
                pass

    def test_unknown_from_hub(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            Model.from_hub("nonexistent-xyz")

    def test_register_rejects_non_model(self) -> None:
        with pytest.raises(TypeError):
            register_model("not-model")(int)  # type: ignore[arg-type]


class TestDatasetRegistry:
    def setup_method(self) -> None:
        self._before = dict(DATASET_REGISTRY)

    def teardown_method(self) -> None:
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(self._before)

    def test_register(self) -> None:
        @register_dataset("fake-ds")
        class FakeDS(_FakeDataset):
            pass

        assert "fake-ds" in DATASET_REGISTRY
        assert FakeDS.name == "fake-ds"


class TestSubset:
    def test_subset_size(self) -> None:
        ds = _FakeDataset(n_samples=10)
        sub = ds.subset(3)
        assert len(sub) == 3
        samples = list(sub)
        assert len(samples) == 3
        ids = [s.sample_id for s in samples]
        # Deterministic: linspace(0, 9, 3) = [0, 4.5, 9]; numpy banker's
        # rounding → [0, 4, 9].
        assert ids == ["s0", "s4", "s9"]

    def test_subset_larger_than_source(self) -> None:
        ds = _FakeDataset(n_samples=3)
        sub = ds.subset(100)
        assert len(sub) == 3

    def test_subset_zero_errors(self) -> None:
        ds = _FakeDataset(n_samples=3)
        with pytest.raises(ValueError, match="> 0"):
            ds.subset(0)

    def test_subset_twice_is_deterministic(self) -> None:
        ds = _FakeDataset(n_samples=20)
        a = [s.sample_id for s in ds.subset(5)]
        b = [s.sample_id for s in ds.subset(5)]
        assert a == b


class TestSample:
    def test_num_views(self) -> None:
        s = Sample(
            sample_id="x",
            images=np.zeros((3, 4, 4, 3), dtype=np.uint8),
            intrinsics=np.eye(3, dtype=np.float32)[None].repeat(3, axis=0),
            extrinsics_gt=np.eye(4, dtype=np.float32)[None].repeat(3, axis=0),
        )
        assert s.num_views == 3
