"""Shared pytest fixtures.

Fixtures are auto-discovered by pytest, so test modules use them as parameters
without importing anything (which keeps ruff from flagging the fixture-as-arg
pattern as a redefinition). The fake model/dataset *classes* the fixtures
register live in :mod:`tests._fakes`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from tests._fakes import (
    _FakeDataset,
    _FixedDepthModel,
    _MultiViewPointCloudDataset,
    _PointMapModel,
)


@pytest.fixture
def registered_fakes() -> Iterator[None]:
    """Register the fakes in MODEL_REGISTRY and DATASET_REGISTRY for one test."""
    # Register built-ins BEFORE snapshotting so the teardown restore doesn't
    # permanently wipe them. Module imports are cached, so a later
    # register_builtin_adapters() call can't re-run the @register decorators —
    # if we snapshotted an empty registry and restored to it, every subsequent
    # test that relies on a built-in adapter would fail with a KeyError. (This
    # is order-dependent: it only bites when a fakes-using test runs before the
    # first built-in registration.)
    from plumbline._discover import register_builtin_adapters

    register_builtin_adapters()
    model_name = "test-fixed-depth"
    dataset_name = "test-synthetic"
    before_models = dict(MODEL_REGISTRY)
    before_datasets = dict(DATASET_REGISTRY)
    _FixedDepthModel.name = model_name  # type: ignore[attr-defined]
    _FakeDataset.name = dataset_name  # type: ignore[attr-defined]
    MODEL_REGISTRY[model_name] = _FixedDepthModel
    DATASET_REGISTRY[dataset_name] = _FakeDataset
    try:
        yield
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(before_models)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(before_datasets)


@pytest.fixture
def registered_pointmap_fakes() -> Iterator[None]:
    """Register point-map model + point-cloud dataset for chamfer-path tests."""
    # See registered_fakes: register built-ins before snapshotting so the
    # teardown restore can't permanently wipe them.
    from plumbline._discover import register_builtin_adapters

    register_builtin_adapters()
    model_name = "test-pointmap"
    dataset_name = "test-pointcloud"
    before_models = dict(MODEL_REGISTRY)
    before_datasets = dict(DATASET_REGISTRY)
    _PointMapModel.name = model_name  # type: ignore[attr-defined]
    _MultiViewPointCloudDataset.name = dataset_name  # type: ignore[attr-defined]
    MODEL_REGISTRY[model_name] = _PointMapModel
    DATASET_REGISTRY[dataset_name] = _MultiViewPointCloudDataset
    try:
        yield
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(before_models)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(before_datasets)


@pytest.fixture
def repro_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect reproductions discovery + prediction cache to a tmp dir.

    The cache redirect matters: ``run_reproduction`` uses the default cache
    under ``~/.cache/plumbline`` if the YAML doesn't set one, which would
    leak predictions across tests (cache key doesn't include model kwargs
    by default).
    """
    d = tmp_path / "reproductions"
    d.mkdir()
    monkeypatch.setattr("plumbline.reproduce.REPRODUCTIONS_DIR", d)
    monkeypatch.setenv("PLUMBLINE_CACHE_DIR", str(tmp_path / "cache"))
    return d
