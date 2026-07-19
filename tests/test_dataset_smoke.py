"""Self-maintaining smoke net over *every* built-in dataset loader.

The dataset counterpart to :mod:`tests.test_adapter_smoke`. Per-loader
behavioural assertions live in :mod:`tests.test_datasets` (synthetic on-disk
trees); this module instead derives its coverage from the canonical discovery
list in :mod:`plumbline._discover`, so a newly-registered loader is exercised
automatically.

The failure mode this closes: ``register_builtin_adapters`` swallows import
errors by design (one broken optional loader must not take down
``list-datasets``). A dataset module that stops importing — a renamed vendor
symbol, a dropped dep — therefore vanishes from ``DATASET_REGISTRY`` silently,
and every protocol YAML naming it fails at *run* time on the box rather than in
CI. Pinning a minimum roster here turns that into a red test.

Nothing here touches disk beyond the repo: loaders are never instantiated
(construction requires a staged dataset root), so this runs CPU-only, offline,
in milliseconds.
"""

from __future__ import annotations

import pytest

from plumbline._discover import (
    builtin_adapter_modules,
    register_builtin_adapters,
)
from plumbline.datasets.base import Dataset
from plumbline.datasets.registry import DATASET_REGISTRY, list_datasets

# Import every built-in adapter so the @register_dataset decorators have run
# before pytest evaluates the parametrize lists below (collection time).
register_builtin_adapters()

# Dataset modules, filtered out of the mixed model+dataset discovery list.
_DATASET_MODULES = sorted(
    m for m in builtin_adapter_modules() if m.startswith("plumbline.datasets.")
)

# Registered dataset names, frozen at collection time.
_DATASET_NAMES = sorted(DATASET_REGISTRY)

# Floor on the roster: these must never silently vanish from the registry. Add
# to this set when you ship a loader you want pinned; it is a *minimum*, not the
# full list — the full list is whatever the registry discovers.
_MIN_EXPECTED_DATASETS = frozenset(
    {
        "7scenes",
        "bonn",
        "booster",
        "co3dv2",
        "co3dv2-vggt-pose-eval",
        "ddad-moge-eval",
        "diode",
        "diode-moge-eval",
        "dtu",
        "eth3d",
        "eth3d-moge-eval",
        "eth3d-native-depth",
        "gso",
        "ibims1",
        "kitti",
        "kitti-moge-eval",
        "nyu-cut3r-eval",
        "nyuv2",
        "realestate10k-pose-eval",
        "sintel",
        "sintel-moge-eval",
        "sun-rgbd-native",
        "tum-dynamics",
    }
)


def test_no_builtin_dataset_module_fails_to_import() -> None:
    """Every built-in dataset module imports in a clean base env."""
    failures = [
        (mod, exc)
        for mod, exc in register_builtin_adapters()
        if mod.startswith("plumbline.datasets.")
    ]
    assert failures == [], "dataset modules failed to import: " + "; ".join(
        f"{mod}: {type(exc).__name__}: {exc}" for mod, exc in failures
    )


def test_every_dataset_module_registers_a_loader() -> None:
    """No dataset module imports cleanly yet silently fails to register.

    A module can import without error but never hit its ``@register_dataset``
    decorator (class body raised, decorator dropped in an edit). Then the
    loader vanishes from the registry while nothing else fails. Some modules
    register more than one loader (e.g. ``ddad_sintel_moge_eval``), so this is
    a floor, not an equality.
    """
    assert len(_DATASET_NAMES) >= len(_DATASET_MODULES), (
        f"{len(_DATASET_MODULES)} dataset modules discovered but only "
        f"{len(_DATASET_NAMES)} loaders registered: {_DATASET_NAMES}"
    )


def test_min_expected_datasets_present() -> None:
    missing = _MIN_EXPECTED_DATASETS - set(_DATASET_NAMES)
    assert not missing, f"expected datasets missing from registry: {sorted(missing)}"


def test_list_datasets_matches_registry() -> None:
    assert list_datasets() == _DATASET_NAMES


@pytest.mark.parametrize("name", _DATASET_NAMES)
def test_loader_subclasses_dataset(name: str) -> None:
    cls = DATASET_REGISTRY[name]
    assert isinstance(cls, type)
    assert issubclass(cls, Dataset)
    # register_dataset stamps the class with its registry name.
    assert cls.name == name


@pytest.mark.parametrize("name", _DATASET_NAMES)
def test_loader_is_iterable_and_sized(name: str) -> None:
    """Every loader implements the Sample-iteration protocol the runner uses.

    The runner does ``len(ds)`` for progress and ``for s in ds`` for eval; a
    loader missing either blows up mid-run on the box.
    """
    cls = DATASET_REGISTRY[name]
    assert callable(getattr(cls, "__iter__", None)), f"{name} has no __iter__"
    assert callable(getattr(cls, "__len__", None)), f"{name} has no __len__"


def test_dataset_names_are_unique_and_normalized() -> None:
    """Registry keys are the strings protocol YAMLs and the CLI type by hand."""
    for name in _DATASET_NAMES:
        assert name == name.strip().lower(), f"{name!r} is not lower-case/stripped"
        assert "_" not in name, f"{name!r} uses '_'; registry names are hyphenated"
