"""Self-maintaining smoke net over *every* built-in model adapter.

Unlike :mod:`tests.test_model_adapters` (which holds adapter-*specific*
behavioural assertions), this module derives its coverage from the canonical
discovery list in :mod:`plumbline._discover`. A newly-registered adapter is
therefore exercised automatically — no hand-maintained list to forget to
update. That omission is exactly what let the UniK3D ``build_losses`` vendoring
bug ship: the adapter was added to ``_BUILTIN_ADAPTER_MODULES`` but not to the
smoke test's hardcoded ``EXPECTED_ADAPTERS``, so CI never imported it.

Everything here runs CPU-only and loads no weights. The base install already
carries torch + every vendored adapter's runtime deps (see ``pyproject.toml``
``[project] dependencies``), and the git/extra-installed adapters lazy-import
their heavy deps inside ``predict``/loaders — so in a clean base env *all*
adapter modules must import and *all* adapters must instantiate. Anything that
doesn't is a bug in our code or vendor tree, not a missing optional dep.

Inference correctness lives in the ``gpu``/``weights``-marked tests.
"""

from __future__ import annotations

import math

import pytest

from plumbline._discover import (
    builtin_adapter_modules,
    register_builtin_adapters,
)
from plumbline.models.base import Model, ModelCapabilities
from plumbline.models.registry import MODEL_REGISTRY

# Import every built-in adapter so the @register_model decorators have run
# before pytest evaluates the parametrize lists below (collection time).
register_builtin_adapters()

# Model adapter modules, filtered out of the mixed model+dataset discovery list.
_MODEL_MODULES = sorted(m for m in builtin_adapter_modules() if m.startswith("plumbline.models."))

# Registered adapter names, frozen at collection time.
_ADAPTER_NAMES = sorted(MODEL_REGISTRY)

# Floor on the roster: these must never silently vanish from the registry (a
# broken import would drop an adapter without otherwise failing a test). Add to
# this set when you ship an adapter you want pinned; it is a *minimum*, not the
# full list — the full list is whatever the registry discovers.
_MIN_EXPECTED_ADAPTERS = frozenset(
    {
        "depth-anything-v2",
        "metric3d-v2",
        "mast3r",
        "dust3r",
        "vggt",
        "depth-anything-3",
        "moge",
        "marigold",
        "depth-pro",
        "geowizard",
        "cut3r",
        "monst3r",
        "dage",
        "unik3d",
        "vda",
        "pi3",
        "streamvggt",
        "vggt-omega",
    }
)


def test_no_builtin_adapter_fails_to_import() -> None:
    """Every built-in adapter module imports in a clean base env.

    This is the guard that catches vendoring/import regressions (e.g. a
    vendored module referencing an un-vendored sibling, or an adapter whose
    third-party dep was never added to base). ``register_builtin_adapters``
    swallows import errors by design so one broken optional adapter can't take
    down ``list-models``; here we assert the swallowed-failure list is empty.
    """
    failures = register_builtin_adapters()
    assert failures == [], "adapter modules failed to import: " + "; ".join(
        f"{mod}: {type(exc).__name__}: {exc}" for mod, exc in failures
    )


def test_every_model_module_registers_an_adapter() -> None:
    """No model module imports cleanly yet silently fails to register.

    A module can import without error but never hit its ``@register_model``
    decorator (e.g. the class body raised, or the decorator was dropped in an
    edit). Then the adapter vanishes from the registry while every per-adapter
    parametrized test below simply doesn't get generated for it — a silent
    coverage hole. Pinning the count closes it.
    """
    assert len(_ADAPTER_NAMES) >= len(_MODEL_MODULES), (
        f"{len(_MODEL_MODULES)} model modules discovered but only "
        f"{len(_ADAPTER_NAMES)} adapters registered: {_ADAPTER_NAMES}"
    )


def test_min_expected_adapters_present() -> None:
    missing = _MIN_EXPECTED_ADAPTERS - set(_ADAPTER_NAMES)
    assert not missing, f"expected adapters missing from registry: {sorted(missing)}"


@pytest.mark.parametrize("name", _ADAPTER_NAMES)
def test_adapter_subclasses_model(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    assert isinstance(cls, type)
    assert issubclass(cls, Model)
    # register_model stamps the class with its registry name.
    assert cls.name == name


@pytest.mark.parametrize("name", _ADAPTER_NAMES)
def test_adapter_declares_valid_capabilities(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert isinstance(caps, ModelCapabilities)
    assert len(caps.tasks) > 0, f"{name} declares no tasks"
    # Tasks must be drawn from the vocabulary the runner dispatches on.
    assert caps.tasks <= {"mono_depth", "mvs_depth", "pose"}, (
        f"{name} declares unknown task(s): {caps.tasks}"
    )
    assert isinstance(caps.is_metric, bool)
    assert caps.min_views >= 1
    assert caps.max_views >= caps.min_views
    # NB: a model can advertise "mono_depth" yet still require >= 2 views —
    # vggt/mast3r/pi3 emit per-view depth from a multi-view reconstruction,
    # and vda needs a clip. So min_views is intentionally not pinned to 1 here.
    if not math.isinf(caps.max_views):
        assert caps.max_views == int(caps.max_views), f"{name} max_views not integral"


@pytest.mark.parametrize("name", _ADAPTER_NAMES)
def test_adapter_instantiates_on_cpu_without_weights(name: str) -> None:
    """``__init__(device="cpu")`` must not load weights or touch a GPU.

    Adapters defer checkpoint loading to first ``predict`` (or an explicit
    load), so construction is cheap and weight-free. This is the check that
    would have flagged the UniK3D import bug at construction time.
    """
    model = MODEL_REGISTRY[name](device="cpu")  # type: ignore[call-arg]
    assert isinstance(model, Model)
    # Instance-level capabilities (some adapters override per variant) stay
    # valid and consistent with the task vocabulary.
    assert model.capabilities.tasks <= {"mono_depth", "mvs_depth", "pose"}


@pytest.mark.parametrize("name", _ADAPTER_NAMES)
def test_adapter_config_hash_is_stable_and_wellformed(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    a = cls(device="cpu").config_hash()  # type: ignore[call-arg]
    b = cls(device="cpu").config_hash()  # type: ignore[call-arg]
    assert isinstance(a, str) and a, f"{name} config_hash is empty"
    assert a == b, f"{name} config_hash is non-deterministic: {a!r} != {b!r}"


def test_config_hashes_are_unique_across_adapters() -> None:
    """Two different adapters must never share a cache key.

    The prediction cache is keyed on ``config_hash``; a collision between
    distinct models would silently serve one model's predictions for another.
    """
    hashes = {name: MODEL_REGISTRY[name](device="cpu").config_hash() for name in _ADAPTER_NAMES}  # type: ignore[call-arg]
    by_hash: dict[str, list[str]] = {}
    for name, h in hashes.items():
        by_hash.setdefault(h, []).append(name)
    collisions = {h: names for h, names in by_hash.items() if len(names) > 1}
    assert not collisions, f"config_hash collisions across adapters: {collisions}"
