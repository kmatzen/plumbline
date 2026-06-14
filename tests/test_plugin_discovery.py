"""Out-of-tree adapter discovery via the ``plumbline.adapters`` entry-point group.

This is the contract that lets a contributor `pip install plumbline-bench` and add
their own model/dataset from a *separate* package, without cloning or editing
plumbline. A third-party distribution declares::

    [project.entry-points."plumbline.adapters"]
    my_adapters = "my_package.adapters"

and plumbline imports that target during adapter discovery so its
`@register_model` / `@register_dataset` decorators run.
"""

from __future__ import annotations

import pytest

import plumbline
from plumbline import Model, ModelCapabilities, Prediction, register_model
from plumbline._discover import (
    ADAPTER_ENTRY_POINT_GROUP,
    load_plugin_adapters,
    register_builtin_adapters,
)
from plumbline.models.registry import MODEL_REGISTRY

_PLUGIN_NAME = "_test_plugin_model"


def test_public_api_exports_registration_decorators() -> None:
    # Contributors should reach the registration API from the top level.
    assert "register_model" in plumbline.__all__
    assert "register_dataset" in plumbline.__all__
    assert plumbline.register_model is register_model


class _FakeEntryPoint:
    """Mimics importlib.metadata.EntryPoint: ``.name`` + ``.load()``."""

    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


def _patch_entry_points(monkeypatch, *eps) -> None:
    monkeypatch.setattr(
        "plumbline._discover.importlib_metadata.entry_points",
        lambda group=None: list(eps) if group == ADAPTER_ENTRY_POINT_GROUP else [],
    )


@pytest.fixture
def plugin_ep(monkeypatch):
    """A fake ``module:setup`` entry point that registers a model when called."""

    def _setup() -> None:
        @register_model(_PLUGIN_NAME)
        class _PluginAdapter(Model):
            capabilities = ModelCapabilities(tasks=frozenset({"mono_depth"}), is_metric=True)

            def predict(self, images, intrinsics=None):  # pragma: no cover - never run
                return Prediction(depth=images)

    # entry point value "my_package.adapters:setup" -> load() returns the callable
    _patch_entry_points(monkeypatch, _FakeEntryPoint("my_adapters", lambda: _setup))
    try:
        yield _PLUGIN_NAME
    finally:
        MODEL_REGISTRY.pop(_PLUGIN_NAME, None)


def test_plugin_model_is_discovered(plugin_ep) -> None:
    assert plugin_ep not in MODEL_REGISTRY
    failures = load_plugin_adapters(raise_on_error=True)
    assert failures == []
    assert plugin_ep in MODEL_REGISTRY, "entry-point adapter was not discovered"


def test_register_builtin_adapters_also_loads_plugins(plugin_ep) -> None:
    register_builtin_adapters()
    assert plugin_ep in MODEL_REGISTRY


def test_broken_plugin_is_soft_by_default(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("plugin import blew up")

    _patch_entry_points(monkeypatch, _FakeEntryPoint("bad", _boom))
    # Soft by default: one broken plugin must not block the harness.
    failures = load_plugin_adapters()
    assert len(failures) == 1
    assert failures[0][0] == f"{ADAPTER_ENTRY_POINT_GROUP}:bad"
    # Strict mode propagates.
    with pytest.raises(RuntimeError, match="blew up"):
        load_plugin_adapters(raise_on_error=True)
