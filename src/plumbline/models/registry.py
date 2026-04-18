"""Model registry: name -> factory for :class:`~plumbline.models.base.Model`.

Adapters register themselves via the :func:`register_model` decorator at
module import time. The CLI and test suite look up adapters through
:data:`MODEL_REGISTRY` rather than importing them directly, so that an optional
adapter with missing dependencies doesn't block the rest of the harness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from plumbline.models.base import Model

__all__ = ["MODEL_REGISTRY", "list_models", "register_model"]

MODEL_REGISTRY: dict[str, type[Model]] = {}

T = TypeVar("T", bound=type[Model])


def register_model(name: str) -> Callable[[T], T]:
    """Register a :class:`Model` subclass under ``name``.

    Use as a class decorator::

        @register_model("my-model")
        class MyAdapter(Model):
            ...
    """

    def decorator(cls: T) -> T:
        if not isinstance(cls, type) or not issubclass(cls, Model):
            raise TypeError(f"{cls!r} is not a Model subclass")
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def list_models() -> list[str]:
    """Return registered model names, sorted."""
    return sorted(MODEL_REGISTRY)
