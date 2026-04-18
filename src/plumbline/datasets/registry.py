"""Dataset registry: name -> factory for :class:`~plumbline.datasets.base.Dataset`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from plumbline.datasets.base import Dataset

__all__ = ["DATASET_REGISTRY", "list_datasets", "register_dataset"]

DATASET_REGISTRY: dict[str, type[Dataset]] = {}

T = TypeVar("T", bound=type[Dataset])


def register_dataset(name: str) -> Callable[[T], T]:
    """Register a :class:`Dataset` subclass under ``name``.

    Use as a class decorator::

        @register_dataset("my-dataset")
        class MyDataset(Dataset):
            ...
    """

    def decorator(cls: T) -> T:
        if not isinstance(cls, type) or not issubclass(cls, Dataset):
            raise TypeError(f"{cls!r} is not a Dataset subclass")
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered")
        cls.name = name
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)
