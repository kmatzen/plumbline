"""Smoke tests for model adapters.

These exercise:
- Module import (no torch required at import time).
- Registry registration.
- Capability declarations.
- Deterministic ``config_hash`` values.
- Adapter instantiation (does not load weights).

Actual inference tests require GPU + weights and are marked ``weights``/``gpu``.
"""

from __future__ import annotations

import pytest

import plumbline.models.depth_anything_3

# Force import so decorators run.
import plumbline.models.depth_anything_v2
import plumbline.models.mast3r
import plumbline.models.metric3d_v2
import plumbline.models.moge
import plumbline.models.vggt  # noqa: F401
from plumbline.models.registry import MODEL_REGISTRY

EXPECTED_ADAPTERS = [
    "depth-anything-v2",
    "metric3d-v2",
    "mast3r",
    "vggt",
    "depth-anything-3",
    "moge",
]


@pytest.mark.parametrize("name", EXPECTED_ADAPTERS)
def test_adapter_is_registered(name: str) -> None:
    assert name in MODEL_REGISTRY


@pytest.mark.parametrize("name", EXPECTED_ADAPTERS)
def test_capabilities_present(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert len(caps.tasks) > 0
    assert caps.min_views >= 1


@pytest.mark.parametrize("name", EXPECTED_ADAPTERS)
def test_can_instantiate_without_gpu(name: str) -> None:
    cls = MODEL_REGISTRY[name]
    # cpu-only; should not attempt to load weights in __init__.
    cls(device="cpu")  # type: ignore[call-arg]


def test_config_hash_is_deterministic() -> None:
    cls = MODEL_REGISTRY["depth-anything-v2"]
    a = cls(device="cpu", variant="small").config_hash()  # type: ignore[call-arg]
    b = cls(device="cpu", variant="small").config_hash()  # type: ignore[call-arg]
    assert a == b


def test_config_hash_varies_by_variant() -> None:
    cls = MODEL_REGISTRY["depth-anything-v2"]
    a = cls(device="cpu", variant="small").config_hash()  # type: ignore[call-arg]
    b = cls(device="cpu", variant="large").config_hash()  # type: ignore[call-arg]
    assert a != b


def test_unknown_variant_errors() -> None:
    cls = MODEL_REGISTRY["depth-anything-v2"]
    with pytest.raises(ValueError):
        cls(device="cpu", variant="not-a-size")  # type: ignore[call-arg]


def test_vggt_enforces_view_bounds() -> None:
    import numpy as np

    cls = MODEL_REGISTRY["vggt"]
    model = cls(device="cpu")  # type: ignore[call-arg]
    images_single = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 views"):
        model.predict(images_single)


def test_mast3r_requires_two_views() -> None:
    import numpy as np

    cls = MODEL_REGISTRY["mast3r"]
    model = cls(device="cpu")  # type: ignore[call-arg]
    images_single = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2"):
        model.predict(images_single)


def test_metric3d_requires_intrinsics() -> None:
    import numpy as np

    cls = MODEL_REGISTRY["metric3d-v2"]
    model = cls(device="cpu")  # type: ignore[call-arg]
    images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires intrinsics"):
        model.predict(images)


def test_moge_v1_variant_is_relative() -> None:
    """MoGe v1 ('vitl') is affine-invariant → the instance overrides
    capabilities to is_metric=False so the runner picks scale_shift
    alignment. Regression: the class-level capabilities declares
    is_metric=True for the default MoGe-2 case."""

    cls = MODEL_REGISTRY["moge"]
    v1 = cls(device="cpu", variant="vitl")  # type: ignore[call-arg]
    v2 = cls(device="cpu", variant="2-vitl")  # type: ignore[call-arg]
    assert v1.capabilities.is_metric is False
    assert v2.capabilities.is_metric is True


def test_moge_unknown_variant_errors() -> None:
    cls = MODEL_REGISTRY["moge"]
    with pytest.raises(ValueError, match="variant"):
        cls(device="cpu", variant="not-a-variant")  # type: ignore[call-arg]


def test_moge_config_hash_varies_by_variant() -> None:
    cls = MODEL_REGISTRY["moge"]
    a = cls(device="cpu", variant="vitl").config_hash()  # type: ignore[call-arg]
    b = cls(device="cpu", variant="2-vitl").config_hash()  # type: ignore[call-arg]
    assert a != b


def test_moge_checkpoint_matches_variant() -> None:
    """Locks the variant → HF-checkpoint mapping so silently retargeting a
    variant at the dict level can't sneak into a release."""

    cls = MODEL_REGISTRY["moge"]
    assert cls(device="cpu", variant="vitl").checkpoint == "Ruicheng/moge-vitl"  # type: ignore[call-arg]
    assert cls(device="cpu", variant="2-vitl").checkpoint == "Ruicheng/moge-2-vitl"  # type: ignore[call-arg]
    assert (
        cls(device="cpu", variant="2-vitb-normal").checkpoint  # type: ignore[call-arg]
        == "Ruicheng/moge-2-vitb-normal"
    )
