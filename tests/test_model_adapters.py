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
import plumbline.models.geowizard
import plumbline.models.mast3r
import plumbline.models.metric3d_v2
import plumbline.models.depth_pro
import plumbline.models.marigold
import plumbline.models.moge
import plumbline.models.pi3
import plumbline.models.vggt  # noqa: F401
from plumbline.models.registry import MODEL_REGISTRY

EXPECTED_ADAPTERS = [
    "depth-anything-v2",
    "metric3d-v2",
    "mast3r",
    "vggt",
    "depth-anything-3",
    "moge",
    "marigold",
    "depth-pro",
    "geowizard",
    "pi3",
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


def test_mast3r_supports_multiview() -> None:
    """MASt3R adapter should advertise N-view support (post-PointCloudOptimizer
    rewrite). The 2-view-only capability cap was a v0.1 limitation."""
    cls = MODEL_REGISTRY["mast3r"]
    assert cls.capabilities.min_views == 2
    assert cls.capabilities.max_views >= 10  # need at least 10 for CO3Dv2 protocol


def test_mast3r_rejects_too_many_views() -> None:
    import numpy as np

    cls = MODEL_REGISTRY["mast3r"]
    model = cls(device="cpu")  # type: ignore[call-arg]
    over = np.zeros((cls.capabilities.max_views + 1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="capped at"):
        model.predict(over)


def test_mast3r_config_hash_depends_on_ga_hyperparams() -> None:
    """Changing PointCloudOptimizer hyperparams changes predictions for
    N>=3, so the prediction cache must invalidate when they change."""
    cls = MODEL_REGISTRY["mast3r"]
    base = cls(device="cpu")  # type: ignore[call-arg]
    # Each hyperparam should produce a distinct hash.
    variants = {
        "default": base.config_hash(),
        "niter": cls(device="cpu", ga_niter=50).config_hash(),  # type: ignore[call-arg]
        "lr": cls(device="cpu", ga_lr=0.005).config_hash(),  # type: ignore[call-arg]
        "schedule": cls(device="cpu", ga_schedule="cosine").config_hash(),  # type: ignore[call-arg]
        "init": cls(device="cpu", ga_init="known_poses").config_hash(),  # type: ignore[call-arg]
    }
    assert len(set(variants.values())) == len(variants), (
        f"hashes collided: {variants}"
    )


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


def test_marigold_rejects_unknown_variant() -> None:
    cls = MODEL_REGISTRY["marigold"]
    with pytest.raises(ValueError, match="variant"):
        cls(device="cpu", variant="nope")  # type: ignore[call-arg]


def test_marigold_rejects_bad_inference_params() -> None:
    cls = MODEL_REGISTRY["marigold"]
    with pytest.raises(ValueError, match="num_inference_steps"):
        cls(device="cpu", num_inference_steps=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="ensemble_size"):
        cls(device="cpu", ensemble_size=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="dtype"):
        cls(device="cpu", dtype="bfloat16")  # type: ignore[call-arg]  # Marigold wants fp16/fp32


def test_marigold_config_hash_varies_by_inference_params() -> None:
    """Changing inference knobs should change config_hash so the prediction
    cache doesn't silently re-use a 1-step result when the user upgraded to
    paper-protocol 4-step 10-ensemble."""

    cls = MODEL_REGISTRY["marigold"]
    fast = cls(device="cpu", num_inference_steps=1, ensemble_size=1).config_hash()  # type: ignore[call-arg]
    paper = cls(device="cpu", num_inference_steps=4, ensemble_size=10).config_hash()  # type: ignore[call-arg]
    assert fast != paper


def test_marigold_capabilities_are_mono_relative() -> None:
    cls = MODEL_REGISTRY["marigold"]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert "mono_depth" in caps.tasks
    assert caps.is_metric is False  # affine-invariant
    assert caps.min_views == 1


def test_depth_pro_rejects_bad_dtype() -> None:
    cls = MODEL_REGISTRY["depth-pro"]
    with pytest.raises(ValueError, match="dtype"):
        cls(device="cpu", dtype="bfloat16")  # type: ignore[call-arg]


def test_geowizard_rejects_bad_domain() -> None:
    cls = MODEL_REGISTRY["geowizard"]
    with pytest.raises(ValueError, match="domain"):
        cls(device="cpu", domain="underwater")  # type: ignore[call-arg]


def test_geowizard_rejects_bad_inference_params() -> None:
    cls = MODEL_REGISTRY["geowizard"]
    with pytest.raises(ValueError, match="num_inference_steps"):
        cls(device="cpu", num_inference_steps=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="ensemble_size"):
        cls(device="cpu", ensemble_size=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="processing_res"):
        cls(device="cpu", processing_res=60)  # not multiple of 8  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="dtype"):
        cls(device="cpu", dtype="bfloat16")  # type: ignore[call-arg]


def test_geowizard_config_hash_varies_by_domain_and_inference_params() -> None:
    """Domain conditioning matters for the prediction; indoor/outdoor
    produce different depths for the same input. The cache key must
    reflect it so swapping domains doesn't return a stale cached result."""

    cls = MODEL_REGISTRY["geowizard"]
    indoor = cls(device="cpu", domain="indoor").config_hash()  # type: ignore[call-arg]
    outdoor = cls(device="cpu", domain="outdoor").config_hash()  # type: ignore[call-arg]
    fast = cls(device="cpu", domain="indoor", num_inference_steps=4).config_hash()  # type: ignore[call-arg]
    paper = cls(device="cpu", domain="indoor", num_inference_steps=10).config_hash()  # type: ignore[call-arg]
    assert indoor != outdoor
    assert fast != paper


def test_geowizard_capabilities_are_mono_relative() -> None:
    cls = MODEL_REGISTRY["geowizard"]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert "mono_depth" in caps.tasks
    assert caps.is_metric is False  # affine-invariant
    assert caps.min_views == 1


def test_depth_pro_capabilities_are_mono_metric() -> None:
    """Depth Pro declares is_metric=True so the runner defaults to
    scale_alignment=none when YAML doesn't override. Regression: if
    someone accidentally flips this to False, every Depth Pro repro
    silently switches to scale+shift fitting and hides the model's
    actual metric accuracy."""

    cls = MODEL_REGISTRY["depth-pro"]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert "mono_depth" in caps.tasks
    assert caps.is_metric is True  # key assertion
    assert caps.requires_intrinsics is False


def test_depth_pro_config_hash_varies_by_dtype() -> None:
    cls = MODEL_REGISTRY["depth-pro"]
    h16 = cls(device="cpu", dtype="float16").config_hash()  # type: ignore[call-arg]
    h32 = cls(device="cpu", dtype="float32").config_hash()  # type: ignore[call-arg]
    assert h16 != h32


def test_pi3_rejects_bad_variant() -> None:
    cls = MODEL_REGISTRY["pi3"]
    with pytest.raises(ValueError, match="variant"):
        cls(device="cpu", variant="pi4")  # type: ignore[call-arg]


def test_pi3_rejects_bad_dtype() -> None:
    cls = MODEL_REGISTRY["pi3"]
    with pytest.raises(ValueError, match="dtype"):
        cls(device="cpu", dtype="int8")  # type: ignore[call-arg]


def test_pi3_config_hash_varies_by_variant_and_dtype() -> None:
    """Pi3 vs Pi3X are different trained models; their cached predictions must
    not collide. Dtype also changes the cache key so a bf16 run doesn't serve
    a pre-cached fp32 result (or vice versa)."""

    cls = MODEL_REGISTRY["pi3"]
    h1 = cls(device="cpu", variant="pi3", dtype="bfloat16").config_hash()  # type: ignore[call-arg]
    h2 = cls(device="cpu", variant="pi3x", dtype="bfloat16").config_hash()  # type: ignore[call-arg]
    h3 = cls(device="cpu", variant="pi3x", dtype="float32").config_hash()  # type: ignore[call-arg]
    assert len({h1, h2, h3}) == 3


def test_pi3_capabilities_are_multi_view_metric() -> None:
    cls = MODEL_REGISTRY["pi3"]
    caps = cls.capabilities  # type: ignore[attr-defined]
    assert "mono_depth" in caps.tasks
    assert "mvs_depth" in caps.tasks
    assert "pose" in caps.tasks
    assert caps.is_metric is True
    assert caps.min_views == 2
