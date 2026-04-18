"""Metric3Dv2 adapter.

Upstream: https://github.com/YvanYin/Metric3D
Paper: "Metric3D v2: A Versatile Monocular Geometric Foundation Model for
Zero-shot Metric Depth and Surface Normal Estimation" (Yin et al. 2024).

Metric3Dv2 predicts **metric** depth (meters) and surface normals from a
single image. It takes intrinsics as an input to condition the metric scale.

The upstream repo distributes weights via torch.hub. When run on plumbline we
wrap the ``torch.hub.load`` path. If the user installs the optional ``models``
extra and has network access, this adapter will pull weights on first use.

Calibration & input
-------------------
- Inputs are sRGB uint8 in OpenCV convention (ours).
- Metric3D's public entry point consumes ``(fx, fy, cx, cy)`` as a list; we
  extract them from the canonical 3x3 ``K``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image, assert_valid_intrinsics
from plumbline.models._torch_utils import ensure_torch, numpy_to_torch_images, torch_to_numpy
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["Metric3Dv2Adapter"]

_HUB_MODELS = {
    # torch.hub load strings per upstream README (as of late 2024).
    "vit_small": "metric3d_vit_small",
    "vit_large": "metric3d_vit_large",
    "vit_giant2": "metric3d_vit_giant2",
}


@register_model("metric3d-v2")
class Metric3Dv2Adapter(Model):
    """Metric monocular depth (+ normals) adapter."""

    version = "2.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=True,
        default_resolution=(616, 1064),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "vit_large",
        repo: str = "YvanYin/Metric3D",
    ) -> None:
        if variant not in _HUB_MODELS:
            raise ValueError(f"variant must be one of {list(_HUB_MODELS)}; got {variant!r}")
        self.device = device
        self.variant = variant
        self.repo = repo
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = ensure_torch()
        model = torch.hub.load(self.repo, _HUB_MODELS[self.variant], pretrained=True)
        self._model = model.to(self.device).eval()

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="metric3d-v2/input")
        if intrinsics is None:
            raise ValueError("Metric3Dv2 requires intrinsics (for metric calibration).")
        assert_valid_intrinsics(intrinsics, name="metric3d-v2/input_K")
        self._load()
        torch = ensure_torch()

        # Upstream calls expect per-image processing; we iterate. N is usually
        # 1 for monocular, but we preserve the batched interface.
        depths: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for i in range(images.shape[0]):
                tensor = numpy_to_torch_images(images[i : i + 1], device=self.device)[0]
                K = intrinsics[i]
                intr = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
                # Upstream `infer(tensor, intrinsics=[fx, fy, cx, cy])` returns
                # metric depth in meters and normals. Stay defensive; wrap the
                # possible API shapes in a try.
                result = _invoke_metric3d(self._model, tensor, intr)
                depths.append(np.asarray(result["depth"], dtype=np.float32))

        depth = np.stack(depths, axis=0)
        assert_valid_depth(depth, name="metric3d-v2/output")
        return Prediction(
            depth=depth,
            metadata={
                "variant": self.variant,
                "space": "depth",
                "native_space": "depth",
                "alignment_hint": "none",
                "checkpoint": f"torchhub://{self.repo}/{_HUB_MODELS[self.variant]}",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _invoke_metric3d(model: Any, tensor: Any, intrinsics: list[float]) -> dict[str, Any]:
    """Call the upstream Metric3D model. Tolerates minor API variations.

    Upstream has shuffled their entry point between ``infer_depth``, ``infer``,
    and direct ``forward`` in different releases. This helper tries the known
    spellings; if none fit, raise with a clear pointer to upstream.
    """
    if hasattr(model, "infer"):
        return model.infer(tensor, intrinsics=intrinsics)
    if hasattr(model, "infer_depth"):
        return {"depth": model.infer_depth(tensor, intrinsics=intrinsics)}
    # Last resort: direct forward. Shape contract is upstream-specific.
    out = model(tensor, intrinsics=intrinsics)
    if isinstance(out, dict) and "depth" in out:
        return out  # type: ignore[return-value]
    raise RuntimeError(
        "Could not find a known Metric3Dv2 entry point (tried `infer`, `infer_depth`, "
        "forward). Check that torch.hub pulled a compatible revision from "
        "YvanYin/Metric3D."
    )


_ = torch_to_numpy  # reserved for future use
