"""Depth Anything V2 adapter.

Upstream: https://github.com/DepthAnything/Depth-Anything-V2
Paper: "Depth Anything V2" (Yang et al. 2024).

Depth Anything V2 is a monocular depth transformer that predicts **relative**
inverse depth (disparity-like). It is *not* metric. Evaluation on ScanNet /
NYU uses MiDaS-style scale-and-shift alignment in inverse-depth space.

We use the HuggingFace Transformers integration (``DepthAnythingForDepthEstimation``)
because it's the closest thing to a stable, pip-installable, non-CLI surface.
Model cards:

- small:  depth-anything/Depth-Anything-V2-Small-hf
- base:   depth-anything/Depth-Anything-V2-Base-hf
- large:  depth-anything/Depth-Anything-V2-Large-hf

Outputs native **disparity** (higher = closer). We invert to depth before
returning; the scale_shift alignment path recovers the MiDaS protocol.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS, assert_valid_depth, assert_valid_image
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["DepthAnythingV2Adapter"]

_HF_CHECKPOINTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


@register_model("depth-anything-v2")
class DepthAnythingV2Adapter(Model):
    """Monocular, relative-depth adapter for Depth Anything V2.

    Parameters
    ----------
    device
        torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    variant
        One of ``"small"``, ``"base"``, ``"large"``. Default: ``"large"`` matches
        the paper table.
    input_size
        Transformer patch size multiple; 518 is the DA2 default. Higher
        resolutions improve fine detail at quadratic cost.
    """

    version = "2.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(518, 518),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "large",
        input_size: int = 518,
    ) -> None:
        if variant not in _HF_CHECKPOINTS:
            raise ValueError(f"variant must be one of {list(_HF_CHECKPOINTS)}; got {variant!r}")
        self.device = device
        self.variant = variant
        self.input_size = int(input_size)
        self._model: Any = None
        self._processor: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = ensure_torch()
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DepthAnythingV2Adapter needs `transformers`. Install with "
                "`uv pip install -e '.[models]'`."
            ) from exc
        checkpoint = _HF_CHECKPOINTS[self.variant]
        # transformers' auto classes are not fully typed; ignore untyped-call.
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)  # type: ignore[no-untyped-call]
        self._model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device).eval()
        _ = torch  # used lazily in predict()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="da-v2/input")
        self._load()
        torch = ensure_torch()

        n, h, w, _ = images.shape
        # HF processor expects a list of (H, W, 3) uint8 arrays or PIL images.
        batch = [images[i] for i in range(n)]
        with torch.inference_mode():
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            # DepthAnythingForDepthEstimation returns "predicted_depth" as the
            # disparity prediction at the model's native feature resolution;
            # HF's post_process_depth_estimation resizes to input-image size.
            resized = self._processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(h, w)] * n,
            )

        # post_process_depth_estimation returns a list of dicts with
        # "predicted_depth" as (H, W) torch tensor in disparity (higher=closer).
        disparity = np.stack(
            [r["predicted_depth"].detach().cpu().numpy().astype(np.float32) for r in resized],
            axis=0,
        )
        # Convert disparity to depth. DA2 disparity is dimensionless; we clip
        # to avoid division by zero at sky / texture-less regions.
        depth = 1.0 / np.maximum(disparity, EPS)
        assert_valid_depth(depth, name="da-v2/output")
        return Prediction(
            depth=depth.astype(np.float32),
            metadata={
                "variant": self.variant,
                "space": "depth",
                "native_space": "disparity",
                "alignment_hint": "scale_shift",
                "checkpoint": _HF_CHECKPOINTS[self.variant],
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/input={self.input_size}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
