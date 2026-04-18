"""Depth Anything V2 adapter.

Upstream: https://github.com/DepthAnything/Depth-Anything-V2
Paper: "Depth Anything V2" (Yang et al. 2024).

Depth Anything V2 ships as two model families with the same DPT backbone:

- **Relative** variants predict disparity-like inverse depth. Paper
  evaluation uses MiDaS-style scale-and-shift alignment in inverse-depth
  space. ``alignment_hint="scale_shift"``.
- **Metric** variants are fine-tuned on Hypersim (indoor) or VKITTI
  (outdoor) with metric supervision, and predict depth in meters directly.
  No alignment at eval time. ``alignment_hint="none"``.

We use the HuggingFace Transformers integration
(``DepthAnythingForDepthEstimation``) because it's the closest thing to a
stable, pip-installable, non-CLI surface. HF post-processing
(``post_process_depth_estimation``) resizes ``predicted_depth`` to the
target size but does *not* change its units — so the adapter must branch
on the variant to decide whether to invert (relative) or pass through
(metric).
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
    # Relative (zero-shot) disparity models.
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    # Metric, indoor (Hypersim fine-tune).
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-indoor-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    # Metric, outdoor (VKITTI fine-tune).
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def _is_metric_variant(variant: str) -> bool:
    """True if the DA-V2 variant outputs depth in meters directly."""
    return variant.startswith("metric-")


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
        # "predicted_depth" as (H, W) torch tensor. For *relative* variants
        # this is disparity (higher=closer); for *metric* variants it is
        # depth in meters (lower=closer).
        raw = np.stack(
            [r["predicted_depth"].detach().cpu().numpy().astype(np.float32) for r in resized],
            axis=0,
        )
        if _is_metric_variant(self.variant):
            # Metric variant: output is depth in meters. Clamp non-finite /
            # non-positive pixels to 0 (canonical invalid marker).
            depth = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0).astype(np.float32)
            native_space = "depth"
            alignment_hint = "none"
        else:
            # Relative variant: output is disparity. Invert to depth with a
            # floor to avoid division by zero at sky / texture-less regions.
            depth = (1.0 / np.maximum(raw, EPS)).astype(np.float32)
            native_space = "disparity"
            alignment_hint = "scale_shift"
        assert_valid_depth(depth, name="da-v2/output")
        return Prediction(
            depth=depth,
            metadata={
                "variant": self.variant,
                "space": "depth",
                "native_space": native_space,
                "alignment_hint": alignment_hint,
                "checkpoint": _HF_CHECKPOINTS[self.variant],
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/input={self.input_size}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
