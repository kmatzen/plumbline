"""Depth Anything 3 adapter.

Upstream: https://github.com/DepthAnything/Depth-Anything-V3 (tentative;
paper and repo naming may shift before release).

Depth Anything 3 is the multi-view successor to DA-V2. Feed-forward
prediction of depth + camera pose. Metric (paper reports metric KITTI,
ScanNet).

Canonical conversion
--------------------
- Depth: metric meters; no alignment needed in most benchmark tables.
- Pose: world_from_camera with first view as identity (we assert / rebase).
- Intrinsics: returned in input-image pixels when the model predicts them;
  else passed through from the caller.

This adapter is v0.1 scaffolding. Wire into the upstream repo's entry point
in :func:`_run_depth_anything_3` once the public API stabilizes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["DepthAnything3Adapter"]


@register_model("depth-anything-3")
class DepthAnything3Adapter(Model):
    """Multi-view depth + pose foundation model (Depth Anything V3)."""

    version = "3.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,
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
        checkpoint: str | None = None,
    ) -> None:
        self.device = device
        self.variant = variant
        self.checkpoint = checkpoint or f"depth-anything/Depth-Anything-V3-{variant.title()}-hf"
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            # When DA3 lands on HuggingFace, this should auto-detect via
            # AutoModel. The exact class name may change; handle at that time.
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DepthAnything3Adapter needs `transformers`. Install with "
                "`uv pip install -e '.[models]'`."
            ) from exc
        self._model = (
            AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="da3/input")
        self._load()

        out = _run_depth_anything_3(self._model, images, device=self.device)

        depth = out["depth"].astype(np.float32)
        K = out.get("intrinsics")
        E = out.get("extrinsics")

        if E is not None:
            if not world_from_camera_is_identity(E):
                E = rebase_to_first_camera(E.astype(np.float64)).astype(np.float32)
            assert_valid_extrinsics(E, name="da3/output_E")
        if K is not None:
            assert_valid_intrinsics(K, name="da3/output_K")
        assert_valid_depth(depth, name="da3/output_depth")

        return Prediction(
            depth=depth,
            intrinsics=(K.astype(np.float32) if K is not None else None),
            extrinsics=(E.astype(np.float32) if E is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "variant": self.variant,
                "native_space": "depth",
                "alignment_hint": "none",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/ckpt={self.checkpoint}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _run_depth_anything_3(
    model: Any, images: NDArray[np.uint8], *, device: str
) -> dict[str, NDArray[Any]]:
    """Run DA3 end-to-end. Placeholder until the upstream API stabilizes.

    Expected return dict (after conversion):
      - depth:      (N, H, W), float32, meters
      - intrinsics: (N, 3, 3), float32, input-image pixels  (optional)
      - extrinsics: (N, 4, 4), float32, world_from_camera   (optional; N>1)
    """
    raise NotImplementedError(
        "Depth Anything 3 inference pipeline not yet wired. Wire into "
        "upstream and follow the output contract in the module docstring. "
        f"File: {__file__}"
    )
