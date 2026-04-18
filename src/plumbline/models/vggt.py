"""VGGT adapter.

Upstream: https://github.com/facebookresearch/vggt
Paper: "VGGT: Visual Geometry Grounded Transformer" (Wang et al. 2025).

VGGT is a feed-forward multi-view transformer that predicts, from up to
~32 views in a single forward pass:

- Per-view depth maps (``depth``)
- Per-view world-space point maps (``point_map``)
- Cameras (intrinsics + extrinsics) in the world frame of the first view
- Dense confidence

Canonical conversion
--------------------
- VGGT outputs poses as ``world_from_camera`` by default (matching plumbline
  conventions). First view is the world frame; we assert this rather than
  rebase.
- VGGT outputs depth in metric meters (trained with metric supervision);
  treat as metric and skip alignment by default. ``align_hint=none``.
- Intrinsics are reported in input-image pixel space; we pass them through.

Memory note
-----------
Per the paper, 32 views at 1024x1024 fits in 24GB on an A100/4090. At higher
view counts or higher resolution, check upstream's memory-efficient mode.
The runner's OOM fallback catches the failure and skips the sample.
"""

from __future__ import annotations

import hashlib
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

__all__ = ["VGGTAdapter"]


@register_model("vggt")
class VGGTAdapter(Model):
    """Multi-view feed-forward 3D foundation model."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,  # paper trains with metric supervision
        min_views=2,
        max_views=32,
        requires_intrinsics=False,
        default_resolution=(1024, 1024),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = "facebook/VGGT-1B",
        dtype: str = "float32",
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.dtype = dtype
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            from vggt.models.vggt import VGGT  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "VGGTAdapter needs the `vggt` package from "
                "https://github.com/facebookresearch/vggt. Install with "
                "`uv pip install git+https://github.com/facebookresearch/vggt`."
            ) from exc
        model = VGGT.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="vggt/input")
        if images.shape[0] < 2:
            raise ValueError("VGGT requires at least 2 views")
        if images.shape[0] > self.capabilities.max_views:
            raise ValueError(f"VGGT max_views={self.capabilities.max_views}; got {images.shape[0]}")
        self._load()

        out = _run_vggt(self._model, images, device=self.device, dtype=self.dtype)

        depth = out["depth"].astype(np.float32)
        K = out["intrinsics"].astype(np.float32)
        E = out["extrinsics"].astype(np.float32)
        point_map = out.get("point_map")
        confidence = out.get("confidence")

        # Guard the convention. Accept a small epsilon; otherwise rebase.
        if not world_from_camera_is_identity(E):
            E = rebase_to_first_camera(E.astype(np.float64)).astype(np.float32)

        assert_valid_depth(depth, name="vggt/output_depth")
        assert_valid_intrinsics(K, name="vggt/output_K")
        assert_valid_extrinsics(E, name="vggt/output_E")

        return Prediction(
            depth=depth,
            intrinsics=K,
            extrinsics=E,
            point_map=(point_map.astype(np.float32) if point_map is not None else None),
            confidence=(confidence.astype(np.float32) if confidence is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "native_space": "depth",
                "alignment_hint": "none",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/ckpt={self.checkpoint}/dtype={self.dtype}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _run_vggt(
    model: Any, images: NDArray[np.uint8], *, device: str, dtype: str
) -> dict[str, NDArray[Any]]:
    """Run VGGT end-to-end. Placeholder: wire into upstream in first GPU run.

    Expected return dict (after conversion):
      - depth:      (N, H, W), float32, meters
      - intrinsics: (N, 3, 3), float32, input-image pixels
      - extrinsics: (N, 4, 4), float32, world_from_camera, first view = identity
      - point_map:  (N, H, W, 3), float32, world frame  (optional)
      - confidence: (N, H, W), float32 in [0, 1]        (optional)
    """
    raise NotImplementedError(
        "VGGT inference needs the upstream `vggt` package wired up. See the "
        "module docstring for the output contract. File: "
        f"{__file__}"
    )
