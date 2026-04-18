"""MASt3R adapter.

Upstream: https://github.com/naver/mast3r
Paper: "Grounding Image Matching in 3D with MASt3R" (Leroy et al. 2024).

MASt3R is a pair-based (2-view) model that predicts per-pixel 3D point maps
in a shared frame for a pair of images, plus a confidence map. Chained
pairwise inference gives multi-view reconstructions.

For plumbline we treat MASt3R as a "min_views=2, max_views=2" primitive and
stitch at the runner level (or here, internally, as the plan recommends).

Inputs
------
- Two sRGB uint8 images of matching size (internally cropped / padded if not).
- No intrinsics required — MASt3R predicts them.

Outputs (in canonical conventions)
----------------------------------
- ``depth``: ``(2, H, W)`` meters, metric up to the paper's residual scale
  ambiguity (rescale with the paper's median alignment when comparing to GT).
- ``intrinsics``: ``(2, 3, 3)`` — estimated by MASt3R.
- ``extrinsics``: ``(2, 4, 4)`` — ``world_from_camera``, first camera is
  identity by our convention.
- ``point_map``: ``(2, H, W, 3)`` — MASt3R's primary output. World frame.
- ``confidence``: ``(2, H, W)``.

For pairs longer than 2, this adapter does the simple thing: pair camera 0 to
each subsequent view and compose into the world frame. Paper-exact numbers
should use MASt3R's own ``sparse_global_alignment`` — wire that in when
running reproductions.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    assert_valid_point_map,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["MASt3RAdapter"]


@register_model("mast3r")
class MASt3RAdapter(Model):
    """Pair-based 3D + pose foundation model."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # recovers a scale up to residual ambiguity
        min_views=2,
        max_views=8,  # upstream handles any N via chained pairs; we cap.
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            from mast3r.model import AsymmetricMASt3R  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MASt3RAdapter needs the `mast3r` package. Install from "
                "https://github.com/naver/mast3r (not on PyPI)."
            ) from exc
        model = AsymmetricMASt3R.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="mast3r/input")
        if images.shape[0] < 2:
            raise ValueError("MASt3R requires at least 2 views")
        self._load()

        # See upstream ``mast3r.utils.image.load_images`` and
        # ``dust3r.inference.inference`` for the canonical pipeline. We defer
        # that to a small wrapper below, which accepts numpy arrays directly.
        out = _run_mast3r(self._model, images, device=self.device)

        point_map = out["point_map"]  # (N, H, W, 3), world frame
        depth = out["depth"]  # (N, H, W), derived from point_map in cam frame
        K = out["intrinsics"]  # (N, 3, 3) pixel space
        extrinsics = out["extrinsics"]  # (N, 4, 4) world_from_camera, E[0] = I
        confidence = out.get("confidence")  # (N, H, W) or None

        assert_valid_image(images)
        assert_valid_intrinsics(K, name="mast3r/output_K")
        assert_valid_extrinsics(extrinsics, name="mast3r/output_E")
        assert_valid_point_map(point_map, name="mast3r/output_pmap")

        return Prediction(
            depth=depth.astype(np.float32),
            intrinsics=K.astype(np.float32),
            extrinsics=extrinsics.astype(np.float32),
            point_map=point_map.astype(np.float32),
            confidence=(confidence.astype(np.float32) if confidence is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "native_space": "point_map",
                "alignment_hint": "median",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/{self.checkpoint}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Upstream wrapper
# ---------------------------------------------------------------------------


def _run_mast3r(model: Any, images: NDArray[np.uint8], *, device: str) -> dict[str, NDArray[Any]]:
    """Run MASt3R on a batch of images and return plumbline-shaped arrays.

    This is a thin wrapper over the upstream ``inference`` API. The exact
    call depends on the upstream commit; see mast3r's README for the current
    pattern. We document here what plumbline expects back so future-us can
    update this if upstream changes.

    Expected return dict (after conversion):
      - point_map: (N, H, W, 3), float32, in the world frame (first camera)
      - depth:     (N, H, W), float32, meters-ish (rescale in metric step)
      - intrinsics: (N, 3, 3), float32, pixel space of the input image
      - extrinsics: (N, 4, 4), float32, world_from_camera, first = identity
      - confidence: optional (N, H, W), float32
    """
    raise NotImplementedError(
        "MASt3R inference requires wiring into the `mast3r` package. See the "
        "module docstring for the expected output format. File: "
        f"{__file__}"
    )
