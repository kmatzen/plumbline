"""Video Depth Anything (VDA) adapter.

Upstream: https://github.com/DepthAnything/Video-Depth-Anything (ByteDance + HKU,
CVPR 2025 Highlight, arXiv:2501.12375). VENDORED (inference subset) under
``plumbline/_vendor/vda`` — the **code is Apache-2.0**; see THIRD_PARTY_NOTICES.md.

VDA produces temporally-consistent depth over arbitrarily long videos: a
spatial-temporal head + motion module on a Depth Anything V2 backbone, with a
keyframe overlap-and-align scheme so per-frame depth stays consistent across the
clip. It is a depth-only model (no camera pose / intrinsics).

Depth scale
-----------
Two checkpoint families:

- **relative** (default) — affine-invariant depth; ``is_metric=False``,
  ``alignment_hint="scale_shift"``.
- **metric** (``metric-*`` variants) — metric depth; ``is_metric=True``,
  ``alignment_hint="none"``.

Weights are split by license: the **Small** checkpoint is Apache-2.0, **Base /
Large** are CC-BY-NC-4.0 (the *code* is Apache either way). Weights download from
HuggingFace on first use; pick ``variant="vits"`` for a commercially-clean run.

Install
-------
Vendored — no install step. A plain ``uv sync`` provides the runtime deps
(easydict + opencv-python, both base). ``$VDA_ROOT`` overrides the vendored path
with a dev checkout.

Implementation notes
--------------------
- VDA is a **video** model: ``predict`` feeds the whole image batch (the clip)
  through ``infer_video_depth`` in one call, which returns temporally-aligned
  per-frame depth ``(N, H, W)``. A single frame (N=1) works too (internally
  padded), just without temporal benefit.
- ``infer_video_depth`` autocasts unless ``fp32=True``. On Pascal GPUs
  (sm_61, slow fp16) pass ``compute_fp32=True`` to force full precision.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["VideoDepthAnythingAdapter"]


def _ensure_vda_on_path() -> None:
    """Put the vendored ``video_depth_anything`` package on ``sys.path``.

    Vendored under ``plumbline/_vendor/vda`` (inference subset of the Apache-2.0
    release; see THIRD_PARTY_NOTICES.md). Internal imports are absolute
    (``from video_depth_anything... import``, ``from utils.util import``), so the
    vendor root must be importable. ``$VDA_ROOT`` overrides for a dev checkout.
    """
    root = os.environ.get("VDA_ROOT")
    if not root:
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_vendor", "vda")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


# variant -> (encoder, features, out_channels, is_metric, hf_repo, checkpoint_filename)
_VARIANTS: dict[str, tuple[str, int, list[int], bool, str, str]] = {
    "vits": (
        "vits",
        64,
        [48, 96, 192, 384],
        False,
        "depth-anything/Video-Depth-Anything-Small",
        "video_depth_anything_vits.pth",
    ),
    "vitb": (
        "vitb",
        128,
        [96, 192, 384, 768],
        False,
        "depth-anything/Video-Depth-Anything-Base",
        "video_depth_anything_vitb.pth",
    ),
    "vitl": (
        "vitl",
        256,
        [256, 512, 1024, 1024],
        False,
        "depth-anything/Video-Depth-Anything-Large",
        "video_depth_anything_vitl.pth",
    ),
    "metric-vits": (
        "vits",
        64,
        [48, 96, 192, 384],
        True,
        "depth-anything/Metric-Video-Depth-Anything-Small",
        "metric_video_depth_anything_vits.pth",
    ),
    "metric-vitl": (
        "vitl",
        256,
        [256, 512, 1024, 1024],
        True,
        "depth-anything/Metric-Video-Depth-Anything-Large",
        "metric_video_depth_anything_vitl.pth",
    ),
}


@register_model("vda")
class VideoDepthAnythingAdapter(Model):
    """Temporally-consistent video depth foundation model (Video Depth Anything)."""

    version = "1.0"
    # Class-level default is the relative (affine-invariant) path; the metric
    # variants override self.capabilities in __init__ (mirrors the MoGe adapter).
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=float("inf"),  # processes a whole video clip at once
        requires_intrinsics=False,
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "vitl",
        compute_fp32: bool = False,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}; got {variant!r}.")
        self.device = device
        self.variant = variant
        self.compute_fp32 = compute_fp32
        _, _, _, is_metric, _, _ = _VARIANTS[variant]
        if is_metric:
            self.capabilities = ModelCapabilities(
                tasks=frozenset({"mono_depth"}),
                is_metric=True,
                min_views=1,
                max_views=float("inf"),
                requires_intrinsics=False,
            )
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_vda_on_path()
        try:
            from video_depth_anything.video_depth import VideoDepthAnything
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('vda')}") from exc
        import torch
        from huggingface_hub import hf_hub_download

        encoder, features, out_channels, is_metric, repo, fname = _VARIANTS[self.variant]
        model = VideoDepthAnything(
            encoder=encoder, features=features, out_channels=out_channels, metric=is_metric
        )
        ckpt = hf_hub_download(repo_id=repo, filename=fname)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self._model = model.to(self.device).eval()

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="vda/input")
        self._load()

        _, _, _, is_metric, _, _ = _VARIANTS[self.variant]
        # infer_video_depth uses `device` for the autocast device_type, which
        # must be the bare accelerator ("cuda", not "cuda:0").
        device_type = self.device.split(":")[0]
        depths, _fps = self._model.infer_video_depth(
            np.ascontiguousarray(images),
            target_fps=-1,
            device=device_type,
            fp32=self.compute_fp32,
        )
        depth = np.asarray(depths).astype(np.float32)  # (N, H, W)
        assert_valid_depth(depth, name="vda/output_depth")

        return Prediction(
            depth=depth,
            metadata={
                "variant": self.variant,
                "is_metric": is_metric,
                "alignment_hint": "none" if is_metric else "scale_shift",
                "native_space": "depth_metric" if is_metric else "depth_affine_invariant",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/fp32={self.compute_fp32}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
