"""Metric3Dv2 adapter.

Upstream: https://github.com/YvanYin/Metric3D
Paper: "Metric3D v2: A Versatile Monocular Geometric Foundation Model for
Zero-shot Metric Depth and Surface Normal Estimation" (Yin et al. 2024).

Metric3Dv2 predicts **metric** depth (meters) and surface normals from a
single image. It takes intrinsics as an input to condition the metric scale.

The upstream repo distributes weights via torch.hub. When run on plumbline we
wrap the ``torch.hub.load`` path. If the user installs the optional ``models``
extra and has network access, this adapter will pull weights on first use.

Canonical camera space protocol
-------------------------------
Metric3Dv2 is trained at a canonical focal length (1000 px) and a fixed
input resolution ((616, 1064) for ViT). Correct inference requires:

1. Resize RGB + intrinsics so the image fits inside (616, 1064) at its
   native aspect ratio (``cv2.INTER_LINEAR``).
2. Pad with the ImageNet mean colour to exactly (616, 1064).
3. Normalise with ImageNet mean/std (pixel values in [0, 255], not [0, 1]).
4. ``model.inference({'input': rgb})`` — returns ``(pred_depth, conf, out)``.
5. Un-pad, upsample to the original (H, W), multiply by
   ``scaled_fx / 1000`` to de-canonicalise to real metric depth.

The public ``hubconf.py`` exec-example is the canonical recipe; this
adapter mirrors it verbatim so future upstream changes are easy to
transcribe.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image, assert_valid_intrinsics
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["Metric3Dv2Adapter"]

_HUB_MODELS = {
    # torch.hub load strings per upstream hubconf.py (as of 2024-10).
    "vit_small": "metric3d_vit_small",
    "vit_large": "metric3d_vit_large",
    "vit_giant2": "metric3d_vit_giant2",
}

# Upstream input size for ViT backbones. ConvNeXt uses (544, 1216); when we
# add ConvNeXt support, key this off the variant.
_VIT_INPUT_SIZE = (616, 1064)
# ImageNet mean/std in [0, 255] (not [0, 1]) — the raw pixel convention the
# canonical Metric3D pipeline expects. Do not normalise pixels to [0, 1].
_IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
_IMAGENET_STD_255 = (58.395, 57.12, 57.375)
# Canonical focal length Metric3D was trained at. De-canonicalise depth by
# multiplying by ``scaled_fx / 1000`` after the forward.
_CANONICAL_FX = 1000.0


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
        # ``trust_repo=True`` skips torch.hub's interactive confirmation prompt
        # when pulling an unlisted GitHub repo; the user already opts in by
        # choosing this adapter.
        # Upstream spelling is ``pretrain`` (no "ed"); passing
        # ``pretrained=True`` is silently swallowed by ``**kwargs`` so weights
        # never load — the model produces NaN on the first forward.
        model = torch.hub.load(self.repo, _HUB_MODELS[self.variant], pretrain=True, trust_repo=True)
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

        depths: list[NDArray[np.float32]] = []
        for i in range(images.shape[0]):
            K = intrinsics[i]
            depth_i = _run_metric3d_one(
                self._model,
                images[i],
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                device=self.device,
            )
            depths.append(depth_i.astype(np.float32))

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


def _run_metric3d_one(
    model: Any,
    rgb: NDArray[np.uint8],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: str,
) -> NDArray[np.float32]:
    """Run Metric3Dv2 on a single sRGB uint8 image; return (H, W) metric depth.

    Mirrors the upstream ``hubconf.py`` exec-example step-for-step. The
    canonical-camera trick hinges on scaling intrinsics + image together,
    then rescaling the predicted depth by ``scaled_fx / 1000`` at the end.
    """
    torch = ensure_torch()
    import cv2

    h, w = rgb.shape[:2]
    scale = min(_VIT_INPUT_SIZE[0] / h, _VIT_INPUT_SIZE[1] / w)
    resized = cv2.resize(
        rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR
    )
    scaled_fx = fx * scale  # only fx enters the de-canonical scale

    pad_h = _VIT_INPUT_SIZE[0] - resized.shape[0]
    pad_w = _VIT_INPUT_SIZE[1] - resized.shape[1]
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=list(_IMAGENET_MEAN_255)
    )

    mean = torch.tensor(_IMAGENET_MEAN_255).float()[:, None, None]
    std = torch.tensor(_IMAGENET_STD_255).float()[:, None, None]
    t = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float()
    t = (t - mean) / std
    t = t[None].to(device)

    with torch.no_grad():
        pred_depth, _conf, _out = model.inference({"input": t})

    # Un-pad, upsample to native, de-canonicalise.
    pd = pred_depth.squeeze()
    pd = pd[top : pd.shape[0] - bottom, left : pd.shape[1] - right]
    pd = torch.nn.functional.interpolate(
        pd[None, None, :, :], size=(h, w), mode="bilinear"
    ).squeeze()
    pd = pd * (scaled_fx / _CANONICAL_FX)
    pd = pd.clamp(0.0, 300.0)
    return pd.detach().cpu().numpy()
