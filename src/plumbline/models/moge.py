"""MoGe adapter.

Upstream: https://github.com/microsoft/MoGe
Paper: "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-
Domain Images with Optimal Training Supervision" (Wang et al. 2024,
CVPR 2025).

MoGe ships two generations. Both predict a dense camera-frame point map
plus a validity mask plus intrinsics:

- **MoGe-1** (``Ruicheng/moge-vitl``) — affine-invariant point map; no
  metric scale. Evaluation uses MiDaS-style scale + shift alignment in
  depth space. ``capabilities.is_metric=False``; ``alignment_hint="scale_shift"``.
- **MoGe-2** (``Ruicheng/moge-2-*``) — metric point map. No alignment
  needed at eval time. ``capabilities.is_metric=True``;
  ``alignment_hint="none"``.

Install surface:

    uv pip install 'git+https://github.com/microsoft/MoGe.git'

(MoGe is not in ``transformers`` or PyTorch Hub — the upstream repo is
the only supported install path.)

Native MoGe output is a dict with keys:
- ``points`` — ``(H, W, 3)`` camera-frame point map in OpenCV coords
  (x right, y down, z forward).
- ``depth`` — ``(H, W)`` depth (meters for MoGe-2; affine-invariant for
  MoGe-1).
- ``mask`` — ``(H, W)`` bool validity.
- ``intrinsics`` — ``(3, 3)`` **normalized**: ``fx, fy`` divided by
  ``W, H`` respectively, likewise ``cx, cy``. The adapter un-normalises
  back to pixel-space K before returning.
- ``normal`` — ``(H, W, 3)`` surface normals when the ``*-normal``
  variant is loaded. plumbline's ``Prediction`` has no normals field
  yet; we log their presence in metadata but don't return them.

MoGe is mono (single-view). Multi-view input is handled by looping over
views and stacking predictions — same pattern used by the DA-V2
adapter.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["MoGeAdapter"]


# Variant -> (HuggingFace checkpoint, is_metric, moge.model submodule)
# The submodule differs between MoGe-1 (moge.model.v1) and MoGe-2
# (moge.model.v2); upstream chose separate classes rather than a version
# switch. Stick to the upstream naming so users can cross-reference the
# repo without translation.
_VARIANTS: dict[str, tuple[str, bool, str]] = {
    # MoGe-1, affine-invariant (no metric scale).
    "vitl": ("Ruicheng/moge-vitl", False, "v1"),
    # MoGe-2, metric.
    "2-vitl": ("Ruicheng/moge-2-vitl", True, "v2"),
    "2-vitl-normal": ("Ruicheng/moge-2-vitl-normal", True, "v2"),
    "2-vitb-normal": ("Ruicheng/moge-2-vitb-normal", True, "v2"),
    "2-vits-normal": ("Ruicheng/moge-2-vits-normal", True, "v2"),
}


@register_model("moge")
class MoGeAdapter(Model):
    """Monocular geometry adapter for MoGe v1 / v2.

    Parameters
    ----------
    device
        torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    variant
        One of :data:`_VARIANTS`. Default ``"2-vitl"`` — the MoGe-2
        large metric checkpoint, which is the paper's headline model.
    """

    version = "2.0"
    # Capabilities are set at class level for MoGe-2 (is_metric=True); the
    # v1 relative checkpoint overrides this in __init__ by rebinding
    # ``self.capabilities`` so ``runner._compute_metrics`` picks the right
    # alignment default.
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "2-vitl",
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}; got {variant!r}")
        self.device = device
        self.variant = variant
        checkpoint, is_metric, submodule = _VARIANTS[variant]
        self.checkpoint = checkpoint
        self._submodule = submodule
        if not is_metric:
            # MoGe-1 is affine-invariant; override the class-level capabilities
            # on this instance so the runner applies scale_shift alignment by
            # default (same as DA-V2 relative variants).
            self.capabilities = ModelCapabilities(
                tasks=frozenset({"mono_depth"}),
                is_metric=False,
                min_views=1,
                max_views=math.inf,
                requires_intrinsics=False,
            )
        self._model: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            if self._submodule == "v2":
                from moge.model.v2 import MoGeModel
            else:
                from moge.model.v1 import MoGeModel
        except ImportError as exc:  # pragma: no cover — exercised only on real installs
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('moge')}") from exc
        self._model = MoGeModel.from_pretrained(self.checkpoint).to(self.device).eval()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="moge/input")
        self._load()
        torch = ensure_torch()

        n, h, w, _ = images.shape
        depths = np.empty((n, h, w), dtype=np.float32)
        pmaps = np.empty((n, h, w, 3), dtype=np.float32)
        confidences = np.empty((n, h, w), dtype=np.float32)
        Ks = np.empty((n, 3, 3), dtype=np.float32)
        has_normal = False

        with torch.inference_mode():
            for i in range(n):
                # MoGe expects (3, H, W) float in [0, 1].
                img_t = (
                    torch.from_numpy(images[i])
                    .to(self.device, dtype=torch.float32)
                    .permute(2, 0, 1)
                    / 255.0
                )
                out = self._model.infer(img_t)
                depths[i] = out["depth"].detach().cpu().numpy().astype(np.float32)
                pmaps[i] = out["points"].detach().cpu().numpy().astype(np.float32)
                # MoGe's mask is a bool validity indicator; stash it as a
                # [0, 1] confidence. The runner doesn't use confidence for
                # metric computation today, but persisting it lets future
                # uncertainty-aware eval paths light up.
                confidences[i] = out["mask"].detach().cpu().numpy().astype(np.float32)
                # Unnormalise intrinsics: MoGe returns a (3, 3) with fx/W,
                # fy/H, cx/W, cy/H in the [0, 0], [1, 1], [0, 2], [1, 2]
                # slots. Scale back to input-image pixels.
                K_norm = out["intrinsics"].detach().cpu().numpy().astype(np.float64)
                K_pix = K_norm.copy()
                K_pix[0, 0] *= w  # fx
                K_pix[0, 2] *= w  # cx
                K_pix[1, 1] *= h  # fy
                K_pix[1, 2] *= h  # cy
                Ks[i] = K_pix.astype(np.float32)
                if "normal" in out:
                    has_normal = True

        # Invalid pixels are conveyed via the confidence mask; keep raw depth
        # values so downstream metrics can decide what to do with them. We
        # still guard against non-finite values so the canonical-depth
        # assertion doesn't fail on NaNs escaping the model.
        depths = np.where(np.isfinite(depths), depths, 0.0).astype(np.float32)
        assert_valid_depth(depths, name="moge/output/depth")
        assert_valid_intrinsics(Ks, name="moge/output/intrinsics")

        _, is_metric, _ = _VARIANTS[self.variant]
        alignment_hint = "none" if is_metric else "scale_shift"

        return Prediction(
            depth=depths,
            intrinsics=Ks,
            point_map=pmaps,  # MoGe's point map is in camera frame = world
            #                   frame (mono model, first-camera-as-world).
            confidence=confidences,
            metadata={
                "variant": self.variant,
                "checkpoint": self.checkpoint,
                "is_metric": is_metric,
                "alignment_hint": alignment_hint,
                "has_normal": has_normal,
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
