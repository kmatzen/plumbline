"""UniK3D adapter.

Upstream: https://github.com/lpiccinelli-eth/UniK3D (Piccinelli et al., CVPR 2025,
arXiv:2503.16591). VENDORED (inference subset) under
``plumbline/_vendor/unik3d`` — CC-BY-NC-SA-4.0; see THIRD_PARTY_NOTICES.md.

UniK3D is a universal monocular **metric** 3D estimator: from a single RGB image
(no camera input required) it predicts a metric point cloud, metric depth, and
camera rays, for any camera model (pinhole → fisheye → panoramic) via a spherical
3D representation that disentangles camera from scene.

Depth scale
-----------
UniK3D outputs **metric** depth (meters), so this adapter declares
``is_metric=True`` and no alignment hint — the runner scores it without
scale/shift alignment (unlike the affine-invariant DA-V2 / MoGe-1 / DA3 path).

Install
-------
Vendored — no install step. A plain ``uv sync`` provides the runtime deps
(timm / scipy / opencv-python, all base). ``$UNIK3D_ROOT`` overrides the vendored
path with a dev checkout.

Models (HuggingFace):

- vits: ``lpiccinelli/unik3d-vits``
- vitb: ``lpiccinelli/unik3d-vitb``
- vitl: ``lpiccinelli/unik3d-vitl`` (default)

Canonical conversion
--------------------
- ``infer(rgb)`` takes a ``(3, H, W)`` uint8 tensor and returns a dict with
  ``depth`` (metric, ``(1, 1, H, W)``), ``points`` (camera-frame metric point
  cloud), and ``rays``. We take ``depth`` squeezed to ``(H, W)``.
- Predicted intrinsics (when exposed by the model) live in the input image's
  pixel space; surfaced on ``Prediction.intrinsics`` when present, else None.
- UniK3D is monocular (one view at a time); a batch is looped and stacked.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image, assert_valid_intrinsics
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction, config_digest
from plumbline.models.registry import register_model

__all__ = ["UniK3DAdapter"]


def _ensure_unik3d_on_path() -> None:
    """Put the vendored ``unik3d`` package on ``sys.path``.

    Vendored under ``plumbline/_vendor/unik3d`` (inference subset of the
    CC-BY-NC-SA package; see THIRD_PARTY_NOTICES.md). Its internal imports are
    absolute (``from unik3d.utils.camera import …``), so the vendor root — the
    directory *containing* the ``unik3d/`` package — must be importable.
    ``$UNIK3D_ROOT`` overrides for a dev checkout.
    """
    root = os.environ.get("UNIK3D_ROOT")
    if not root:
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_vendor", "unik3d")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


_HF_CHECKPOINTS = {
    "vits": "lpiccinelli/unik3d-vits",
    "vitb": "lpiccinelli/unik3d-vitb",
    "vitl": "lpiccinelli/unik3d-vitl",
}


@register_model("unik3d")
class UniK3DAdapter(Model):
    """Universal monocular metric 3D foundation model (UniK3D)."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,  # UniK3D predicts metric depth (meters)
        min_views=1,
        max_views=1,
        requires_intrinsics=False,
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "vitl",
        checkpoint: str | None = None,
    ) -> None:
        if checkpoint is None and variant not in _HF_CHECKPOINTS:
            raise ValueError(
                f"variant must be one of {list(_HF_CHECKPOINTS)}; got {variant!r}. "
                "Pass `checkpoint=<hf-repo-id>` to target a variant not in the table."
            )
        self.device = device
        self.variant = variant
        self.checkpoint = checkpoint or _HF_CHECKPOINTS[variant]
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_unik3d_on_path()
        try:
            from unik3d.models import UniK3D
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('unik3d')}") from exc
        self._model = UniK3D.from_pretrained(self.checkpoint).to(device=self.device).eval()

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="unik3d/input")
        self._load()

        out = _run_unik3d(self._model, images, device=self.device)
        depth = out["depth"].astype(np.float32)
        K = out.get("intrinsics")

        assert_valid_depth(depth, name="unik3d/output_depth")
        if K is not None:
            assert_valid_intrinsics(K, name="unik3d/output_K")

        return Prediction(
            depth=depth,
            intrinsics=(K.astype(np.float32) if K is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "variant": self.variant,
                "native_space": "depth_metric",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/ckpt={self.checkpoint}"
        return config_digest(s)


def _run_unik3d(model: Any, images: NDArray[np.uint8], *, device: str) -> dict[str, Any]:
    """Run UniK3D on a batch of sRGB uint8 images (looped — UniK3D is monocular).

    Returns:
      - depth:      (N, H, W) float32, metric (meters)
      - intrinsics: (N, 3, 3) float32 if the model exposes them, else absent
    """
    import torch

    depths: list[NDArray[np.float32]] = []
    intr: list[NDArray[np.float32]] = []
    have_intr = True
    with torch.no_grad():
        for i in range(images.shape[0]):
            rgb = torch.from_numpy(np.ascontiguousarray(images[i])).permute(
                2, 0, 1
            )  # (3,H,W) uint8
            pred = model.infer(rgb.to(device))
            d = pred["depth"]
            d = d.squeeze().detach().cpu().numpy().astype(np.float32)  # (H,W)
            depths.append(d)
            k = pred.get("intrinsics") if hasattr(pred, "get") else None
            if k is None:
                have_intr = False
            else:
                intr.append(np.asarray(k.detach().cpu()).reshape(3, 3).astype(np.float32))

    out: dict[str, Any] = {"depth": np.stack(depths, axis=0)}
    if have_intr and intr:
        out["intrinsics"] = np.stack(intr, axis=0)
    return out
