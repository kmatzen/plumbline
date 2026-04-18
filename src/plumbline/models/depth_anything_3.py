"""Depth Anything 3 adapter.

Upstream: https://github.com/ByteDance-Seed/Depth-Anything-3
Package: ``depth_anything_3`` (pip-installable).

Depth Anything 3 is the multi-view successor to DA-V2. Feed-forward
prediction of depth + camera pose + intrinsics from 1..N views. Metric
(paper reports metric depth benchmarks).

Install
-------
::

    uv pip install depth-anything-3

Models (as of release, HuggingFace):

- large:  ``depth-anything/DA3-LARGE`` (~0.41B params)

Canonical conversion
--------------------
- DA3's ``Prediction.extrinsics`` is ``w2c = camera_from_world`` shape
  ``(N, 3, 4)``. We pad to ``(N, 4, 4)`` and invert to
  ``world_from_camera`` to match plumbline's convention. The pose encoder
  biases view 0 to identity, so our "first camera is world" usually holds
  within float noise; we rebase if it doesn't.
- ``Prediction.depth`` is metric meters; default ``alignment_hint=none``.
- ``Prediction.intrinsics`` live in DA3's processed-image pixel space
  (504 long-edge). The runner resizes depth to GT for metric computation.

Implementation notes
--------------------
- ``model.inference`` mutates the filesystem if ``export_dir`` is set; we
  pass ``export_dir=None`` (no side effects). It still allocates Gaussian /
  mini_npz structures in memory; we don't touch those.
- Outputs land as numpy arrays, not torch — no ``.cpu().numpy()`` needed.
- DA3's point_map isn't exposed in the top-level ``Prediction``; leave it
  None. Users who need one can unproject the depth map.
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
    invert_pose,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["DepthAnything3Adapter"]

_HF_CHECKPOINTS = {
    "large": "depth-anything/DA3-LARGE",
}


@register_model("depth-anything-3")
class DepthAnything3Adapter(Model):
    """Multi-view depth + pose foundation model (Depth Anything 3)."""

    version = "3.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(378, 504),  # DA3's default long-edge=504 preserves aspect
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "large",
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
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DepthAnything3Adapter needs the `depth_anything_3` package. "
                "Install with `VIRTUAL_ENV=<project>/.venv uv pip install "
                "depth-anything-3`."
            ) from exc
        self._model = (
            DepthAnything3.from_pretrained(self.checkpoint).to(device=self.device).eval()
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
        conf = out.get("confidence")

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
            confidence=(conf.astype(np.float32) if conf is not None else None),
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
    """Run DA3 end-to-end on a batch of sRGB uint8 images.

    Returns arrays at DA3's processed resolution (long edge = 504,
    short edge rounded to a multiple of DA3's patch size):

      - depth:      (N, H_p, W_p), float32, meters
      - intrinsics: (N, 3, 3), float32, in processed pixel space
      - extrinsics: (N, 4, 4), float32, world_from_camera, first view = identity
      - confidence: (N, H_p, W_p), float32
    """
    import torch

    imgs_list = [images[i] for i in range(images.shape[0])]
    with torch.no_grad():
        # export_dir=None skips all file writes; export_format must be a
        # non-None string because upstream does ``if "gs" in export_format``
        # before checking export_dir.
        pred = model.inference(imgs_list, export_dir=None, export_format="mini_npz")

    depth_np = np.asarray(pred.depth)  # (N, H, W), already numpy, metric meters
    intr_np = np.asarray(pred.intrinsics)  # (N, 3, 3)
    extr_np = np.asarray(pred.extrinsics)  # (N, 3, 4), w2c (camera_from_world)
    conf_np = np.asarray(pred.conf) if pred.conf is not None else None

    # w2c 3x4 → camera_from_world 4x4 → world_from_camera 4x4.
    n = extr_np.shape[0]
    cam_from_world = np.zeros((n, 4, 4), dtype=np.float64)
    cam_from_world[:, :3, :] = extr_np.astype(np.float64)
    cam_from_world[:, 3, 3] = 1.0
    world_from_cam = invert_pose(cam_from_world)

    # Canonical invalid marker: negative / non-finite depth → 0.
    depth_np = np.where(np.isfinite(depth_np) & (depth_np > 0), depth_np, 0.0)

    return {
        "depth": depth_np.astype(np.float32),
        "intrinsics": intr_np.astype(np.float32),
        "extrinsics": world_from_cam.astype(np.float32),
        "confidence": conf_np.astype(np.float32) if conf_np is not None else None,
    }
