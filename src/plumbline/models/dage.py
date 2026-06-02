"""DAGE adapter — feed-forward video geometry + camera pose.

Upstream: https://github.com/ngoductuanlhp/DAGE
Paper: "DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry
Estimation" (arXiv:2603.03744, 2026).

DAGE is a feed-forward multi-view model: one ``infer`` pass over a clip yields
per-frame camera poses, global/local point maps, intrinsics and a metric scale,
with no per-scene optimization (unlike DUSt3R/MonST3R global alignment).

``DAGE.infer(video, lr_max_size=...)`` takes ``(B, N, 3, H, W)`` in ``[0, 1]``
and returns a dict with:

- ``camera_poses``  ``(N, 4, 4)`` — camera-to-world (== plumbline ``world_from_camera``), metric.
- ``global_points`` ``(N, h, w, 3)`` — world-frame points (at the lr stream resolution).
- ``local_points``  ``(N, h, w, 3)`` — metric camera-frame points.
- ``intrinsics``    ``(N, 3, 3)`` — at the lr stream resolution.
- ``metric_scale``, ``mask``.

This adapter wires the **pose** task (the camera_poses → ``Prediction.extrinsics``
mapping). Depth / point-map tasks are deferred: ``infer`` returns geometry at the
internal lr resolution (default 252 px max side), so a faithful depth cell needs
the outputs unscaled back to input-image pixels — handled in a follow-up.

Pascal note: ``infer`` autocasts to **bfloat16** by default, which pre-Ampere
GPUs (e.g. a GTX 1080Ti, sm_61) do not support. We pass ``enable_autocast=False``
so it runs in fp32 on such hardware (≈8 GB at 252 px for short clips).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_extrinsics,
    assert_valid_image,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["DAGEAdapter"]

_DEFAULT_CHECKPOINT = "TuanNgo/DAGE"  # HuggingFace hub (model.pt + config)


def _ensure_dage_on_path() -> None:
    """Put the DAGE repo on ``sys.path`` so ``import dage`` resolves.

    The repo's editable install does not expose the ``dage`` package on the
    import path, so (like the CUT3R/MonST3R adapters) we add ``$DAGE_ROOT``
    explicitly. Defaults to ``~/git/DAGE``.
    """
    root = os.environ.get("DAGE_ROOT", str(Path.home() / "git" / "DAGE"))
    if root not in sys.path:
        sys.path.insert(0, root)


@register_model("dage")
class DAGEAdapter(Model):
    """Feed-forward video geometry + pose adapter for DAGE.

    Parameters
    ----------
    device
        torch device string.
    checkpoint
        HF hub id or local ``model.pt`` path (default ``TuanNgo/DAGE``).
    lr_max_size
        Max side (px) of the low-resolution stream that pose is read from
        (DAGE's pose eval uses 252).
    enable_autocast
        Leave False on pre-Ampere GPUs (no bf16); the default infer path
        autocasts to bfloat16.
    """

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"pose"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(252, 252),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str | Path = _DEFAULT_CHECKPOINT,
        lr_max_size: int = 252,
        enable_autocast: bool = False,
    ) -> None:
        self.device = device
        self.checkpoint = str(checkpoint)
        self.lr_max_size = int(lr_max_size)
        self.enable_autocast = bool(enable_autocast)
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_dage_on_path()
        try:
            from dage.models.dage import DAGE  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — needs the repo
            raise ImportError(
                "DAGEAdapter needs the DAGE repo importable: clone "
                "https://github.com/ngoductuanlhp/DAGE, set $DAGE_ROOT to it, and "
                "install its deps (torch, einops, omegaconf, safetensors, utils3d, "
                "kornia, roma, segmentation_models_pytorch)."
            ) from exc
        self._model = DAGE.from_pretrained(self.checkpoint).to(self.device).eval()

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="dage/input")
        self._load()
        torch = ensure_torch()

        # (N, H, W, 3) uint8 → (1, N, 3, H, W) float in [0, 1].
        video = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        video = video.float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(0).contiguous()

        with torch.inference_mode():
            out = self._model.infer(
                video,
                lr_max_size=self.lr_max_size,
                enable_autocast=self.enable_autocast,
            )

        # camera_poses: (N, 4, 4) camera-to-world == world_from_camera, metric.
        E = np.asarray(out["camera_poses"].detach().to("cpu"), dtype=np.float64)
        if E.ndim == 3 and E.shape[1:] == (4, 4):
            if not world_from_camera_is_identity(E.astype(np.float32)):
                E = rebase_to_first_camera(E)
        E = E.astype(np.float32)
        assert_valid_extrinsics(E, name="dage/output_E")

        return Prediction(
            extrinsics=E,
            metadata={
                "native_space": "feedforward_pose",
                "lr_max_size": self.lr_max_size,
                "enable_autocast": self.enable_autocast,
            },
        )

    def config_hash(self) -> str:
        import hashlib

        s = f"{self.name}@{self.version}/lr={self.lr_max_size}/autocast={self.enable_autocast}/ckpt={self.checkpoint}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
