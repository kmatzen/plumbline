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
GPUs (e.g. a GTX 1080Ti, sm_61) do not support. We disable DAGE's bf16 autocast
and apply our own (``compute_dtype``): fp32 for short clips, or fp16 mixed
precision (Pascal-compatible) so 50-frame clips fit in 11 GB.
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
    """Put the ``dage`` package on ``sys.path`` so ``import dage`` resolves.

    By default this uses the **vendored** copy bundled under
    ``plumbline/_vendor/dage`` (DAGE is CC BY-NC 4.0 — redistributable with
    attribution; see ``_vendor/dage/LICENSE`` and ``THIRD_PARTY_NOTICES.md``),
    so no separate clone is needed. Set ``$DAGE_ROOT`` to a checkout to override
    (e.g. for development against upstream).
    """
    root = os.environ.get("DAGE_ROOT")
    if root is None:
        root = str(Path(__file__).resolve().parent.parent / "_vendor")
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
    compute_dtype
        Mixed-precision mode for inference: ``"float32"`` (safe, most memory),
        ``"float16"`` (Pascal-compatible mixed precision — roughly halves
        activation memory so long clips fit in 11 GB), or ``"bfloat16"``
        (Ampere+ only). DAGE's own infer autocast is bf16, which pre-Ampere
        GPUs don't support; we disable it and wrap the call in our own
        autocast of the chosen dtype instead.
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
        hr_max_size: int | None = None,
        compute_dtype: str = "float32",
    ) -> None:
        if compute_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"compute_dtype must be float32/float16/bfloat16; got {compute_dtype!r}"
            )
        self.device = device
        self.checkpoint = str(checkpoint)
        self.lr_max_size = int(lr_max_size)
        # Pose comes from the low-res stream; the high-res stream only refines
        # local_points. Capping hr_max_size shrinks the HR upsampling (the memory
        # bottleneck on an 11 GB card) without changing camera_poses. Defaults to
        # lr_max_size so pose-only runs stay cheap.
        self.hr_max_size = int(hr_max_size) if hr_max_size is not None else int(lr_max_size)
        self.compute_dtype = compute_dtype
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_dage_on_path()
        try:
            from dage.models.dage import DAGE  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — needs the deps
            raise ImportError(
                "DAGEAdapter could not import the (vendored) `dage` package. Its "
                "runtime deps must be installed: einops, omegaconf, safetensors, "
                "utils3d==0.0.2 (EasternJournalist@3913c65), kornia, roma, "
                "segmentation_models_pytorch. (`plumbline install dage`)."
            ) from exc
        model = DAGE.from_pretrained(self.checkpoint).to(self.device).eval()
        # On an 11 GB card the fp32 weights (~5.5 GB) leave too little for a
        # 50-frame clip's activations. Halving the weights (fp16) frees ~2.7 GB;
        # combined with the fp16 autocast in predict, long clips then fit.
        if self.compute_dtype == "float16":
            model = model.half()
            # DAGE casts the camera-head input to fp32 (dage.py: `camera_hidden =
            # camera_hidden.float()` before `self.camera_head(...)`), so the head
            # must stay fp32 or its weights mismatch the fp32 activations. It is
            # tiny (512-dim MLPs), so this costs no meaningful memory.
            if hasattr(model, "camera_head"):
                model.camera_head.float()
        elif self.compute_dtype == "bfloat16":
            model = model.bfloat16()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="dage/input")
        self._load()
        torch = ensure_torch()

        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.compute_dtype
        ]
        # (N, H, W, 3) uint8 → (1, N, 3, H, W) in [0, 1], matching the model dtype.
        video = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        video = video.to(dtype).div_(255.0).permute(0, 3, 1, 2).unsqueeze(0).contiguous()

        # DAGE's infer opens its OWN autocast hardcoded to bfloat16 (Ampere-only),
        # and an ``enabled=False`` autocast would override any outer context and
        # leave fp16-halved weights mismatched against fp32 activations. So when
        # running fp16 on Pascal we let DAGE's autocast run but monkeypatch its
        # bf16 → fp16 for the duration of the call.
        amp_enabled = self.compute_dtype != "float32"
        orig_autocast = torch.amp.autocast

        def _coerced_autocast(*a: Any, **k: Any) -> Any:
            if k.get("dtype") is torch.bfloat16:
                k["dtype"] = dtype
            return orig_autocast(*a, **k)

        if amp_enabled:
            torch.amp.autocast = _coerced_autocast
        try:
            with torch.inference_mode():
                out = self._model.infer(
                    video,
                    lr_max_size=self.lr_max_size,
                    hr_max_size=self.hr_max_size,
                    enable_autocast=amp_enabled,
                )
        finally:
            torch.amp.autocast = orig_autocast

        # camera_poses: (N, 4, 4) camera-to-world == world_from_camera, metric.
        E = np.asarray(out["camera_poses"].detach().to("cpu"), dtype=np.float64)
        if (
            E.ndim == 3
            and E.shape[1:] == (4, 4)
            and not world_from_camera_is_identity(E.astype(np.float32))
        ):
            E = rebase_to_first_camera(E)
        extrinsics: NDArray[np.float32] = E.astype(np.float32)
        assert_valid_extrinsics(extrinsics, name="dage/output_E")

        return Prediction(
            extrinsics=extrinsics,
            metadata={
                "native_space": "feedforward_pose",
                "lr_max_size": self.lr_max_size,
                "compute_dtype": self.compute_dtype,
            },
        )

    def config_hash(self) -> str:
        import hashlib

        s = f"{self.name}@{self.version}/lr={self.lr_max_size}/hr={self.hr_max_size}/dtype={self.compute_dtype}/ckpt={self.checkpoint}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
