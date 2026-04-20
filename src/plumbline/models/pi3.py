"""π³ (Pi-Cubed) multi-view 3D adapter.

Upstream: https://github.com/yyfz/Pi3
Paper: π³, ICLR 2026 submission (Yuan et al., ByteDance).

π³ is a feed-forward multi-view 3D foundation model in the VGGT /
DUSt3R / MASt3R / DA3 family. It ingests N views in a single forward
pass and emits:

- Per-view **local point maps** (camera-frame (x, y, z); z = depth)
- Per-view **global point maps** (world-frame, anchored on view 0)
- Per-view **camera_poses** as 4x4 ``camera_from_view_to_world``
  (OpenCV convention — exactly plumbline's ``world_from_camera``)
- Per-view **confidence** (pre-sigmoid logits; convert with
  ``torch.sigmoid``)

The model ships two variants:

- ``"pi3"``   — the original (canonical for paper A/B comparisons)
- ``"pi3x"``  — the December 2025 improved revision (authors' recommendation)

Default is ``"pi3x"``. Switch to ``"pi3"`` when reproducing numbers
from the original paper exactly.

Install
-------
Upstream has no PyPI distribution; follow the MASt3R / GeoWizard
pattern:

    git clone https://github.com/yyfz/Pi3 /workspace/deps/pi3
    cd /workspace/deps/pi3 && pip install -r requirements.txt
    export PI3_ROOT=/workspace/deps/pi3

Weights live on HuggingFace at ``yyfz233/Pi3`` / ``yyfz233/Pi3X`` and
are fetched via ``from_pretrained`` on first use.

Alignment
---------
π³ outputs metric 3D points (aligned to the paper's training scale).
Treat as metric; use ``scale_alignment: none`` in reproductions. If a
specific paper's protocol realigns predictions with ICP or Umeyama,
apply that at the runner level — not in the adapter.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    rebase_to_first_camera,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["Pi3Adapter"]

_VARIANT_HF: dict[str, tuple[str, str, str]] = {
    # variant -> (import_module, class_name, hf_checkpoint)
    "pi3": ("pi3.models.pi3", "Pi3", "yyfz233/Pi3"),
    "pi3x": ("pi3.models.pi3x", "Pi3X", "yyfz233/Pi3X"),
}


@register_model("pi3")
class Pi3Adapter(Model):
    """Multi-view feed-forward 3D foundation model (π³).

    Parameters
    ----------
    device
        torch device string.
    variant
        ``"pi3"`` (original) or ``"pi3x"`` (recommended, December 2025 rev).
    dtype
        ``"bfloat16"`` (default, fast on A100/4090), ``"float16"``, or
        ``"float32"`` (fallback for debugging / exact-match runs).
    """

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,  # trained with metric supervision; no per-scene fit by default
        min_views=2,
        # Upstream doesn't document a hard cap. 16 is a conservative
        # default that fits comfortably on a 24GB card; the runner's
        # OOM fallback catches anything larger.
        max_views=16,
        requires_intrinsics=False,
        default_resolution=(1024, 1024),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "pi3x",
        dtype: str = "bfloat16",
    ) -> None:
        if variant not in _VARIANT_HF:
            raise ValueError(f"variant must be one of {sorted(_VARIANT_HF)}; got {variant!r}")
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16|float16|float32; got {dtype!r}")
        self.device = device
        self.variant = variant
        self.dtype = dtype
        self._model: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_pi3_on_path()
        module, cls_name, hf = _VARIANT_HF[self.variant]
        try:
            mod = __import__(module, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            raise ImportError(
                f"Pi3Adapter could not import {module}.{cls_name}. Clone "
                "https://github.com/yyfz/Pi3 and point $PI3_ROOT at it "
                "(see the module docstring)."
            ) from exc
        import torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
        # Upstream's README shows:
        #     model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        self._model = cls.from_pretrained(hf).to(self.device).to(torch_dtype).eval()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="pi3/input")
        if images.shape[0] < self.capabilities.min_views:
            raise ValueError(
                f"pi3 needs >= {self.capabilities.min_views} views; got {images.shape[0]}"
            )
        self._load()
        import torch

        n, h, w, _ = images.shape
        # Upstream expects (N, 3, H, W) float in [0, 1], batched via [None].
        t = torch.from_numpy(images).to(self.device)
        t = t.permute(0, 3, 1, 2).to(
            dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[self.dtype]
        )
        t = t.float() / 255.0 if self.dtype == "float32" else t / 255.0
        # (N, 3, H, W) -> (1, N, 3, H, W)
        batch = t[None]

        with torch.no_grad():
            out = self._model(batch)

        # Move to CPU float32 for numpy handoff.
        local_points = out["local_points"][0].detach().float().cpu().numpy()  # (N, H, W, 3)
        points = out["points"][0].detach().float().cpu().numpy()               # (N, H, W, 3)
        camera_poses = out["camera_poses"][0].detach().float().cpu().numpy()   # (N, 4, 4)
        conf_logits = out["conf"][0].detach().float().cpu().numpy()            # (N, H, W)
        conf = 1.0 / (1.0 + np.exp(-conf_logits))  # sigmoid → [0, 1]

        # Per-view depth is the z-coordinate of the camera-local point map.
        # Negative / non-finite → invalid (0 per plumbline convention).
        depth = local_points[..., 2].astype(np.float32)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        assert_valid_depth(depth, name="pi3/depth")

        # camera_poses are camera-to-world in OpenCV convention (==
        # plumbline's world_from_camera). No inversion needed. Rebase so
        # camera 0 is the world origin — π³ usually anchors on view 0
        # already but float noise can drift it.
        extrinsics = rebase_to_first_camera(camera_poses.astype(np.float64)).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name="pi3/extrinsics")

        return Prediction(
            depth=depth,
            extrinsics=extrinsics,
            point_map=points.astype(np.float32),
            confidence=conf.astype(np.float32),
            metadata={
                "variant": self.variant,
                "checkpoint": _VARIANT_HF[self.variant][2],
                "dtype": self.dtype,
                "n_views": n,
                "native_space": "camera_local_xyz",
                "alignment_hint": "none",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/dtype={self.dtype}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_pi3_on_path() -> None:
    """Add ``$PI3_ROOT`` to sys.path so `from pi3.models...` resolves."""
    root = os.environ.get("PI3_ROOT", "/workspace/deps/pi3")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
