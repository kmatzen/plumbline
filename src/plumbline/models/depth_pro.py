"""Depth Pro adapter.

Upstream: https://github.com/apple/ml-depth-pro
Paper: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"
(Bochkovskii et al. 2024, Apple).

Depth Pro predicts **metric** depth (meters) and a per-image focal length
in pixels, from a single RGB. No GT intrinsics required at inference time;
the model infers its own from the image content.

Install surface:

    uv pip install 'git+https://github.com/apple/ml-depth-pro.git'
    # Then fetch weights:
    wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \\
        -P ~/.cache/plumbline/weights/depth-pro/

The upstream ``depth_pro.create_model_and_transforms()`` expects the
weights at ``checkpoints/depth_pro.pt`` relative to the repo root by
default. The adapter accepts a ``weights_path`` kwarg to override.

Native output is a dict::

    {
      "depth": (H, W) float32 tensor, meters
      "focallength_px": scalar, pixels
    }

We map to plumbline's canonical Prediction:
- ``depth``: (N, H, W) float32 meters — same as the native key.
- ``intrinsics``: (N, 3, 3) float32 with ``fx = fy = focallength_px``,
  ``cx = W/2, cy = H/2`` (Depth Pro's model is trained as a pinhole with
  the image centre as principal point). Un-normalised to input pixels.
- ``extrinsics``: identity (mono model; first-camera-is-world).
- ``point_map``: left as None so the runner's depth→pointmap
  back-projection fallback (commit 27995a1) populates it on demand for
  chamfer evaluations.

Depth Pro is designed for metric evaluation (no scale alignment); the
``is_metric=True`` capability flag directs the runner to default to
``scale_alignment: none`` unless the YAML overrides.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
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

__all__ = ["DepthProAdapter"]

_DEFAULT_WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
_DEFAULT_WEIGHTS_PATH = (
    Path.home() / ".cache" / "plumbline" / "weights" / "depth-pro" / "depth_pro.pt"
)


@register_model("depth-pro")
class DepthProAdapter(Model):
    """Metric monocular depth adapter for Apple's Depth Pro.

    Parameters
    ----------
    device
        torch device string.
    weights_path
        Path to the ``depth_pro.pt`` checkpoint. If omitted, falls back
        to ``~/.cache/plumbline/weights/depth-pro/depth_pro.pt`` (the
        conventional plumbline location — adapter prints the download
        URL at first use if missing).
    dtype
        ``"float16"`` (default, ~2x faster on a 3090) or ``"float32"``.
    """

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(1536, 1536),  # Depth Pro's native training resolution
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        weights_path: Path | str | None = None,
        dtype: str = "float16",
        use_gt_focal: bool = False,
    ) -> None:
        if dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be 'float16' or 'float32'; got {dtype!r}")
        self.device = device
        self.dtype = dtype
        self.use_gt_focal = use_gt_focal
        # Depth Pro self-estimates focal length by default (its headline
        # zero-shot capability). For metric benchmarks whose published number
        # is produced with the dataset's GT focal (e.g. SUN-RGBD Table 1,
        # δ₁ 0.890 — reproduces only with GT focal, not the self-estimate),
        # opt in: the runner then feeds the sample's GT intrinsics and we pass
        # fx as ``f_px`` to ``infer`` so the estimated focal is ignored. The
        # default path is unchanged, so Booster et al. are unaffected.
        if use_gt_focal:
            import dataclasses as _dc

            self.capabilities = _dc.replace(type(self).capabilities, requires_intrinsics=True)
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH
        self._model: Any = None
        self._transform: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = ensure_torch()
        try:
            import depth_pro
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('depth-pro')}") from exc
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Depth Pro weights not found at {self.weights_path}. Download:\n"
                f"    mkdir -p {self.weights_path.parent}\n"
                f"    wget {_DEFAULT_WEIGHTS_URL} -O {self.weights_path}"
            )
        # Upstream's create_model_and_transforms() looks for ./checkpoints/
        # by default. We use the `checkpoint_uri` config override so the
        # adapter works regardless of cwd.
        import dataclasses

        from depth_pro import depth_pro as dp_module

        config = dataclasses.replace(
            dp_module.DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=str(self.weights_path),
        )
        self._model, self._transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device(self.device)
        )
        self._model.eval()
        if self.dtype == "float16":
            self._model = self._model.half()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="depth-pro/input")
        self._load()
        torch = ensure_torch()
        from PIL import Image as PImage

        n, h, w, _ = images.shape
        depths = np.empty((n, h, w), dtype=np.float32)
        Ks = np.empty((n, 3, 3), dtype=np.float32)
        with torch.inference_mode():
            for i in range(n):
                pil = PImage.fromarray(images[i])
                # The upstream transform expects a PIL image and returns
                # a preprocessed tensor at (possibly) a different
                # resolution; `model.infer` resizes the output back.
                img_t = self._transform(pil)
                if self.dtype == "float16":
                    img_t = img_t.half()
                # When opted in, pass the GT focal (fx, in the input image's
                # pixel units — the transform keeps native size) so the
                # estimated focal is ignored. ``infer`` uses ``W / f_px`` to
                # scale metric depth, so fx must match the image width passed.
                f_px = None
                if self.use_gt_focal and intrinsics is not None:
                    f_px = torch.as_tensor(float(intrinsics[i][0, 0]), device=img_t.device)
                out = self._model.infer(img_t, f_px=f_px)
                # depth: (H_out, W_out) — Depth Pro resizes to the input
                # resolution internally when f_px isn't passed, but belt-
                # and-suspenders: resize explicitly if shape mismatches.
                depth_t = out["depth"]
                if depth_t.dim() == 3:
                    depth_t = depth_t.squeeze(0)
                depth_arr = depth_t.detach().float().cpu().numpy()
                if depth_arr.shape != (h, w):
                    # Resize with nearest-ish — PIL's BILINEAR is fine
                    # for a metric depth upsampling, but avoid introducing
                    # negative values.
                    depth_pil = PImage.fromarray(depth_arr.astype(np.float32), mode="F")
                    depth_arr = np.asarray(
                        depth_pil.resize((w, h), resample=PImage.Resampling.BILINEAR),
                        dtype=np.float32,
                    )
                # Zero-out non-finite / non-positive (canonical invalid).
                depth_arr = np.where(
                    np.isfinite(depth_arr) & (depth_arr > 0), depth_arr, 0.0
                ).astype(np.float32)
                depths[i] = depth_arr

                fx = float(out["focallength_px"])
                Ks[i] = np.array(
                    [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )

        assert_valid_depth(depths, name="depth-pro/output/depth")
        assert_valid_intrinsics(Ks, name="depth-pro/output/intrinsics")
        return Prediction(
            depth=depths,
            intrinsics=Ks,
            metadata={
                "dtype": self.dtype,
                "native_space": "depth_metric",
                "alignment_hint": "none",
                "weights": str(self.weights_path),
            },
        )

    def config_hash(self) -> str:
        # Version + dtype is enough; the weights are pinned to one URL.
        s = f"{self.name}@{self.version}/dtype={self.dtype}/gt_focal={self.use_gt_focal}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
