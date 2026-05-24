"""Depth Anything V2 adapter.

Upstream: https://github.com/DepthAnything/Depth-Anything-V2
Paper: "Depth Anything V2" (Yang et al. 2024, arXiv:2406.09414).

Depth Anything V2 ships as two model families with the same DPT backbone:

- **Relative** variants predict disparity-like inverse depth. Paper
  evaluation uses MiDaS-style scale-and-shift alignment in inverse-depth
  space. ``alignment_hint="scale_shift"``.
- **Metric** variants are fine-tuned on Hypersim (indoor) or VKITTI
  (outdoor) with metric supervision, and predict depth in meters directly.
  No alignment at eval time. ``alignment_hint="none"``.

Two ways to load weights, selected via the ``source`` kwarg:

- ``source="paper"`` (default for relative variants): load the paper's
  original ``.pth`` checkpoint from the ``depth-anything/Depth-Anything-V2-*``
  HF repo (NOT the ``*-hf`` one) and construct the model via the paper's
  own ``depth_anything_v2.dpt.DepthAnythingV2`` class cloned into
  ``$DAV2_ROOT`` (default ``/workspace/deps/depth-anything-v2``). This is
  what the paper tables were computed on, so it's the right path for
  reproducibility against Table 2.
- ``source="hf"``: load via HF transformers ``AutoModelForDepthEstimation``
  from the ``*-hf`` re-exports. Handy for quick smoke tests and required
  for all metric variants (the paper only released `.pth` for the three
  relative ones). Metric variants force this path.

Earlier plumbline versions only supported the HF path; the paper's own
checkpoints produce a small (~0.002 AbsRel) but systematic shift relative
to the HF re-exports on NYU, enough to tip the Base variant outside the
5 % paper-match gate. See ``docs/runs/20260421.md § da-v2-base-nyuv2``.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS, assert_valid_depth, assert_valid_image
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["DepthAnythingV2Adapter"]

# HF repos that ship the paper's original `.pth` alongside a README. The
# filename is always `depth_anything_v2_{encoder}.pth`.
_PAPER_REPOS = {
    "small": ("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
    "base":  ("depth-anything/Depth-Anything-V2-Base",  "depth_anything_v2_vitb.pth"),
    "large": ("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"),
}

# HF transformers-format re-exports. Small / Base / Large and all metric
# variants live here.
_HF_CHECKPOINTS = {
    # Relative (zero-shot) disparity models.
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    # Metric, indoor (Hypersim fine-tune).
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-indoor-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    # Metric, outdoor (VKITTI fine-tune).
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}

# Per-variant DPT head config (from the paper repo's run.py).
_MODEL_CONFIGS = {
    "small": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "base":  {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _is_metric_variant(variant: str) -> bool:
    """True if the DA-V2 variant outputs depth in meters directly."""
    return variant.startswith("metric-")


def _ensure_dav2_on_path() -> None:
    """Add ``$DAV2_ROOT`` to ``sys.path`` so `depth_anything_v2.dpt` imports.

    Default: ``/workspace/deps/depth-anything-v2``. Clone via
    ``git clone https://github.com/DepthAnything/Depth-Anything-V2 <root>``.
    """
    root = os.environ.get("DAV2_ROOT", "/workspace/deps/depth-anything-v2")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


@register_model("depth-anything-v2")
class DepthAnythingV2Adapter(Model):
    """Monocular, relative-depth adapter for Depth Anything V2.

    Parameters
    ----------
    device
        torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    variant
        One of ``"small"``, ``"base"``, ``"large"`` for relative; or
        ``"metric-{indoor,outdoor}-{small,base,large}"`` for metric.
        Default: ``"large"`` matches the paper Table 2 anchor.
    input_size
        Square input the network consumes; 518 is the DA-V2 default. Must
        be a multiple of 14 (ViT patch size).
    source
        ``"paper"`` → load from the paper's original ``.pth`` checkpoints
        via ``depth-anything/Depth-Anything-V2-{S,B,L}`` (requires
        ``$DAV2_ROOT`` pointing at the github clone). ``"hf"`` → load via
        HF transformers ``AutoModelForDepthEstimation`` from the ``-hf``
        re-exports. Metric variants always use ``"hf"`` — the paper
        checkpoints don't cover them. Default: ``"paper"`` for relative,
        forced to ``"hf"`` for metric.
    """

    version = "2.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(518, 518),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "large",
        input_size: int = 518,
        source: str = "paper",
    ) -> None:
        if variant not in _HF_CHECKPOINTS:
            raise ValueError(f"variant must be one of {list(_HF_CHECKPOINTS)}; got {variant!r}")
        if input_size % 14 != 0 or input_size < 14:
            raise ValueError(f"input_size must be a positive multiple of 14; got {input_size}")
        if source not in ("paper", "hf"):
            raise ValueError(f"source must be 'paper' or 'hf'; got {source!r}")
        # Metric variants only have HF checkpoints.
        if _is_metric_variant(variant):
            source = "hf"
        self.device = device
        self.variant = variant
        self.input_size = int(input_size)
        self.source = source
        self._model: Any = None
        self._processor: Any = None  # only for source="hf"

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.source == "paper":
            self._load_paper()
        else:
            self._load_hf()

    def _load_paper(self) -> None:
        """Instantiate the paper's DPT and load its ``.pth`` state_dict."""
        torch = ensure_torch()
        _ensure_dav2_on_path()
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                "DepthAnythingV2Adapter(source='paper') needs the paper's "
                "repo. Clone https://github.com/DepthAnything/Depth-Anything-V2 "
                "and point $DAV2_ROOT at it (default "
                "/workspace/deps/depth-anything-v2)."
            ) from exc
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DepthAnythingV2Adapter(source='paper') needs huggingface_hub. "
                "Install with `uv pip install -e '.[models]'`."
            ) from exc

        repo_id, filename = _PAPER_REPOS[self.variant]
        ckpt_path = hf_hub_download(repo_id, filename=filename)
        model = DepthAnythingV2(**_MODEL_CONFIGS[self.variant])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(self.device).eval()
        self._model = model

    def _load_hf(self) -> None:
        """Load via HF transformers for the `-hf` re-exports + metric variants."""
        torch = ensure_torch()
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DepthAnythingV2Adapter needs `transformers`. Install with "
                "`uv pip install -e '.[models]'`."
            ) from exc
        checkpoint = _HF_CHECKPOINTS[self.variant]
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)  # type: ignore[no-untyped-call]
        self._model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device).eval()
        _ = torch

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="da-v2/input")
        self._load()
        if self.source == "paper":
            return self._predict_paper(images)
        return self._predict_hf(images)

    def _predict_paper(self, images: NDArray[np.uint8]) -> Prediction:
        """Paper's inference path with explicit device handling.

        The paper repo's `infer_image` hardcodes DEVICE=cuda-if-available
        inside `image2tensor`, which breaks CPU eval and fights with
        plumbline's own `self.device`. We replicate their preprocessing
        (BGR input, MiDaS Resize + ImageNet normalize + PrepareForNet)
        but move tensors to `self.device` ourselves, then run the model
        via its ``forward`` directly and interpolate back to the input
        resolution.
        """
        torch = ensure_torch()
        import torch.nn.functional as F

        n, h, w, _ = images.shape
        disp = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            bgr = images[i][..., ::-1].copy()  # RGB → BGR
            # Paper's image2tensor does BGR2RGB then /255. Inline:
            rgb = bgr[..., ::-1].astype(np.float32) / 255.0
            # Paper's Resize + Normalize + PrepareForNet pipeline. We import
            # from the paper repo so any future tweak they ship is picked up.
            from depth_anything_v2.util.transform import (
                NormalizeImage,
                PrepareForNet,
                Resize,
            )
            from torchvision.transforms import Compose

            transform = Compose(
                [
                    Resize(
                        width=self.input_size,
                        height=self.input_size,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        # NOTE (source audit 2026-05-23): 3 == cv2.INTER_AREA,
                        # NOT cv2.INTER_CUBIC. Upstream `image2tensor` uses
                        # cv2.INTER_CUBIC (==2). Left as-is to preserve the 8
                        # verified DA-V2 cells; switching to 2 for source
                        # fidelity needs GPU re-validation. See docs/SOURCE_AUDIT.md.
                        image_interpolation_method=3,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )
            prepared = transform({"image": rgb})["image"]
            tensor = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                out = self._model.forward(tensor)  # (1, H', W') disparity
                # Resize back to input (h, w).
                out = F.interpolate(out[:, None], (h, w), mode="bilinear", align_corners=True)
            disp[i] = out[0, 0].cpu().numpy().astype(np.float32)
        depth = (1.0 / np.maximum(disp, EPS)).astype(np.float32)
        assert_valid_depth(depth, name="da-v2/output")
        repo_id, filename = _PAPER_REPOS[self.variant]
        return Prediction(
            depth=depth,
            metadata={
                "variant": self.variant,
                "source": "paper",
                "space": "depth",
                "native_space": "disparity",
                "alignment_hint": "scale_shift",
                "checkpoint": f"{repo_id}/{filename}",
                "input_size": self.input_size,
            },
        )

    def _predict_hf(self, images: NDArray[np.uint8]) -> Prediction:
        torch = ensure_torch()
        n, h, w, _ = images.shape
        batch = [images[i] for i in range(n)]
        with torch.inference_mode():
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            resized = self._processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(h, w)] * n,
            )
        raw = np.stack(
            [r["predicted_depth"].detach().cpu().numpy().astype(np.float32) for r in resized],
            axis=0,
        )
        if _is_metric_variant(self.variant):
            depth = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0).astype(np.float32)
            native_space = "depth"
            alignment_hint = "none"
        else:
            depth = (1.0 / np.maximum(raw, EPS)).astype(np.float32)
            native_space = "disparity"
            alignment_hint = "scale_shift"
        assert_valid_depth(depth, name="da-v2/output")
        return Prediction(
            depth=depth,
            metadata={
                "variant": self.variant,
                "source": "hf",
                "space": "depth",
                "native_space": native_space,
                "alignment_hint": alignment_hint,
                "checkpoint": _HF_CHECKPOINTS[self.variant],
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/variant={self.variant}"
            f"/input={self.input_size}/source={self.source}"
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]
