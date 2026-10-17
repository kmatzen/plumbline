"""Marigold diffusion-depth adapter.

Upstream: https://github.com/prs-eth/marigold
Paper: "Repurposing Diffusion-Based Image Generators for Monocular Depth
Estimation" (Ke et al. 2024).

Marigold is the first widely-adopted diffusion-based monocular depth model —
a Stable-Diffusion-v2-initialised UNet fine-tuned to produce affine-invariant
depth maps in [0, 1]. Unlike the transformer-based DA-V2 / Metric3D / MoGe
adapters, inference is iterative (N denoising steps × E ensemble passes) and
the output is stochastic unless the random generator is seeded.

We use the ``diffusers.MarigoldDepthPipeline`` since it's the only stable,
pip-installable interface; the upstream Marigold repo also works but requires
extra vendored code.

Paper protocol for NYU / KITTI table numbers:
  - ``num_inference_steps=4``
  - ``ensemble_size=10``
  - fp32 weights (not fp16)
  - deterministic seed
Run time on a 3090: ~5 s / sample at paper settings (vs ~1 s for the ViT
adapters). Speed mode (``num_inference_steps=1, ensemble_size=1``) is ~85 ms
per sample and sacrifices some precision — see module docstring on the
diffusers side for the tradeoff.

Output: relative depth in [0, 1] (affine-invariant). Scale + shift
alignment is required for metric comparison; the paper fits in DEPTH
space (not inverse-depth like DA-V2 / MiDaS). Use YAML
``scale_alignment: scale_shift`` for the default inv_depth fit, or wire
``alignment_space: depth`` once plumbline's align_depth supports that
as a kwarg.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction, config_digest
from plumbline.models.registry import register_model

__all__ = ["MarigoldAdapter"]

# Variant -> HuggingFace checkpoint. Only one "depth" family ships today; we
# leave the dict form so adding GeoWizard / DepthFM as sibling variants is
# additive. Marigold-normals and Marigold-iid are intentionally excluded —
# plumbline is a depth/geometry harness, not a normals/lighting one.
_HF_CHECKPOINTS: dict[str, str] = {
    "v1-1": "prs-eth/marigold-depth-v1-1",
    "v1-0": "prs-eth/marigold-depth-v1-0",
}


@register_model("marigold")
class MarigoldAdapter(Model):
    """Monocular, relative-depth adapter for Marigold (diffusion-based).

    Parameters
    ----------
    device
        torch device string.
    variant
        One of ``"v1-1"`` (default, recommended) or ``"v1-0"`` (original).
    num_inference_steps
        Denoising steps per sample. Paper protocol: 4. Speed mode: 1.
        More steps = higher quality at linear cost. Default 4.
    ensemble_size
        Number of random-noise passes to average per sample. Paper
        protocol: 10. Speed mode: 1. Trades wall time for precision.
    dtype
        ``"float16"`` (fp16, fast) or ``"float32"`` (paper protocol for
        leaderboard numbers). Default fp16 — fp32 is only needed when
        exactly matching paper NYU/KITTI row reproductions.
    seed
        Integer seed for the diffusion latent generator. Set this for
        reproducibility; plumbline caches predictions by config_hash so
        the seed is part of the cache key.
    per_sample_seed
        Reset the generator at the start of every ``predict`` call so cached
        output is independent of evaluation order. Enabled by default. Set to
        ``False`` only to reproduce legacy streaming-generator runs.
    """

    version = "1.1"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(768, 768),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "v1-1",
        num_inference_steps: int = 4,
        ensemble_size: int = 10,
        dtype: str = "float16",
        seed: int = 0,
        processing_res: int | None = None,
        per_sample_seed: bool = True,
    ) -> None:
        if variant not in _HF_CHECKPOINTS:
            raise ValueError(f"variant must be one of {list(_HF_CHECKPOINTS)}; got {variant!r}")
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1; got {num_inference_steps}")
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1; got {ensemble_size}")
        if dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be 'float16' or 'float32'; got {dtype!r}")
        if processing_res is not None and processing_res < 0:
            raise ValueError(
                f"processing_res must be >= 0 (0 = native resolution) or None; got {processing_res}"
            )
        self.device = device
        self.variant = variant
        self.num_inference_steps = int(num_inference_steps)
        self.ensemble_size = int(ensemble_size)
        self.dtype = dtype
        self.seed = int(seed)
        # ``processing_res`` mirrors prs-eth/Marigold's CLI flag:
        #   - None → use the diffusers pipeline default (768 long-edge).
        #   - 0 → no resize, feed the pipeline at native input resolution
        #         (the paper's eval convention for KITTI + NYU; otherwise
        #         a 1216×352 benchmark-cropped input gets squished to
        #         768×222 before denoising and produces visibly different
        #         predictions).
        #   - int > 0 → explicit long-edge resize target.
        self.processing_res = processing_res
        # One generator is created at load and threaded through every
        # ``predict`` call, so its state advances across samples: the depth
        # for a given sample_id depends on how many samples were denoised
        # before it. A `subset(n)`, pinned-ID, or resumed run therefore
        # produces different output for the same sample than a full run —
        # and prediction-cache entries, keyed on (config_hash, sample_id),
        # silently disagree between them.
        #
        # ``per_sample_seed=True`` reseeds per call, making each sample's
        # output independent of evaluation order. Legacy reproduction configs
        # explicitly opt out because their recorded runs used the streaming
        # generator. The mode is part of ``config_hash``, so the two semantics
        # never share cache entries. GeoWizard remains order-dependent by
        # design (see geowizard.py — it matches upstream `seed_all`).
        self.per_sample_seed = bool(per_sample_seed)
        self._pipe: Any = None
        self._generator: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return
        torch = ensure_torch()
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MarigoldAdapter needs `diffusers`. Install with "
                "`uv pip install -e '.[models]'` (diffusers is in the models extra) "
                "or `uv pip install diffusers`."
            ) from exc
        checkpoint = _HF_CHECKPOINTS[self.variant]
        kwargs: dict[str, Any] = {}
        if self.dtype == "float16":
            kwargs["variant"] = "fp16"
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        self._pipe = diffusers.MarigoldDepthPipeline.from_pretrained(  # type: ignore[no-untyped-call]
            checkpoint, **kwargs
        ).to(self.device)
        # Silence tqdm — plumbline's runner already shows per-sample progress.
        self._pipe.set_progress_bar_config(disable=True)
        # Seeded generator for reproducibility. Marigold's output depends
        # heavily on the random latent, so a fixed seed is necessary for a
        # useful cache — but note it is not sufficient unless
        # ``per_sample_seed`` is set; see __init__ for why.
        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="marigold/input")
        self._load()
        from PIL import Image as PImage

        torch = ensure_torch()
        n, h, w, _ = images.shape
        depths = np.empty((n, h, w), dtype=np.float32)
        if self.per_sample_seed:
            # Reset before the batch so this call's latents don't depend on
            # how many samples the runner happened to denoise earlier.
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for i in range(n):
            # Marigold pipeline takes PIL (or a list of PIL). It resizes
            # internally to the model's preferred 768-long-edge, runs the
            # denoising loop, and returns the prediction remapped to the
            # input resolution.
            pil = PImage.fromarray(images[i])
            pipe_kwargs: dict[str, Any] = {
                "num_inference_steps": self.num_inference_steps,
                "ensemble_size": self.ensemble_size,
                "generator": self._generator,
            }
            if self.processing_res is not None:
                # Diffusers' MarigoldDepthPipeline accepts ``processing_resolution``
                # (long-edge target). 0 = native (no resize) — matches
                # prs-eth/Marigold CLI's ``--processing_res 0``.
                pipe_kwargs["processing_resolution"] = self.processing_res
            out = self._pipe(pil, **pipe_kwargs)
            pred = out.prediction  # shape (1, H, W, 1) typically
            arr = np.asarray(pred).squeeze()
            if arr.shape != (h, w):
                raise RuntimeError(f"marigold returned {arr.shape}, expected ({h}, {w})")
            # Output is in [0, 1] (affine-invariant disparity-like). plumbline
            # convention stores depth where 0 = invalid; Marigold's 0 means
            # "nearest plane" which IS valid. Clamp to EPS so the downstream
            # depth-validity check doesn't treat near-plane pixels as
            # invalid. Alignment (scale_shift or scale_shift_robust) will
            # then fit this to GT metric depth.
            arr = np.clip(arr, 1e-6, 1.0).astype(np.float32)
            depths[i] = arr

        assert_valid_depth(depths, name="marigold/output")
        return Prediction(
            depth=depths,
            metadata={
                "variant": self.variant,
                "checkpoint": _HF_CHECKPOINTS[self.variant],
                "num_inference_steps": self.num_inference_steps,
                "ensemble_size": self.ensemble_size,
                "dtype": self.dtype,
                "seed": self.seed,
                "per_sample_seed": self.per_sample_seed,
                "native_space": "depth_affine_invariant",
                "alignment_hint": "scale_shift",
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/variant={self.variant}"
            f"/steps={self.num_inference_steps}/ens={self.ensemble_size}"
            f"/dtype={self.dtype}/seed={self.seed}"
            f"/processing_res={self.processing_res}"
            # Only contributes when enabled, so existing cache entries and
            # the published paper-match cells keep their current hash.
            f"{'/per_sample_seed=1' if self.per_sample_seed else ''}"
        )
        return config_digest(s)
