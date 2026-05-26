"""GeoWizard diffusion depth+normals adapter.

Upstream: https://github.com/fuxiao0719/GeoWizard
Paper: "GeoWizard: Unleashing the Diffusion Priors for 3D Geometry
Estimation from a Single Image" (Fu et al. 2024, arXiv:2403.12013).

GeoWizard is a diffusion-based monocular depth + surface-normals model,
built on a Stable-Diffusion-v2-initialised UNet. Conceptually a close
cousin of Marigold (this plumbline's `marigold` adapter), with two
differentiators:

  1. Joint depth + normals in a single forward pass (we only use depth).
  2. Domain conditioning: passing ``domain="indoor"`` vs ``"outdoor"``
     swaps an embedding that trades off near-field detail vs far-field
     range. Indoor conditioning is the correct setting for NYU / DIODE
     indoor / iBims; outdoor is for KITTI / DIODE outdoor.

Upstream does **not** ship on PyPI. Pattern mirrors the MASt3R adapter:

    git clone https://github.com/fuxiao0719/GeoWizard /workspace/deps/geowizard
    export GEOWIZARD_ROOT=/workspace/deps/geowizard

The adapter lazy-adds ``$GEOWIZARD_ROOT/geowizard`` to ``sys.path`` on
first ``predict()`` so it can `from models.geowizard_pipeline import
DepthNormalEstimationPipeline`. Weights come from the HuggingFace repo
``lemonaddie/Geowizard`` (safetensors, loaded via
``DepthNormalEstimationPipeline.from_pretrained``).

Alignment: GeoWizard output is affine-invariant depth in [0, 1], same
as Marigold. Use ``scale_alignment: scale_shift_depth`` to match the
paper's depth-space alignment protocol (not the inverse-depth fit
DA-V2 / MoGe use). The author's own alignment helper at
``geowizard/utils/de_normalized.py::align_scale_shift`` uses
``np.polyfit(deg=1)`` over masked depth-space — structurally
identical to plumbline's ``scale_shift_depth``.

D17 / D18 note (resolved 2026-05-26): single-seed eval on the
publicly-released ``lemonaddie/Geowizard`` checkpoint lands AbsRel
~0.057 on NYU and ~0.11-0.13 on KITTI, vs paper Table 1's 0.052 /
0.097. Adapter was audited end-to-end (dtype, xformers, full
``seed_all``, denoise_steps 10 vs 50) — none move the metric beyond
±1 %. Independent reproducer @anonymous on
`fuxiao0719/GeoWizard#36` measured 0.0576 / 0.9615 (NYU, 50-step,
ens-10), matching plumbline's 0.0574 / 0.9594 exactly. The author
replied on that issue: *"we perform multiple inferences with
different initialized seeds for each test dataset, along with the
[ensemble] operation, and select the best result for the metric
report."* The paper number is best-of-N seeds, not single-seed mean
— an undocumented eval-protocol detail. Single-seed numbers from
this adapter are the methodologically defensible baseline; the
paper's 0.052 / 0.097 are not exactly reproducible without
cherry-picking across seed draws.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import assert_valid_depth, assert_valid_image
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["GeoWizardAdapter"]

# GeoWizard ships one depth+normals checkpoint at this HF repo. The
# value below matches the default used by upstream's run_infer.py.
_HF_CHECKPOINT: str = "lemonaddie/Geowizard"

_VALID_DOMAINS: frozenset[str] = frozenset({"indoor", "outdoor", "object"})


@register_model("geowizard")
class GeoWizardAdapter(Model):
    """Monocular, relative-depth adapter for GeoWizard (diffusion-based).

    Parameters
    ----------
    device
        torch device string.
    domain
        ``"indoor"``, ``"outdoor"``, or ``"object"``. GeoWizard trains a
        single model with per-sample domain conditioning; pick the value
        the paper's target table uses. NYU / DIODE-indoor / iBims-1 use
        ``"indoor"``; KITTI / DIODE-outdoor use ``"outdoor"``; GSO
        synthetic-object runs use ``"object"``.
    num_inference_steps
        Denoising steps per sample. Upstream default for paper-row
        reproductions: 10. Speed mode: 4.
    ensemble_size
        Number of random-noise passes to average per sample. Upstream
        default: 10. Speed mode: 3.
    processing_res
        Long-edge the diffusion model consumes; upstream default 768.
        Drop to 512 for speed on <24 GB VRAM.
    dtype
        ``"float16"`` (fast) or ``"float32"`` (paper protocol).
    seed
        Integer seed for the diffusion latent generator. Fixed for
        reproducibility + plumbline prediction-cache key stability.
    """

    version = "1.0"
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
        domain: str = "indoor",
        num_inference_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        dtype: str = "float16",
        seed: int = 0,
    ) -> None:
        if domain not in _VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(_VALID_DOMAINS)}; got {domain!r}")
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1; got {num_inference_steps}")
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1; got {ensemble_size}")
        if processing_res < 64 or processing_res % 8 != 0:
            raise ValueError(
                f"processing_res must be >= 64 and a multiple of 8 (diffusion VAE); got {processing_res}"
            )
        if dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be 'float16' or 'float32'; got {dtype!r}")
        self.device = device
        self.domain = domain
        self.num_inference_steps = int(num_inference_steps)
        self.ensemble_size = int(ensemble_size)
        self.processing_res = int(processing_res)
        self.dtype = dtype
        self.seed = int(seed)
        self._pipe: Any = None
        self._generator: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return
        torch = ensure_torch()
        _shim_diffusers_for_geowizard()
        _ensure_geowizard_on_path()
        try:
            # Upstream's pipeline class; discovered at runtime from
            # $GEOWIZARD_ROOT/geowizard/models/geowizard_pipeline.py.
            from models.geowizard_pipeline import DepthNormalEstimationPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GeoWizardAdapter could not import the upstream pipeline. "
                "Clone https://github.com/fuxiao0719/GeoWizard and point "
                "$GEOWIZARD_ROOT at it (see the module docstring)."
            ) from exc

        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
        self._pipe = DepthNormalEstimationPipeline.from_pretrained(
            _HF_CHECKPOINT,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        # Match upstream ``run_infer.py``: try to enable xformers memory-
        # efficient attention; silently fall back to vanilla SDP if the
        # xformers wheel isn't compatible. Upstream gates inference behind
        # this call (best-effort), so for paper-protocol parity we do the
        # same. Numerics differ slightly from vanilla SDP but the
        # paper-row was produced with xformers enabled. Speed: ~4-5× for
        # the 10×10 ensemble at processing_res=768 on a 3090.
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        # Upstream's __call__ does NOT accept a `generator` kwarg; seed the
        # global RNG right before each sample instead (see predict()).
        self._generator = None

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="geowizard/input")
        # Seed the global RNG ONCE on first predict() call, then let the
        # state advance through subsequent samples. Matches upstream
        # ``run_infer.py``'s ``seed_all(seed)`` at startup followed by a
        # for-loop over images that advances the global RNG implicitly.
        # Plumbline previously reseeded per-sample (``manual_seed(seed +
        # i)``) which guaranteed per-sample reproducibility but produced
        # different ensemble denoising trajectories than upstream — D17
        # diagnosis. Also matches upstream's ``seed_all`` body: random,
        # numpy, torch, and torch.cuda. ``ensemble_depths`` calls
        # ``scipy.optimize.minimize(BFGS)`` whose convergence path is
        # deterministic given the input but the helper imports
        # ``random``/``np.random`` upstream — match for paranoia.
        first_call = self._pipe is None
        self._load()
        from PIL import Image as PImage

        torch = ensure_torch()
        if first_call:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        n, h, w, _ = images.shape
        depths = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            pil = PImage.fromarray(images[i])
            out = self._pipe(
                pil,
                denoising_steps=self.num_inference_steps,
                ensemble_size=self.ensemble_size,
                processing_res=self.processing_res,
                match_input_res=True,
                domain=self.domain,
                show_progress_bar=False,
            )
            # Upstream emits `depth_np` as float in [0, 1], affine-invariant.
            arr = np.asarray(out.depth_np).astype(np.float32)
            if arr.shape != (h, w):
                raise RuntimeError(
                    f"geowizard returned {arr.shape}, expected ({h}, {w}). "
                    "Set match_input_res=True (the adapter does) and check for "
                    "upstream API drift."
                )
            arr = np.clip(arr, 1e-6, 1.0)
            depths[i] = arr

        assert_valid_depth(depths, name="geowizard/output")
        return Prediction(
            depth=depths,
            metadata={
                "domain": self.domain,
                "checkpoint": _HF_CHECKPOINT,
                "num_inference_steps": self.num_inference_steps,
                "ensemble_size": self.ensemble_size,
                "processing_res": self.processing_res,
                "dtype": self.dtype,
                "seed": self.seed,
                "native_space": "depth_affine_invariant",
                "alignment_hint": "scale_shift_depth",
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/domain={self.domain}"
            f"/steps={self.num_inference_steps}/ens={self.ensemble_size}"
            f"/res={self.processing_res}/dtype={self.dtype}/seed={self.seed}"
            # rng_mode tracks the per-sample RNG seeding scheme; bumped
            # 2026-04-26 to match upstream ``run_infer.py``'s "seed once
            # at startup" pattern (D17). v2 adds random+numpy seeding
            # for full ``seed_all`` parity. v3 enables xformers
            # attention (changes attention numerics — paper-protocol).
            # Old cache entries stay intact under the previous hash;
            # the new path runs fresh.
            "/rng_mode=once_at_startup_v3_xformers"
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_geowizard_on_path() -> None:
    """Add ``$GEOWIZARD_ROOT/geowizard`` to sys.path for upstream imports.

    Upstream has no PyPI distribution; it's a CLI repo. Callers set
    ``$GEOWIZARD_ROOT`` to the cloned repo root; the pipeline class
    is at ``<root>/geowizard/models/geowizard_pipeline.py`` and its
    internal imports are relative to ``<root>/geowizard``.
    """
    root = os.environ.get("GEOWIZARD_ROOT", "/workspace/deps/geowizard")
    subdir = os.path.join(root, "geowizard")
    if os.path.isdir(subdir) and subdir not in sys.path:
        sys.path.insert(0, subdir)


def _shim_diffusers_for_geowizard() -> None:
    """Patch diffusers' public surface for upstream GeoWizard compat.

    Upstream GeoWizard was forked from a diffusers version several
    releases older than what plumbline runs. Two spots drifted:

    1. ``diffusers.models.embeddings.PositionNet`` was renamed to
       ``GLIGENTextBoundingboxProjection`` — same class body, same
       ``__init__(positive_len, out_dim, feature_type, fourier_freqs)``
       signature.
    2. ``diffusers.models.dual_transformer_2d`` moved under the
       ``diffusers.models.transformers`` subpackage. Aliased via
       ``sys.modules`` so the old import path resolves.

    Idempotent; no-op if either name is already where GeoWizard expects.
    """
    import diffusers.models.embeddings as _dme

    if not hasattr(_dme, "PositionNet"):
        replacement = getattr(_dme, "GLIGENTextBoundingboxProjection", None)
        if replacement is not None:
            _dme.PositionNet = replacement

    try:
        import diffusers.models.dual_transformer_2d  # noqa: F401
    except ModuleNotFoundError:
        try:
            import diffusers.models.transformers.dual_transformer_2d as _dt

            sys.modules["diffusers.models.dual_transformer_2d"] = _dt
        except ModuleNotFoundError:  # pragma: no cover
            pass
