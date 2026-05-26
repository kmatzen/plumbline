"""MonST3R adapter (dynamic-scene / video 3D).

Upstream: https://github.com/Junyi42/monst3r
Paper: "MonST3R: A Simple Approach for Estimating Geometry in the Presence
of Motion" (Zhang et al. 2024, arXiv:2410.03825).

MonST3R is DUSt3R fine-tuned on dynamic-scene data so it estimates geometry
even when objects move — the canonical "video" member of the DUSt3R family.
Architecturally it *is* a DUSt3R model (``AsymmetricCroCo3DStereo``) with
MonST3R weights, so inference reuses dust3r's pairwise ``inference`` +
``global_aligner`` exactly like the MASt3R N-view path. This adapter
therefore delegates to the shared, tested dust3r runner.

Faithfulness scope (read before trusting numbers)
-------------------------------------------------
MonST3R's *full* video pipeline (``demo.py``) adds, on top of the base
global alignment: optical-flow consistency loss (``flow_loss_weight``),
temporal smoothing, per-frame motion masks, and window-wise alignment for
long sequences. Those refinements are **not** wired here — this adapter runs
MonST3R weights through the base dust3r ``PointCloudOptimizer``
(``flow_loss_weight=0`` equivalent), which yields genuine MonST3R per-view
geometry but the *plain* global alignment, not MonST3R's flow-refined
trajectory. That refinement is a scoped follow-up (it needs the bundled flow
network + motion masks and is best validated on a GPU against MonST3R's own
video-depth eval). Treat this as "MonST3R base inference"; do not pin a
flow-dependent paper cell to it without that work.

Install
-------
Recursive clone (ships its own ``dust3r``/``croco`` fork), not on PyPI::

    git clone --recursive https://github.com/Junyi42/monst3r /workspace/deps/monst3r
    export MONST3R_ROOT=/workspace/deps/monst3r
    uv pip install roma scikit-learn trimesh   # dust3r transitive deps

Weights are pulled from HuggingFace
(``Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt``) via
``from_pretrained`` on first use.

Note: MonST3R and MASt3R each ship their own ``dust3r`` fork. Importing both
adapters in one process will resolve ``import dust3r`` to whichever loaded
first (Python caches it). Run one DUSt3R-family model per process.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    assert_valid_point_map,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction

# Reuse the MASt3R adapter's tested dust3r runner. ``_run_mast3r`` is
# model-agnostic: it runs dust3r ``inference`` over the complete pairwise
# graph, aligns with PairViewer (N==2) / PointCloudOptimizer (N>=3), and
# returns plumbline-shaped arrays + view-0 rebase. Passing MonST3R's model
# in runs MonST3R through that same path.
from plumbline.models.mast3r import _run_mast3r
from plumbline.models.registry import register_model

__all__ = ["MonST3RAdapter"]

_DEFAULT_HF = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"


@register_model("monst3r")
class MonST3RAdapter(Model):
    """Dynamic-scene 3D + pose model (MonST3R), base (no-flow) inference.

    Parameters
    ----------
    device
        torch device string.
    checkpoint
        HuggingFace model id (or local path) for ``from_pretrained``.
        Defaults to the released PO-TA-S-W ViT-Large 512 DPT weights.
    long_edge
        dust3r ``load_images`` long-edge resize target (512 for the DPT
        checkpoint).
    ga_niter, ga_lr, ga_schedule, ga_init
        PointCloudOptimizer global-alignment hyperparameters (N>=3),
        dust3r defaults — fold into ``config_hash`` so changing them
        invalidates cached predictions.
    """

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # relative; realign per-protocol at the runner
        # Single frames are supported by duplicating the view (MonST3R
        # demo does this), so the model is usable for single-frame depth.
        min_views=1,
        max_views=32,
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = _DEFAULT_HF,
        long_edge: int = 512,
        ga_niter: int = 300,
        ga_lr: float = 0.01,
        ga_schedule: str = "linear",
        ga_init: str = "mst",
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.long_edge = int(long_edge)
        self.ga_niter = int(ga_niter)
        self.ga_lr = float(ga_lr)
        self.ga_schedule = str(ga_schedule)
        self.ga_init = str(ga_init)
        self._model: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_monst3r_on_path()
        try:
            from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - needs the repo
            raise ImportError(
                "MonST3RAdapter needs the MonST3R repo (ships its own dust3r "
                "fork). Clone https://github.com/Junyi42/monst3r recursively "
                "and set $MONST3R_ROOT (default /workspace/deps/monst3r)."
            ) from exc
        # demo.py: AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
        self._model = AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint).to(self.device).eval()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="monst3r/input")
        n = int(images.shape[0])
        if n > self.capabilities.max_views:
            raise ValueError(
                f"monst3r adapter capped at {self.capabilities.max_views} views; got {n}"
            )
        self._load()

        # MonST3R demo duplicates a lone frame to form a pair.
        single = n == 1
        run_images = np.concatenate([images, images], axis=0) if single else images

        _ensure_monst3r_on_path()
        out = _run_mast3r(
            self._model,
            run_images,
            device=self.device,
            long_edge=self.long_edge,
            ga_niter=self.ga_niter,
            ga_lr=self.ga_lr,
            ga_schedule=self.ga_schedule,
            ga_init=self.ga_init,
        )

        depth = out["depth"]
        K = out["intrinsics"]
        extrinsics = out["extrinsics"]
        point_map = out["point_map"]
        confidence = out.get("confidence")
        if single:
            # Keep only the original frame's outputs.
            depth = depth[:1]
            K = K[:1]
            extrinsics = extrinsics[:1]
            point_map = point_map[:1]
            if confidence is not None:
                confidence = confidence[:1]

        assert_valid_depth(depth, name="monst3r/depth")
        assert_valid_intrinsics(K, name="monst3r/intrinsics")
        assert_valid_extrinsics(extrinsics, name="monst3r/extrinsics")
        assert_valid_point_map(point_map, name="monst3r/point_map")

        return Prediction(
            depth=depth.astype(np.float32),
            intrinsics=K.astype(np.float32),
            extrinsics=extrinsics.astype(np.float32),
            point_map=point_map.astype(np.float32),
            confidence=(confidence.astype(np.float32) if confidence is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "n_views": n,
                "single_frame_duplicated": single,
                "native_space": "point_map",
                "alignment_hint": "median",
                "flow_refinement": False,  # base dust3r global alignment only
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/{self.checkpoint}"
            f"/le{self.long_edge}/ga_n{self.ga_niter}_lr{self.ga_lr}"
            f"_sch{self.ga_schedule}_init{self.ga_init}/noflow"
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _ensure_monst3r_on_path() -> None:
    """Add ``$MONST3R_ROOT`` to sys.path so ``from dust3r... import`` resolves.

    MonST3R ships its own ``dust3r`` fork at the repo root.
    """
    root = os.environ.get("MONST3R_ROOT", "/workspace/deps/monst3r")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
    _shim_sam2_for_monst3r()


def _shim_sam2_for_monst3r() -> None:
    """Stub the ``sam2`` package so MonST3R's vendored dust3r can import.

    MonST3R's ``dust3r/cloud_opt/optimizer.py`` ships a *module-level*
    ``from sam2.build_sam import build_sam2_video_predictor``. The
    symbol is only invoked inside ``refine_motion_mask_w_sam2()`` —
    part of the motion-mask refinement path that the base-aligner
    adapter (this one) never triggers. Installing sam2 + its
    checkpoint file (~2 GB) just to import a never-called function
    isn't worth it; instead we register a no-op ``sam2`` module in
    ``sys.modules`` before the upstream import resolves. If the
    motion-mask path ever runs, calling
    ``build_sam2_video_predictor`` from the stub raises a clear
    ``ImportError`` pointing at the real install.

    Idempotent; no-op if a real ``sam2`` is already importable.
    """
    if "sam2" in sys.modules:
        return
    try:
        import sam2  # noqa: F401
        return
    except ImportError:
        pass

    import types

    sam2_pkg = types.ModuleType("sam2")
    sam2_pkg.__path__ = []  # mark as a package so `from sam2.X import` works
    build_sam_mod = types.ModuleType("sam2.build_sam")

    def _build_sam2_video_predictor(*args: object, **kwargs: object) -> object:
        raise ImportError(
            "sam2 is stubbed by plumbline (motion-mask refinement is out "
            "of scope for the base MonST3R adapter). Install MonST3R's "
            "third_party/sam2 + checkpoint if you need refine_motion_mask_w_sam2()."
        )

    build_sam_mod.build_sam2_video_predictor = _build_sam2_video_predictor  # type: ignore[attr-defined]
    sys.modules["sam2"] = sam2_pkg
    sys.modules["sam2.build_sam"] = build_sam_mod
