"""DUSt3R adapter.

Upstream: https://github.com/naver/dust3r
Paper: "DUSt3R: Geometric 3D Vision Made Easy" (Wang et al. 2024,
arXiv:2312.14132, CVPR 2024).

DUSt3R is the foundation model that MASt3R and MonST3R extend; it predicts
per-pixel 3D point maps in a shared frame for a pair of images. The
adapter dispatches on view count the same way MASt3R does:

- ``N == 1``: paper §4.3 monocular depth protocol — "we simply feed the
  same input image I to the network as F(I, I)". Implemented as the
  view-duplicate trick (load the image twice, symmetrize the pair,
  average the two pred1.pts3d directions). This is the same path
  MonST3R's ``eval_mono_depth`` helper uses; it's the canonical
  reproduction of DUSt3R Table 2 (NYU / KITTI / Bonn / DDAD / TUM).
- ``N == 2``: feed-forward via dust3r's ``PairViewer`` global aligner
  (fast, no optimization).
- ``N >= 3``: dust3r's ``PointCloudOptimizer`` global aligner — iterative
  optimization over the chosen pairwise graph (``scene_graph`` kwarg;
  default ``"complete"``, set to ``"swinstride-5-noncyclic"`` for long
  sequences where the complete graph blows past 24 GB).

The model + global aligner are functionally identical to what the MASt3R
adapter drives (MASt3R is "dust3r weights swapped for the MASt3R matching
checkpoint" + a per-pair confidence head); we deliberately reuse the
shared ``_run_mast3r`` helper. The only DUSt3R-specific piece is the
upstream package + checkpoint.

Install
-------
Upstream is a self-contained git repo with no PyPI release. Set
``$DUST3R_ROOT`` to the clone (default ``/workspace/deps/dust3r``);
the adapter adds it to ``sys.path`` lazily.

    git clone --recursive https://github.com/naver/dust3r
    uv pip install roma scikit-learn trimesh

Inputs / outputs
----------------
Same canonical shapes as MASt3R: ``depth (N,H,W)``, ``intrinsics (N,3,3)``,
``extrinsics (N,4,4)`` ``world_from_camera`` rebased to view 0,
``point_map (N,H,W,3)`` in the rebased world frame. DUSt3R is not metric
(``alignment_hint="median"``).
"""

from __future__ import annotations

import hashlib
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
from plumbline.models.mast3r import (
    _images_to_dust3r_dicts,
    _run_mast3r,  # share the dust3r runner
)
from plumbline.models.registry import register_model

__all__ = ["DUSt3RAdapter"]


@register_model("dust3r")
class DUSt3RAdapter(Model):
    """DUSt3R foundation model (Wang et al. 2024)."""

    # 1.0 → 1.1: added single-frame (N=1) view-duplicate path so DUSt3R Table 2
    # (Wang 2024, arXiv:2312.14132) NYU/KITTI/Bonn/DDAD/TUM monocular cells
    # can be reproduced. Pre-1.1 the adapter raised on N<2; the paper §4.3
    # explicitly does F(I, I), matching MonST3R's `eval_mono_depth` recipe.
    version = "1.1"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # recovers a scale up to residual ambiguity
        # Paper §4.3: monocular = F(I, I) view duplication. The adapter
        # mirrors MonST3R's single-frame branch for the Table 2 cells.
        min_views=1,
        # Bumped to 60 so long Sintel-final clips (~50 frames) fit when
        # the YAML overrides scene_graph to "swinstride-5-noncyclic".
        max_views=60,
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        long_edge: int = 512,
        ga_niter: int = 300,
        ga_lr: float = 0.01,
        ga_schedule: str = "linear",
        ga_init: str = "mst",
        # scene_graph: default "complete" for short multi-view sets (matches
        # the MASt3R adapter and dust3r's quick-start). For long sequences
        # (e.g. 50-frame Sintel clips for pose eval) set to one of dust3r's
        # sliding-window graphs — "swinstride-5-noncyclic" keeps pair count
        # ~5N instead of N(N-1)/2.
        scene_graph: str = "complete",
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.long_edge = int(long_edge)
        self.ga_niter = int(ga_niter)
        self.ga_lr = float(ga_lr)
        self.ga_schedule = str(ga_schedule)
        self.ga_init = str(ga_init)
        self.scene_graph = str(scene_graph)
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_dust3r_on_path()
        try:
            from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('dust3r')}") from exc
        model = AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="dust3r/input")
        n = int(images.shape[0])
        if n < 1:
            raise ValueError(f"DUSt3R requires at least 1 view; got {n}")
        if n > self.capabilities.max_views:
            raise ValueError(
                f"DUSt3R adapter capped at {self.capabilities.max_views} views; got {n}"
            )
        self._load()

        # N=1: paper §4.3 "F(I, I)" — view-duplicate trick, then average
        # the two symmetric pred1.pts3d directions. Same path MonST3R's
        # `eval_mono_depth` helper uses on its vendored dust3r fork; here
        # we resolve the dust3r symbols from $DUST3R_ROOT instead.
        single = n == 1
        if single:
            depth, point_map, K = _dust3r_single_frame_eval(
                self._model,
                images,
                device=self.device,
                long_edge=self.long_edge,
            )
            extrinsics = np.eye(4, dtype=np.float32)[None]
            confidence = None
        else:
            out = _run_mast3r(
                self._model,
                images,
                device=self.device,
                long_edge=self.long_edge,
                ga_niter=self.ga_niter,
                ga_lr=self.ga_lr,
                ga_schedule=self.ga_schedule,
                ga_init=self.ga_init,
                scene_graph=self.scene_graph,
            )

            depth = out["depth"]
            K = out["intrinsics"]
            extrinsics = out["extrinsics"]
            point_map = out["point_map"]
            confidence = out.get("confidence")

        assert_valid_depth(depth, name="dust3r/output_depth")
        assert_valid_intrinsics(K, name="dust3r/output_K")
        assert_valid_extrinsics(extrinsics, name="dust3r/output_E")
        assert_valid_point_map(point_map, name="dust3r/output_pmap")

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
                "single_frame_path": "eval_mono_depth_avg" if single else None,
                "native_space": "point_map",
                "alignment_hint": "median",
                "scene_graph": self.scene_graph,
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/{self.checkpoint}"
            f"/le{self.long_edge}/ga_n{self.ga_niter}_lr{self.ga_lr}"
            f"_sch{self.ga_schedule}_init{self.ga_init}"
            f"/sg{self.scene_graph}"
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _ensure_dust3r_on_path() -> None:
    """Add $DUST3R_ROOT to sys.path so `from dust3r.model import …` works.

    Upstream is a recursive git clone (no PyPI). Callers are expected to
    set $DUST3R_ROOT or accept the default.
    """
    import os
    import sys

    root = os.environ.get("DUST3R_ROOT", "/workspace/deps/dust3r")
    if root not in sys.path and os.path.isdir(root):
        sys.path.insert(0, root)


def _dust3r_single_frame_eval(
    model: Any,
    images: NDArray[np.uint8],
    *,
    device: str,
    long_edge: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Reproduce DUSt3R §4.3 monocular depth: feed `F(I, I)` and average the
    two symmetric pred1.pts3d directions.

    Paper §4.3: *"For this monocular task, we simply feed the same input
    image I to the network as F(I, I). By design, depth prediction is
    simply the z coordinate in the predicted 3D pointmap."* The MonST3R
    fork later concretised this into the ``eval_mono_depth`` helper
    (load the image twice → ``symmetrize=True`` → average across the
    leading axis) — we mirror that helper here so the two adapters
    score the same Table 2 / Table 3 cells with bit-identical logic,
    only differing in checkpoint + which ``dust3r`` package is on
    sys.path.

    Returns ``(depth, point_map, K)`` shaped ``(1, H, W)``, ``(1, H, W, 3)``,
    ``(1, 3, 3)`` so the caller can wrap them into a :class:`Prediction`.
    K is a synthetic centre-principal-point camera with focal = max(H, W) —
    monocular depth scoring is scale-only (per-frame median), so K never
    enters the metric; we just need a shape-valid value to satisfy
    :func:`assert_valid_intrinsics`.
    """
    from copy import deepcopy

    from dust3r.image_pairs import make_pairs  # type: ignore[import-not-found]
    from dust3r.inference import inference  # type: ignore[import-not-found]

    if images.shape[0] != 1:
        raise ValueError(
            f"_dust3r_single_frame_eval expects exactly 1 image; got {images.shape[0]}"
        )

    # Use the same image-prep helper the multi-view path uses (lives in
    # mast3r.py) — guarantees the resize / center-crop is bit-identical
    # to dust3r's ``load_images``.
    dust3r_imgs = _images_to_dust3r_dicts(images, long_edge=long_edge)
    img0 = dust3r_imgs[0]
    img1 = deepcopy(img0)
    img1["idx"] = 1
    img1["instance"] = "1"
    pairs = make_pairs([img0, img1], symmetrize=True, prefilter=None)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    # pred1.pts3d: (2, H, W, 3) — average across the 2 symmetric pairs.
    pts3d = output["pred1"]["pts3d"].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    depth = pts3d[..., -1]  # (H, W) — z-component
    # Drop non-finite / non-positive values, matching the multi-view runner.
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)

    H, W = depth.shape
    f = float(max(H, W))
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return depth[None], pts3d[None], K[None]
