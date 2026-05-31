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
from plumbline.models.mast3r import _images_to_dust3r_dicts, _run_mast3r
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

    # Bumped 2026-05-26 from "1.0" → "1.1": single-frame `n == 1` branch
    # now matches MonST3R's `eval_mono_depth` exactly (mean of pred1.pts3d
    # across the two symmetric pairs) instead of routing through the
    # MASt3R-shared PairViewer N=2 path. PairViewer returns a single
    # direction's depth without averaging, which empirically diverged ~9 %
    # on Sintel / ~14 % on Bonn in PR #5. Version bump invalidates the
    # prediction cache for any monst3r-* reproduction.
    # 1.2 (2026-05-27): added the full MonST3R video-pose pipeline path —
    # swinstride-5-noncyclic pair graph + MonST3R's extended PCO with flow
    # consistency / motion mask / temporal smoothness. Gated on the new
    # `video_pose: bool = False` constructor kwarg; existing N=1 / N>=2
    # depth callsites are unchanged so depth-side cache hits are preserved.
    version = "1.2"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # relative; realign per-protocol at the runner
        # Single frames are supported by duplicating the view (MonST3R
        # demo does this), so the model is usable for single-frame depth.
        min_views=1,
        # 60 covers Sintel's longest scene (~50 frames) for trajectory
        # pose evaluation (MonST3R Table 4); CO3Dv2 pose uses 10 views,
        # well under this cap.
        max_views=60,
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
        # ── MonST3R video-pose mode ─────────────────────────────────
        # When True AND n>=3, route inference through MonST3R's full
        # video-pose pipeline: sliding-window pair graph + flow-loss +
        # motion-mask + temporal-smoothness terms in the global
        # alignment. Matches what `launch.py --mode=eval_pose` does for
        # Table 4 / sintel / TUM / scannet rows. When False (default),
        # n>=3 routes through the MASt3R-shared base PointCloudOptimizer
        # (no flow / motion / temporal terms) — appropriate for the
        # mono-depth / unordered-multi-view cells.
        video_pose: bool = False,
        # Per MonST3R `dust3r/training.py` defaults:
        scene_graph_type: str = "swinstride-5-noncyclic",
        flow_loss_weight: float = 0.01,
        flow_loss_fn: str = "smooth_l1",
        flow_loss_start_epoch: float = 0.1,
        flow_loss_thre: float = 20.0,
        pxl_thresh: float = 50.0,
        motion_mask_thre: float = 0.35,
        temporal_smoothing_weight: float = 0.01,
        translation_weight: float = 1.0,
        depth_regularize_weight: float = 0.0,
        # When True, signals to MonST3R's optimizer to use raft-sintel
        # (vs the Tartan-C-T-TSKH default) for flow consistency. Set this
        # via the YAML when the dataset is Sintel.
        sintel_ckpt: bool = False,
        use_self_mask: bool = True,
        sam2_mask_refine: bool = False,
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.long_edge = int(long_edge)
        self.ga_niter = int(ga_niter)
        self.ga_lr = float(ga_lr)
        self.ga_schedule = str(ga_schedule)
        self.ga_init = str(ga_init)
        self.video_pose = bool(video_pose)
        self.scene_graph_type = str(scene_graph_type)
        self.flow_loss_weight = float(flow_loss_weight)
        self.flow_loss_fn = str(flow_loss_fn)
        self.flow_loss_start_epoch = float(flow_loss_start_epoch)
        self.flow_loss_thre = float(flow_loss_thre)
        self.pxl_thresh = float(pxl_thresh)
        self.motion_mask_thre = float(motion_mask_thre)
        self.temporal_smoothing_weight = float(temporal_smoothing_weight)
        self.translation_weight = float(translation_weight)
        self.depth_regularize_weight = float(depth_regularize_weight)
        self.sintel_ckpt = bool(sintel_ckpt)
        self.use_self_mask = bool(use_self_mask)
        self.sam2_mask_refine = bool(sam2_mask_refine)
        self._model: Any = None

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_monst3r_on_path()
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:  # pragma: no cover - needs the repo
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('monst3r')}") from exc
        # demo.py: AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
        self._model = (
            AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint).to(self.device).eval()
        )

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
        _ensure_monst3r_on_path()

        # MonST3R demo duplicates a lone frame to form a pair. For the
        # single-frame Table 3 protocol we faithfully reproduce
        # `dust3r/depth_eval.py:eval_mono_depth`, which AVERAGES the two
        # symmetric pair predictions:
        #
        #     pairs = make_pairs(symmetrize=True)        # (0,1) + (1,0)
        #     output = inference(pairs, ...)             # pred1.pts3d (2,H,W,3)
        #     depth_map = output['pred1']['pts3d'][..., -1].mean(dim=0)
        #
        # Pre-1.1 we routed N=2 through MASt3R's PairViewer path, which
        # returns a single direction's depth without averaging — fine for
        # NYU/KITTI (within 2-3 % of paper) but ~9 % off on Sintel and ~14 %
        # off on Bonn (see PR #5 + DISCREPANCIES D26).
        single = n == 1
        if single:
            depth, point_map, K = _monst3r_single_frame_eval(
                self._model,
                images,
                device=self.device,
                long_edge=self.long_edge,
            )
            extrinsics = np.eye(4, dtype=np.float32)[None]
            confidence = None
        elif self.video_pose and n >= 3:
            # Full MonST3R video-pose pipeline: swinstride pair graph +
            # MonST3R's extended PointCloudOptimizer with flow / motion /
            # temporal kwargs. Matches `launch.py --mode=eval_pose`.
            out = _run_monst3r_video_pose(
                self._model,
                images,
                device=self.device,
                long_edge=self.long_edge,
                ga_niter=self.ga_niter,
                ga_lr=self.ga_lr,
                ga_schedule=self.ga_schedule,
                ga_init=self.ga_init,
                scene_graph_type=self.scene_graph_type,
                flow_loss_weight=self.flow_loss_weight,
                flow_loss_fn=self.flow_loss_fn,
                flow_loss_start_epoch=self.flow_loss_start_epoch,
                flow_loss_thre=self.flow_loss_thre,
                pxl_thresh=self.pxl_thresh,
                motion_mask_thre=self.motion_mask_thre,
                temporal_smoothing_weight=self.temporal_smoothing_weight,
                translation_weight=self.translation_weight,
                depth_regularize_weight=self.depth_regularize_weight,
                sintel_ckpt=self.sintel_ckpt,
                use_self_mask=self.use_self_mask,
                sam2_mask_refine=self.sam2_mask_refine,
            )
            depth = out["depth"]
            K = out["intrinsics"]
            extrinsics = out["extrinsics"]
            point_map = out["point_map"]
            confidence = out.get("confidence")
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
            )
            depth = out["depth"]
            K = out["intrinsics"]
            extrinsics = out["extrinsics"]
            point_map = out["point_map"]
            confidence = out.get("confidence")

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
                "single_frame_path": "eval_mono_depth_avg" if single else None,
            },
        )

    def config_hash(self) -> str:
        s = (
            f"{self.name}@{self.version}/{self.checkpoint}"
            f"/le{self.long_edge}/ga_n{self.ga_niter}_lr{self.ga_lr}"
            f"_sch{self.ga_schedule}_init{self.ga_init}"
        )
        if self.video_pose:
            # Video-pose path is materially different — cache must not
            # collide with the depth-path cache for the same N.
            s += (
                f"/vp_sg{self.scene_graph_type}"
                f"_fw{self.flow_loss_weight}_ffn{self.flow_loss_fn}"
                f"_fse{self.flow_loss_start_epoch}_flt{self.flow_loss_thre}"
                f"_pxl{self.pxl_thresh}_mmt{self.motion_mask_thre}"
                f"_tsw{self.temporal_smoothing_weight}_tw{self.translation_weight}"
                f"_drw{self.depth_regularize_weight}"
                f"_sin{int(self.sintel_ckpt)}_sm{int(self.use_self_mask)}"
                f"_smr{int(self.sam2_mask_refine)}"
            )
        else:
            s += "/noflow"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _monst3r_single_frame_eval(
    model: Any,
    images: NDArray[np.uint8],
    *,
    device: str,
    long_edge: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Reproduce MonST3R's ``eval_mono_depth`` single-frame path verbatim.

    The view-duplicate trick: build a 2-element list with the same image
    twice, ``symmetrize=True`` makes both directed pairs (0,1) and (1,0),
    and the resulting ``output['pred1']['pts3d']`` has shape ``(2, H, W, 3)``
    — one prediction per pair, both expressed in the (duplicated) view-1
    coordinate frame. Averaging across the leading axis is what the upstream
    eval helper does (`pts3d.mean(dim=0)`).

    Returns ``(depth, point_map, K)`` shaped ``(1, H, W)``, ``(1, H, W, 3)``,
    ``(1, 3, 3)`` so the caller can wrap them into a :class:`Prediction`. K
    is a synthetic centre-principal-point camera with focal = max(H, W) — the
    Table 3 protocol scores depth only (per-frame median scaling), so K
    doesn't enter the metric; we just need a shape-valid value to satisfy
    :func:`assert_valid_intrinsics`.
    """
    from copy import deepcopy

    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference

    if images.shape[0] != 1:
        raise ValueError(
            f"_monst3r_single_frame_eval expects exactly 1 image; got {images.shape[0]}"
        )

    # Use the same image-prep helper the multi-view path uses — guarantees
    # the resize / center-crop is bit-identical to dust3r's load_images.
    dust3r_imgs = _images_to_dust3r_dicts(images, long_edge=long_edge)
    img0 = dust3r_imgs[0]
    # Duplicate the view (eval_mono_depth: `[imgs[0], deepcopy(imgs[0])]`).
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


def _run_monst3r_video_pose(
    model: Any,
    images: NDArray[np.uint8],
    *,
    device: str,
    long_edge: int,
    ga_niter: int,
    ga_lr: float,
    ga_schedule: str,
    ga_init: str,
    scene_graph_type: str,
    flow_loss_weight: float,
    flow_loss_fn: str,
    flow_loss_start_epoch: float,
    flow_loss_thre: float,
    pxl_thresh: float,
    motion_mask_thre: float,
    temporal_smoothing_weight: float,
    translation_weight: float,
    depth_regularize_weight: float,
    sintel_ckpt: bool,
    use_self_mask: bool,
    sam2_mask_refine: bool,
) -> dict[str, Any]:
    """MonST3R full video-pose pipeline (replicates ``launch.py
    --mode=eval_pose`` for Table 4 rows).

    Verbatim port of ``dust3r/pose_eval.py:eval_pose_estimation_dist`` minus
    the file I/O. Differences from the MASt3R-shared base ``_run_mast3r``:

    1. **Scene graph**: ``swinstride-5-noncyclic`` (sliding-window pair set,
       ~5 N pairs instead of N(N-1)/2) — required for long sequences to
       fit in VRAM. Base ``_run_mast3r`` uses ``"complete"``.
    2. **Global aligner**: MonST3R's extended ``PointCloudOptimizer``
       accepts ``flow_loss_weight``, ``motion_mask_thre``,
       ``temporal_smoothing_weight``, ``sintel_ckpt``, ``use_self_mask``,
       ``sam2_mask_refine``, ``flow_loss_start_epoch``, ``flow_loss_thre``,
       ``translation_weight``, ``depth_regularize_weight``, ``pxl_thre`` —
       all the kwargs the paper's pipeline uses. Base PCO ignores them.

    Returns a dict shaped like ``_run_mast3r`` (``depth``, ``intrinsics``,
    ``extrinsics``, ``point_map``, optional ``confidence``) so the adapter
    body can splice it in interchangeably.
    """
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference

    from plumbline.conventions import (
        invert_pose,
        rebase_to_first_camera,
        world_from_camera_is_identity,
    )

    n = int(images.shape[0])
    dust3r_imgs = _images_to_dust3r_dicts(images, long_edge=long_edge)
    pairs = make_pairs(dust3r_imgs, scene_graph=scene_graph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=False,
            shared_focal=True,
            flow_loss_weight=flow_loss_weight,
            flow_loss_fn=flow_loss_fn,
            depth_regularize_weight=depth_regularize_weight,
            num_total_iter=ga_niter,
            temporal_smoothing_weight=temporal_smoothing_weight,
            motion_mask_thre=motion_mask_thre,
            flow_loss_start_epoch=flow_loss_start_epoch,
            flow_loss_thre=flow_loss_thre,
            translation_weight=translation_weight,
            sintel_ckpt=sintel_ckpt,
            use_self_mask=use_self_mask,
            sam2_mask_refine=sam2_mask_refine,
            # ``empty_cache`` mirrors MonST3R's heuristic: scenes with
            # ~80+ frames AND ~600+ pairs need explicit cache drops to
            # fit on 48 GB GPUs. Use the same gate.
            empty_cache=(n >= 80 and len(pairs) > 600),
            pxl_thre=pxl_thresh,
        )
        scene.compute_global_alignment(
            init=ga_init,
            niter=ga_niter,
            schedule=ga_schedule,
            lr=ga_lr,
        )

    depthmaps = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    depth = np.stack(depthmaps).astype(np.float32)

    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    pps = scene.get_principal_points().detach().cpu().numpy().reshape(n, 2)
    K = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        K[i] = np.array(
            [[focals[i], 0.0, pps[i, 0]], [0.0, focals[i], pps[i, 1]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    E_scene = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
    if not world_from_camera_is_identity(E_scene):
        E_rebased = rebase_to_first_camera(E_scene)
    else:
        E_rebased = E_scene

    pts3d = [p.detach().cpu().numpy().astype(np.float64) for p in scene.get_pts3d()]
    T_rebase = invert_pose(E_scene[0])
    pts3d_rebased = []
    for p in pts3d:
        p_h = np.concatenate([p, np.ones_like(p[..., :1])], axis=-1)
        p_new = p_h @ T_rebase.T
        pts3d_rebased.append(p_new[..., :3])
    point_map = np.stack(pts3d_rebased).astype(np.float32)

    return {
        "depth": depth,
        "intrinsics": K,
        "extrinsics": E_rebased.astype(np.float32),
        "point_map": point_map,
        "confidence": None,
    }


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
