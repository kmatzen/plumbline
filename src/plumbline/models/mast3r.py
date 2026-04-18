"""MASt3R adapter.

Upstream: https://github.com/naver/mast3r
Paper: "Grounding Image Matching in 3D with MASt3R" (Leroy et al. 2024).

MASt3R is a pair-based (2-view) model that predicts per-pixel 3D point maps
in a shared frame for a pair of images, plus a confidence + descriptor map.
For v0.1 we treat it strictly as a **2-view** primitive: min_views = max_views
= 2. Extension to N>2 via ``mast3r.cloud_opt.sparse_ga.sparse_global_alignment``
is a v0.2 concern (it's an iterative optimizer, orders of magnitude slower
than the feed-forward path).

Install
-------
Upstream is not pip-installable; it expects a recursive clone with the
``dust3r`` + ``croco`` submodules. Set ``$MAST3R_ROOT`` to that clone and
``$DUST3R_ROOT`` to the ``dust3r`` submodule path; the adapter adds both
to ``sys.path`` lazily on first use. Defaults: ``/workspace/deps/mast3r``
and ``/workspace/deps/mast3r/dust3r``. Install transitive deps into the
project venv::

    uv pip install roma scikit-learn trimesh

Inputs
------
- Two sRGB uint8 images of matching size.
- No intrinsics required — MASt3R + PairViewer estimate them.

Outputs (in canonical conventions)
----------------------------------
- ``depth``: ``(2, H, W)`` — camera-frame depth in MASt3R-native units
  (metric variant is scaled but has residual ambiguity; use
  ``alignment_hint="median"`` when comparing to GT).
- ``intrinsics``: ``(2, 3, 3)`` — from PairViewer's focal + principal-point
  estimates; intrinsics live in MASt3R's processed-image pixel space (the
  runner resizes depth to GT for metric computation).
- ``extrinsics``: ``(2, 4, 4)`` — ``world_from_camera``, **rebased** so the
  first camera is identity. PairViewer may choose either view as its scene
  origin based on pair confidence; we invert and propagate the rebase
  through the point map to match our convention.
- ``point_map``: ``(2, H, W, 3)`` — in the rebased world frame.

Implementation notes
--------------------
- We mirror dust3r's ``load_images`` (long-edge resize to 512, centre-crop
  to multiples of ``patch_size``) in-memory so plumbline's numpy inputs
  don't need to be written to disk.
- PairViewer's ``get_depthmaps`` returns camera-frame depth regardless of
  which view it picked as the scene origin — safe to use directly.
- The pairwise scene graph is ``[(v0, v1), (v1, v0)]``; ``inference`` is
  called with ``batch_size=1`` to keep memory bounded.
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
    invert_pose,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["MASt3RAdapter"]


@register_model("mast3r")
class MASt3RAdapter(Model):
    """Pair-based 3D + pose foundation model."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # recovers a scale up to residual ambiguity
        min_views=2,
        # v0.1 wires MASt3R via PairViewer (2-view only). N>2 requires
        # sparse_global_alignment, which is iterative and a v0.2 concern.
        max_views=2,
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        long_edge: int = 512,
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.long_edge = int(long_edge)
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_mast3r_on_path()
        try:
            from mast3r.model import AsymmetricMASt3R  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MASt3RAdapter needs the `mast3r` package. Clone "
                "https://github.com/naver/mast3r recursively and set "
                "$MAST3R_ROOT (default /workspace/deps/mast3r). The repo is "
                "not on PyPI."
            ) from exc
        model = AsymmetricMASt3R.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="mast3r/input")
        n = images.shape[0]
        if n < 2:
            raise ValueError(f"MASt3R requires at least 2 views; got {n}")
        if n != 2:
            raise ValueError(
                f"MASt3R v0.1 adapter supports exactly 2 views; got {n}. "
                "N>2 requires sparse_global_alignment (v0.2)."
            )
        self._load()

        out = _run_mast3r(
            self._model, images, device=self.device, long_edge=self.long_edge
        )

        point_map = out["point_map"]  # (N, H, W, 3), world frame (view 0)
        depth = out["depth"]  # (N, H, W), camera-frame depth
        K = out["intrinsics"]  # (N, 3, 3) in processed-image pixel space
        extrinsics = out["extrinsics"]  # (N, 4, 4) world_from_camera, E[0] = I
        confidence = out.get("confidence")  # (N, H, W) or None

        assert_valid_depth(depth, name="mast3r/output_depth")
        assert_valid_intrinsics(K, name="mast3r/output_K")
        assert_valid_extrinsics(extrinsics, name="mast3r/output_E")
        assert_valid_point_map(point_map, name="mast3r/output_pmap")

        return Prediction(
            depth=depth.astype(np.float32),
            intrinsics=K.astype(np.float32),
            extrinsics=extrinsics.astype(np.float32),
            point_map=point_map.astype(np.float32),
            confidence=(confidence.astype(np.float32) if confidence is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "native_space": "point_map",
                "alignment_hint": "median",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/{self.checkpoint}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Upstream wrapper
# ---------------------------------------------------------------------------


def _ensure_mast3r_on_path() -> None:
    """Add $MAST3R_ROOT and $DUST3R_ROOT to sys.path so imports resolve.

    Upstream ships as a recursive git clone (no PyPI). Callers are expected
    to set the env vars or accept the defaults below.
    """
    import os
    import sys

    mast3r_root = os.environ.get("MAST3R_ROOT", "/workspace/deps/mast3r")
    dust3r_root = os.environ.get("DUST3R_ROOT", os.path.join(mast3r_root, "dust3r"))
    for p in (mast3r_root, dust3r_root):
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)


def _images_to_dust3r_dicts(
    images: NDArray[np.uint8], *, long_edge: int, patch_size: int = 16
) -> list[dict[str, Any]]:
    """Replicate dust3r's ``load_images`` for in-memory uint8 arrays.

    Long-edge resize to ``long_edge`` (LANCZOS if downscaling, BICUBIC if up),
    centre-crop to ``patch_size`` multiples with the 3:4 ratio rule for square
    inputs. Yields the same dicts ``dust3r.inference.inference`` expects.
    """
    import torch
    import torchvision.transforms as tvf
    from PIL import Image as PImage

    norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dicts: list[dict[str, Any]] = []
    for idx in range(images.shape[0]):
        pil = PImage.fromarray(images[idx])
        W1, H1 = pil.size
        S = max(W1, H1)
        interp = PImage.Resampling.LANCZOS if S > long_edge else PImage.Resampling.BICUBIC
        new_size = (int(round(W1 * long_edge / S)), int(round(H1 * long_edge / S)))
        pil = pil.resize(new_size, interp)
        W, H = pil.size
        cx, cy = W // 2, H // 2
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if W == H:  # enforce 3:4 for square sources (dust3r default)
            halfh = 3 * halfw / 4
        pil = pil.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        dicts.append(
            {
                "img": norm(pil)[None],  # (1, 3, H, W)
                "true_shape": np.int32([pil.size[::-1]]),
                "idx": idx,
                "instance": str(idx),
            }
        )
    return dicts


def _run_mast3r(
    model: Any, images: NDArray[np.uint8], *, device: str, long_edge: int = 512
) -> dict[str, NDArray[Any]]:
    """Run MASt3R on an image pair; return plumbline-shaped arrays.

    Uses dust3r's ``inference`` + ``global_aligner(mode=PairViewer)`` to
    recover per-view depth, intrinsics, and world_from_camera poses from
    MASt3R's raw pair predictions. Rebases poses + point map so view 0
    is the canonical world frame.
    """
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.geometry import geotrf

    dust3r_imgs = _images_to_dust3r_dicts(images, long_edge=long_edge)
    pairs = make_pairs(dust3r_imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

    # Per-view camera-frame depth (already camera-frame regardless of which
    # view PairViewer chose as the scene origin).
    depthmaps = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    depth = np.stack(depthmaps).astype(np.float32)  # (2, H, W)

    # Intrinsics from PairViewer: per-view focal + principal point.
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    pps = scene.pp.detach().cpu().numpy()  # (2, 2)
    n = len(depthmaps)
    K = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        K[i] = np.array(
            [[focals[i], 0.0, pps[i, 0]], [0.0, focals[i], pps[i, 1]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    # World_from_camera poses. PairViewer may already have view 0 = identity
    # (higher conf path) or view 1 = identity. Always rebase so view 0 wins.
    E_scene = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
    if not world_from_camera_is_identity(E_scene):
        E_rebased = rebase_to_first_camera(E_scene)
    else:
        E_rebased = E_scene

    # Rebase the point map by the same transform:  X_new = inv(E_scene[0]) @ X_old.
    pts3d = [p.detach().cpu().numpy().astype(np.float64) for p in scene.get_pts3d()]
    T_rebase = invert_pose(E_scene[0])  # (4, 4); identity if no rebase happened
    # Apply via dust3r's helper which handles the (H, W, 3) shape fluently.
    pts3d_world = [geotrf(torch.from_numpy(T_rebase), torch.from_numpy(p)).numpy() for p in pts3d]
    point_map = np.stack(pts3d_world).astype(np.float32)  # (2, H, W, 3)

    # Per-view mean-confidence over its own-frame prediction edges.
    conf_per_view = _extract_pairviewer_confidence(scene)

    # Depth sanity: replace non-finite / non-positive with 0 (canonical invalid).
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)

    return {
        "depth": depth,
        "intrinsics": K,
        "extrinsics": E_rebased.astype(np.float32),
        "point_map": point_map,
        "confidence": conf_per_view,
    }


def _extract_pairviewer_confidence(scene: Any) -> NDArray[np.float32] | None:
    """Per-view confidence map from PairViewer's pair outputs.

    PairViewer stores per-edge confidence under ``conf_i[edge_str(i, j)]``;
    we take the self-view edges and stack them. Returns None if the attributes
    aren't present on a given upstream pin.
    """
    try:
        from dust3r.cloud_opt.commons import edge_str
    except ImportError:  # pragma: no cover
        return None
    try:
        confs: list[NDArray[np.float32]] = []
        for i in range(scene.n_imgs):
            key = edge_str(i, 1 - i)
            c = scene.conf_i[key].detach().cpu().numpy().astype(np.float32)
            confs.append(c)
        return np.stack(confs)
    except (AttributeError, KeyError):  # pragma: no cover
        return None
