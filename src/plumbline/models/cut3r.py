"""CUT3R (Continuous 3D Reconstruction) adapter.

Upstream: https://github.com/CUT3R/CUT3R
Paper: "Continuous 3D Perception Model with Persistent State"
(Wang et al., CVPR 2025, arXiv:2501.12387).

CUT3R is a recurrent DUSt3R-family model with a persistent internal state.
It ingests views one at a time (online), updating state per view, and emits
a per-view 3D reconstruction in a shared world frame. Because the state is
order-agnostic and the model is trained on both video and photo collections,
a single adapter covers two task families plumbline was otherwise missing:

- **video** — ordered frames streamed through the recurrent state
- **unordered image collections** — the same forward pass, any frame order

Per view the model produces (see upstream ``demo.py::prepare_output``):

- ``pts3d_in_self_view``  — point map in the view's own camera frame; the
  z-channel is metric depth.
- ``pts3d_in_other_view`` — point map in the shared world frame.
- ``camera_pose``         — an ``absT_quaR`` pose *encoding*; decode with
  ``pose_encoding_to_camera`` -> 4x4 **camera-to-world** (OpenCV), which is
  exactly plumbline's ``world_from_camera``.
- ``conf_self`` / ``conf`` — per-pixel confidence (self / other view).

Install
-------
No PyPI distribution; follow the MASt3R clone pattern:

    git clone https://github.com/CUT3R/CUT3R /workspace/deps/cut3r
    cd /workspace/deps/cut3r && pip install -r requirements.txt   # + submodules
    export CUT3R_ROOT=/workspace/deps/cut3r
    # download weights per the repo README, then point at the .pth:
    export CUT3R_CKPT=$CUT3R_ROOT/src/cut3r_512_dpt_4_64.pth

Alignment
---------
CUT3R outputs metric 3D. Most published video-depth / pose protocols still
realign per sequence (scale or scale+shift for depth, Sim(3) for trajectory);
apply that at the runner level via the reproduction's ``scale_alignment`` /
``pointcloud_alignment`` — not in the adapter.

Status: adapter + conversion unit tests landed; **GPU validation pending**
Queued in ``reproductions/gpu_queue.yaml``.
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
    assert_valid_point_map,
    rebase_to_first_camera,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["CUT3RAdapter"]

_DEFAULT_CKPT = "src/cut3r_512_dpt_4_64.pth"


@register_model("cut3r")
class CUT3RAdapter(Model):
    """Recurrent multi-view 3D foundation model (CUT3R).

    Parameters
    ----------
    device
        torch device string.
    checkpoint
        Path to the CUT3R ``.pth`` checkpoint. Defaults to ``$CUT3R_CKPT``,
        else ``$CUT3R_ROOT/src/cut3r_512_dpt_4_64.pth`` (the 512-DPT model
        the README recommends).
    size
        Input rescale target passed to upstream ``load_images``. ``512`` for
        the DPT checkpoint, ``224`` for the linear checkpoint. Only the 512
        (long-edge) preprocessing is replicated here; 224 raises.
    """

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,  # metric-trained; realign per-protocol at the runner
        min_views=1,
        # README: 4-64 views. Memory grows linearly (recurrent state); the
        # runner's OOM fallback catches anything larger.
        max_views=64,
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str | None = None,
        size: int = 512,
    ) -> None:
        if size not in (224, 512):
            raise ValueError(f"size must be 224 or 512; got {size}")
        if size == 224:
            # The 224 model uses short-edge resize + square crop, which we
            # don't replicate. Fail loudly rather than feed it 512-shaped input.
            raise NotImplementedError(
                "CUT3RAdapter currently replicates only the 512 long-edge "
                "preprocessing; use the 512-DPT checkpoint (size=512)."
            )
        self.device = device
        self.size = int(size)
        self._checkpoint = checkpoint
        self._model: Any = None

    @property
    def checkpoint(self) -> str:
        if self._checkpoint:
            return self._checkpoint
        env = os.environ.get("CUT3R_CKPT")
        if env:
            return env
        root = os.environ.get("CUT3R_ROOT", "/workspace/deps/cut3r")
        return os.path.join(root, _DEFAULT_CKPT)

    # -- lazy load -------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = ensure_torch()
        _ensure_cut3r_on_path()
        try:
            from src.dust3r.model import ARCroco3DStereo  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - needs the repo
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('cut3r')}") from exc
        # demo.py: ARCroco3DStereo.from_pretrained(args.model_path).to(device).eval()
        # `from_pretrained` accepts a local .pth checkpoint path here.
        #
        # CUT3R's from_pretrained calls torch.load internally with the default
        # weights_only. torch>=2.6 (our pinned base dep) defaults that to True,
        # which refuses this checkpoint because it embeds an omegaconf
        # ``DictConfig`` (UnpicklingError: Unsupported global
        # omegaconf.dictconfig.DictConfig). The checkpoint is operator-supplied
        # via $CUT3R_CKPT and therefore trusted, so force the legacy full
        # unpickle for the duration of the load (the clone's code, which we
        # don't control, never passes weights_only itself).
        orig_load = torch.load

        def _full_unpickle_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        torch.load = _full_unpickle_load
        try:
            model = ARCroco3DStereo.from_pretrained(self.checkpoint)
        finally:
            torch.load = orig_load
        self._model = model.to(self.device).eval()

    # -- predict ---------------------------------------------------------

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="cut3r/input")
        n = images.shape[0]
        if n < self.capabilities.min_views:
            raise ValueError(f"cut3r needs >= {self.capabilities.min_views} views; got {n}")
        if n > self.capabilities.max_views:
            raise ValueError(
                f"cut3r adapter capped at {self.capabilities.max_views} views; got {n}"
            )
        self._load()
        import torch

        views = _build_views(images, long_edge=self.size)

        _ensure_cut3r_on_path()
        from src.dust3r.inference import inference  # type: ignore[import-not-found]
        from src.dust3r.utils.camera import (  # type: ignore[import-not-found]
            pose_encoding_to_camera,
        )

        with torch.no_grad():
            outputs, _state = inference(views, self._model, self.device)

        preds = outputs["pred"]  # one dict per view
        self_pts = np.stack(
            [p["pts3d_in_self_view"][0].detach().float().cpu().numpy() for p in preds]
        )  # (N, H, W, 3), camera frame
        conf = np.stack(
            [p["conf_self"][0].detach().float().cpu().numpy() for p in preds]
        )  # (N, H, W)
        # Decode each view's pose encoding → (1,4,4) camera-to-world (OpenCV).
        c2w = np.stack(
            [
                pose_encoding_to_camera(p["camera_pose"].clone())[0].detach().float().cpu().numpy()
                for p in preds
            ]
        )  # (N, 4, 4)

        # Rebase so view 0 is the world origin (plumbline convention). CUT3R
        # anchors on view 0 already, but float noise can drift it.
        extrinsics = rebase_to_first_camera(c2w.astype(np.float64))  # (N,4,4), E[0]=I

        # Depth = z of the self-view (camera-frame) point map. Non-finite or
        # non-positive → invalid (0), our canonical invalid-depth marker.
        depth = self_pts[..., 2].astype(np.float32)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
        assert_valid_depth(depth, name="cut3r/depth")

        # World point map consistent with (rebased extrinsics, self-view depth):
        # P_world[i] = E[i] @ [x, y, z, 1]. This keeps depth, pose, and point
        # map mutually consistent (matters for both pose and chamfer tasks) and
        # equals demo.py's geotrf(c2w, self) up to the view-0 rebase.
        point_map = _transform_points(extrinsics, self_pts).astype(np.float32)

        extrinsics = extrinsics.astype(np.float32)
        assert_valid_extrinsics(extrinsics, name="cut3r/extrinsics")
        assert_valid_point_map(point_map, name="cut3r/point_map")

        return Prediction(
            depth=depth,
            extrinsics=extrinsics,
            point_map=point_map,
            confidence=conf.astype(np.float32),
            metadata={
                "checkpoint": os.path.basename(self.checkpoint),
                "size": self.size,
                "n_views": n,
                "native_space": "camera_local_xyz",
                "alignment_hint": "none",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/ckpt={os.path.basename(self.checkpoint)}/size={self.size}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cut3r_on_path() -> None:
    """Add ``$CUT3R_ROOT`` to sys.path so ``from src.dust3r... import`` resolves.

    Upstream imports are rooted at the repo (``src.dust3r.*``), mirroring how
    ``demo.py`` calls ``add_path_to_dust3r``.
    """
    root = os.environ.get("CUT3R_ROOT", "/workspace/deps/cut3r")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


def _transform_points(extrinsics: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply per-view SE(3) ``extrinsics[i]`` to ``pts[i]`` (camera→world).

    ``extrinsics``: (N,4,4) world_from_camera. ``pts``: (N,H,W,3). Returns
    (N,H,W,3) in the world frame.
    """
    n, h, w, _ = pts.shape
    out = np.empty((n, h, w, 3), dtype=np.float64)
    p = pts.astype(np.float64)
    for i in range(n):
        r = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        flat = p[i].reshape(-1, 3) @ r.T + t  # (H*W, 3)
        out[i] = flat.reshape(h, w, 3)
    return out


def _build_views(
    images: NDArray[np.uint8], *, long_edge: int, patch_size: int = 16
) -> list[dict[str, Any]]:
    """Build CUT3R view dicts from in-memory uint8 images.

    Replicates dust3r's ``load_images`` 512-branch preprocessing (long-edge
    resize, centre-crop to patch multiples, [-1,1] normalization) — identical
    to the MASt3R adapter's ``_images_to_dust3r_dicts`` — then wraps each view
    with the extra keys CUT3R's ``inference`` expects (``ray_map`` NaN,
    identity ``camera_pose``, ``img_mask``/``ray_mask``/``update``/``reset``),
    matching upstream ``demo.py::prepare_input`` (images-only branch).
    """
    import torch
    import torchvision.transforms as tvf
    from PIL import Image as PImage

    norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    views: list[dict[str, Any]] = []
    for idx in range(images.shape[0]):
        pil = PImage.fromarray(images[idx])
        w1, h1 = pil.size
        s = max(w1, h1)
        interp = PImage.Resampling.LANCZOS if s > long_edge else PImage.Resampling.BICUBIC
        pil = pil.resize((round(w1 * long_edge / s), round(h1 * long_edge / s)), interp)
        w, h = pil.size
        cx, cy = w // 2, h // 2
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if w == h:  # dust3r enforces 3:4 on square sources
            halfh = 3 * halfw / 4
        pil = pil.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        img = norm(pil)[None]  # (1, 3, H, W) in [-1, 1]
        views.append(
            {
                "img": img,
                "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan),
                "true_shape": torch.from_numpy(np.array([pil.size[::-1]], dtype=np.int32)),
                "idx": idx,
                "instance": str(idx),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views
