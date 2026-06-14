"""VGGT adapter.

Upstream: https://github.com/facebookresearch/vggt
Paper: "VGGT: Visual Geometry Grounded Transformer" (Wang et al. 2025).

VGGT is a feed-forward multi-view transformer that predicts, from up to
~32 views in a single forward pass:

- Per-view depth maps (``depth``)
- Per-view world-space point maps (``point_map``)
- Cameras (intrinsics + extrinsics) via a compact pose encoding
- Dense confidence

Canonical conversion
--------------------
- VGGT's preprocessing (``load_and_preprocess_images``) resizes each input
  image so width=518 and height is the nearest multiple of 14 that preserves
  aspect ratio, optionally centre-cropping to 518 on the tall axis. Depth /
  point_map / intrinsics are returned at the *processed* resolution; the
  runner later resizes depth to GT for metric computation. Intrinsics live
  in that processed pixel space, which is the resolution at which the
  returned depth map makes sense.
- ``pose_encoding_to_extri_intri`` yields extrinsics as **camera_from_world**
  3x4 matrices (per ``vggt.utils.pose_enc``). We pad to 4x4 and invert to
  world_from_camera. VGGT's pose encoding biases view 0 toward identity,
  so our canonical "first camera is world" already holds within float noise;
  we rebase if it drifts.
- VGGT outputs depth in metric meters (trained with metric supervision);
  treat as metric and skip alignment by default. ``alignment_hint=none``.
- Negative or non-finite depth values are mapped to ``0`` (our canonical
  invalid marker) to keep ``assert_valid_depth`` happy.

Memory note
-----------
Per the paper, 32 views at 1024x1024 fits in 24GB on an A100/4090. At the
upstream default (518px width, preserved aspect), inference of 8 views on a
24GB card is comfortably under 10GB. The runner's OOM fallback catches the
failure and skips the sample when it doesn't fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    invert_pose,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction, config_digest
from plumbline.models.registry import register_model

__all__ = ["VGGTAdapter"]


@register_model("vggt")
class VGGTAdapter(Model):
    """Multi-view feed-forward 3D foundation model."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,  # paper trains with metric supervision
        min_views=2,
        # Probed 2026-04-26 on a 24 GB 3090: 49 views at the upstream
        # default 518-px width fits in ~19 GB peak. The previous 32-view
        # cap was a conservative leftover from before that probe; lifting
        # it lets DTU/ETH3D feed the full rig (49 / 38-76 views) into one
        # forward pass and removes a real source of the D3 / D4 paper gap
        # (we were missing 17 of 49 DTU rig views, ~30 % of the surface
        # arc, in every scan).
        max_views=49,
        requires_intrinsics=False,
        default_resolution=(1024, 1024),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str = "facebook/VGGT-1B",
        dtype: str = "bfloat16",
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.dtype = dtype
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('vggt')}") from exc
        model = VGGT.from_pretrained(self.checkpoint).to(self.device).eval()
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="vggt/input")
        if images.shape[0] < 2:
            raise ValueError("VGGT requires at least 2 views")
        if images.shape[0] > self.capabilities.max_views:
            raise ValueError(f"VGGT max_views={self.capabilities.max_views}; got {images.shape[0]}")
        self._load()

        out = _run_vggt(self._model, images, device=self.device, dtype=self.dtype)

        depth = out["depth"].astype(np.float32)
        K = out["intrinsics"].astype(np.float32)
        E = out["extrinsics"].astype(np.float32)
        point_map = out.get("point_map")
        confidence = out.get("confidence")

        # Guard the convention. Accept a small epsilon; otherwise rebase.
        if not world_from_camera_is_identity(E):
            E = rebase_to_first_camera(E.astype(np.float64)).astype(np.float32)

        assert_valid_depth(depth, name="vggt/output_depth")
        assert_valid_intrinsics(K, name="vggt/output_K")
        assert_valid_extrinsics(E, name="vggt/output_E")

        return Prediction(
            depth=depth,
            intrinsics=K,
            extrinsics=E,
            point_map=(point_map.astype(np.float32) if point_map is not None else None),
            confidence=(confidence.astype(np.float32) if confidence is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "native_space": "depth",
                "alignment_hint": "none",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/ckpt={self.checkpoint}/dtype={self.dtype}"
        return config_digest(s)


def _run_vggt(
    model: Any, images: NDArray[np.uint8], *, device: str, dtype: str
) -> dict[str, NDArray[Any]]:
    """Run VGGT end-to-end on a batch of sRGB uint8 images.

    Returns arrays at VGGT's processed resolution (width=518, height is the
    nearest multiple of 14 to width*orig_h/orig_w, capped at 518):

      - depth:      (N, H_p, W_p), float32, meters
      - intrinsics: (N, 3, 3), float32, in processed pixel space
      - extrinsics: (N, 4, 4), float32, world_from_camera, first view = identity
      - point_map:  (N, H_p, W_p, 3), float32, world frame
      - confidence: (N, H_p, W_p), float32 in [0, 1]  (world_points_conf)
    """
    import torch
    from PIL import Image as PImage
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    target_width = 518

    # Mirror vggt.utils.load_fn.load_and_preprocess_images(mode="crop") in-memory.
    tensors: list[torch.Tensor] = []
    for i in range(images.shape[0]):
        img = images[i]  # (H, W, 3) uint8
        h, w = img.shape[:2]
        new_w = target_width
        # Height: nearest-divisible-by-14 multiple that preserves aspect ratio.
        new_h = max(14, round(h * (new_w / w) / 14) * 14)
        pil = PImage.fromarray(img).resize((new_w, new_h), PImage.Resampling.BICUBIC)
        arr = np.array(pil, dtype=np.uint8, copy=True)  # writable copy → no torch warning
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if new_h > target_width:  # centre-crop on the tall axis
            start = (new_h - target_width) // 2
            t = t[:, start : start + target_width, :]
        tensors.append(t)

    # Pad to a common (H, W) with white (1.0), matching upstream's behaviour
    # when batch members have different aspect ratios.
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) > 1:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        padded: list[torch.Tensor] = []
        for t in tensors:
            hp = max_h - t.shape[1]
            wp = max_w - t.shape[2]
            if hp or wp:
                t = torch.nn.functional.pad(
                    t,
                    (wp // 2, wp - wp // 2, hp // 2, hp - hp // 2),
                    mode="constant",
                    value=1.0,
                )
            padded.append(t)
        tensors = padded

    batched = torch.stack(tensors).to(device)  # (N, 3, H, W)

    # Autocast dtype per ``dtype`` arg. ``float32`` runs unmoved; bf16/fp16
    # match the VGGT demo's recommended path on sm >= 8.
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"VGGTAdapter: unsupported dtype {dtype!r}")

    with torch.no_grad():
        if torch_dtype is torch.float32 or device.startswith("cpu"):
            preds = model(batched)
        else:
            with torch.amp.autocast("cuda", dtype=torch_dtype):  # type: ignore[attr-defined]
                preds = model(batched)

    # pose_enc: (1, N, 9) → extri (1, N, 3, 4), intri (1, N, 3, 3)
    extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], batched.shape[-2:])
    extri_np = extri[0].float().cpu().numpy()  # (N, 3, 4), camera_from_world
    intri_np = intri[0].float().cpu().numpy()  # (N, 3, 3)
    depth_np = preds["depth"][0, ..., 0].float().cpu().numpy()  # (N, H, W)
    world_points_np = preds["world_points"][0].float().cpu().numpy()  # (N, H, W, 3)
    conf_np = preds["world_points_conf"][0].float().cpu().numpy()  # (N, H, W)

    # camera_from_world 3x4 → world_from_camera 4x4.
    n = extri_np.shape[0]
    cam_from_world = np.zeros((n, 4, 4), dtype=np.float64)
    cam_from_world[:, :3, :] = extri_np.astype(np.float64)
    cam_from_world[:, 3, 3] = 1.0
    world_from_cam = invert_pose(cam_from_world)

    # Mark any non-finite / negative depth as invalid (0) so assert_valid_depth
    # passes and downstream valid-mask computation is correct.
    depth_np = np.where(np.isfinite(depth_np) & (depth_np > 0), depth_np, 0.0)

    return {
        "depth": depth_np.astype(np.float32),
        "intrinsics": intri_np.astype(np.float32),
        "extrinsics": world_from_cam.astype(np.float32),
        "point_map": world_points_np.astype(np.float32),
        "confidence": conf_np.astype(np.float32),
    }
