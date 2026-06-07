"""StreamVGGT adapter.

Upstream: https://github.com/wzzheng/StreamVGGT (Zheng et al., ICLR 2026,
arXiv:2507.11539). VENDORED (self-contained model subset) under
``plumbline/_vendor/streamvggt`` — CC-BY-NC(-SA)-4.0; see THIRD_PARTY_NOTICES.md.

StreamVGGT is a *causal/streaming* variant of VGGT: it processes a frame
sequence incrementally with a temporal KV-cache, emitting per-frame depth,
camera pose, and point maps for low-latency online 4D reconstruction. Distilled
from ``facebook/VGGT-1B``; the prediction surface mirrors VGGT (camera-frame,
scale-normalised — *not* metric).

Depth scale
-----------
Like VGGT, depth is scale-normalised / affine-invariant rather than metric, so
``is_metric=False`` and ``alignment_hint="scale_shift"``.

Compute
-------
Built on the VGGT alternating-attention backbone — runs bf16 on Ampere+, fp16
on older GPUs (the upstream demo picks bf16 iff ``device_capability[0] >= 8``,
else fp16). FlashAttention is the fast path; on Pascal it falls back to eager
attention. ``dtype`` selects the autocast precision.

Install
-------
Vendored — no install step. Inference deps (torch/einops/transformers/
huggingface-hub) are all base. ``$STREAMVGGT_ROOT`` overrides the vendored path.

Canonical conversion
--------------------
- Input frames are preprocessed VGGT-style (width=518, height a multiple of 14),
  wrapped as per-frame ``{"img": tensor}`` dicts, and fed to ``model.inference``
  which returns a per-frame result list (``output.ress``).
- Per-frame ``depth`` is stacked to ``(N, H, W)``; ``camera_pose`` encodings are
  decoded to extrinsics via ``pose_encoding_to_extri_intri`` (camera_from_world
  → world_from_camera), rebased so view 0 is identity.

NOTE: StreamVGGT is bf16/Ampere-class; this adapter's inference path has not yet
been GPU-validated (no Ampere/H100 available at authoring). Treat depth/pose
wiring as provisional until a forward-pass run confirms it.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = ["StreamVGGTAdapter"]


def _ensure_streamvggt_on_path() -> None:
    """Put the vendored ``streamvggt`` package on ``sys.path``.

    Vendored under ``plumbline/_vendor/streamvggt`` (self-contained model subset
    of the CC-BY-NC release; see THIRD_PARTY_NOTICES.md). Internal imports are
    absolute (``from streamvggt.models... import``), so the vendor root must be
    importable. ``$STREAMVGGT_ROOT`` overrides for a dev checkout.
    """
    root = os.environ.get("STREAMVGGT_ROOT")
    if not root:
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_vendor", "streamvggt")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


_HF_CHECKPOINT = "lch01/StreamVGGT"


@register_model("streamvggt")
class StreamVGGTAdapter(Model):
    """Causal/streaming VGGT — per-frame depth + pose over a sequence."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # scale-normalised like VGGT
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(518, 518),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        checkpoint: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16|float16|float32; got {dtype!r}")
        self.device = device
        self.checkpoint = checkpoint or _HF_CHECKPOINT
        self.dtype = dtype
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        _ensure_streamvggt_on_path()
        try:
            from streamvggt.models.streamvggt import StreamVGGT
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('streamvggt')}") from exc
        self._model = StreamVGGT.from_pretrained(self.checkpoint).to(self.device).eval()

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="streamvggt/input")
        self._load()

        out = _run_streamvggt(self._model, images, device=self.device, dtype=self.dtype)
        depth = out["depth"].astype(np.float32)
        E = out.get("extrinsics")
        conf = out.get("confidence")

        assert_valid_depth(depth, name="streamvggt/output_depth")
        if E is not None:
            if not world_from_camera_is_identity(E):
                E = rebase_to_first_camera(E.astype(np.float64)).astype(np.float32)
            assert_valid_extrinsics(E, name="streamvggt/output_E")

        return Prediction(
            depth=depth,
            extrinsics=(E.astype(np.float32) if E is not None else None),
            confidence=(conf.astype(np.float32) if conf is not None else None),
            metadata={
                "checkpoint": self.checkpoint,
                "native_space": "depth_affine_invariant",
                "alignment_hint": "scale_shift",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/ckpt={self.checkpoint}/dtype={self.dtype}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _run_streamvggt(
    model: Any, images: NDArray[np.uint8], *, device: str, dtype: str
) -> dict[str, NDArray[Any]]:
    """Run StreamVGGT over a frame sequence; return stacked per-frame depth + pose.

    - depth:      (N, H_p, W_p), float32, scale-normalised
    - extrinsics: (N, 4, 4), float32, world_from_camera, view 0 = identity
    - confidence: (N, H_p, W_p), float32  (depth_conf)
    """
    import torch
    from PIL import Image as PImage
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    # VGGT-style preprocessing (StreamVGGT reuses VGGT's load_fn): width=518,
    # height the nearest multiple of 14, centre-crop tall frames.
    target_width = 518
    tensors: list[torch.Tensor] = []
    for i in range(images.shape[0]):
        img = images[i]
        h, w = img.shape[:2]
        new_h = max(14, round(h * (target_width / w) / 14) * 14)
        pil = PImage.fromarray(img).resize((target_width, new_h), PImage.Resampling.BICUBIC)
        arr = np.array(pil, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if new_h > target_width:
            start = (new_h - target_width) // 2
            t = t[:, start : start + target_width, :]
        tensors.append(t)
    # pad mixed aspect ratios to a common size (white), matching VGGT
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) > 1:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        tensors = [
            torch.nn.functional.pad(
                t,
                (
                    (max_w - t.shape[2]) // 2,
                    max_w - t.shape[2] - (max_w - t.shape[2]) // 2,
                    (max_h - t.shape[1]) // 2,
                    max_h - t.shape[1] - (max_h - t.shape[1]) // 2,
                ),
                value=1.0,
            )
            for t in tensors
        ]
    # Each frame's img needs a leading S=1 dim: StreamVGGT.inference() does one
    # more unsqueeze(0) (-> B), and the aggregator unpacks (B, S, C, H, W). So pass
    # (1, C, H, W) per frame, not (C, H, W).
    frames = [{"img": t.unsqueeze(0).to(device)} for t in tensors]

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.split(":")[0], dtype=torch_dtype, enabled=(dtype != "float32")
        ),
    ):
        output = model.inference(frames)

    ress = output.ress if hasattr(output, "ress") else output["ress"]
    depths = np.stack([np.asarray(r["depth"].squeeze().float().cpu()) for r in ress], axis=0)
    out: dict[str, NDArray[Any]] = {"depth": depths.astype(np.float32)}

    if all("depth_conf" in r for r in ress):
        out["confidence"] = np.stack(
            [np.asarray(r["depth_conf"].squeeze().float().cpu()) for r in ress], axis=0
        ).astype(np.float32)

    if all("camera_pose" in r and r["camera_pose"] is not None for r in ress):
        pose_enc = torch.stack([r["camera_pose"].reshape(-1) for r in ress], dim=0).unsqueeze(0)
        h_p, w_p = depths.shape[1], depths.shape[2]
        extri, _intri = pose_encoding_to_extri_intri(pose_enc.float(), (h_p, w_p))
        extri = extri[0].cpu().numpy()  # (N, 3, 4) camera_from_world
        homog = np.tile(np.eye(4, dtype=np.float32), (extri.shape[0], 1, 1))
        homog[:, :3, :4] = extri
        # camera_from_world -> world_from_camera
        out["extrinsics"] = np.linalg.inv(homog).astype(np.float32)

    return out
