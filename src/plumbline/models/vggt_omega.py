"""VGGT-Omega (VGGT-Ω) adapter.

Upstream: https://github.com/facebookresearch/vggt-omega (Meta FAIR, CVPR 2026
Oral, arXiv:2605.15195). The official successor to VGGT — a scaled-up, single-
dense-head redesign (~30% of VGGT's GPU memory) predicting depth + camera pose
+ intrinsics + point maps from a multi-view set.

NOT vendored: VGGT-Ω ships under the **FAIR Noncommercial Research License v1**
(custom, non-redistributable) and its weights are **gated** on HuggingFace
(``facebook/VGGT-Omega``, access-request required). So it stays a git-install,
like VGGT — ``plumbline install vggt-omega`` (``pip install
git+https://github.com/facebookresearch/vggt-omega``), and the operator must
have accepted the gated-weights terms for the checkpoint to download.

Depth scale
-----------
Like VGGT, depth is camera-frame / scale-normalised, so ``is_metric=False`` and
``alignment_hint="scale_shift"``.

Compute
-------
Built on the VGGT alternating-attention backbone → **bf16 on Ampere+ (A100/H100
class)**. ``dtype`` selects the autocast precision; the default bf16 will not run
on pre-Ampere GPUs.

NOTE: gated weights + bf16/Ampere — this adapter has NOT been GPU-validated
(neither import nor forward pass) at authoring; the package isn't installed in
the dev env and no Ampere/H100 was available. Treat the inference wiring
(preprocessing, ``encoding_to_camera`` pose decode) as provisional until a run
confirms it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    rebase_to_first_camera,
    world_from_camera_is_identity,
)
from plumbline.models._torch_utils import ensure_torch
from plumbline.models.base import Model, ModelCapabilities, Prediction, config_digest
from plumbline.models.registry import register_model

__all__ = ["VGGTOmegaAdapter"]

_HF_REPO = "facebook/VGGT-Omega"
_CHECKPOINTS = {
    "1b-512": "vggt_omega_1b_512.pt",
    "1b-256-text": "vggt_omega_1b_256_text.pt",
}


@register_model("vggt-omega")
class VGGTOmegaAdapter(Model):
    """VGGT-Ω — multi-view depth + pose + intrinsics (Meta FAIR)."""

    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=False,  # camera-frame / scale-normalised, like VGGT
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(512, 512),
    )

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        variant: str = "1b-512",
        dtype: str = "bfloat16",
    ) -> None:
        if variant not in _CHECKPOINTS:
            raise ValueError(f"variant must be one of {list(_CHECKPOINTS)}; got {variant!r}.")
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16|float16|float32; got {dtype!r}")
        self.device = device
        self.variant = variant
        self.dtype = dtype
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_torch()
        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:  # pragma: no cover
            from plumbline.install import install_hint

            raise ImportError(f"{type(self).__name__} {install_hint('vggt-omega')}") from exc
        import torch
        from huggingface_hub import hf_hub_download

        model = VGGTOmega().to(self.device).eval()
        ckpt = hf_hub_download(repo_id=_HF_REPO, filename=_CHECKPOINTS[self.variant])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._model = model

    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        assert_valid_image(images, name="vggt-omega/input")
        self._load()

        out = _run_vggt_omega(self._model, images, device=self.device, dtype=self.dtype)
        depth = out["depth"].astype(np.float32)
        E = out.get("extrinsics")
        K = out.get("intrinsics")
        conf = out.get("confidence")

        assert_valid_depth(depth, name="vggt-omega/output_depth")
        if E is not None:
            if not world_from_camera_is_identity(E):
                E = rebase_to_first_camera(E.astype(np.float64)).astype(np.float32)
            assert_valid_extrinsics(E, name="vggt-omega/output_E")
        if K is not None:
            assert_valid_intrinsics(K, name="vggt-omega/output_K")

        return Prediction(
            depth=depth,
            extrinsics=(E.astype(np.float32) if E is not None else None),
            intrinsics=(K.astype(np.float32) if K is not None else None),
            confidence=(conf.astype(np.float32) if conf is not None else None),
            metadata={
                "variant": self.variant,
                "native_space": "depth_affine_invariant",
                "alignment_hint": "scale_shift",
            },
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}/dtype={self.dtype}"
        return config_digest(s)


def _run_vggt_omega(
    model: Any, images: NDArray[np.uint8], *, device: str, dtype: str
) -> dict[str, NDArray[Any]]:
    """Run VGGT-Ω on a multi-view image set; return depth + decoded pose.

    - depth:      (N, H_p, W_p), float32, scale-normalised
    - extrinsics: (N, 4, 4), float32, world_from_camera, view 0 = identity
    - intrinsics: (N, 3, 3), float32, processed pixel space
    - confidence: (N, H_p, W_p), float32  (depth_conf)
    """
    import torch
    from PIL import Image as PImage
    from vggt_omega.utils.pose_enc import encoding_to_camera

    # VGGT-Ω preprocesses at resolution 512 (multiple of 14). Mirror
    # load_and_preprocess_images(image_resolution=512) in-memory.
    target = 512
    tensors: list[torch.Tensor] = []
    for i in range(images.shape[0]):
        img = images[i]
        h, w = img.shape[:2]
        new_h = max(14, round(h * (target / w) / 14) * 14)
        pil = PImage.fromarray(img).resize((target, new_h), PImage.Resampling.BICUBIC)
        arr = np.array(pil, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if new_h > target:
            start = (new_h - target) // 2
            t = t[:, start : start + target, :]
        tensors.append(t)
    batched = torch.stack(tensors).to(device)  # (N, 3, H, W)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.split(":")[0], dtype=torch_dtype, enabled=(dtype != "float32")
        ),
    ):
        predictions = model(batched)

    depth = np.asarray(predictions["depth"].squeeze().float().cpu())  # (N, H, W)
    out: dict[str, NDArray[Any]] = {"depth": depth.astype(np.float32)}
    if "depth_conf" in predictions:
        out["confidence"] = np.asarray(predictions["depth_conf"].squeeze().float().cpu()).astype(
            np.float32
        )

    if "pose_enc" in predictions:
        hw = batched.shape[-2:]
        extri, intri = encoding_to_camera(predictions["pose_enc"], hw)
        extri = np.asarray(extri.squeeze(0).float().cpu())  # (N, 3, 4) camera_from_world
        homog = np.tile(np.eye(4, dtype=np.float32), (extri.shape[0], 1, 1))
        homog[:, :3, :4] = extri
        out["extrinsics"] = np.linalg.inv(homog).astype(np.float32)
        out["intrinsics"] = np.asarray(intri.squeeze(0).float().cpu()).astype(np.float32)

    return out
