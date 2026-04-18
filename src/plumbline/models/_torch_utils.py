"""Shared torch helpers for adapters. Imported lazily.

These helpers never import torch at module-scope. All public functions take
numpy in and return numpy out, so the adapter surface stays torch-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ensure_torch",
    "imagenet_normalize",
    "numpy_to_torch_images",
    "torch_to_numpy",
]


def ensure_torch() -> Any:
    """Import torch or raise a helpful error pointing at the install extra."""
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError as exc:  # pragma: no cover - requires absence of torch
        raise ImportError(
            "This adapter requires torch. Install plumbline with the 'models' "
            "extra:\n    uv pip install -e '.[models]'\n"
            "or install torch directly: uv pip install torch torchvision"
        ) from exc


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(
    x: Any,
    *,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
) -> Any:
    """Normalize a ``(N, 3, H, W)`` torch tensor in ``[0, 1]`` by ImageNet stats."""
    torch = ensure_torch()
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def numpy_to_torch_images(
    images: NDArray[np.uint8], device: str, dtype_str: str = "float32"
) -> Any:
    """Convert ``(N, H, W, 3)`` uint8 sRGB numpy to ``(N, 3, H, W)`` torch in ``[0, 1]``."""
    torch = ensure_torch()
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {images.dtype}")
    dtype = getattr(torch, dtype_str)
    tensor = torch.from_numpy(images).to(device=device, dtype=dtype)
    # HWC -> CHW and scale to [0, 1].
    tensor = tensor.permute(0, 3, 1, 2).contiguous() / 255.0
    return tensor


def torch_to_numpy(x: Any) -> NDArray:
    """Bring a torch tensor back to numpy, off GPU, detached."""
    return x.detach().cpu().numpy()
