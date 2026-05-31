"""Evaluation masks that refine a dataset's valid-pixel mask.

Depth benchmarks ship a validity mask per sample, but modern mono-depth
papers add per-eval refinements on top: boundary masking (exclude pixels
near depth discontinuities where sensor GT is unreliable), segmentation
masking (exclude sky / dynamic objects), etc.

These helpers build refinements on top of an existing boolean mask; the
runner ANDs them in when the corresponding YAML flag is set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS

__all__ = ["boundary_edge_mask"]


def boundary_edge_mask(
    depth: NDArray[Any],
    valid: NDArray[Any],
    *,
    thickness: int = 1,
    tol: float = 0.1,
) -> NDArray[np.bool_]:
    """Port of MoGe's ``depth_occlusion_edge_numpy`` — the boundary mask
    that NYU / DIODE / KITTI mono-depth evaluations use to exclude pixels
    near GT depth discontinuities (sensor noise at edges).

    Algorithm (matching MoGe's reference implementation):
      1. Convert GT depth to disparity (1 / depth) on valid pixels.
      2. For each pixel, compute the MASK-WEIGHTED mean disparity in a
         ``(2*thickness+1) × (2*thickness+1)`` window.
      3. Foreground edge: pixel-disparity > ``(1 + tol)`` × window-mean
         (the pixel is much closer than its neighbours).
      4. Background edge: window-mean > ``(1 + tol)`` × pixel-disparity
         (the pixel is much farther than its neighbours).
      5. Dilate each edge mask with a 3×3 structuring element,
         ``thickness`` iterations.
      6. Edge = fg-dilated ∩ bg-dilated. (Both kinds of edges must
         co-occur — a single "cliff" has a fg side + bg side nearby.)

    Returns a boolean array of the same ``(H, W)`` shape as ``depth``,
    ``True`` where the pixel is classified as a boundary and should be
    EXCLUDED from evaluation. Caller ANDs ``~edge`` into the existing
    ``valid`` mask before computing metrics.

    Parameters
    ----------
    depth, valid
        ``(H, W)`` float / bool arrays. Invalid pixels' depths are
        ignored (disparity is zeroed out there).
    thickness
        Window radius. ``1`` = 3×3 (MoGe default).
    tol
        Relative-disparity threshold for declaring an edge. ``0.1`` =
        10% (MoGe default).

    References
    ----------
    MoGe upstream: ``moge/utils/geometry_numpy.py::depth_occlusion_edge_numpy``.
    """
    try:
        from scipy.ndimage import binary_dilation, uniform_filter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "boundary_edge_mask needs scipy. Install plumbline with the 'scipy' "
            "extra: `uv sync --extra scipy`."
        ) from exc

    if depth.shape != valid.shape:
        raise ValueError(f"depth and valid must share shape; got {depth.shape} vs {valid.shape}")
    if depth.ndim != 2:
        raise ValueError(f"boundary_edge_mask operates on (H, W); got ndim={depth.ndim}")
    kernel_size = 2 * thickness + 1

    valid_f = valid.astype(np.float32)
    disp = np.where(valid, 1.0 / np.maximum(depth.astype(np.float64), EPS), 0.0)
    disp_masked = np.where(valid, disp, 0.0)

    # Weighted mean of disparity over the window:
    #   sum(disp * valid) / sum(valid)
    # uniform_filter returns the *mean* over a window; multiplying by
    # window-area gives the sum.
    area = kernel_size * kernel_size
    disp_sum = uniform_filter(disp_masked, size=kernel_size, mode="constant", cval=0.0) * area
    mask_sum = uniform_filter(valid_f, size=kernel_size, mode="constant", cval=0.0) * area
    with np.errstate(invalid="ignore", divide="ignore"):
        disp_mean = np.where(mask_sum > 0, disp_sum / np.maximum(mask_sum, EPS), 0.0)

    fg_edge = valid & (disp > (1.0 + tol) * disp_mean)
    bg_edge = valid & (disp_mean > (1.0 + tol) * disp)

    kernel_3x3 = np.ones((3, 3), dtype=bool)
    fg_dilated = binary_dilation(fg_edge, structure=kernel_3x3, iterations=thickness)
    bg_dilated = binary_dilation(bg_edge, structure=kernel_3x3, iterations=thickness)

    edge = fg_dilated & bg_dilated
    return np.asarray(edge, dtype=bool)
