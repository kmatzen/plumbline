"""Scale alignment modes for relative-depth predictions.

Some models predict depth up to an unknown scale (Depth Anything V2) or up to
an unknown affine transform (MiDaS, many transformer depth models). Compute
the aligning transform on ground-truth-valid pixels, then apply it to the
whole prediction before metric computation.

The runner logs which mode was used; reports display it; cached predictions
store raw (unaligned) values so the alignment can be changed without
re-running inference.

Modes
-----
- ``"none"``      — identity. Use for metric models.
- ``"median"``    — scalar ``s`` minimizing ``|s*pred / gt|`` via median ratio.
                    Cheap, robust to outliers, standard for "up-to-scale" eval.
- ``"lstsq"``     — scalar ``s`` minimizing ``||s*pred - gt||_2``. Closed form.
- ``"scale_shift"`` — affine ``(s, b)`` minimizing ``||s*pred + b - gt||_2`` in
                    inverse-depth or log-depth space. Used by MiDaS-family
                    eval protocols. Operates on inverse depth by default
                    (Ranftl et al. 2020, "Towards Robust Monocular Depth
                    Estimation").
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS

__all__ = [
    "align_depth",
    "align_scale_and_shift",
    "align_scale_lstsq",
    "align_scale_median",
]


def align_scale_median(pred: NDArray, gt: NDArray, valid: NDArray | None = None) -> float:
    """Return the scalar ``s`` such that ``s * pred`` matches ``gt`` in median ratio.

    ``s = median(gt / pred)`` over valid pixels where both are positive.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.median(g / np.maximum(p, EPS)))


def align_scale_lstsq(pred: NDArray, gt: NDArray, valid: NDArray | None = None) -> float:
    """Return the scalar ``s`` minimizing ``||s*pred - gt||_2``.

    Closed form: ``s = (pred . gt) / (pred . pred)``.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0 or float(p @ p) < EPS:
        return float("nan")
    return float((p @ g) / (p @ p))


def align_scale_and_shift(
    pred: NDArray,
    gt: NDArray,
    valid: NDArray | None = None,
    *,
    space: str = "inv_depth",
) -> tuple[float, float]:
    """Return ``(s, b)`` minimizing ``||s*pred + b - gt||_2``.

    Parameters
    ----------
    space
        ``"inv_depth"`` (default) fits in inverse-depth space, then the caller
        applies the transform to inverse predictions. ``"depth"`` fits directly
        in depth space. ``"log"`` fits in log-depth.

    Notes
    -----
    The MiDaS evaluation protocol (Ranftl et al. 2020) uses inverse-depth space
    because MiDaS predicts disparity-like quantities; matches their reported
    numbers on NYU/KITTI/Sintel.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size < 2:
        return float("nan"), float("nan")
    if space == "inv_depth":
        p = 1.0 / np.maximum(p, EPS)
        g = 1.0 / np.maximum(g, EPS)
    elif space == "log":
        p = np.log(p)
        g = np.log(g)
    elif space != "depth":
        raise ValueError(f"unknown space '{space}'; use 'depth', 'inv_depth', or 'log'")
    A = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(coef[0]), float(coef[1])


def align_depth(
    pred: NDArray,
    gt: NDArray,
    valid: NDArray | None = None,
    *,
    mode: str = "median",
) -> NDArray:
    """Apply the named alignment and return the aligned prediction.

    Parameters
    ----------
    pred, gt
        Same shape; any ndim. Alignment is computed on valid pixels.
    valid
        Boolean mask of pixels to use for fitting. ``None`` = use all pixels
        where both pred and gt are positive and finite.
    mode
        One of ``"none"``, ``"median"``, ``"lstsq"``, ``"scale_shift"``.
    """
    if mode == "none":
        return pred
    out = pred.astype(np.float64, copy=True)
    if mode == "median":
        s = align_scale_median(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "lstsq":
        s = align_scale_lstsq(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "scale_shift":
        s, b = align_scale_and_shift(pred, gt, valid, space="inv_depth")
        if np.isfinite(s) and np.isfinite(b):
            inv = 1.0 / np.maximum(out, EPS)
            inv = s * inv + b
            out = 1.0 / np.maximum(inv, EPS)
        return out
    raise ValueError(f"unknown alignment mode '{mode}'")


def _valid_pairs(
    pred: NDArray, gt: NDArray, valid: NDArray | None
) -> tuple[NDArray, NDArray, NDArray]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    mask = (
        (np.ones(pred.shape, dtype=bool) if valid is None else valid)
        & np.isfinite(pred)
        & np.isfinite(gt)
        & (pred > 0)
        & (gt > 0)
    )
    return pred[mask].astype(np.float64), gt[mask].astype(np.float64), mask
