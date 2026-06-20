"""Depth metrics: ``AbsRel``, ``SqRel``, ``RMSE``, ``RMSE-log``, ``δ₁/₂/₃``, ``SILog``, ``log10``.

All functions are pure numpy and side-effect free. Inputs are float arrays in
canonical conventions; ``valid`` is a boolean mask of the same shape as
``pred``/``gt`` indicating pixels to include. Invalid pixels are ignored.

Conventions
-----------
- ``pred`` and ``gt`` must have identical shapes. If the prediction is at a
  different resolution than ground truth, resize **the prediction to GT**
  before calling these functions; never the other way around.
- ``valid`` may be ``None`` or an all-zero mask, in which case the metric
  returns ``NaN`` to signal "no data". Callers decide whether to drop NaNs or
  propagate them.
- Metrics reduce over all valid pixels with uniform weight (no per-image
  averaging). Per-image averaging is a policy decision and lives in the
  runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS

__all__ = [
    "abs_rel",
    "delta_threshold",
    "log10_error",
    "rmse",
    "rmse_log",
    "silog",
    "sq_rel",
]


def _flat_valid(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return 1D arrays of prediction/GT values on valid pixels only.

    Rejects non-finite values and non-positive GT (depth must be > 0).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if valid is None:
        valid = np.ones(pred.shape, dtype=bool)
    elif valid.shape != pred.shape:
        raise ValueError(f"valid/pred shape mismatch: {valid.shape} vs {pred.shape}")
    mask = valid & np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)
    return pred[mask].astype(np.float64), gt[mask].astype(np.float64)


def abs_rel(pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None) -> float:
    """Absolute relative error: ``mean(|pred - gt| / gt)``.

    Eigen et al., "Depth Map Prediction from a Single Image using a
    Multi-Scale Deep Network" (2014). Standard depth-estimation metric.
    """
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - g) / np.maximum(g, EPS)))


def sq_rel(pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None) -> float:
    """Squared relative error: ``mean((pred - gt)**2 / gt)``.

    Eigen et al. (2014); a standard column in every KITTI Eigen-split depth
    table alongside ``AbsRel``. Penalises large errors more than ``abs_rel``.
    """
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - g) ** 2 / np.maximum(g, EPS)))


def rmse(pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None) -> float:
    """Root mean squared error, in the same units as the input."""
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - g) ** 2)))


def rmse_log(pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None) -> float:
    """RMSE in log space: ``sqrt(mean((log(pred) - log(gt))**2))``.

    Eigen et al. (2014), natural log (matching :func:`silog`). The fourth
    classic KITTI Eigen-split column (``AbsRel, SqRel, RMSE, RMSE-log``).
    Unlike :func:`silog` it is *not* scale-invariant.
    """
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    d = np.log(p) - np.log(g)
    return float(np.sqrt(np.mean(d**2)))


def delta_threshold(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None = None,
    *,
    threshold: float = 1.25,
) -> float:
    """Fraction of pixels where ``max(pred/gt, gt/pred) < threshold``.

    With ``threshold = 1.25**i`` for ``i = 1, 2, 3``, this is the classic
    δ₁, δ₂, δ₃ metric from Eigen et al.
    """
    if threshold <= 1:
        raise ValueError(f"threshold must be > 1; got {threshold}")
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < threshold))


def silog(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None = None,
    *,
    lambda_: float = 1.0,
) -> float:
    """Scale-invariant logarithmic error (SILog).

    ``sqrt(mean(d^2) - lambda * mean(d)^2) * 100``, where ``d = log(pred) - log(gt)``.
    With ``lambda_ = 1.0`` this is scale-invariant. Used by the KITTI
    benchmark with ``lambda_ = 0.85`` and reported as a percentage.
    """
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    d = np.log(p) - np.log(g)
    val = np.mean(d**2) - lambda_ * (np.mean(d) ** 2)
    # Floating-point can push tiny variances slightly negative when lambda_=1
    # and pred exactly equals gt; clamp to 0 before sqrt.
    return float(np.sqrt(max(val, 0.0)) * 100.0)


def log10_error(pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None) -> float:
    """Mean |log10(pred) - log10(gt)|. Common on NYUv2 depth tables."""
    p, g = _flat_valid(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.log10(p) - np.log10(g))))
