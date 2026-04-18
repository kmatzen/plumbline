"""Pose metrics: rotation error, translation error, AUC@angles.

Poses are canonical ``world_from_camera`` 4x4 matrices. All functions accept
batched or single poses. Errors are per-pose floats or arrays; aggregation
(mean / median / AUC) is a separate function so the runner can apply its own
policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS

__all__ = [
    "auc",
    "pose_auc",
    "rotation_error_degrees",
    "translation_cosine_error",
    "translation_error",
]


def rotation_error_degrees(R_pred: NDArray[Any], R_gt: NDArray[Any]) -> NDArray[Any]:
    """Geodesic rotation error in degrees.

    Accepts rotations as ``(3, 3)`` or ``(4, 4)`` (upper-left taken), or batches
    with matching leading dims. Returns an array of the same leading shape (or
    a scalar if both inputs are single).
    """
    Rp = _extract_rot(R_pred)
    Rg = _extract_rot(R_gt)
    if Rp.shape != Rg.shape:
        raise ValueError(f"rotation shape mismatch: {Rp.shape} vs {Rg.shape}")
    # Trace of R_pred @ R_gt.T gives 1 + 2*cos(theta).
    rel = Rp @ np.swapaxes(Rg, -1, -2)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def translation_error(t_pred: NDArray[Any], t_gt: NDArray[Any]) -> NDArray[Any]:
    """Euclidean translation error, same units as input (meters when metric).

    Accepts ``(3,)``, ``(N, 3)``, ``(4, 4)``, or ``(N, 4, 4)`` (last: takes the
    translation column).
    """
    tp = _extract_trans(t_pred)
    tg = _extract_trans(t_gt)
    if tp.shape != tg.shape:
        raise ValueError(f"translation shape mismatch: {tp.shape} vs {tg.shape}")
    return np.linalg.norm(tp - tg, axis=-1)


def translation_cosine_error(t_pred: NDArray[Any], t_gt: NDArray[Any]) -> NDArray[Any]:
    """Angular error between translation directions, in degrees.

    Used for up-to-scale monocular pose evaluation, where the magnitude of ``t``
    is arbitrary but its direction is meaningful.
    """
    tp = _extract_trans(t_pred)
    tg = _extract_trans(t_gt)
    if tp.shape != tg.shape:
        raise ValueError(f"translation shape mismatch: {tp.shape} vs {tg.shape}")
    np_ = tp / np.maximum(np.linalg.norm(tp, axis=-1, keepdims=True), EPS)
    ng = tg / np.maximum(np.linalg.norm(tg, axis=-1, keepdims=True), EPS)
    cos = np.clip(np.sum(np_ * ng, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def auc(errors: NDArray[Any], thresholds: list[float]) -> dict[float, float]:
    """Area under the accuracy-vs-threshold curve, in the SuperGlue style.

    Given a 1D array of per-sample errors (lower is better) and a list of
    thresholds ``t``, each AUC is the area under the step function
    ``acc(x) = fraction of errors <= x`` integrated over ``x in [0, t]`` and
    normalized by ``t`` so the result lies in ``[0, 1]``.

    Each error ``e_i <= t`` contributes a step of height ``1/N`` starting at
    ``e_i`` and ending at ``t``, i.e. ``(t - e_i)/N`` to the area. Errors
    exceeding ``t`` contribute nothing.

    Following the SuperGlue / LoFTR / MASt3R convention.
    """
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    out: dict[float, float] = {}
    n = int(errors.size)
    for t in thresholds:
        if n == 0:
            out[t] = float("nan")
            continue
        contributions = np.clip(t - errors, 0.0, t)
        out[t] = float(contributions.sum() / (n * t))
    return out


def pose_auc(
    R_pred: NDArray[Any],
    R_gt: NDArray[Any],
    t_pred: NDArray[Any],
    t_gt: NDArray[Any],
    *,
    thresholds: tuple[float, ...] = (5.0, 10.0, 30.0),
    translation_mode: str = "cosine",
) -> dict[float, float]:
    """Pose AUC at the given degree thresholds.

    Per-pose error is ``max(rot_deg_err, trans_deg_err)``, matching SuperGlue
    and MASt3R. With ``translation_mode="cosine"``, the translation error is
    the direction error in degrees (up-to-scale); with ``"metric"`` it is the
    euclidean error in meters and the threshold is compared to ``max(rot_deg,
    trans_m * 10)`` — not used here; stick to cosine unless a paper specifies
    otherwise.
    """
    rot_err = rotation_error_degrees(R_pred, R_gt)
    if translation_mode == "cosine":
        trans_err = translation_cosine_error(t_pred, t_gt)
    elif translation_mode == "metric":
        trans_err = translation_error(t_pred, t_gt)
    else:
        raise ValueError(f"unknown translation_mode '{translation_mode}'")
    combined = np.maximum(np.asarray(rot_err), np.asarray(trans_err))
    return auc(combined, list(thresholds))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_rot(x: NDArray[Any]) -> NDArray[Any]:
    x = np.asarray(x)
    if x.shape[-2:] == (3, 3):
        return x
    if x.shape[-2:] == (4, 4):
        return x[..., :3, :3]
    raise ValueError(f"expected (..., 3, 3) or (..., 4, 4); got {x.shape}")


def _extract_trans(x: NDArray[Any]) -> NDArray[Any]:
    x = np.asarray(x)
    if x.shape[-1] == 3 and x.shape[-2:] != (4, 4):
        return x
    if x.shape[-2:] == (4, 4):
        return x[..., :3, 3]
    raise ValueError(f"expected (..., 3) or (..., 4, 4); got {x.shape}")
