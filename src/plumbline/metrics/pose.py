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
    "accuracy_at_threshold",
    "auc",
    "pairwise_pose_errors",
    "pairwise_relative_poses",
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


def translation_cosine_error(
    t_pred: NDArray[Any], t_gt: NDArray[Any], *, antipodal: bool = False
) -> NDArray[Any]:
    """Angular error between translation directions, in degrees.

    Used for up-to-scale monocular pose evaluation, where the magnitude of ``t``
    is arbitrary but its direction is meaningful.

    Parameters
    ----------
    antipodal
        When ``True``, returns ``min(angle, 180 - angle)`` so a sign-flipped
        prediction reads as 0° instead of 180°. This is the convention in
        VGGT / PoseDiffusion / RelPose++ CO3Dv2 pose eval — pred translation
        magnitude/sign is genuinely ambiguous up to scale, so antipodal pairs
        are treated as equivalent. Default ``False`` preserves the
        SuperGlue/MASt3R 7-Scenes-style raw direction error.
    """
    tp = _extract_trans(t_pred)
    tg = _extract_trans(t_gt)
    if tp.shape != tg.shape:
        raise ValueError(f"translation shape mismatch: {tp.shape} vs {tg.shape}")
    np_ = tp / np.maximum(np.linalg.norm(tp, axis=-1, keepdims=True), EPS)
    ng = tg / np.maximum(np.linalg.norm(tg, axis=-1, keepdims=True), EPS)
    cos = np.clip(np.sum(np_ * ng, axis=-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    if antipodal:
        angle = np.minimum(angle, 180.0 - angle)
    return angle


def auc(
    errors: NDArray[Any],
    thresholds: list[float],
    *,
    mode: str = "analytic",
) -> dict[float, float]:
    """Area under the accuracy-vs-threshold curve.

    Given a 1D array of per-sample errors (lower is better) and a list of
    thresholds ``t``, each AUC is the area under the step function
    ``acc(x) = fraction of errors <= x`` integrated over ``x in [0, t]`` and
    normalized by ``t`` so the result lies in ``[0, 1]``.

    Parameters
    ----------
    mode
        ``"analytic"`` (default): exact integral of the step function.
        Each error ``e_i <= t`` contributes ``(t - e_i) / (N * t)``. This
        is the SuperGlue / LoFTR / MASt3R 7-Scenes form.

        ``"vggt_co3d_histogram"``: Riemann approximation with 1°-wide bins,
        cumulative-sum, mean. Matches VGGT's `evaluation/test_co3d.py
        ::calculate_auc_np` and PoseDiffusion / RelPose++ exactly. Produces
        ~1-3% higher values than analytic on realistic distributions due
        to the upper-bin-edge convention. Use this when reproducing
        CO3Dv2 / RealEstate10K pose-AUC paper cells.
    """
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    out: dict[float, float] = {}
    n = int(errors.size)
    if mode == "analytic":
        for t in thresholds:
            if n == 0:
                out[t] = float("nan")
                continue
            contributions = np.clip(t - errors, 0.0, t)
            # Clamp to [0, 1]: the integral is bounded by construction,
            # but floating-point can nudge it to 1 + eps.
            out[t] = float(min(1.0, max(0.0, contributions.sum() / (n * t))))
        return out
    if mode == "vggt_co3d_histogram":
        for t in thresholds:
            if n == 0:
                out[t] = float("nan")
                continue
            t_int = int(round(float(t)))
            if t_int <= 0:
                raise ValueError(f"vggt_co3d_histogram requires t > 0; got {t}")
            bins = np.arange(t_int + 1, dtype=np.float64)
            hist, _ = np.histogram(errors, bins=bins)
            hist = hist.astype(np.float64) / float(n)
            out[t] = float(min(1.0, max(0.0, float(np.mean(np.cumsum(hist))))))
        return out
    raise ValueError(f"unknown auc mode '{mode}' (use 'analytic' or 'vggt_co3d_histogram')")


def accuracy_at_threshold(
    errors: NDArray[Any], thresholds: list[float]
) -> dict[float, float]:
    """Fraction of errors at or below each threshold.

    The shape VGGT / DUSt3R / MASt3R papers report as RRA@τ (rotation
    accuracy) and RTA@τ (translation accuracy). Each threshold returns a
    value in ``[0, 1]``. Non-finite errors are dropped, matching :func:`auc`.
    """
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    out: dict[float, float] = {}
    n = int(errors.size)
    for t in thresholds:
        if n == 0:
            out[t] = float("nan")
            continue
        out[t] = float((errors <= t).sum() / n)
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
# Pairwise relative pose (the metric papers like VGGT / MASt3R report)
# ---------------------------------------------------------------------------


def pairwise_relative_poses(
    extrinsics: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """All unordered pairwise relative poses from N ``world_from_camera`` matrices.

    For ``E_i = [R_i | t_i]`` (world-from-cam), the relative pose of cam ``j``
    in cam ``i``'s frame is ``E_i^-1 @ E_j``, giving
    ``R_ij = R_i^T @ R_j`` and ``t_ij = R_i^T @ (t_j - t_i)``.

    Returns two arrays of length ``N*(N-1)/2``:
    ``(R_rel, t_rel)`` over pairs ``(i, j)`` with ``i < j``.
    """
    E = np.asarray(extrinsics, dtype=np.float64)
    if E.ndim != 3 or E.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must be (N, 4, 4); got {E.shape}")
    n = E.shape[0]
    if n < 2:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    i_idx, j_idx = np.triu_indices(n, k=1)
    Ri = E[i_idx, :3, :3]  # (P, 3, 3)
    Rj = E[j_idx, :3, :3]
    ti = E[i_idx, :3, 3]  # (P, 3)
    tj = E[j_idx, :3, 3]
    Ri_T = np.transpose(Ri, (0, 2, 1))
    R_rel = Ri_T @ Rj
    t_rel = np.einsum("pij,pj->pi", Ri_T, tj - ti)
    return R_rel, t_rel


def pairwise_pose_errors(
    E_pred: NDArray[Any],
    E_gt: NDArray[Any],
    *,
    translation_antipodal: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-pair rotation + translation-direction errors (degrees) over all (i, j), i < j.

    Frame-invariant: uses relative poses, so no "same world frame"
    assumption between pred and GT. This is what VGGT / MASt3R / DUSt3R
    papers report as "relative pose AUC". Returns ``(rot_deg, trans_cos_deg)``
    each of shape ``(N*(N-1)/2,)``.

    ``translation_antipodal`` toggles VGGT/PoseDiffusion's
    ``min(angle, 180 - angle)`` convention on the translation error.
    """
    E_pred = np.asarray(E_pred, dtype=np.float64)
    E_gt = np.asarray(E_gt, dtype=np.float64)
    if E_pred.shape != E_gt.shape:
        raise ValueError(f"pred/gt extrinsics shape mismatch: {E_pred.shape} vs {E_gt.shape}")
    R_p, t_p = pairwise_relative_poses(E_pred)
    R_g, t_g = pairwise_relative_poses(E_gt)
    rot = rotation_error_degrees(R_p, R_g)
    trans = translation_cosine_error(t_p, t_g, antipodal=translation_antipodal)
    return np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)


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
