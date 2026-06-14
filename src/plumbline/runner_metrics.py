"""Per-sample metric aggregation for the evaluation runner.

These helpers turn a model's raw prediction + the dataset's ground truth into
the flat ``{metric_name: value}`` dicts the runner records per sample. They were
factored out of :mod:`plumbline.runner` (which had grown past 1400 lines) to
keep the orchestration loop (``evaluate``) separate from the numeric
metric-aggregation it delegates to. Pure-numpy, no torch, no model imports.

Three entry points are called by ``runner._compute_metrics``:

- :func:`_depth_metrics`      — aligned depth errors (AbsRel/RMSE/δ-thresholds).
- :func:`_point_cloud_metrics` — chamfer + F-score on a flattened point cloud.
- :func:`_pose_metrics`        — per-view + pairwise + trajectory pose errors.

The remaining functions are internal helpers of the depth path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import depth_is_valid
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import (
    abs_rel,
    delta_threshold,
    log10_error,
    rmse,
    rmse_log,
    silog,
    sq_rel,
)
from plumbline.metrics.pointmap import chamfer_distance, f_score
from plumbline.metrics.pose import accuracy_at_threshold as pose_acc_fn
from plumbline.metrics.pose import auc as pose_auc_fn
from plumbline.metrics.pose import (
    pairwise_pose_errors,
    rotation_error_degrees,
    trajectory_ate_rmse_sim3,
    trajectory_rpe_rmse_sim3,
    translation_cosine_error,
)


def _depth_metrics(
    *,
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
    scale_alignment: str,
    delta_thresholds: tuple[float, ...],
    depth_clip: tuple[float, float] | None = None,
) -> dict[str, float]:
    pred_aligned, gt_flat, valid_flat = _flatten_pred_gt(pred, gt, valid)
    if pred_aligned.size == 0:
        return {k: float("nan") for k in _depth_metric_keys(delta_thresholds)}

    aligned = align_depth(pred_aligned, gt_flat, valid_flat, mode=scale_alignment)
    # Post-alignment clip to a physical depth range. Standard NYU/KITTI eval
    # protocol clips predictions to the same range as the valid-GT mask so a
    # handful of alignment-induced outliers don't pull the mean AbsRel up by
    # orders of magnitude (seen on DA-V2 Large sample 88 where aligned
    # depth blew up to 1e8m without this clip).
    if depth_clip is not None:
        lo, hi = depth_clip
        aligned = np.clip(aligned, lo, hi)

    metrics: dict[str, float] = {
        "abs_rel": abs_rel(aligned, gt_flat, valid_flat),
        "sq_rel": sq_rel(aligned, gt_flat, valid_flat),
        "rmse": rmse(aligned, gt_flat, valid_flat),
        "rmse_log": rmse_log(aligned, gt_flat, valid_flat),
        "log10": log10_error(aligned, gt_flat, valid_flat),
        "silog": silog(aligned, gt_flat, valid_flat),
    }
    for i, t in enumerate(delta_thresholds, start=1):
        metrics[f"delta_{i}"] = delta_threshold(aligned, gt_flat, valid_flat, threshold=t)
    return metrics


def _depth_metric_keys(deltas: tuple[float, ...]) -> list[str]:
    return [
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "log10",
        "silog",
        *[f"delta_{i}" for i in range(1, len(deltas) + 1)],
    ]


def _flatten_pred_gt(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Resize prediction to GT resolution per-view and flatten batches."""
    pred_out = pred.astype(np.float64) if pred.shape == gt.shape else _resize_depth_to_gt(pred, gt)
    if valid is None:
        valid = depth_is_valid(gt)
    return pred_out, gt.astype(np.float64), valid


def _resize_depth_to_gt(pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
    """Resize a batch of depth maps to GT resolution with bilinear sampling.

    Never bilinear-resizes ground-truth; the runner only resizes predictions.
    Uses PIL so it runs on numpy alone, no torch dependency.
    """
    from PIL import Image

    if pred.shape[-2:] == gt.shape[-2:]:
        return pred.astype(np.float64)
    if pred.ndim != gt.ndim:
        raise ValueError(f"ndim mismatch: pred {pred.shape} vs gt {gt.shape}")
    batch_shape = pred.shape[:-2]
    tgt_h, tgt_w = gt.shape[-2:]
    flat = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    out = np.empty((flat.shape[0], tgt_h, tgt_w), dtype=np.float64)
    for i, m in enumerate(flat):
        img = Image.fromarray(m.astype(np.float32), mode="F")
        resample = Image.Resampling.BILINEAR
        out[i] = np.asarray(img.resize((tgt_w, tgt_h), resample=resample), dtype=np.float64)
    return out.reshape(*batch_shape, tgt_h, tgt_w)


def _point_cloud_metrics(
    *,
    point_map: NDArray[Any],
    point_cloud_gt: NDArray[Any],
    f_score_threshold: float,
    outlier_distance: float | None = None,
) -> dict[str, float]:
    """Chamfer + F-score between the flattened prediction and GT point cloud.

    Flattens ``(N, H, W, 3)`` prediction into a single ``(M, 3)`` cloud,
    dropping NaNs/zeros, and compares against the GT cloud with the given
    distance threshold (meters, matching the units of the point map).

    Parameters
    ----------
    outlier_distance
        If set, discard predicted points farther than this many units from
        their nearest GT neighbour before computing chamfer + precision.
        Matches the ETH3D / Tanks & Temples paper protocol, where the
        reported chamfer/F-score is on the inlier pool so a tiny fraction
        of hallucinations can't dominate the metric.
    """
    if point_map.ndim < 2 or point_map.shape[-1] != 3:
        raise ValueError(f"point_map must end in 3 for xyz; got {point_map.shape}")
    pts = point_map.reshape(-1, 3).astype(np.float64)
    # Drop NaN-marked invalid points.
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        return {
            "chamfer": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f_score": float("nan"),
        }

    chamfer = chamfer_distance(
        pts.astype(np.float32),
        point_cloud_gt.astype(np.float32),
        outlier_distance=outlier_distance,
    )
    f = f_score(
        pts.astype(np.float32),
        point_cloud_gt.astype(np.float32),
        threshold=f_score_threshold,
        outlier_distance=outlier_distance,
    )
    return {"chamfer": float(chamfer), **{k: float(v) for k, v in f.items()}}


def _pose_metrics(
    E_pred: NDArray[Any],
    E_gt: NDArray[Any],
    auc_thresholds: tuple[float, ...],
    acc_thresholds: tuple[float, ...],
    auc_mode: str,
    translation_antipodal: bool,
    trajectory_metrics: bool = False,
) -> dict[str, float]:
    if E_pred.shape != E_gt.shape:
        raise ValueError(f"pose shape mismatch: {E_pred.shape} vs {E_gt.shape}")
    # Absolute per-view errors: compare pred[i] vs GT[i] in their shared
    # world frame (view-0 identity by convention). Skip camera 0.
    if E_pred.ndim == 3 and E_pred.shape[0] > 1:
        Ep = E_pred[1:]
        Eg = E_gt[1:]
    else:
        Ep = E_pred
        Eg = E_gt
    abs_rot = np.asarray(rotation_error_degrees(Ep, Eg)).reshape(-1)
    abs_trans = np.asarray(
        translation_cosine_error(Ep[..., :3, 3], Eg[..., :3, 3], antipodal=translation_antipodal)
    ).reshape(-1)
    abs_combined = np.maximum(abs_rot, abs_trans)
    abs_aucs = pose_auc_fn(abs_combined, list(auc_thresholds), mode=auc_mode)

    out: dict[str, float] = {
        "rotation_error_deg_mean": float(np.nanmean(abs_rot)),
        "translation_cos_err_deg_mean": float(np.nanmean(abs_trans)),
    }
    for t, v in abs_aucs.items():
        out[f"pose_auc@{t:g}"] = float(v)

    # Pairwise relative pose errors: frame-invariant, compared over all
    # N*(N-1)/2 unordered pairs. This is the metric VGGT / MASt3R /
    # DUSt3R papers report — directly comparable to their tables.
    if E_pred.ndim == 3 and E_pred.shape[0] >= 2:
        pw_rot, pw_trans = pairwise_pose_errors(
            E_pred, E_gt, translation_antipodal=translation_antipodal
        )
        pw_combined = np.maximum(pw_rot, pw_trans)
        pw_aucs = pose_auc_fn(pw_combined, list(auc_thresholds), mode=auc_mode)
        pw_rra = pose_acc_fn(pw_rot, list(acc_thresholds))
        pw_rta = pose_acc_fn(pw_trans, list(acc_thresholds))
        out["pairwise_rot_err_deg_mean"] = float(np.nanmean(pw_rot))
        out["pairwise_trans_cos_err_deg_mean"] = float(np.nanmean(pw_trans))
        for t, v in pw_aucs.items():
            out[f"pairwise_pose_auc@{t:g}"] = float(v)
        for t, v in pw_rra.items():
            out[f"pairwise_RRA@{t:g}"] = float(v)
        for t, v in pw_rta.items():
            out[f"pairwise_RTA@{t:g}"] = float(v)

    # Trajectory pose metrics (TUM-RGBD-style ATE / RPE). These are what
    # MonST3R Table 4 / SLAM-style papers report for video-pose evaluation.
    # The metric is meaningful only when each Sample is a *trajectory* (N
    # ordered frames from one sequence) AND the loader provides
    # metric-scale GT (Sim(3) alignment recovers scale, so any
    # consistently-scaled pred lands on the same number). Plumbline
    # delegates to `evo` to match MonST3R's bit-for-bit eval shape (their
    # `dust3r/utils/vo_eval.py:eval_metrics` is the same evo wrapper).
    if E_pred.ndim == 3 and E_pred.shape[0] >= 3 and trajectory_metrics:
        try:
            ate = trajectory_ate_rmse_sim3(E_pred, E_gt)
            rpe_t, rpe_r = trajectory_rpe_rmse_sim3(E_pred, E_gt, delta=1)
            out["trajectory_ate_rmse"] = float(ate)
            out["trajectory_rpe_trans_rmse"] = float(rpe_t)
            out["trajectory_rpe_rot_deg_rmse"] = float(rpe_r)
        except ImportError:
            # `evo` is an optional extra; silently skip if not installed.
            pass
        except Exception:
            # Umeyama can fail on degenerate trajectories (collinear /
            # near-stationary cameras). Emit NaN so the per-sample row
            # records the failure and aggregation skips it via nanmean —
            # don't crash the whole eval.
            out["trajectory_ate_rmse"] = float("nan")
            out["trajectory_rpe_trans_rmse"] = float("nan")
            out["trajectory_rpe_rot_deg_rmse"] = float("nan")

    return out
