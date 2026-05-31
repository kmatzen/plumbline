"""Point-cloud metrics: Chamfer distance and F-score.

These are the standard point-set distance metrics used for multi-view stereo
(ETH3D, Tanks & Temples) and recent MVS foundation models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["accuracy_completeness", "chamfer_distance", "f_score", "voxel_downsample"]


def _nn_distances(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """For each point in ``a``, distance to its nearest neighbor in ``b``.

    Falls back to a numpy pairwise computation when scipy is unavailable. Uses
    ``scipy.spatial.cKDTree`` when available for O(M log N) behavior, with
    ``workers=-1`` so queries parallelise across all CPU cores. The
    single-threaded default was an ~8x slowdown on the 1.45M-point MVS
    predictions (VGGT at 518×350 × 8 views) — enough to turn a 7-minute
    chamfer reproduction into 50+ min.
    """
    if a.size == 0 or b.size == 0:
        return np.empty(a.shape[0], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(b)
        d, _ = tree.query(a, k=1, workers=-1)
        return d.astype(np.float64)
    except ImportError:  # pragma: no cover — exercised only without scipy
        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        d = np.empty(a64.shape[0], dtype=np.float64)
        # Chunk to avoid blowing up memory when both arrays are large.
        # NB: parenthesise (1 << 20) — ``//`` binds tighter than ``<<`` in
        # Python, so ``1 << 20 // b`` is ``1 << (20 // b)`` (== 1 for any
        # b >= 21), which collapses the chunk to a single row and makes the
        # fallback pathologically slow. We want ~1M / b rows per chunk.
        chunk = max(1, (1 << 20) // max(b64.shape[0], 1))
        for i in range(0, a64.shape[0], chunk):
            block = a64[i : i + chunk]
            diff = block[:, None, :] - b64[None, :, :]
            d[i : i + chunk] = np.sqrt((diff * diff).sum(-1)).min(axis=1)
        return d


def chamfer_distance(
    pred: NDArray[Any],
    gt: NDArray[Any],
    *,
    two_sided: bool = True,
    outlier_distance: float | None = None,
) -> float:
    """Symmetric Chamfer distance between two point sets.

    ``mean_{x in pred} min_{y in gt} ||x - y||  +  mean_{y in gt} min_{x in pred} ||y - x||``

    When ``two_sided=False`` returns only the first term (accuracy, pred->gt).

    Parameters
    ----------
    pred, gt
        ``(N, 3)`` and ``(M, 3)`` float arrays in the same world frame.
    outlier_distance
        If set, discard predicted points whose nearest GT distance exceeds
        this threshold BEFORE computing the mean. Matches the
        standard MVS paper protocol (ETH3D / Tanks & Temples) where the
        mean is reported over "inlier" predictions — far-outlier
        hallucinations are excluded so a tiny fraction of bad predictions
        can't dominate the metric.

        The GT→pred direction keeps all GT points (recall is meant to
        reflect coverage and shouldn't be pruned). Effectively this turns
        the reported chamfer from
            mean(d_pred→gt) + mean(d_gt→pred)
        into
            mean(d_pred→gt | d_pred→gt < outlier_distance) + mean(d_gt→pred).

        ``None`` (default) reports the untrimmed chamfer — matches prior
        plumbline behaviour.
    """
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"pred must be (N, 3); got {pred.shape}")
    if gt.ndim != 2 or gt.shape[-1] != 3:
        raise ValueError(f"gt must be (M, 3); got {gt.shape}")
    if pred.size == 0 or gt.size == 0:
        return float("nan")
    d_pg = _nn_distances(pred, gt)
    if outlier_distance is not None:
        if outlier_distance <= 0:
            raise ValueError(f"outlier_distance must be > 0; got {outlier_distance}")
        inlier = d_pg < outlier_distance
        if not inlier.any():
            # No inliers means the prediction is entirely outside the GT
            # region — mean-over-empty is undefined, report NaN.
            return float("nan")
        d_pg = d_pg[inlier]
    if not two_sided:
        return float(d_pg.mean())
    d_gp = _nn_distances(gt, pred)
    return float(d_pg.mean() + d_gp.mean())


def f_score(
    pred: NDArray[Any],
    gt: NDArray[Any],
    *,
    threshold: float,
    outlier_distance: float | None = None,
) -> dict[str, float]:
    """F-score / precision / recall at a distance threshold.

    - ``precision`` — fraction of ``pred`` points within ``threshold`` of any
      ``gt`` point (accuracy).
    - ``recall`` — fraction of ``gt`` points within ``threshold`` of any
      ``pred`` point (completeness).
    - ``f_score`` — harmonic mean of the two.

    Returned as percentages in ``[0, 100]``, matching the Tanks & Temples and
    ETH3D reporting convention.

    Parameters
    ----------
    threshold
        Inlier distance for precision/recall (typically 5 cm or 10 cm).
    outlier_distance
        If set, discard predicted points whose nearest GT distance exceeds
        this threshold before computing precision. Precision is always
        already insensitive to outliers when ``outlier_distance >= threshold``
        — a point outside threshold isn't counted either way — so this
        only matters when you want the *base* denominator (all pred
        points including far hallucinations) replaced by the inlier pool.
        Matches the chamfer-side outlier masking protocol used by MVS
        papers when they report F-score alongside chamfer. Recall is
        unchanged (coverage-on-GT shouldn't be pruned).
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold}")
    if pred.size == 0 or gt.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f_score": float("nan")}
    d_pg = _nn_distances(pred, gt)
    d_gp = _nn_distances(gt, pred)
    if outlier_distance is not None:
        if outlier_distance <= 0:
            raise ValueError(f"outlier_distance must be > 0; got {outlier_distance}")
        inlier = d_pg < outlier_distance
        if not inlier.any():
            return {"precision": float("nan"), "recall": float("nan"), "f_score": float("nan")}
        d_pg = d_pg[inlier]
    precision = float((d_pg < threshold).mean()) * 100.0
    recall = float((d_gp < threshold).mean()) * 100.0
    f = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f}


def voxel_downsample(points: NDArray[Any], voxel_size: float) -> NDArray[np.float64]:
    """One representative point per occupied voxel (centroid of cell members).

    Matches the density normalization ETH3D's multi-view-evaluation tool
    applies before chamfer/accuracy — prevents dense regions of the
    prediction from dominating mean-distance statistics.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0; got {voxel_size}")
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    p = points.astype(np.float64)
    keys = np.floor(p / voxel_size).astype(np.int64)
    # Unique voxel indices with an inverse map so we can bincount means.
    # numpy 2.0/2.1 returned a 2-D ``(N, 1)`` inverse from
    # ``np.unique(axis=0, return_inverse=True)`` (a regression reverted in
    # 2.2); ``np.bincount`` rejects a 2-D arg. ``reshape(-1)`` is a no-op on
    # the 1-D result every other version gives and keeps us correct across
    # the declared ``numpy>=1.24`` range.
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.reshape(inv, -1)
    n_cells = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=n_cells).astype(np.float64)
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    for i in range(3):
        sums[:, i] = np.bincount(inv, weights=p[:, i], minlength=n_cells)
    return sums / counts[:, None]


def accuracy_completeness(
    pred: NDArray[Any],
    gt: NDArray[Any],
    *,
    voxel_size: float | None = 0.01,
    outlier_distance: float | None = None,
) -> dict[str, float]:
    """Paper-protocol Acc/Comp/Overall, all in input units.

    - ``accuracy``     — mean pred→GT nearest-neighbor distance.
    - ``completeness`` — mean GT→pred nearest-neighbor distance.
    - ``overall``      — (accuracy + completeness) / 2, matches VGGT paper
      Table 3 "Overall (Chamfer distance)" convention.

    Parameters
    ----------
    voxel_size
        Cell size for ``voxel_downsample`` of the prediction cloud before
        computing chamfer. ``None`` skips the downsample (matches the
        DUSt3R-/MASt3R-/VGGT-family convention — see CUT3R's
        ``eval/mv_recon/utils.py`` which calls KDTree on the raw masked
        cloud directly). The plumbline scene-aggregation path already
        voxel-downsamples each per-sample chunk before accumulating
        (see ``runner._scene_aggregation``), so passing ``None`` here
        avoids a redundant second downsample at the cost of slightly
        more work in the KDTree query. Default ``0.01`` mirrors the
        ETH3D evaluation tool when chunks are NOT pre-downsampled.
        Units match ``pred`` / ``gt``.
    outlier_distance
        If set, drop pred points whose nearest-GT distance exceeds this
        threshold BEFORE computing either Acc or Comp. Matches the MVS
        convention (e.g. CUT3R's ``conf_thresh`` + 224×224 center crop)
        where wild outlier predictions are excluded; without it Acc is
        dominated by a handful of far-outlier points while Comp is
        unaffected. Units match ``pred`` / ``gt``.
    """
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"pred must be (N, 3); got {pred.shape}")
    if gt.ndim != 2 or gt.shape[-1] != 3:
        raise ValueError(f"gt must be (M, 3); got {gt.shape}")
    if pred.size == 0 or gt.size == 0:
        nan = float("nan")
        return {"accuracy": nan, "completeness": nan, "overall": nan}
    pred_ds = voxel_downsample(pred, voxel_size) if voxel_size is not None else pred
    d_pg = _nn_distances(pred_ds, gt)
    if outlier_distance is not None:
        if outlier_distance <= 0:
            raise ValueError(f"outlier_distance must be > 0; got {outlier_distance}")
        inlier = d_pg < outlier_distance
        if not inlier.any():
            nan = float("nan")
            return {"accuracy": nan, "completeness": nan, "overall": nan}
        pred_ds = pred_ds[inlier]
        d_pg = d_pg[inlier]
    d_gp = _nn_distances(gt, pred_ds)
    acc = float(d_pg.mean())
    comp = float(d_gp.mean())
    # Median variants — CUT3R's ``eval/mv_recon/utils.py::accuracy`` reports
    # both mean and median; some MVS papers use the median to discount
    # outliers. Cheap to compute (both NN distance arrays are already
    # materialised) so we always emit them under distinct keys.
    acc_med = float(np.median(d_pg))
    comp_med = float(np.median(d_gp))
    return {
        "accuracy": acc,
        "completeness": comp,
        "overall": 0.5 * (acc + comp),
        "accuracy_median": acc_med,
        "completeness_median": comp_med,
        "overall_median": 0.5 * (acc_med + comp_med),
    }
