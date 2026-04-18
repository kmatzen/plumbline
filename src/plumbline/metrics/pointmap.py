"""Point-cloud metrics: Chamfer distance and F-score.

These are the standard point-set distance metrics used for multi-view stereo
(ETH3D, Tanks & Temples) and recent MVS foundation models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["chamfer_distance", "f_score"]


def _nn_distances(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """For each point in ``a``, distance to its nearest neighbor in ``b``.

    Falls back to a numpy pairwise computation when scipy is unavailable. Uses
    ``scipy.spatial.cKDTree`` when available for O(M log N) behavior.
    """
    if a.size == 0 or b.size == 0:
        return np.empty(a.shape[0], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(b)
        d, _ = tree.query(a, k=1)
        return d.astype(np.float64)
    except ImportError:  # pragma: no cover — exercised only without scipy
        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        d = np.empty(a64.shape[0], dtype=np.float64)
        # Chunk to avoid blowing up memory when both arrays are large.
        chunk = max(1, 1 << 20 // max(b64.shape[0], 1))
        for i in range(0, a64.shape[0], chunk):
            block = a64[i : i + chunk]
            diff = block[:, None, :] - b64[None, :, :]
            d[i : i + chunk] = np.sqrt((diff * diff).sum(-1)).min(axis=1)
        return d


def chamfer_distance(pred: NDArray[Any], gt: NDArray[Any], *, two_sided: bool = True) -> float:
    """Symmetric Chamfer distance between two point sets.

    ``mean_{x in pred} min_{y in gt} ||x - y||  +  mean_{y in gt} min_{x in pred} ||y - x||``

    When ``two_sided=False`` returns only the first term (accuracy, pred->gt).

    Parameters
    ----------
    pred, gt
        ``(N, 3)`` and ``(M, 3)`` float arrays in the same world frame.
    """
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(f"pred must be (N, 3); got {pred.shape}")
    if gt.ndim != 2 or gt.shape[-1] != 3:
        raise ValueError(f"gt must be (M, 3); got {gt.shape}")
    if pred.size == 0 or gt.size == 0:
        return float("nan")
    d_pg = _nn_distances(pred, gt)
    if not two_sided:
        return float(d_pg.mean())
    d_gp = _nn_distances(gt, pred)
    return float(d_pg.mean() + d_gp.mean())


def f_score(
    pred: NDArray[Any],
    gt: NDArray[Any],
    *,
    threshold: float,
) -> dict[str, float]:
    """F-score / precision / recall at a distance threshold.

    - ``precision`` — fraction of ``pred`` points within ``threshold`` of any
      ``gt`` point (accuracy).
    - ``recall`` — fraction of ``gt`` points within ``threshold`` of any
      ``pred`` point (completeness).
    - ``f_score`` — harmonic mean of the two.

    Returned as percentages in ``[0, 100]``, matching the Tanks & Temples and
    ETH3D reporting convention.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold}")
    if pred.size == 0 or gt.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f_score": float("nan")}
    d_pg = _nn_distances(pred, gt)
    d_gp = _nn_distances(gt, pred)
    precision = float((d_pg < threshold).mean()) * 100.0
    recall = float((d_gp < threshold).mean()) * 100.0
    f = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f}
