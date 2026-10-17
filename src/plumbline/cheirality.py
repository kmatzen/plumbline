"""Classical two-ray diagnostics, not a learned metric or a usability verdict."""

from __future__ import annotations

import numpy as np

from plumbline.conventions import assert_valid_extrinsics


def triangulation_diagnostics(first, second, world_from_first, world_from_second):
    """Least-squares ray depths in baseline units, with explicit degeneracy.

    Inputs are calibrated (x/z,y/z) points. Positive depths are necessary for a
    static point in front of both cameras, but not sufficient for valid tracking.
    Numerical rank is not a practical observability threshold. Small nonzero
    angles may still give unreliable depths. No matches are silently filtered.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
        raise ValueError("correspondences must have matching (N,2) shapes")
    p, q = np.asarray(world_from_first, dtype=float), np.asarray(world_from_second, dtype=float)
    if p.shape != (4, 4) or q.shape != (4, 4):
        raise ValueError("expected two 4x4 poses")
    assert_valid_extrinsics(p)
    assert_valid_extrinsics(q)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite coordinates required")
    relative = np.linalg.inv(q) @ p
    t = relative[:3, 3]
    baseline = np.linalg.norm(t)
    if baseline <= np.finfo(float).tiny:
        return None
    t = t / baseline
    u = np.c_[a, np.ones(len(a))] @ relative[:3, :3].T
    v = np.c_[b, np.ones(len(b))]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    system = np.stack((u, -v), axis=-1)
    left, singular, right = np.linalg.svd(system, full_matrices=False)
    resolved = singular[:, 1] > 1e-10 * singular[:, 0]
    depths = np.full((len(a), 2), np.nan)
    depths[resolved] = np.einsum(
        "nji,nj->ni",
        right[resolved],
        np.einsum("nji,j->ni", left[resolved], -t) / singular[resolved],
    )
    gap = np.linalg.norm(np.einsum("nij,nj->ni", system, depths) + t, axis=1)
    return {
        "resolved": resolved,
        "depths_baseline_units": depths,
        "positive_both": resolved & (depths > 0).all(axis=1),
        "negative_both": resolved & (depths < 0).all(axis=1),
        "ray_angle_degrees": np.degrees(
            np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.sum(u * v, axis=1))
        ),
        "ray_gap_baseline_units": gap,
    }
