"""Exact paired squared-error sensitivity to shared image-location uncertainty.

Conditional deterministic bounds: every true target must lie in its declared
Euclidean disk. This module neither establishes those disks nor handles missing
or undefined tasks. It does not produce statistical confidence or DNF labels.
"""

import numpy as np


def squared_difference_bounds(left, right, observed, radii):
    """Bound mean(||left-target||^2 - ||right-target||^2).

    The shared target makes the squared-error difference affine. Each target
    may independently lie anywhere in a disk around its measured location.
    Extremes are attained by moving along/opposite the candidate separation.
    Negative difference favors left, unlike a success-fraction difference.
    """
    a, b, q = [np.asarray(v, dtype=float) for v in (left, right, observed)]
    if a.ndim != 2 or a.shape[1:] != (2,) or not len(a) or b.shape != a.shape or q.shape != a.shape:
        raise ValueError('nonempty matched Nx2 locations required')
    radius = np.asarray(radii, dtype=float)
    if radius.ndim == 0:
        radius = np.full(len(a), float(radius))
    if radius.shape != (len(a),) or np.any(radius < 0) or not all(np.isfinite(v).all() for v in (a,b,q,radius)):
        raise ValueError('finite coordinates and nonnegative finite radii required')
    delta = np.sum((a-q)**2-(b-q)**2, axis=1)
    width = 2*radius*np.linalg.norm(a-b, axis=1)
    center = float(delta.mean())
    low, high = float((delta-width).mean()), float((delta+width).mean())
    return dict(observations=len(a), estimated_mean_squared_error_difference=center,
                mean_squared_error_difference_bounds=[low,high],
                ordering='left' if high < 0 else 'right' if low > 0 else 'equal' if low == high == 0 else 'unresolved',
                radius_pixels=radius.tolist(),
                assumption='every true target lies within its declared image-location disk; fixed valid task set; not a confidence interval or camera-quality guarantee')
