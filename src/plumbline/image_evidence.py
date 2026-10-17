"""Camera consistency against candidate-independent image correspondences.

Sampson residuals are a conventional geometric baseline, not preference labels.
Coordinates must be calibrated normalized image coordinates. A zero baseline
cannot constrain epipolar geometry and returns no score rather than zero error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def rays_from_normalized_intrinsics(
    pixels: NDArray[Any], intrinsics: NDArray[Any], image_size_wh: tuple[int, int]
) -> NDArray[np.float64]:
    """Convert integer-center pixel coordinates using width/height-normalized K.

    DAGE's utils3d UV convention puts pixel centers at (j+0.5)/width,
    (i+0.5)/height. OpenCV feature locations use integer centers, hence +0.5.
    This is estimated calibration when K comes from DAGE, not ground truth.
    """
    xy = np.asarray(pixels, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    size = np.asarray(image_size_wh, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or k.shape != (3, 3):
        raise ValueError("expected (N,2) pixels and (3,3) intrinsics")
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError("positive width and height required")
    if not all(np.isfinite(a).all() for a in (xy, k, size)):
        raise ValueError("inputs must be finite")
    if (
        not np.allclose(k[2], [0, 0, 1])
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or abs(np.linalg.det(k)) < 1e-12
    ):
        raise ValueError("invalid normalized pinhole intrinsics")
    uv = (xy + 0.5) / size
    rays = np.linalg.solve(k, np.column_stack((uv, np.ones(len(xy)))).T).T
    return rays[:, :2] / rays[:, 2:]


def sampson_residuals(
    first: NDArray[Any],
    second: NDArray[Any],
    world_from_first: NDArray[Any],
    world_from_second: NDArray[Any],
) -> NDArray[np.float64] | None:
    """Return square-root Sampson errors, or None for zero translation.

    The result is in normalized camera units, not pixels or meters. The essential
    matrix is normalized so the result is invariant to global trajectory scale.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
        raise ValueError("correspondences must have matching (N,2) shapes")
    p, q = np.asarray(world_from_first, dtype=float), np.asarray(world_from_second, dtype=float)
    if p.shape != (4, 4) or q.shape != (4, 4):
        raise ValueError("poses must have shape (4,4)")
    if not all(np.isfinite(x).all() for x in (a, b, p, q)):
        raise ValueError("inputs must be finite")
    relative = np.linalg.inv(q) @ p
    translation = q[:3, :3].T @ (p[:3, 3] - q[:3, 3])
    norm = float(np.linalg.norm(translation))
    if norm <= np.finfo(float).tiny:
        return None
    x, y, z = translation / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    essential = skew @ relative[:3, :3]
    a_h = np.column_stack((a, np.ones(len(a))))
    b_h = np.column_stack((b, np.ones(len(b))))
    ea = a_h @ essential.T
    etb = b_h @ essential
    numerator = np.abs(np.sum(b_h * ea, axis=1))
    denominator = np.sqrt(np.sum(ea[:, :2] ** 2 + etb[:, :2] ** 2, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(a), np.inf), where=denominator > 1e-15)


def target_epipolar_lines(
    first_pixels: NDArray[Any],
    first_intrinsics: NDArray[Any],
    second_intrinsics: NDArray[Any],
    first_size_wh: tuple[int, int],
    second_size_wh: tuple[int, int],
    world_from_first: NDArray[Any],
    world_from_second: NDArray[Any],
) -> NDArray[np.float64] | None:
    """Return ax+by+c=0 lines in target integer-center pixels; K is normalized.

    Lines constrain matches of static scene points only. They do not locate
    an exact correspondence, and a point on a line does not prove correct pose.
    """
    from plumbline.conventions import assert_valid_extrinsics

    p, q = np.asarray(world_from_first, dtype=float), np.asarray(world_from_second, dtype=float)
    if p.shape != (4, 4) or q.shape != (4, 4):
        raise ValueError("need two 4x4 camera-to-world poses")
    assert_valid_extrinsics(p)
    assert_valid_extrinsics(q)
    rays = rays_from_normalized_intrinsics(first_pixels, first_intrinsics, first_size_wh)
    # Validate target calibration and dimensions using the same public contract.
    rays_from_normalized_intrinsics(np.empty((0, 2)), second_intrinsics, second_size_wh)
    relative = np.linalg.inv(q) @ p
    t = relative[:3, 3]
    norm = np.linalg.norm(t)
    if norm <= np.finfo(float).tiny:
        return None
    x, y, z = t / norm
    essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ relative[:3, :3]
    w, h = second_size_wh
    pixel_from_uv = np.array([[w, 0, -0.5], [0, h, -0.5], [0, 0, 1]])
    k_pixels = pixel_from_uv @ np.asarray(second_intrinsics)
    lines = np.column_stack((rays, np.ones(len(rays)))) @ essential.T @ np.linalg.inv(k_pixels)
    length = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return np.divide(lines, length, out=np.full_like(lines, np.nan), where=length > 1e-15)
