"""Development ORE-style baseline, not a verified reproduction of published ORE.

Based on Chen et al., Object Reprojection Error (NeurIPS 2023), sections 3.2–3.3.
Requires independently verified static-object boxes and explicit intrinsics.
This implementation uses a caller-supplied finite depth grid, the first box
center as anchor, and rejects depths projecting behind any observed camera.
An audit of the NeurIPS supplement confirms differences: its demo uses radial
ray distance, width-plus-height normalization, clipped penalties, image-border
suppression, and does not reject negative-depth projections. This prototype is
not numerically equivalent and must not be called published ORE. It does not
assign functional-DNF labels. See paper/audit_ore_semantics.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def score_static_track(
    world_from_camera: ArrayLike,
    intrinsics: ArrayLike,
    boxes_xyxy: ArrayLike,
    image_size_wh: tuple[int, int],
    depths: ArrayLike,
    *,
    static_verified: bool,
) -> dict[str, Any]:
    """Score one object's observed frames, in matching order for all arrays.

    Boxes use input-image pixels; intrinsics must use the same pixel convention.
    Missing observations must be removed consistently before calling. The score
    is mean distance outside each box: dx / width + dy / height. No clipping of
    off-image projections is performed. Depth is in the trajectory's units.
    """
    if static_verified is not True:
        raise ValueError("independent static-object verification is required")
    poses = np.asarray(world_from_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    grid = np.asarray(depths, dtype=np.float64)
    size = np.asarray(image_size_wh, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("need at least two 4x4 camera-to-world poses")
    n = len(poses)
    if k.shape == (3, 3):
        k = np.broadcast_to(k, (n, 3, 3))
    if k.shape != (n, 3, 3) or boxes.shape != (n, 4):
        raise ValueError("intrinsics and boxes must match observed frames")
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError("positive image width and height required")
    if grid.ndim != 1 or len(grid) < 2 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("need at least two positive increasing depth samples")
    if not all(np.isfinite(a).all() for a in (poses, k, boxes, grid, size)):
        raise ValueError("inputs must be finite")
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)
    ):
        raise ValueError("poses must contain proper rigid transforms")
    if (
        not np.allclose(k[:, 2], [0, 0, 1])
        or np.any(k[:, 0, 0] <= 0)
        or np.any(k[:, 1, 1] <= 0)
        or np.any(np.abs(np.linalg.det(k)) < 1e-12)
    ):
        raise ValueError("valid pinhole intrinsics required")
    if (
        np.any(boxes[:, :2] > boxes[:, 2:])
        or np.any(boxes < 0)
        or np.any(boxes[:, [0, 2]] > size[0])
        or np.any(boxes[:, [1, 3]] > size[1])
    ):
        raise ValueError("ordered boxes must lie within the image")
    center = (boxes[0, :2] + boxes[0, 2:]) / 2
    ray = np.linalg.solve(k[0], np.r_[center, 1.0])
    local = grid[:, None] * ray
    world = local @ rotations[0].T + poses[0, :3, 3]
    camera = np.einsum(
        "nij,dnj->dni", rotations.swapaxes(1, 2), world[:, None, :] - poses[None, :, :3, 3]
    )
    valid = (camera[:, :, 2] > 1e-9).all(axis=1)
    scores = np.full(len(grid), np.inf)
    projections = np.full((len(grid), n, 2), np.nan)
    if valid.any():
        homogeneous = np.einsum("nij,dnj->dni", k, camera[valid])
        uv = homogeneous[:, :, :2] / homogeneous[:, :, 2:]
        outside = np.maximum(boxes[None, :, :2] - uv, 0) + np.maximum(uv - boxes[None, :, 2:], 0)
        scores[valid] = (outside / size).sum(axis=2).mean(axis=1)
        projections[valid] = uv
    if not np.isfinite(scores).any():
        return {"status": "unavailable", "reason": "no depth projects in front of all cameras"}
    best = int(np.argmin(scores))
    return {
        "status": "ok",
        "baseline": "ore_style_finite_grid_v1",
        "score": float(scores[best]),
        "depth": float(grid[best]),
        "depth_at_grid_boundary": best in (0, len(grid) - 1),
        "valid_depth_count": int(valid.sum()),
        "n_observations": n,
        "projected_xy": projections[best].tolist(),
        "depth_grid": grid.tolist(),
    }
