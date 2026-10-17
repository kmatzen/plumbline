"""Numerical static-point support in box frusta; not a camera-DNF certificate.

A common point is a stronger assumption than a static extended object with
partial visibility. Infeasibility can arise from annotations or calibration.

This reuses classical bounded-reprojection feasibility, not a novel triangulator.
See Kahl, ICCV 2005, Multiple View Geometry and the L-infinity Norm.
Unlike that paper's maximum Euclidean image residual, these axis-wise boxes use
coordinate-wise bounds and therefore linear rather than second-order cones.
"""

import numpy as np

from plumbline.conventions import assert_valid_extrinsics


def box_support(world_to_camera, intrinsics, boxes_xyxy, *, anchor_only=False, first_depth_interval=None):
    """Maximize positive-depth margin with box halfspaces, using a linear program.

    Translation is recentered and scaled by maximum camera-center displacement.
    The numerical margin threshold is not an application-quality threshold.
    Solver infeasibility is reported as such, not as an exact mathematical proof.
    Optional first-camera axial depth bounds use the same units as translation.
    """
    from scipy.optimize import linprog

    poses = np.asarray(world_to_camera, dtype=float)
    if first_depth_interval is not None:
        first_depth_interval = np.asarray(first_depth_interval, dtype=float)
        if (first_depth_interval.shape != (2,) or not np.isfinite(first_depth_interval).all()
                or first_depth_interval[0] <= 0 or first_depth_interval[1] < first_depth_interval[0]):
            raise ValueError("finite positive ordered first depth interval required")
    boxes = np.asarray(boxes_xyxy, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("at least two rigid poses required")
    assert_valid_extrinsics(poses)
    if k.shape == (3, 3):
        k = np.broadcast_to(k, (len(poses), 3, 3))
    if k.shape != (len(poses), 3, 3) or boxes.shape != (len(poses), 4):
        raise ValueError("intrinsics and boxes must match poses")
    if (
        not np.isfinite(k).all()
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, :2] > boxes[:, 2:])
    ):
        raise ValueError("finite calibration and ordered boxes required")
    if (
        not np.allclose(k[:, 2], [0, 0, 1])
        or np.any(k[:, 0, 0] <= 0)
        or np.any(k[:, 1, 1] <= 0)
        or np.any(np.abs(np.linalg.det(k)) < 1e-12)
    ):
        raise ValueError("valid pinhole intrinsics required")
    centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    origin = centers[0]
    scale = float(np.linalg.norm(centers - origin, axis=1).max())
    if scale == 0:
        scale = 1.0
    normalized = poses[:, :3].copy()
    normalized[:, :, 3] = (
        np.einsum("nij,j->ni", poses[:, :3, :3], origin) + poses[:, :3, 3]
    ) / scale
    matrices = k @ normalized
    constraints, limits = [], []
    for index in range(len(poses)):
        if not anchor_only or index == 0:
            p = matrices[index]
            x0, y0, x1, y1 = boxes[index]
            for row in (x0 * p[2] - p[0], p[0] - x1 * p[2], y0 * p[2] - p[1], p[1] - y1 * p[2]):
                constraints.append(np.r_[row[:3], 0.0])
                limits.append(-row[3])
        constraints.append(np.r_[-normalized[index, 2, :3], 1.0])
        limits.append(normalized[index, 2, 3])
    if first_depth_interval is not None:
        lo, hi = first_depth_interval / scale
        depth_row = normalized[0, 2]
        constraints.extend([np.r_[depth_row[:3], 0.0], np.r_[-depth_row[:3], 0.0]])
        limits.extend([hi - depth_row[3], depth_row[3] - lo])
    a, b = np.asarray(constraints), np.asarray(limits)
    norms = np.linalg.norm(a, axis=1)
    a, b = a / norms[:, None], b / norms
    result = linprog(
        [0, 0, 0, -1],
        A_ub=a,
        b_ub=b,
        bounds=[(None, None)] * 3 + [(0, 1)],
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    record = {
        "scope": "anchor box plus all depth constraints"
        if anchor_only
        else "all boxes plus all depth constraints",
        "solver_status": int(result.status),
        "solver_message": result.message,
        "coordinate_scale": scale,
        "numerical_margin_threshold": 1e-7,
        "status": "solver_infeasible" if result.status == 2 else "solver_failure",
    }
    if first_depth_interval is not None:
        record["first_depth_interval"] = first_depth_interval.tolist()
    if result.success:
        xyz = result.x[:3]
        z = normalized[:, 2, :3] @ xyz + normalized[:, 2, 3]
        violation = float(np.maximum(a @ result.x - b, 0).max())
        margin = float(result.x[3])
        projected = matrices @ np.r_[xyz, 1.0]
        if (z > 0).all():
            xy = projected[:, :2] / projected[:, 2:]
            used = slice(0, 1) if anchor_only else slice(None)
            pixel_violation = float(
                np.maximum(
                    np.maximum(boxes[used, :2] - xy[used], xy[used] - boxes[used, 2:]), 0
                ).max()
            )
        else:
            pixel_violation = None
        record.update(
            {
                "status": "positive_witness"
                if margin > 1e-7
                and z.min() > 1e-7
                and violation <= 1e-7
                and pixel_violation is not None
                and pixel_violation <= 1e-5
                else "boundary_or_numerical",
                "margin": margin,
                "minimum_normalized_z": float(z.min()),
                "maximum_normalized_constraint_violation": violation,
                "maximum_box_violation_pixels": pixel_violation,
                "point_world": (origin + scale * xyz).tolist(),
            }
        )
    return record
