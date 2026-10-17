"""Classical linear bounds on first-camera depth, not a calibrated confidence interval."""

import numpy as np
from scipy.optimize import linprog

from plumbline.frustum_support import box_support


def depth_range(world_to_camera, intrinsics, boxes_xyxy):
    """Bound axial depth over the closure of positive-depth box intersections.

    A separate feasibility check requires a strictly positive witness first.
    Range endpoints can lie on the zero-depth boundary. Infinite upper depth
    means unbounded geometry, not missing data. Units match pose translation.
    """
    support = box_support(world_to_camera, intrinsics, boxes_xyxy)
    if support["status"] != "positive_witness":
        return {"status": support["status"], "support": support}
    poses = np.asarray(world_to_camera, dtype=float)
    relative = poses @ np.linalg.inv(poses[0])
    scale = support["coordinate_scale"]
    relative[:, :3, 3] /= scale
    k = np.asarray(intrinsics, dtype=float)
    if k.ndim == 2:
        k = np.broadcast_to(k, (len(poses), 3, 3))
    projection = k @ relative[:, :3]
    constraints, limits = [], []
    for p, camera, box in zip(projection, relative, boxes_xyxy, strict=True):
        x0, y0, x1, y1 = box
        for row in (x0 * p[2] - p[0], p[0] - x1 * p[2],
                    y0 * p[2] - p[1], p[1] - y1 * p[2], -camera[2]):
            constraints.append(row[:3])
            limits.append(-row[3])
    a, b = np.asarray(constraints), np.asarray(limits)
    norms = np.linalg.norm(a, axis=1)
    a, b = a / norms[:, None], b / norms
    endpoints = {}
    for name, sign in (("lower", 1), ("upper", -1)):
        solution = linprog([0, 0, sign], A_ub=a, b_ub=b,
                           bounds=[(None, None)] * 3, method="highs",
                           options={"primal_feasibility_tolerance": 1e-9,
                                    "dual_feasibility_tolerance": 1e-9})
        record = {"solver_status": int(solution.status), "message": solution.message}
        if solution.success:
            violation = float(np.maximum(a @ solution.x - b, 0).max())
            record.update(status="finite" if violation <= 1e-7 else "numerical_failure",
                          depth=float(solution.x[2] * scale),
                          maximum_normalized_violation=violation)
        elif solution.status == 3:
            record["status"] = "unbounded"
        else:
            record["status"] = "solver_failure"
        endpoints[name] = record
    return {"status": "checked", "scope": "closure bounds, not calibrated uncertainty",
            "coordinate_scale": scale, "support": support, **endpoints}
