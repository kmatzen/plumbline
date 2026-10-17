"""Linear-fractional projection bounds for early box intersections."""

import numpy as np
from scipy.optimize import linprog

from plumbline.conventions import assert_valid_extrinsics
from plumbline.frustum_support import box_support


def projection_range(world_to_camera, intrinsics, boxes_xyxy, query_camera, query_intrinsics):
    """Conditional image bounds, without using a query-frame observation.

    Homogeneous coordinates are scaled to query-camera depth one. The final
    homogeneous coordinate is nonnegative, allowing points at infinity in the
    closure. Bounds apply only to query-positive points; they do not certify
    visibility, staticness, or a confidence level. This is classical geometry.
    """
    query = np.asarray(query_camera, dtype=float)
    assert_valid_extrinsics(query[None])
    qk = np.asarray(query_intrinsics, dtype=float)
    if (qk.shape != (3, 3) or not np.isfinite(qk).all()
            or not np.allclose(qk[2], [0, 0, 1]) or qk[0, 0] <= 0 or qk[1, 1] <= 0
            or abs(np.linalg.det(qk)) < 1e-12):
        raise ValueError("valid query calibration required")
    support = box_support(world_to_camera, intrinsics, boxes_xyxy)
    if support["status"] != "positive_witness":
        return {"status": support["status"], "support": support}
    poses = np.asarray(world_to_camera, dtype=float)
    first_inverse = np.linalg.inv(poses[0])
    relative = poses @ first_inverse
    query_relative = query @ first_inverse
    scale = support["coordinate_scale"]
    relative[:, :3, 3] /= scale
    query_relative[:3, 3] /= scale
    k = np.asarray(intrinsics, dtype=float)
    projection = k @ relative[:, :3]
    query_projection = qk @ query_relative[:3]
    constraints = []
    for p, camera, box in zip(projection, relative, boxes_xyxy, strict=True):
        x0, y0, x1, y1 = box
        constraints.extend((x0*p[2]-p[0], p[0]-x1*p[2],
                            y0*p[2]-p[1], p[1]-y1*p[2], -camera[2]))
    a = np.asarray(constraints)
    a /= np.linalg.norm(a, axis=1)[:, None]
    endpoints = {}
    for axis in range(2):
        for name, sign in (("lower", 1), ("upper", -1)):
            solution = linprog(sign * query_projection[axis], A_ub=a, b_ub=np.zeros(len(a)),
                               A_eq=query_projection[2:3], b_eq=[1.],
                               bounds=[(None, None)] * 3 + [(0, None)], method="highs",
                               options={"primal_feasibility_tolerance": 1e-9,
                                        "dual_feasibility_tolerance": 1e-9})
            record = {"solver_status": int(solution.status), "message": solution.message}
            if solution.success:
                violation = float(max(np.maximum(a @ solution.x, 0).max(),
                                      abs(query_projection[2] @ solution.x - 1),
                                      max(-solution.x[3], 0)))
                record.update(status="finite" if violation <= 1e-7 else "numerical_failure",
                              pixel=float(query_projection[axis] @ solution.x),
                              homogeneous_point_first_camera=solution.x.tolist(),
                              maximum_constraint_violation=violation)
            else:
                record["status"] = {2: "infeasible", 3: "unbounded"}.get(solution.status, "solver_failure")
            endpoints[f"{'xy'[axis]}_{name}"] = record
    return {"status": "checked", "scope": "query-positive closure bounds; not a visibility certificate",
            "coordinate_scale": scale, "support": support, "bounds": endpoints}
