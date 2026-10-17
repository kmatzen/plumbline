"""Conditional final-frame consistency, not a trajectory-level DNF label."""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.frustum_support import box_support


def compare_query(fit_cameras, fit_xy, query_camera, query_xy, k, tolerance):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive tolerance required")
    if len(fit_cameras) != len(fit_xy) or len(fit_xy) < 2:
        raise ValueError("matching fitting cameras and observations required")
    if any(xy is None for xy in fit_xy):
        return {"status": "insufficient_fitting_observations"}
    xy = np.array(fit_xy)
    boxes = np.c_[xy - tolerance, xy + tolerance]
    early = box_support(fit_cameras, k, boxes)
    if early["status"] != "positive_witness":
        return {"status": "fitting_geometry_not_established", "early": early}
    if query_xy is None:
        return {"status": "missing_query_observation", "early": early}
    query_xy = np.asarray(query_xy)
    combined = box_support(np.concatenate([fit_cameras, query_camera[None]]), k,
                           np.vstack([boxes, np.r_[query_xy-tolerance, query_xy+tolerance]]))
    status = {"positive_witness": "compatible", "solver_infeasible": "query_contradiction"}.get(
        combined["status"], "numerically_unresolved")
    return {"status": status, "early": early, "combined": combined}


def run(source, control_path, output):
    if output.exists():
        raise ValueError("preserve previous comparisons")
    manifest = json.loads(source.read_text())
    control = json.loads(control_path.read_text())
    if control["source_manifest_sha256"] != digest(source):
        raise ValueError("source mismatch")
    poses = {i: tum_pose_to_matrix(*map(float, manifest["frames"][i]["reference"]["matches"][0]["values"]))
             for i in range(1, 90)}
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    cases = []
    for end in (10, 30, 60):
        frames = list(range(1, end + 1))
        cameras = np.linalg.inv(np.stack([poses[i] for i in frames]))
        for tolerance in (1, 2, 4, 8):
            for angle in (-5, -1, -.25, 0, .25, 1, 5):
                query = poses[89].copy()
                query[:3, :3] = query[:3, :3] @ Rotation.from_euler("y", angle, degrees=True).as_matrix()
                slots = [{"slot": slot, **compare_query(cameras, [track[i]["xy"] for i in frames],
                    np.linalg.inv(query), track[89]["xy"], k, tolerance)}
                    for slot, track in enumerate(control["tracks"])]
                cases.append({"fit_end": end, "query_frame": 89, "yaw_degrees": angle,
                              "tolerance_pixels": tolerance, "slots": slots,
                              "counts": dict(Counter(s["status"] for s in slots))})
    result = {"scope": "single final-frame conditional consistency; not camera DNF or independent validation",
              "policy": "fit reference poses unchanged; only query rotation perturbed; all slots/settings retained",
              "limitations": "a compatible point may differ between queries; one endpoint cannot distinguish step from ramp",
              "source_sha256": digest(source), "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)),
              "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
              "cases": cases}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "control", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print("Evaluated", len(run(args.source, args.control, args.output)["cases"]), "cases")
