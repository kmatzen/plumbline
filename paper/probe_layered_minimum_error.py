"""Classical bounded-reprojection baseline on layered oracle controls, not DNF."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from paper.probe_evidence_sensitivity import alternatives
from paper.render_layered_camera_control import K, anchors, camera_pose, project_anchors
from paper.stage_static_rgbd_control import digest
from plumbline.frustum_support import box_support


def minimum_box_error(cameras, xy, *, upper=64., resolution=.01):
    """Numerical bracket; unresolved solves never advance the lower bound."""
    if not (np.isfinite(upper) and np.isfinite(resolution) and 0 < resolution < upper):
        raise ValueError("finite positive ordered search limits required")
    xy = np.asarray(xy, dtype=float)
    calls = []

    def solve(tolerance):
        result = box_support(cameras, K, np.c_[xy-tolerance, xy+tolerance])
        calls.append(dict(tolerance=tolerance, result=result))
        return result

    high = solve(upper)
    if high["status"] != "positive_witness":
        return dict(status="upper_bound_not_established", solves=calls)
    low = 0.
    while upper-low > resolution:
        middle = (low+upper)/2
        result = solve(middle)
        if result["status"] == "positive_witness":
            upper, high = middle, result
        elif result["status"] == "solver_infeasible":
            low = middle
        else:
            return dict(status="numerically_unresolved", lower=low, upper=upper, solves=calls)
    return dict(status="numerical_bracket", lower=low, upper=upper,
                upper_point_world=high["point_world"], solves=calls)


def run(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    root = Path(__file__).parents[1]
    for name, expected in data["dependencies_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("control dependency changed")
    points, ids = anchors()
    reference = [project_anchors(points, ids, t) for t in range(24)]
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    cases = []
    for change, changed in alternatives(cameras):
        slots = []
        for slot in range(12):
            indices = [t for t in range(24) if reference[t][1][slot]]
            xy = np.array([reference[t][0][slot] for t in indices])
            result = minimum_box_error(changed[indices], xy)
            if result["status"] == "numerical_bracket":
                point = np.r_[result["upper_point_world"], 1.]
                camera = (changed[indices] @ point)[:, :3]
                projected = camera @ K.T
                residual = float(np.abs(projected[:, :2]/projected[:, 2:]-xy).max())
                if not (camera[:, 2] > 0).all() or residual > result["upper"]+1e-5:
                    raise AssertionError("upper witness failed independent projection check")
                result["audited_upper_witness_error"] = residual
            slots.append(dict(slot=slot, visible_indices=indices, **result))
        cases.append(dict(change=change, slots=slots))
    return dict(scope="oracle classical minimum coordinate-reprojection baseline; not novel or DNF",
                protocol="all 14 fixed camera variants; all 12 unique geometric tracks; all visible frames; [0,64] pixel search to .01; no parameter selection",
                limitations="numerical infeasibility not formal proof; refitted structure; oracle visibility; single scene/path; no independent texture replicates",
                source_sha256=digest(source), script_sha256=digest(Path(__file__)), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior analysis")
    result = run(args.source)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    for case in result["cases"]:
        print(case["change"], [(s["status"], s.get("lower"), s.get("upper")) for s in case["slots"]])
