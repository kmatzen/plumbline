"""Available-observation sensitivity arm; feasibility is not observability."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_consistency_attribution import yaw_suffix
from paper.probe_joint_query_consistency import compare_queries
from paper.probe_rendered_consistency import tracks_for
from paper.render_camera_control import K, camera_pose
from paper.stage_static_rgbd_control import digest


def compare_available(fit_cameras, fit_xy, query_cameras, query_xy, k, tolerance):
    if len(fit_cameras) != len(fit_xy):
        raise ValueError("matching fitting cameras and observations required")
    if len(query_cameras) != len(query_xy) or not len(query_xy):
        raise ValueError("matching nonempty query cameras and observations required")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive tolerance required")
    observed = [i for i, xy in enumerate(fit_xy) if xy is not None]
    record = {"planned_fitting_observations": len(fit_xy),
              "observed_fitting_indices": observed, "observed_fitting_observations": len(observed)}
    if len(observed) < 2:
        return {**record, "status": "insufficient_fitting_observations",
                "planned_queries": len(query_xy),
                "observed_queries": sum(xy is not None for xy in query_xy)}
    calibration = np.asarray(k)
    if calibration.shape != (3, 3):
        raise ValueError("this diagnostic requires shared calibration")
    return {**record, **compare_queries(np.asarray(fit_cameras)[observed],
            [fit_xy[i] for i in observed], query_cameras, query_xy, k, tolerance)}


def run(source, previous, output):
    if output.exists():
        raise ValueError("preserve prior results")
    data = json.loads(source.read_text())
    with gzip.open(previous, "rt") as stream:
        old = json.load(stream)
    if old["source_sha256"] != digest(source):
        raise ValueError("previous comparison used different source")
    root = Path(__file__).parents[1]
    for name, expected in old["dependencies_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("previous comparison implementation changed")
    reference = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    cases, totals, transitions = [], {}, {}
    lookup = {(c["seed"], c["occluder"], c["track_arm"], c["suffix_yaw_degrees"]): c for c in old["cases"]}
    for case in data["cases"]:
        for arm in ("observed", "exact_visible"):
            tracks = tracks_for(case, arm)
            for angle in (0, 1, 5):
                key = (case["seed"], case["occluder"], arm, angle)
                prior = lookup.pop(key)
                cameras = yaw_suffix(reference, angle)
                slots = [{"slot": slot, **compare_available(cameras[:12], track[:12],
                          cameras[12:], track[12:], K, 2)} for slot, track in enumerate(tracks)]
                label = f"occluder={case['occluder']},arm={arm},yaw={angle}"
                totals.setdefault(label, Counter()).update(s["status"] for s in slots)
                transitions.setdefault(label, Counter()).update(
                    f"{before['status']} -> {after['status']}"
                    for before, after in zip(prior["slots"], slots, strict=True))
                cases.append({"seed": case["seed"], "occluder": case["occluder"],
                              "track_arm": arm, "suffix_yaw_degrees": angle, "slots": slots})
    if lookup:
        raise ValueError("previous cases unmatched")
    result = {"scope": "missing-aware fitting sensitivity, not camera DNF or observability validation",
              "policy": "same inputs and tolerances as prior arm; retain all slots; fit all available prefix observations if at least two; no gap filling or replacement of image coordinates; joint feasibility still uses observed query constraints",
              "limitations": "two views and a positive witness do not establish triangulation precision; missing data weakens constraints; exact arm is oracle only",
              "source_sha256": digest(source), "previous_sha256": digest(previous),
              "script_sha256": digest(Path(__file__)),
              "dependencies_sha256": old["dependencies_sha256"],
              "totals": {k: dict(v) for k, v in totals.items()},
              "transitions": {k: dict(v) for k, v in transitions.items()}, "cases": cases}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    return result["totals"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "previous", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.previous, args.output), indent=2))
