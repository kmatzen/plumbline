"""Require one static point across later observations; no camera-failure attribution."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_static_point_support import load_prediction
from paper.probe_static_query_consistency import compare_query
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest
from plumbline.frustum_support import box_support


def compare_queries(fit_cameras, fit_xy, query_cameras, query_xy, k, tolerance):
    """Compare independent query witnesses with a single shared witness.

    Missing query observations contribute no constraints, but remain in coverage.
    This cannot distinguish camera error, correspondence error, object motion,
    or calibration error. Compatibility is not a quality certificate.
    """
    if len(query_cameras) != len(query_xy) or not len(query_xy):
        raise ValueError("matching nonempty query cameras and observations required")
    individual = [compare_query(fit_cameras, fit_xy, camera, xy, k, tolerance)
                  for camera, xy in zip(query_cameras, query_xy)]
    observed = [i for i, xy in enumerate(query_xy) if xy is not None]
    record = {"planned_queries": len(query_xy), "observed_queries": len(observed),
              "observed_query_indices": observed,
              "individual_counts": dict(Counter(item["status"] for item in individual)),
              "individual": individual}
    if individual[0]["status"] in ("insufficient_fitting_observations", "fitting_geometry_not_established"):
        return {**record, "status": individual[0]["status"]}
    if not observed:
        return {**record, "status": "missing_query_observations"}
    xy = np.asarray(list(fit_xy) + [query_xy[i] for i in observed])
    combined = box_support(np.concatenate([fit_cameras, np.asarray(query_cameras)[observed]]),
                           k, np.c_[xy - tolerance, xy + tolerance])
    status = {"positive_witness": "jointly_compatible",
              "solver_infeasible": "joint_contradiction"}.get(combined["status"], "numerically_unresolved")
    return {**record, "status": status, "combined": combined}


def evaluate(poses, tracks):
    if poses.shape != (90, 4, 4) or len(tracks) != 12 or any(len(t) != 90 for t in tracks):
        raise ValueError("retain all 90 poses and 12 tracks")
    cameras = np.linalg.inv(poses)
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    cases = []
    for end in (10, 30, 60):
        for tolerance in (1, 2, 4, 8):
            slots = [{"slot": slot, **compare_queries(cameras[1:end+1],
                [t[i]["xy"] for i in range(1, end+1)], cameras[end+1:],
                [t[i]["xy"] for i in range(end+1, 90)], k, tolerance)}
                for slot, t in enumerate(tracks)]
            cases.append({"fit_end": end, "query_frames": list(range(end+1, 90)),
                          "tolerance_pixels": tolerance, "slots": slots,
                          "counts": dict(Counter(s["status"] for s in slots))})
    return cases


def run(root, control_path, dage_run, da3_run, output):
    if output.exists():
        raise ValueError("preserve previous comparisons")
    control = json.loads(control_path.read_text())
    images, frames, source_hash = load_rgb(root)
    if control["source_manifest_sha256"] != source_hash:
        raise ValueError("control source mismatch")
    input_hash = hashlib.sha256(images.tobytes()).hexdigest()
    del images
    methods = {}
    for name, path, expected in (("dage", dage_run, "DAGE"), ("da3", da3_run, "DA3-LARGE-1.1")):
        poses, provenance = load_prediction(path, source_hash, frames, input_hash)
        if provenance["model"] != expected:
            raise ValueError("model manifest does not match label")
        methods[name] = {"provenance": provenance, "cases": evaluate(poses, control["tracks"])}
    result = {"scope": "joint static-point consistency, not camera DNF or independent validation",
              "inputs": "RGB tracks and saved estimated cameras; published calibration; no depth or reference poses",
              "limitations": "models saw all 90 frames; track identity and staticness unverified; missing queries unconstrained",
              "policy": "all 12 slots, prefixes 10/30/60, tolerances 1/2/4/8, every later frame retained",
              "source_manifest_sha256": source_hash, "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)),
              "dependencies_sha256": {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in (
                  Path(__file__).with_name("probe_static_query_consistency.py"),
                  Path(__file__).with_name("probe_static_point_support.py"),
                  Path(__file__).with_name("run_static_dage.py"),
                  Path(__file__).parents[1] / "src/plumbline/frustum_support.py")},
              "methods": methods}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "dage_run", "da3_run", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.control, args.dage_run, args.da3_run, args.output)
    for name, method in result["methods"].items():
        for case in method["cases"]:
            print(name, case["fit_end"], case["tolerance_pixels"], case["counts"])
