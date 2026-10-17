"""Combine image-position bounds and final-query tests for saved RGB-only predictions."""

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
from plumbline.frustum_projection_range import projection_range


def evaluate(poses, tracks):
    if poses.shape != (90, 4, 4) or len(tracks) != 12 or any(len(t) != 90 for t in tracks):
        raise ValueError("retain all 90 poses and 12 tracks")
    cameras = np.linalg.inv(poses)
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    cases = []
    for end in (10, 30, 60):
        for tolerance in (1, 2, 4, 8):
            slots = []
            for slot, track in enumerate(tracks):
                fit_xy = [track[i]["xy"] for i in range(1, end + 1)]
                fit_cameras = cameras[1:end+1]
                if any(xy is None for xy in fit_xy):
                    bounds = {"status": "insufficient_fitting_observations"}
                else:
                    xy = np.array(fit_xy)
                    bounds = projection_range(fit_cameras, k, np.c_[xy-tolerance, xy+tolerance], cameras[89], k)
                comparison = compare_query(fit_cameras, fit_xy, cameras[89], track[89]["xy"], k, tolerance)
                slots.append({"slot": slot, "prediction_bounds": bounds, "query_comparison": comparison})
            cases.append({"fit_end": end, "query_frame": 89, "tolerance_pixels": tolerance,
                          "counts": dict(Counter(s["query_comparison"]["status"] for s in slots)), "slots": slots})
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
    result = {"scope": "development consistency and observability on saved full-clip predictions; not DNF validation",
              "inputs": "RGB-derived tracks, estimated cameras, common published calibration; no depth or reference poses in evaluation",
              "limitations": "models saw all 90 RGB frames; this is not causal prefix-only model inference or independent validation",
              "policy": "all original slots, prefixes and tolerances retained; no quality threshold selected",
              "source_manifest_sha256": source_hash, "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)),
              "dependencies_sha256": {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in (
                  Path(__file__).with_name("probe_static_query_consistency.py"),
                  Path(__file__).with_name("probe_static_point_support.py"),
                  Path(__file__).with_name("run_static_dage.py"),
                  Path(__file__).parents[1] / "src/plumbline/frustum_support.py",
                  Path(__file__).parents[1] / "src/plumbline/frustum_projection_range.py")},
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
