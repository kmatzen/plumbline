"""Sensor-depth oracle diagnostic, explicitly not reference-free evaluation."""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

from paper.probe_static_rotation_response import perturb
from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.frustum_support import box_support


def run(source, control_path, response_path, output):
    if output.exists():
        raise ValueError("preserve previous results")
    manifest = json.loads(source.read_text())
    control = json.loads(control_path.read_text())
    response = json.loads(response_path.read_text())
    if response["source_manifest_sha256"] != digest(source) or response["control_sha256"] != digest(control_path):
        raise ValueError("source evidence mismatch")
    poses = [None] + [tum_pose_to_matrix(*map(float, r["reference"]["matches"][0]["values"]))
                      for r in manifest["frames"][1:]]
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    cases = []
    for case in response["cases"]:
        cameras = perturb(poses, case["yaw_degrees"], case["schedule"])
        for fraction in (.1, .25):
            slots = []
            for slot, anchor in enumerate(case["anchors"]):
                measured = control["depth_tracks"][slot]["rows"][1]
                if anchor["status"] != "checked":
                    slots.append({"slot": slot, "status": anchor["status"]})
                    continue
                if measured["status"] != "measured":
                    slots.append({"slot": slot, "status": "missing_frame_one_depth"})
                    continue
                frames = anchor["observed_original_frames"]
                expected = [i for i in range(1, 90) if control["tracks"][slot][i]["xy"] is not None]
                if frames != expected or frames[0] != 1:
                    raise ValueError("observation mask mismatch")
                xy = np.array([control["tracks"][slot][i]["xy"] for i in frames])
                tolerance = case["tolerance_pixels"]
                interval = measured["depth_m"] * np.array([1 - fraction, 1 + fraction])
                result = box_support(np.linalg.inv(np.stack([cameras[i] for i in frames])),
                                     k, np.c_[xy - tolerance, xy + tolerance],
                                     first_depth_interval=interval)
                slots.append({"slot": slot, "sensor_depth_m": measured["depth_m"], **result})
            cases.append({key: case[key] for key in ("schedule", "yaw_degrees", "tolerance_pixels")} |
                         {"depth_fraction": fraction, "slots": slots,
                          "counts": dict(Counter(s["status"] for s in slots))})
    result = {"scope": "sensor-depth oracle; not a reference-free metric or calibrated sensor confidence interval",
              "policy": "frame one depth only; missing values never replaced; both 10% and 25% bands retained",
              "source_sha256": digest(source), "control_sha256": digest(control_path),
              "response_sha256": digest(response_path), "script_sha256": digest(Path(__file__)),
              "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
              "cases": cases}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "control", "response", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.source, args.control, args.response, args.output)
    print("Completed", len(result["cases"]), "cases")
