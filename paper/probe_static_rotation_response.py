"""Controlled rotation response; diagnostic, not a calibrated failure metric."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.probe_static_point_support import run as baseline_run, support
from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def perturb(poses, angle, schedule):
    if schedule not in {"step", "ramp"} or not np.isfinite(angle):
        raise ValueError("finite angle and step/ramp schedule required")
    if len(poses) != 90 or poses[0] is not None:
        raise ValueError("expected fixed 90-frame reference window")
    result = [None] + [p.copy() for p in poses[1:]]
    for frame in range(11, 90):
        fraction = 1 if schedule == "step" else (frame - 10) / 79
        result[frame][:3, :3] = poses[frame][:3, :3] @ Rotation.from_euler(
            "y", angle * fraction, degrees=True).as_matrix()
    return result


def run(root, control_path, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    output.mkdir()
    # This validates the source reference index and associations against the
    # control before constructing any new camera variants.
    baseline = baseline_run(root, control_path, output / "baseline.json")
    manifest = json.loads((root / "staging-manifest.json").read_text())
    poses = [None] + [tum_pose_to_matrix(*map(float, row["reference"]["matches"][0]["values"]))
                      for row in manifest["frames"][1:]]
    tracks = json.loads(control_path.read_text())["tracks"]
    calibration = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    cases = []
    for schedule in ("step", "ramp"):
        for angle in (-5, -1, -.25, 0, .25, 1, 5):
            cameras = perturb(poses, angle, schedule)
            for tolerance in (1, 2, 4, 8):
                anchors = [support(t, cameras, calibration, tolerance) for t in tracks]
                counts = dict(Counter(a["all_observed"]["status"] if a["status"] == "checked"
                                      else a["status"] for a in anchors))
                cases.append({"schedule": schedule, "yaw_degrees": angle,
                              "tolerance_pixels": tolerance, "counts": counts, "anchors": anchors})
    result = {
        "scope": "controlled local-yaw response on one development sequence; not functional DNF validation",
        "policy": "all signs, schedules, angles and tolerances retained; no decision threshold selected",
        "schedule": "frames 1–10 unchanged; step applies full yaw at 11; ramp reaches full yaw at 89",
        "limitations": "full-track joint consistency, not held-out prediction; fallible tracks and nearest reference timing",
        "script_sha256": digest(Path(__file__)),
        "support_script_sha256": baseline["script_sha256"],
        "kernel_sha256": baseline["kernel_sha256"],
        "control_sha256": baseline["control_sha256"],
        "source_manifest_sha256": baseline["source_manifest_sha256"],
        "baseline_sha256": digest(output / "baseline.json"), "cases": cases,
    }
    (output / "response.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    for row in run(args.root, args.control, args.output)["cases"]:
        print(row["schedule"], row["yaw_degrees"], row["tolerance_pixels"], row["counts"])
