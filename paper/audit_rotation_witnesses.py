"""Independently reproject retained witnesses and expose observation censoring."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.probe_static_rotation_response import perturb
from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def reproject(point, poses, xy):
    camera = np.linalg.inv(np.asarray(poses)) @ np.r_[point, 1.0]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("witness must be finite and in front of every camera")
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    pixels = camera[:, :3] @ k.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    return np.max(np.abs(pixels - np.asarray(xy)), axis=1)


def run(source, control_path, response_path, output):
    if output.exists():
        raise ValueError("preserve previous audits")
    manifest = json.loads(source.read_text())
    control = json.loads(control_path.read_text())
    response = json.loads(response_path.read_text())
    if response["source_manifest_sha256"] != digest(source) or response["control_sha256"] != digest(control_path):
        raise ValueError("source evidence mismatch")
    poses = [None] + [tum_pose_to_matrix(*map(float, r["reference"]["matches"][0]["values"]))
                      for r in manifest["frames"][1:]]
    records = []
    for case in response["cases"]:
        cameras = perturb(poses, case["yaw_degrees"], case["schedule"])
        slots = []
        for slot, anchor in enumerate(case["anchors"]):
            if anchor["status"] != "checked":
                slots.append({"slot": slot, "status": anchor["status"]})
                continue
            frames = anchor["observed_original_frames"]
            expected = [i for i in range(1, 90) if control["tracks"][slot][i]["xy"] is not None]
            if frames != expected:
                raise ValueError("observation mask changed")
            last = max(frames)
            status = anchor["all_observed"]["status"]
            row = {"slot": slot, "status": status, "last_observed_frame": last,
                   "observes_final_frame": last == 89,
                   "later_observations": sum(i >= 11 for i in frames),
                   "last_observed_yaw_degrees": case["yaw_degrees"] *
                       (1 if case["schedule"] == "step" else (last - 10) / 79)}
            if status == "positive_witness":
                residuals = reproject(anchor["all_observed"]["point_world"],
                                     [cameras[i] for i in frames],
                                     [control["tracks"][slot][i]["xy"] for i in frames])
                if residuals.max() > case["tolerance_pixels"] + 1e-5:
                    raise ValueError("retained witness violates pixel bounds")
                row["maximum_axis_residual_pixels"] = float(residuals.max())
                row["last_observed_axis_residual_pixels"] = float(residuals[-1])
            slots.append(row)
        records.append({k: case[k] for k in ("schedule", "yaw_degrees", "tolerance_pixels")} | {
            "slots": slots,
            "final_frame_observed_slots": sum(s.get("observes_final_frame", False) for s in slots),
            "positive_witnesses_observing_final_frame": sum(s.get("observes_final_frame", False) and
                s["status"] == "positive_witness" for s in slots)})
    result = {"scope": "post-hoc witness verification and censoring audit; not functional labels",
              "source_manifest_sha256": digest(source), "control_sha256": digest(control_path),
              "response_sha256": digest(response_path), "script_sha256": digest(Path(__file__)),
              "rotation_script_sha256": digest(Path(__file__).with_name("probe_static_rotation_response.py")),
              "cases": records}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "control", "response", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.source, args.control, args.response, args.output)
    print("Verified", len(result["cases"]), "cases")
