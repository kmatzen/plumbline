"""Early-image-only depth observability diagnostic on the static control."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.frustum_depth_range import depth_range


def run(source, control_path, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    manifest = json.loads(source.read_text())
    control = json.loads(control_path.read_text())
    if control["source_manifest_sha256"] != digest(source):
        raise ValueError("source mismatch")
    frames = list(range(1, 11))
    poses = np.stack([tum_pose_to_matrix(*map(float, manifest["frames"][i]["reference"]["matches"][0]["values"]))
                      for i in frames])
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    rows = []
    for tolerance in (1, 2, 4, 8):
        for slot, track in enumerate(control["tracks"]):
            if any(track[i]["xy"] is None for i in frames):
                result = {"status": "insufficient_early_observations"}
            else:
                xy = np.array([track[i]["xy"] for i in frames])
                result = depth_range(np.linalg.inv(poses), k, np.c_[xy-tolerance, xy+tolerance])
            rows.append({"slot": slot, "tolerance_pixels": tolerance, **result})
    result = {"scope": "reference-camera development diagnostic; no sensor depth or later image coordinates used",
              "fit_original_frames": frames, "policy": "all slots and tolerances retained; no admission threshold selected",
              "source_sha256": digest(source), "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)),
              "range_kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_depth_range.py"),
              "support_kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
              "rows": rows}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "control", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print("Evaluated", len(run(args.source, args.control, args.output)["rows"]), "slot/tolerance pairs")
