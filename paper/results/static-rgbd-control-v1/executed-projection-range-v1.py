"""Later-image bounds from fixed early observations; no later tracks used in fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.frustum_projection_range import projection_range


def run(source, control_path, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    manifest = json.loads(source.read_text())
    control = json.loads(control_path.read_text())
    if control["source_manifest_sha256"] != digest(source):
        raise ValueError("source mismatch")
    fit = list(range(1, 11))
    queries = (11, 45, 89)
    cameras = {i: np.linalg.inv(tum_pose_to_matrix(*map(float,
                    manifest["frames"][i]["reference"]["matches"][0]["values"])))
               for i in (*fit, *queries)}
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    rows = []
    for tolerance in (1, 2, 4, 8):
        for slot, track in enumerate(control["tracks"]):
            for frame in queries:
                if any(track[i]["xy"] is None for i in fit):
                    result = {"status": "insufficient_early_observations"}
                else:
                    xy = np.array([track[i]["xy"] for i in fit])
                    result = projection_range(np.stack([cameras[i] for i in fit]), k,
                                              np.c_[xy-tolerance, xy+tolerance], cameras[frame], k)
                rows.append({"slot": slot, "tolerance_pixels": tolerance, "query_frame": frame, **result})
    result = {"scope": "reference-camera development diagnostic; no depth or query-image observations used",
              "fit_original_frames": fit, "query_frames": queries,
              "limitations": "conditional on positive query depth; separate coordinate extrema need not form a feasible corner",
              "policy": "no image-bound clipping; retain unbounded, infeasible, and insufficient results",
              "source_sha256": digest(source), "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)),
              "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_projection_range.py"),
              "support_kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
              "rows": rows}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "control", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print("Evaluated", len(run(args.source, args.control, args.output)["rows"]), "queries")
