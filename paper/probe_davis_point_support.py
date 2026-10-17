"""Fixed-tolerance point-support development probe with native estimated cameras."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from paper.build_reference_free_review import digest
from paper.probe_point_support import track_support


def pixel_calibration(normalized, shape):
    k = np.array(normalized, dtype=float, copy=True)
    height, width = shape[1:3]
    k[:, 0] *= width
    k[:, 1] *= height
    k[:, :2, 2] -= 0.5
    return k


def run(root, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    result = {
        "status": "complete",
        "scope": "DAVIS training development; not held-out validation, pose accuracy or DNF",
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": {
            name: digest(root / name)
            for name in ("paper/probe_point_support.py", "src/plumbline/frustum_support.py")
        },
        "policy": "same 1,2,4,8 axis-wise pixel tolerances as TUM; native predicted intrinsics; LK tracks; annotation-zero admission, not certified staticness",
        "cases": [],
    }
    for clip in ("bear", "bmx-bumps", "boat"):
        source_path = root / f"paper/results/davis/grid-anchors-v1/{clip}.json"
        source = json.loads(source_path.read_text())
        tracks = [
            [{"xy": r["xy"] if r["annotation_label"] == 0 else None} for r in track]
            for track in source["tracks"]
        ]
        excluded = [
            [i for i, r in enumerate(track) if r["xy"] is not None and r["annotation_label"] != 0]
            for track in source["tracks"]
        ]
        for method in source["methods"]:
            run_path = root / method["run"]
            manifest = json.loads((run_path / "manifest.json").read_text())
            if digest(run_path / "manifest.json") != method["manifest_sha256"]:
                raise ValueError("source manifest changed")
            if digest(run_path / "prediction.npz") != manifest["prediction_file_sha256"]:
                raise ValueError("prediction changed")
            if manifest["split"] != "train":
                raise ValueError("development split required")
            with np.load(run_path / "prediction.npz") as archive:
                poses = archive["extrinsics"]
                k = pixel_calibration(archive["intrinsics_normalized"], manifest["shape"])
            for tolerance in (1, 2, 4, 8):
                anchors = [track_support(t, poses, k, tolerance) for t in tracks]
                counts = {
                    phase: dict(
                        Counter(
                            a["status"]
                            if a["status"] != "checked"
                            else "no_later_observations"
                            if phase == "all_observed" and a["later_observed_frames"] == 0
                            else a[phase]["status"]
                            for a in anchors
                        )
                    )
                    for phase in ("early", "all_observed")
                }
                case = {
                    "clip": clip,
                    "run": method["run"],
                    "source_sha256": digest(source_path),
                    "manifest_sha256": method["manifest_sha256"],
                    "tolerance_pixels": tolerance,
                    "mask_excluded_frame_indices": excluded,
                    "anchors": anchors,
                    "counts": counts,
                }
                result["cases"].append(case)
                print(clip, run_path.name, tolerance, counts)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
