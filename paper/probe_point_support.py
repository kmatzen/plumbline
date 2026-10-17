"""Point-support sensitivity without treating one triangulated point as truth.

Full-track support is a joint consistency check, not a held-out prediction score.
It assumes a single static, correctly tracked point and calibrated cameras.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from paper.analyze_dense_run import load_verified
from paper.build_reference_free_review import digest
from paper.probe_tum_anchors import camera_variants

from plumbline.frustum_support import box_support


def track_support(track, poses, calibration, tolerance):
    if len(track) < 12 or any(r["xy"] is None for r in track[:11]):
        return {"status": "insufficient_early_observations"}
    indices = np.array([i for i, r in enumerate(track) if r["xy"] is not None])
    xy = np.array([track[i]["xy"] for i in indices])
    boxes = np.c_[xy - tolerance, xy + tolerance]
    world_to_camera = np.linalg.inv(poses.astype(float))
    early = box_support(world_to_camera[:11], calibration[:11], boxes[:11])
    full = box_support(world_to_camera[indices], calibration[indices], boxes)
    return {
        "status": "checked",
        "observed_indices": indices.tolist(),
        "later_observed_frames": int(np.sum(indices >= 11)),
        "missing_frames": len(track) - len(indices),
        "early": early,
        "all_observed": full,
    }


def run(run_path, probe_path, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    source = json.loads(probe_path.read_text())
    manifest, prediction, reference = load_verified(run_path)
    if source["source_manifest_sha256"] != digest(run_path / "manifest.json"):
        raise ValueError("track provenance mismatch")
    if source["reference_file_sha256"] != digest(run_path / "reference.npz"):
        raise ValueError("reference calibration changed")
    with np.load(run_path / "reference.npz") as archive:
        k = archive["intrinsics"].astype(float)
    result = {
        "status": "complete",
        "sequence": manifest["sequence"],
        "scope": "joint point-support diagnostic, not camera DNF or held-out prediction accuracy",
        "assumptions": "static correctly tracked points, valid calibration and synchronization; none established by solver",
        "tolerance_policy": "fixed 1,2,4,8 pixel axis-wise boxes; no best tolerance selected",
        "probe_sha256": digest(probe_path),
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
        "cases": [],
    }
    for name, poses in camera_variants(reference, prediction).items():
        for tolerance in (1, 2, 4, 8):
            anchors = [track_support(t, poses, k, tolerance) for t in source["tracks"]]
            counts = {
                phase: dict(
                    Counter(
                        a[phase]["status"] if a["status"] == "checked" else a["status"]
                        for a in anchors
                    )
                )
                for phase in ("early", "all_observed")
            }
            result["cases"].append(
                {
                    "method": name,
                    "tolerance_pixels": tolerance,
                    "anchors": anchors,
                    "counts": counts,
                }
            )
            print(name, tolerance, counts)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "probe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.run, args.probe, args.output)
