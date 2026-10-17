"""Validate fixed-sample reference associations without substituting frames."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.stage_static_rgbd_control import digest, nearest, parse_index
from plumbline.conventions import assert_valid_extrinsics
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def window_availability(poses, fit_frames, query_frame):
    missing = [i for i in fit_frames if poses[i] is None]
    query_missing = poses[query_frame] is None
    return {"status": "available" if not missing and not query_missing else "reference_unavailable",
            "unavailable_fit_frames": missing, "query_reference_unavailable": query_missing}


def load_reference(root):
    source = root / "staging-manifest.json"
    manifest = json.loads(source.read_text())
    reference = root / "groundtruth.txt"
    if digest(reference) != manifest["indices"]["groundtruth.txt"]:
        raise ValueError("reference index hash mismatch")
    index = parse_index(reference.read_bytes(), 8)
    poses, records = [], []
    for i, frame in enumerate(manifest["frames"]):
        if frame["frame"] != i:
            raise ValueError("frame order mismatch")
        expected = nearest(frame["timestamp"], index)
        if frame["reference"] != expected:
            raise ValueError("reference association mismatch")
        pose = None
        status = expected["status"]
        if status == "associated":
            values = np.array(expected["matches"][0]["values"], float)
            if not np.isfinite(values).all() or np.linalg.norm(values[3:]) == 0:
                status = "invalid_reference_pose"
            else:
                pose = tum_pose_to_matrix(*values)
                try:
                    assert_valid_extrinsics(pose[None])
                except AssertionError:
                    pose = None
                    status = "invalid_reference_pose"
        poses.append(pose)
        records.append({"frame": i, "status": status, "association": expected,
                        "camera_to_world": pose.tolist() if pose is not None else None})
    return poses, {"source_manifest_sha256": digest(source), "reference_index_sha256": digest(reference),
                   "rows": records, "counts": dict(Counter(r["status"] for r in records))}


def run(root, output):
    if output.exists():
        raise ValueError("preserve previous audits")
    poses, result = load_reference(root)
    if len(poses) != 90:
        raise ValueError("expected fixed 90-frame sample")
    result.update(scope="reference availability only; not camera-quality or staticness labels",
                  policy="nearest within 20 ms; preserve missing/ambiguous rows; no interpolation or resampling",
                  script_sha256=digest(Path(__file__)),
                  windows=[{"fit_end": end, "query_frame": 89,
                            **window_availability(poses, range(1, end+1), 89)} for end in (10, 30, 60)])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(result["counts"], result["windows"])
