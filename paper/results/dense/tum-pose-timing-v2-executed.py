"""Reference timestamp sensitivity, with unchanged tracks and no pose extrapolation."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from paper.analyze_dense_run import load_verified
from paper.probe_davis_anchors import evaluate
from paper.stage_rgbd_validation import sha
from scipy.spatial.transform import Rotation, Slerp

from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def interpolate(times, poses, query, max_gap=0.05):
    times = np.asarray(times, dtype=float)
    poses = np.asarray(poses, dtype=float)
    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
        or poses.shape != (len(times), 4, 4)
        or not np.isfinite(poses).all()
        or not np.isfinite(query)
        or not np.isfinite(max_gap)
        or max_gap <= 0
    ):
        raise ValueError("invalid pose interpolation input")
    right = int(np.searchsorted(times, query))
    if right < len(times) and times[right] == query:
        return poses[right].copy(), {
            "status": "exact",
            "bracket_seconds": [float(query), float(query)],
            "gap_seconds": 0.0,
        }
    if right == 0 or right == len(times):
        return None, {"status": "outside_reference_span"}
    left = right - 1
    gap = float(times[right] - times[left])
    record = {"bracket_seconds": [float(times[left]), float(times[right])], "gap_seconds": gap}
    if gap > max_gap:
        return None, {**record, "status": "reference_gap"}
    alpha = float((query - times[left]) / gap)
    result = np.eye(4)
    result[:3, 3] = (1 - alpha) * poses[left, :3, 3] + alpha * poses[right, :3, 3]
    result[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(poses[[left, right], :3, :3]))(
        [alpha]
    ).as_matrix()[0]
    return result, {**record, "status": "interpolated", "fraction": alpha}


def run(run_path, tracks_path, groundtruth_path, output, start_frame=0):
    if output.exists():
        raise ValueError("preserve previous timing analyses")
    manifest, _, reference = load_verified(run_path)
    if not isinstance(start_frame, int) or not 0 <= start_frame < len(reference) - 11:
        raise ValueError("invalid comparison start frame")
    source = json.loads(tracks_path.read_text())
    if source["source_manifest_sha256"] != sha(run_path / "manifest.json") or source[
        "reference_file_sha256"
    ] != sha(run_path / "reference.npz"):
        raise ValueError("track provenance mismatch")
    raw = np.loadtxt(groundtruth_path)
    times = raw[:, 0]
    poses = np.stack([tum_pose_to_matrix(*r[1:]) for r in raw])
    by_time = {float(t): p for t, p in zip(times, poses, strict=True)}
    associated = np.stack([by_time[float(f["reference_timestamp"])] for f in manifest["frames"]])
    associated = np.linalg.inv(associated[0]) @ associated
    if not np.allclose(associated, reference, rtol=0, atol=1e-6):
        raise ValueError("groundtruth file does not reproduce retained reference")
    interpolated, timing = [], []
    for frame, row in enumerate(manifest["frames"]):
        pose, record = interpolate(times, poses, float(row["timestamp"]))
        interpolated.append(pose)
        timing.append(
            {
                "frame": frame,
                "rgb_timestamp": row["timestamp"],
                "associated_minus_rgb_seconds": float(row["reference_timestamp"])
                - float(row["timestamp"]),
                **record,
            }
        )
    result = {
        "scope": "reference-timing development sensitivity; not camera quality or DNF labels",
        "script_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(run_path / "manifest.json"),
        "tracks_sha256": sha(tracks_path),
        "groundtruth_sha256": sha(groundtruth_path),
        "reference_file_sha256": sha(run_path / "reference.npz"),
        "kernel_sha256": sha(Path(__file__).with_name("probe_davis_anchors.py")),
        "policy": "linear translation and quaternion SLERP at RGB time; maximum bracket gap 50 ms; no extrapolation; same tracks and 11-frame fit",
        "frames": timing,
        "interpolation_counts": dict(Counter(r["status"] for r in timing)),
        "comparison_start_frame": start_frame,
        "excluded_prefix_frames": list(range(start_frame)),
        "fit_original_frames": list(range(start_frame, start_frame + 11)),
    }
    interpolated = interpolated[start_frame:]
    reference = reference[start_frame:]
    if any(p is None for p in interpolated):
        result["comparison_status"] = "unavailable_incomplete_pose_sequence"
    else:
        interpolated = np.stack(interpolated)
        interpolated = np.linalg.inv(interpolated[0]) @ interpolated
        size = np.array(manifest["shape"][1:3][::-1])
        with np.load(run_path / "reference.npz") as archive:
            k = archive["intrinsics"][start_frame:].astype(float)
        k[:, :2, 2] += 0.5
        k[:, 0] /= size[0]
        k[:, 1] /= size[1]
        cases = {
            name: [evaluate(t[start_frame:], p, k, size) for t in source["tracks"]]
            for name, p in [
                ("associated_reference", reference),
                ("interpolated_reference", interpolated),
            ]
        }
        common = []
        for slot in range(len(source["tracks"])):
            for frame in range(11, len(reference)):
                errors = {
                    name: anchors[slot]["rows"][frame]["distance_pixels"]
                    for name, anchors in cases.items()
                }
                if all(value is not None for value in errors.values()):
                    common.append({"slot": slot, "frame": frame + start_frame, "pixels": errors})
        result.update(
            comparison_status="complete",
            cases=cases,
            common_observations=common,
            common_median_pixels={
                name: float(np.median([r["pixels"][name] for r in common])) if common else None
                for name in cases
            },
        )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("frames", "cases", "common_observations")
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "tracks", "groundtruth", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    run(args.run, args.tracks, args.groundtruth, args.output, args.start_frame)
