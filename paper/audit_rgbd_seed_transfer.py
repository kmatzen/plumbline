"""Reference/depth transfer versus RGB tracks; diagnostic, not failure labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.rgbd_track_audit import world_observation
from paper.probe_tum_pose_timing import interpolate
from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def project(point, pose, k):
    xyz = np.linalg.solve(pose, np.r_[point, 1.])[:3]
    if xyz[2] <= 0:
        return None
    pixel = k @ xyz
    return (pixel[:2]/pixel[2]).tolist()


def run(root, tracks_path, output, pose_timing="nearest"):
    if output.exists():
        raise ValueError("preserve prior audit")
    manifest_path = root / "staging-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence = json.loads(tracks_path.read_text())
    if evidence["source_manifest_sha256"] != digest(manifest_path):
        raise ValueError("track/source mismatch")
    frames, tracks = manifest["frames"], evidence["tracks"]
    if len(frames) != 90 or len(tracks) != 12 or any(len(t) != 90 for t in tracks):
        raise ValueError("fixed 90-frame, 12-slot evidence required")
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    if pose_timing not in {"nearest", "interpolated"}:
        raise ValueError("unknown pose timing policy")
    reference_path = root / "groundtruth.txt"
    if digest(reference_path) != manifest["indices"]["groundtruth.txt"]:
        raise ValueError("reference index changed")
    timing = []
    if pose_timing == "nearest":
        poses = [tum_pose_to_matrix(*map(float, f["reference"]["matches"][0]["values"]))
                 if f["reference"]["status"] == "associated" else None for f in frames]
    else:
        raw = np.loadtxt(reference_path)
        reference_poses = np.stack([tum_pose_to_matrix(*r[1:]) for r in raw])
        poses = []
        for f in frames:
            pose, record = interpolate(raw[:, 0], reference_poses, f["timestamp"])
            poses.append(pose)
            timing.append(dict(frame=f["frame"], **record))
    source_frame = 1  # Fixed before inspecting residuals; frame zero lacks reference in the near control.
    d = frames[source_frame]["depth"]
    depth = None
    if d["status"] == "associated":
        path = root / d["path"]
        if digest(path) != d["file_sha256"]:
            raise ValueError("source depth changed")
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    slots = []
    for slot, track in enumerate(tracks):
        if depth is None or poses[source_frame] is None:
            seed = dict(status="missing_source_reference_or_depth")
        else:
            seed = world_observation(track[source_frame]["xy"], depth, k, poses[source_frame])
        rows = []
        for frame in range(11, 90):
            row = dict(frame=frame, observed_xy=track[frame]["xy"], projected_xy=None, error_pixels=None)
            if seed["status"] != "measured":
                row["status"] = "source_unavailable"
            elif poses[frame] is None:
                row["status"] = "reference_unavailable"
            else:
                row["projected_xy"] = project(seed["world_point_m"], poses[frame], k)
                if row["projected_xy"] is None:
                    row["status"] = "behind_reference_camera"
                elif row["observed_xy"] is None:
                    row["status"] = "image_track_unavailable"
                else:
                    row["status"] = "compared"
                    row["error_pixels"] = float(np.linalg.norm(np.array(row["projected_xy"])-row["observed_xy"]))
            rows.append(row)
        slots.append(dict(slot=slot, source=seed, rows=rows))
    errors = [r["error_pixels"] for s in slots for r in s["rows"] if r["status"] == "compared"]
    result = dict(scope="reference/depth transfer agreement with RGB LK; neither source is exact pixel ground truth",
                  source_frame=source_frame, query_frames=list(range(11, 90)), planned_observations=12*79,
                  policy="fixed frame-1 seed; nearest-pixel depth; all slots retained; no replacement seed or visibility filtering",
                  pose_timing=pose_timing, interpolation_records=timing,
                  interpolation_policy="linear translation and quaternion SLERP; maximum bracket gap 50ms; no extrapolation or ambiguous brackets",
                  reference_index_sha256=digest(reference_path),
                  timing_script_sha256=digest(Path(__file__).with_name("probe_tum_pose_timing.py")),
                  limitations="residuals mix tracking, calibration, timing, depth, and scene effects; target visibility is unverified; no camera failure labels",
                  source_sha256=digest(manifest_path), tracks_sha256=digest(tracks_path), script_sha256=digest(Path(__file__)),
                  dependency_sha256=digest(Path(__file__).with_name("rgbd_track_audit.py")),
                  slots=slots, counts=dict(Counter(r["status"] for s in slots for r in s["rows"])),
                  median_compared_error_pixels=float(np.median(errors)) if errors else None,
                  compared_within_pixels={str(t): sum(e <= t for e in errors) for t in (1, 2, 4, 8)})
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tracks", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--pose-timing", choices=("nearest", "interpolated"), default="nearest")
    args = parser.parse_args()
    result = run(args.root, args.tracks, args.output, args.pose_timing)
    print({k: result[k] for k in ("counts", "median_compared_error_pixels", "compared_within_pixels")})
