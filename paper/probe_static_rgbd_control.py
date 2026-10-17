"""Reference-camera development control, not a functional-DNF benchmark."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from paper.probe_davis_anchors import evaluate, grid_features, track_feature
from paper.rgbd_track_audit import observed_spread, world_observation
from paper.stage_static_rgbd_control import digest, nearest, parse_index
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def reference_cases(poses):
    if len(poses) != 90 or poses[0] is not None:
        raise ValueError("expected fixed control with unavailable reference frame zero")
    if any(p is None or p.shape != (4, 4) or not np.isfinite(p).all() for p in poses[1:]):
        raise ValueError("incomplete reference window")
    rotated = [None, *[p.copy() for p in poses[1:]]]
    yaw = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    for frame in range(11, 90):
        rotated[frame][:3, :3] = rotated[frame][:3, :3] @ yaw
    return {"reference": poses, "reference_later_yaw_5deg": rotated}


def evaluate_window(track, poses, intrinsics, size):
    if len(track) != 90 or len(poses) != 90 or len(intrinsics) != 90:
        raise ValueError("expected all 90 original frames")
    result = evaluate(track[1:], np.stack(poses[1:]), intrinsics[1:], size, fit_count=10)
    result["rows"] = [{"frame": 0, "fit_frame": True,
                       "status": "comparison_reference_unavailable",
                       "predicted_xy": None, "distance_pixels": None}] + [
        {**r, "frame": r["frame"] + 1} for r in result["rows"]
    ]
    return result


def run(root, output):
    if output.exists():
        raise ValueError("preserve previous controls")
    manifest = json.loads((root / "staging-manifest.json").read_text())
    for name, checksum in manifest["indices"].items():
        if digest(root / name) != checksum:
            raise ValueError("index hash mismatch")
    rgb_rows = sorted(parse_index((root / "rgb.txt").read_bytes(), 2), key=lambda r: r["timestamp"])[::3][:90]
    indices = {"reference": parse_index((root / "groundtruth.txt").read_bytes(), 8),
               "depth": parse_index((root / "depth.txt").read_bytes(), 2)}
    gray, depths, poses = [], [], []
    for i, (frame, source) in enumerate(zip(manifest["frames"], rgb_rows, strict=True)):
        if (frame["frame"] != i or frame["rgb_index"] != 3 * i
                or frame["path"] != source["values"][0] or frame["timestamp"] != source["timestamp"]):
            raise ValueError("fixed RGB selection mismatch")
        if digest(root / frame["path"]) != frame["file_sha256"]:
            raise ValueError("RGB hash mismatch")
        image = cv2.imread(str(root / frame["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (480, 640) or image.dtype != np.uint8:
            raise ValueError("invalid native RGB image")
        gray.append(image)
        for kind, index in indices.items():
            expected = nearest(frame["timestamp"], index)
            if any(frame[kind][k] != v for k, v in expected.items()):
                raise ValueError("association mismatch")
        reference = frame["reference"]
        pose = None
        if reference["status"] == "associated":
            values = np.array(reference["matches"][0]["values"], float)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("invalid reference pose")
            pose = tum_pose_to_matrix(*values)
        poses.append(pose)
        depth = None
        if frame["depth"]["status"] == "associated":
            path = root / frame["depth"]["path"]
            if digest(path) != frame["depth"]["file_sha256"]:
                raise ValueError("depth hash mismatch")
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape != (480, 640) or depth.dtype != np.uint16:
                raise ValueError("invalid native depth image")
        depths.append(depth)
    cases = reference_cases(poses)
    masks = [np.zeros_like(image) for image in gray]  # Admission only, not semantic labels.
    tracks = [track_feature(gray, masks, feature) for feature in grid_features(gray[0], masks[0])]
    pixel_k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    normalized_k = pixel_k.copy()
    normalized_k[:2, 2] += .5
    normalized_k[0] /= 640
    normalized_k[1] /= 480
    intrinsics = np.repeat(normalized_k[None], 90, axis=0)
    methods = {name: [evaluate_window(t, p, intrinsics, np.array([640, 480])) for t in tracks]
               for name, p in cases.items()}
    common = []
    depth_tracks = []
    for slot, track in enumerate(tracks):
        rows = []
        for frame, obs in enumerate(track):
            if obs["xy"] is None:
                measured = {"status": "missing_track"}
            elif poses[frame] is None:
                measured = {"status": "missing_reference"}
            elif depths[frame] is None:
                measured = {"status": "missing_depth_association"}
            else:
                measured = world_observation(obs["xy"], depths[frame], pixel_k, poses[frame])
            rows.append({"frame": frame, **measured})
            if frame >= 11:
                errors = {name: anchors[slot]["rows"][frame]["distance_pixels"] for name, anchors in methods.items()}
                if all(v is not None for v in errors.values()):
                    common.append({"slot": slot, "frame": frame, "pixels": errors})
        depth_tracks.append({"slot": slot, "status_counts": dict(Counter(r["status"] for r in rows)),
                             "observed_spread": observed_spread(rows), "rows": rows})
    result = {
        "scope": "static-scene reference/rotation development control; not camera-DNF labels",
        "source_manifest_sha256": digest(root / "staging-manifest.json"),
        "script_sha256": digest(Path(__file__)), "opencv_version": cv2.__version__,
        "kernel_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                          ("probe_davis_anchors.py", "build_anchor_preview.py", "rgbd_track_audit.py", "stage_static_rgbd_control.py")},
        "calibration": "published Freiburg-3 native RGB K; depth /5000; nearest reference/depth, no interpolation",
        "selection": "12 original frame-zero grid slots; all pixels admitted, no staticness labels",
        "fit_original_frames": list(range(1, 11)), "test_original_frames": list(range(11, 90)),
        "unavailable_reference_frames": [0], "planned_test_observations": 12 * 79,
        "tracks": tracks, "methods": methods, "depth_tracks": depth_tracks,
        "common_observations": common,
        "summary": {"common_count": len(common),
                    "fit_counts": {name: dict(Counter(a["fit_status"] for a in anchors)) for name, anchors in methods.items()},
                    "median_common_pixels": {name: float(np.median([r["pixels"][name] for r in common])) if common else None for name in methods}},
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(run(args.root, args.output)["summary"])
