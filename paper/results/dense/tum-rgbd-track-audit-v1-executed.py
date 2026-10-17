"""Reference-assisted depth observations; not reference-free evaluator labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from paper.analyze_dense_run import load_verified
from paper.stage_rgbd_validation import sha


def world_observation(xy, depth, intrinsics, camera_to_world):
    """Backproject nearest pixel with axial depth, retaining local discontinuities.

    Returns missing evidence explicitly. A large world-space change can reflect
    scene motion, a bad track, occlusion, depth noise, or calibration/timing error;
    it is not automatically a camera failure or a moving-object label.
    """
    if xy is None:
        return {"status": "missing_track"}
    xy = np.asarray(xy, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("invalid image point")
    if (
        k.shape != (3, 3)
        or pose.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(pose).all()
    ):
        raise ValueError("invalid camera")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("expected raw uint16 TUM depth image")
    x, y = np.rint(xy).astype(int)
    height, width = depth.shape
    if not (0 <= x < width and 0 <= y < height):
        return {"status": "outside_depth_image"}
    raw = int(depth[y, x])
    patch = depth[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)]
    valid = patch[patch > 0].astype(float) / 5000
    result = {
        "sampled_pixel_xy": [int(x), int(y)],
        "raw_depth": raw,
        "patch_valid_pixels": int(len(valid)),
        "patch_total_pixels": int(patch.size),
        "patch_depth_range_m": [float(valid.min()), float(valid.max())] if len(valid) else None,
    }
    if raw == 0:
        return {**result, "status": "missing_depth"}
    ray = np.linalg.solve(k, [x, y, 1.0])
    if abs(ray[2]) < 1e-12:
        raise ValueError("invalid axial-depth ray")
    camera_point = ray / ray[2] * (raw / 5000)
    world_point = pose[:3, :3] @ camera_point + pose[:3, 3]
    return {
        **result,
        "status": "measured",
        "depth_m": raw / 5000,
        "camera_point_m": camera_point.tolist(),
        "world_point_m": world_point.tolist(),
    }


def audit(run_path, tracks_path, depth_root, output):
    if output.exists():
        raise ValueError("preserve previous audit attempts")
    manifest, _, poses = load_verified(run_path)
    tracks = json.loads(tracks_path.read_text())
    depth_manifest_path = depth_root / "depth-manifest.json"
    depths = json.loads(depth_manifest_path.read_text())
    source_hash = sha(run_path / "manifest.json")
    if any(record["source_manifest_sha256"] != source_hash for record in (tracks, depths)):
        raise ValueError("source manifest mismatch")
    if tracks["reference_file_sha256"] != sha(run_path / "reference.npz"):
        raise ValueError("track reference mismatch")
    count = len(manifest["frames"])
    if len(depths["rows"]) != count or any(len(t) != count for t in tracks["tracks"]):
        raise ValueError("frame count mismatch")
    if depths["depth_scale_divisor"] != 5000 or depths["zero_depth"] != "missing":
        raise ValueError("unexpected depth convention")
    with np.load(run_path / "reference.npz", allow_pickle=False) as archive:
        intrinsics = archive["intrinsics"]
    if intrinsics.shape != (count, 3, 3):
        raise ValueError("intrinsics shape mismatch")
    images = []
    timing = []
    for index, (frame, row) in enumerate(zip(manifest["frames"], depths["rows"], strict=True)):
        if (
            row["frame"] != index
            or row["rgb_path"] != frame["path"]
            or row["rgb_sha256"] != frame["file_sha256"]
            or float(row["rgb_timestamp"]) != float(frame["timestamp"])
        ):
            raise ValueError("depth association frame mismatch")
        image = None
        match = row["nearest_depth"]
        if row["status"] == "staged":
            if not match or not match["within_tolerance"]:
                raise ValueError("invalid staged association")
            path = depth_root / match["path"]
            if sha(path) != row["depth_sha256"]:
                raise ValueError("depth image hash mismatch")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint16 or image.shape != (480, 640):
                raise ValueError("expected native TUM 640x480 uint16 depth")
        elif row["status"] != "no_depth_within_20ms":
            raise ValueError("unknown depth association status")
        images.append(image)
        timing.append(
            {
                "frame": index,
                "depth_minus_rgb_seconds": match["delta_seconds"] if match else None,
                "reference_minus_rgb_seconds": float(frame["reference_timestamp"])
                - float(frame["timestamp"]),
                "depth_status": row["status"],
            }
        )
    result_tracks = []
    for slot, track in enumerate(tracks["tracks"]):
        rows = []
        for frame, observation in enumerate(track):
            if observation["xy"] is None:
                result = {"status": "missing_track"}
            elif images[frame] is None:
                result = {"status": "missing_depth_association"}
            else:
                result = world_observation(
                    observation["xy"], images[frame], intrinsics[frame], poses[frame]
                )
            rows.append({"frame": frame, **result})
        origin = rows[0].get("world_point_m")
        distances = []
        for row in rows:
            point = row.get("world_point_m")
            distance = (
                float(np.linalg.norm(np.array(point) - origin))
                if point is not None and origin is not None
                else None
            )
            row["distance_from_first_frame_m"] = distance
            if row["frame"] > 0 and distance is not None:
                distances.append(distance)
        result_tracks.append(
            {
                "slot": slot,
                "status_counts": dict(Counter(r["status"] for r in rows)),
                "first_frame_measured": origin is not None,
                "later_comparable_count": len(distances),
                "median_later_distance_m": float(np.median(distances)) if distances else None,
                "max_later_distance_m": max(distances) if distances else None,
                "rows": rows,
            }
        )
    result = {
        "scope": "reference-assisted development evidence; not staticness or camera-failure labels",
        "sequence": manifest["sequence"],
        "source_manifest_sha256": source_hash,
        "reference_file_sha256": sha(run_path / "reference.npz"),
        "tracks_sha256": sha(tracks_path),
        "depth_manifest_sha256": sha(depth_manifest_path),
        "script_sha256": sha(Path(__file__)),
        "opencv_version": cv2.__version__,
        "policy": "all original slots and frames retained; nearest depth pixel; fixed frame-zero origin, never re-anchor; no threshold labels",
        "pose_timing": "existing nearest-reference RGB association; no interpolation or depth-time correction",
        "frames": timing,
        "tracks": result_tracks,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "tracks", "depth_root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.tracks, args.depth_root, args.output)
    for track in result["tracks"]:
        print({k: v for k, v in track.items() if k != "rows"})
