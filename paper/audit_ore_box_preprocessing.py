"""Trace released box preprocessing without repairing annotations or scores."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paper.audit_ore_depth_support import CLIP
from paper.run_ore_demo import digest


def trace_box(row, calibration):
    x, y, w, h = (row[k] for k in ("x", "y", "width", "height"))
    # Preserve the released construction and assignment semantics.
    box = np.array([x, y, x + w, y + h])
    ow, oh = (
        (row["original_width"], row["original_height"])
        if ("original_width" in row and "original_height" in row)
        else (1920, 1440)
    )
    box[0:3:2] = box[0:3:2] / ow * 640
    box[1:4:2] = box[1:4:2] / oh * 480
    k = np.eye(3)
    k[0, 0], k[1, 1], k[0, 2], k[1, 2] = calibration[:4]
    corners = box.reshape(2, 2).astype(float)
    undistorted = cv2.undistortPoints(corners, k, calibration[4:], P=k).reshape(2, 2)
    normalized = (np.linalg.inv(k) @ np.c_[undistorted, np.ones(2)].T).T
    redistorted = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, calibration[4:])[
        0
    ].reshape(2, 2)
    error = np.linalg.norm(redistorted - corners, axis=1)
    final = undistorted.astype(int)
    return {
        "frame": row["frame_number"],
        "raw_xywh": [x, y, w, h],
        "rescaled_xyxy": box.tolist(),
        "undistorted_xyxy": undistorted.flatten().tolist(),
        "integer_xyxy": final.flatten().tolist(),
        "reversed_before_distortion": bool(np.any(corners[0] > corners[1])),
        "reversed_after_distortion": bool(np.any(undistorted[0] > undistorted[1])),
        "reversed_after_integer_cast": bool(np.any(final[0] > final[1])),
        "roundtrip_error_pixels": error.tolist(),
    }


def audit(source, demo_path, box_audit_path, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    demo = json.loads(demo_path.read_text())
    for name, expected in demo["source_files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"source changed: {name}")
    prior = json.loads(box_audit_path.read_text())
    dataset = json.loads((source / "egotracks_with_additional_annotation.json").read_text())
    clip = next(c for v in dataset["videos"] for c in v["clips"] if c["source_clip_uid"] == CLIP)
    queries = clip["annotations"][0]["query_sets"]
    camera = json.loads((source / "video_meta.json").read_text())[CLIP]
    calibration = np.loadtxt(source / f"calib/{camera}.txt")
    tracks = []
    for previous in prior["methods"][0]["tracks"]:
        track_id = previous["track_id"]
        rows = [trace_box(r, calibration) for r in queries[track_id]["lt_track"]]
        invalid = {r["frame"]: r["integer_xyxy"] for r in rows if r["reversed_after_integer_cast"]}
        assert invalid == dict(
            zip(previous["invalid_box_frames"], previous["invalid_boxes_xyxy"], strict=True)
        )
        errors = np.array([r["roundtrip_error_pixels"] for r in rows])
        tracks.append(
            {
                "track_id": track_id,
                "frames": len(rows),
                "reversed_before_distortion": sum(r["reversed_before_distortion"] for r in rows),
                "reversed_after_distortion": sum(r["reversed_after_distortion"] for r in rows),
                "reversed_after_integer_cast": len(invalid),
                "corners_with_roundtrip_error_above_one_pixel": int((errors > 1).sum()),
                "roundtrip_error_pixel_quantiles": np.quantile(
                    errors, [0, 0.5, 0.9, 0.99, 1]
                ).tolist(),
                "rows": rows,
            }
        )
    result = {
        "status": "complete",
        "scope": "preprocessing diagnostic, not repaired ORE or camera DNF",
        "script_sha256": digest(Path(__file__)),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "demo_sha256": digest(demo_path),
        "box_audit_sha256": digest(box_audit_path),
        "calibration": calibration.tolist(),
        "roundtrip": "undistortPoints default inversion then projectPoints forward distortion; before integer casting",
        "threshold_note": "one pixel describes inversion residual, not a camera quality or DNF threshold",
        "tracks": tracks,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "demo", "box_audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.demo, args.box_audit, args.output)
    print(
        json.dumps(
            [{k: v for k, v in t.items() if k != "rows"} for t in result["tracks"]], indent=2
        )
    )
