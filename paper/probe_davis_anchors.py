"""Fixed-grid, early-fit anchor diagnostics on identical image observations.

This development protocol does not certify static regions or correct tracks.
All slots, lost tracks, failed fits, and undefined projections are retained.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from paper.analyze_davis_regions import labels_at
from paper.build_anchor_preview import fit_point, project
from paper.build_reference_free_review import digest, load_verified
from PIL import Image


def grid_features(gray, mask):
    eligible = cv2.erode((mask == 0).astype(np.uint8) * 255, np.ones((21, 21), np.uint8))
    height, width = gray.shape
    slots = []
    for row in range(3):
        for column in range(4):
            region = np.zeros_like(eligible)
            region[
                row * height // 3 : (row + 1) * height // 3,
                column * width // 4 : (column + 1) * width // 4,
            ] = 255
            points = cv2.goodFeaturesToTrack(gray, 1, 0.01, 30, mask=eligible & region)
            slots.append(points.reshape(2) if points is not None else None)
    return slots


def flow(first, second, xy):
    previous = np.asarray(xy, np.float32).reshape(1, 1, 2)
    current, status, _ = cv2.calcOpticalFlowPyrLK(
        first, second, previous, None, winSize=(21, 21), maxLevel=3
    )
    if current is None or not status.item() or not np.isfinite(current).all():
        return None, None
    backward, status, _ = cv2.calcOpticalFlowPyrLK(
        second, first, current, None, winSize=(21, 21), maxLevel=3
    )
    if backward is None or not status.item() or not np.isfinite(backward).all():
        return None, None
    error = float(np.linalg.norm(backward - previous))
    point = current.reshape(2)
    height, width = second.shape
    if error > 1 or not (point >= 0).all() or not (point < [width, height]).all():
        return None, error
    return point.astype(float), error


def track_feature(gray, masks, first):
    if first is None:
        return [{"status": "empty_slot", "xy": None} for _ in gray]
    previous = first
    rows = []
    for index in range(len(gray)):
        current, fb = (
            (first, 0.0)
            if index == 0
            else (
                flow(gray[index - 1], gray[index], previous)
                if previous is not None
                else (None, None)
            )
        )
        direct, direct_fb = (first, 0.0) if index == 0 else flow(gray[0], gray[index], first)
        delta = (
            float(np.linalg.norm(current - direct))
            if current is not None and direct is not None
            else None
        )
        label = int(labels_at(masks[index], current[None])[0]) if current is not None else None
        rows.append(
            {
                "status": "tracked" if current is not None else "lost",
                "xy": current.tolist() if current is not None else None,
                "forward_backward_pixels": fb,
                "annotation_label": label,
                "direct_xy": direct.tolist() if direct is not None else None,
                "direct_forward_backward_pixels": direct_fb,
                "direct_chain_distance_pixels": delta,
                "crosscheck_pass": delta is not None and delta <= 3 and label == 0,
            }
        )
        previous = current
    return rows


def evaluate(track, poses, intrinsics, size, *, fit_count=11):
    if not 2 <= fit_count < len(track):
        raise ValueError("fit_count must leave at least one evaluation frame")
    fitting = track[:fit_count]
    status = "fit"
    point = None
    if any(r["xy"] is None or r["annotation_label"] != 0 for r in fitting):
        status = "insufficient_early_background_track"
    else:
        point = fit_point(
            np.array([r["xy"] for r in fitting]), poses[:fit_count], intrinsics[:fit_count], size
        )
        if point is None:
            status = "numerically_degenerate_fit"
        elif any(
            project(point, p, k, size) is None
            for p, k in zip(poses[:fit_count], intrinsics[:fit_count], strict=True)
        ):
            point, status = None, "nonpositive_fit_depth"
    rows = []
    for index, observation in enumerate(track):
        prediction = (
            project(point, poses[index], intrinsics[index], size) if point is not None else None
        )
        measured = prediction is not None and observation["xy"] is not None
        rows.append(
            {
                "frame": index,
                "fit_frame": index < fit_count,
                "status": "fit_unavailable"
                if point is None
                else "nonpositive_projection_depth"
                if prediction is None
                else "image_track_lost"
                if observation["xy"] is None
                else "measured",
                "predicted_xy": prediction.tolist() if prediction is not None else None,
                "distance_pixels": float(np.linalg.norm(prediction - observation["xy"]))
                if measured
                else None,
            }
        )
    return {
        "fit_status": status,
        "fit_crosscheck_pass": all(r.get("crosscheck_pass", False) for r in fitting),
        "point_world": point.tolist() if point is not None else None,
        "rows": rows,
    }


def build(first_run, second_run, root, annotations, output):
    if output.exists():
        raise ValueError("preserve previous probes")
    ma, pa, paths = load_verified(first_run, root)
    mb, pb, _ = load_verified(second_run, root)
    if ma["frames"] != mb["frames"] or ma["decoded_input_sha256"] != mb["decoded_input_sha256"]:
        raise ValueError("identical source and decoded inputs required")
    mask_record = json.loads((annotations / "masks.json").read_text())
    if mask_record["source_manifest_sha256"] != digest(first_run / "manifest.json"):
        raise ValueError("mask source mismatch")
    if [r["frame_index"] for r in mask_record["frames"]] != [
        r["frame_index"] for r in ma["frames"]
    ]:
        raise ValueError("mask ordering mismatch")
    gray = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    masks = []
    for row in mask_record["frames"]:
        path = (annotations / row["path"]).resolve()
        if not path.is_relative_to(annotations.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("mask hash/path mismatch")
        with Image.open(path) as image:
            mask = np.array(image)
        if mask.shape != gray[0].shape:
            raise ValueError("mask shape mismatch")
        masks.append(mask)
    features = grid_features(gray[0], masks[0])
    tracks = [track_feature(gray, masks, first) for first in features]
    size = np.array(gray[0].shape[::-1])
    methods = []
    for run, poses in ((first_run, pa), (second_run, pb)):
        with np.load(run / "prediction.npz", allow_pickle=False) as archive:
            intrinsics = archive["intrinsics_normalized"]
        anchors = [evaluate(track, poses, intrinsics, size) for track in tracks]
        methods.append(
            {
                "run": str(run),
                "manifest_sha256": digest(run / "manifest.json"),
                "fit_counts": dict(Counter(a["fit_status"] for a in anchors)),
                "anchors": anchors,
            }
        )
    comparisons = {}
    for checked in (False, True):
        errors = [[], []]
        per_anchor = [[], []]
        for slot, track in enumerate(tracks):
            anchors = [method["anchors"][slot] for method in methods]
            local = [[], []]
            for frame in range(11, len(paths)):
                rows = [a["rows"][frame] for a in anchors]
                qualifies = track[frame].get("annotation_label") == 0
                if checked:
                    qualifies &= track[frame].get("crosscheck_pass", False) and all(
                        a["fit_crosscheck_pass"] for a in anchors
                    )
                if qualifies and all(r["distance_pixels"] is not None for r in rows):
                    for method_id, row in enumerate(rows):
                        local[method_id].append(row["distance_pixels"])
            for method_id in range(2):
                errors[method_id].extend(local[method_id])
                if local[method_id]:
                    per_anchor[method_id].append(float(np.median(local[method_id])))
        comparisons["crosschecked" if checked else "background_chain"] = {
            "common_test_observations": len(errors[0]),
            "common_anchors": len(per_anchor[0]),
            "pooled_median_pixels": [float(np.median(e)) if e else None for e in errors],
            "median_anchor_median_pixels": [float(np.median(e)) if e else None for e in per_anchor],
        }
    result = {
        "scope": "development anchor consistency, not validated camera accuracy or DNF",
        "script_sha256": digest(Path(__file__)),
        "fit_kernel_sha256": digest(Path(__file__).with_name("build_anchor_preview.py")),
        "opencv_version": cv2.__version__,
        "mask_manifest_sha256": digest(annotations / "masks.json"),
        "policy": {
            "grid": [4, 3],
            "fit_frames": list(range(11)),
            "flow_fb_max_pixels": 1,
            "direct_chain_max_pixels": 3,
            "threshold_status": "fixed development choices, not calibrated correctness thresholds",
        },
        "limitations": [
            "Annotated background may move.",
            "Direct and chained LK can share errors; agreement is not independent validation.",
            "Small parallax and predicted calibration confound fitting.",
            "Matched available subsets can be easier; coverage must accompany errors.",
            "Repeated frames/anchors are not independent samples.",
        ],
        "slots": len(tracks),
        "possible_test_observations": len(tracks) * (len(paths) - 11),
        "tracks": tracks,
        "methods": methods,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first_run", "second_run", "root", "annotations", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.first_run, args.second_run, args.root, args.annotations, args.output)
