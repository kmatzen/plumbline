"""Compare placement support using separately inferred prefix/full image tracks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from paper.analyze_davis_regions import labels_at
from paper.build_reference_free_review import digest, load_verified
from paper.probe_davis_anchors import evaluate
from PIL import Image


def observations(prefix, full, prefix_visibility, full_visibility, masks, lk_track):
    # Deliberately do not read full-video coordinates for the fitting frames.
    xy = np.concatenate((prefix, full[11:]))
    visible = np.concatenate((prefix_visibility, full_visibility[11:]))
    labels = [int(labels_at(mask, point[None])[0]) for mask, point in zip(masks, xy, strict=True)]
    height, width = masks[0].shape
    valid = (
        visible
        & np.isfinite(xy).all(axis=1)
        & (xy >= 0).all(axis=1)
        & (xy < [width, height]).all(axis=1)
    )
    result = []
    for index, (point, label) in enumerate(zip(xy, labels, strict=True)):
        other = lk_track[index]["xy"]
        delta = float(np.linalg.norm(point - other)) if other is not None and valid[index] else None
        result.append(
            {
                "xy": point.tolist() if valid[index] else None,
                "raw_xy": point.tolist(),
                "predicted_visible": bool(visible[index]),
                "annotation_label": label,
                "lk_distance_pixels": delta,
                "crosscheck_pass": delta is not None and delta <= 3 and label == 0,
            }
        )
    return result


def analyze(run, baseline_path, root, annotations, output):
    if output.exists():
        raise ValueError("preserve existing analyses")
    record = json.loads((run / "manifest.json").read_text())
    baseline = json.loads(baseline_path.read_text())
    if record["status"] != "complete" or record["baseline_sha256"] != digest(baseline_path):
        raise ValueError("complete matching inference required")
    arrays = {}
    for phase in record["phases"]:
        path = run / f"{phase['name']}.npz"
        if phase["status"] != "complete" or digest(path) != phase["prediction_sha256"]:
            raise ValueError("prediction hash/status mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays[phase["name"]] = {k: archive[k] for k in archive.files}
    if set(arrays) != {"prefix", "full"}:
        raise ValueError("both prefix and full inference required")
    prefix, full = arrays["prefix"], arrays["full"]
    if (
        not np.array_equal(prefix["slots"], full["slots"])
        or prefix["slots"].tolist() != record["slots"]
    ):
        raise ValueError("query slot mismatch")
    masks_record = json.loads((annotations / "masks.json").read_text())
    if masks_record["source_manifest_sha256"] != record["source_manifest_sha256"]:
        raise ValueError("annotation source mismatch")
    masks = []
    for row in masks_record["frames"]:
        path = (annotations / row["path"]).resolve()
        if not path.is_relative_to(annotations.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("annotation hash/path mismatch")
        with Image.open(path) as image:
            masks.append(np.array(image))
    tracks = [
        observations(
            prefix["tracks"][:, j],
            full["tracks"][:, j],
            prefix["visibility"][:, j],
            full["visibility"][:, j],
            masks,
            baseline["tracks"][slot],
        )
        for j, slot in enumerate(record["slots"])
    ]
    methods = []
    for method in baseline["methods"]:
        source = Path(method["run"])
        manifest, poses, _ = load_verified(source, root)
        if digest(source / "manifest.json") != method["manifest_sha256"] or [
            r["frame_index"] for r in manifest["frames"]
        ] != [r["frame_index"] for r in masks_record["frames"]]:
            raise ValueError("camera source mismatch")
        with np.load(source / "prediction.npz", allow_pickle=False) as archive:
            intrinsics = archive["intrinsics_normalized"]
        anchors = [
            evaluate(track, poses, intrinsics, np.array(masks[0].shape[::-1])) for track in tracks
        ]
        methods.append(
            {
                "run": str(source),
                "fit_counts": dict(Counter(a["fit_status"] for a in anchors)),
                "anchors": anchors,
            }
        )
    errors = [[], []]
    common_anchors = 0
    for j, track in enumerate(tracks):
        count = 0
        for frame in range(11, len(track)):
            values = [m["anchors"][j]["rows"][frame]["distance_pixels"] for m in methods]
            if track[frame]["annotation_label"] == 0 and all(v is not None for v in values):
                count += 1
                for method, value in enumerate(values):
                    errors[method].append(value)
        common_anchors += count > 0
    deltas = [
        r["lk_distance_pixels"]
        for track in tracks
        for r in track[11:]
        if r["lk_distance_pixels"] is not None
    ]
    prefix_delta = np.linalg.norm(prefix["tracks"] - full["tracks"][:11], axis=-1)
    result = {
        "scope": "development image-evidence comparison; not accuracy or DNF",
        "inference_manifest_sha256": digest(run / "manifest.json"),
        "baseline_sha256": digest(baseline_path),
        "script_sha256": digest(Path(__file__)),
        "evaluation_kernel_sha256": digest(Path(__file__).with_name("probe_davis_anchors.py")),
        "slot_count": baseline["slots"],
        "queried_slots": record["slots"],
        "common_test_observations": len(errors[0]),
        "common_anchors": common_anchors,
        "possible_test_observations": baseline["possible_test_observations"],
        "pooled_median_pixels": [float(np.median(e)) if e else None for e in errors],
        "lk_overlap_observations": len(deltas),
        "median_lk_distance_pixels": float(np.median(deltas)) if deltas else None,
        "prefix_full_coordinate_delta_quantiles_0_50_100": np.percentile(
            prefix_delta, [0, 50, 100]
        ).tolist(),
        "limitations": [
            "Visibility is a model prediction, not validation.",
            "Background may move; image tracks may be wrong.",
            "Different trackers create different available subsets, so error changes alone do not establish improvement.",
            "Separate prefix inference prevents full-video coordinates from entering the early point fit.",
        ],
        "tracks": tracks,
        "methods": methods,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "baseline", "root", "annotations", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.baseline, args.root, args.annotations, args.output)
