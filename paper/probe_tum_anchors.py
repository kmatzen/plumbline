"""Development check of early-fit anchors against reference cameras and controls.

All image regions are admitted: neither staticness nor image-track correctness
is known. Reference poses are a diagnostic control, not selection criteria.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from paper.analyze_dense_run import load_verified
from paper.build_reference_free_review import digest
from paper.probe_davis_anchors import evaluate, grid_features, track_feature
from scipy.spatial.transform import Rotation


def camera_variants(reference, prediction):
    rotated = reference.copy()
    rotated[11:, :3, :3] = (
        rotated[11:, :3, :3] @ Rotation.from_euler("y", 5, degrees=True).as_matrix()
    )
    reversed_centers = reference.copy()
    reversed_centers[:, :3, 3] *= -1
    return {
        "reference": reference,
        "dage": prediction,
        "reference_later_yaw_5deg": rotated,
        "reference_reversed_centers": reversed_centers,
    }


def run(run_path, root, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    manifest, prediction, reference = load_verified(run_path)
    gray = []
    for row in manifest["frames"]:
        path = root / row["path"]
        if digest(path) != row["file_sha256"]:
            raise ValueError("source image hash mismatch")
        gray.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    size = np.array(gray[0].shape[::-1])
    with np.load(run_path / "reference.npz") as archive:
        k = archive["intrinsics"].astype(float)
    # Convert pixel-coordinate calibration to the overlay kernel's +0.5 convention.
    k[:, 0, 2] += 0.5
    k[:, 1, 2] += 0.5
    k[:, 0] /= size[0]
    k[:, 1] /= size[1]
    admitted = [np.zeros_like(g, dtype=np.uint8) for g in gray]
    features = grid_features(gray[0], admitted[0])
    tracks = [track_feature(gray, admitted, f) for f in features]
    methods = {}
    for name, poses in camera_variants(reference, prediction).items():
        anchors = [evaluate(t, poses, k, size) for t in tracks]
        values = [
            r["distance_pixels"]
            for a in anchors
            for r in a["rows"][11:]
            if r["distance_pixels"] is not None
        ]
        methods[name] = {
            "anchors": anchors,
            "fit_counts": dict(Counter(a["fit_status"] for a in anchors)),
            "later_measured": len(values),
            "median_pixels": float(np.median(values)) if values else None,
        }
    common = []
    for slot in range(len(tracks)):
        for frame in range(11, len(gray)):
            values = {
                name: methods[name]["anchors"][slot]["rows"][frame]["distance_pixels"]
                for name in ("reference", "dage", "reference_later_yaw_5deg")
            }
            if all(v is not None for v in values.values()):
                common.append({"slot": slot, "frame": frame, "pixels": values})
    result = {
        "status": "complete",
        "sequence": manifest["sequence"],
        "scope": "single development clip; not held-out validation or functional DNF",
        "source_manifest_sha256": digest(run_path / "manifest.json"),
        "reference_file_sha256": digest(run_path / "reference.npz"),
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": {
            n: digest(Path(__file__).parent / n)
            for n in ["probe_davis_anchors.py", "build_anchor_preview.py"]
        },
        "opencv_version": cv2.__version__,
        "staticness": "unverified; zero-valued admission masks accept all pixels, NOT background annotations",
        "calibration": "dataset pixel intrinsics for all variants, including DAGE; predicted calibration not tested",
        "fit_frames": 11,
        "possible_later_observations": len(tracks) * (len(gray) - 11),
        "tracks": tracks,
        "methods": methods,
        "common_observations": common,
        "common_median_pixels": {
            n: float(np.median([r["pixels"][n] for r in common])) if common else None
            for n in ("reference", "dage", "reference_later_yaw_5deg")
        },
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.run, args.root, args.output)
    print(
        json.dumps(
            {
                "methods": {
                    n: {k: v for k, v in m.items() if k != "anchors"}
                    for n, m in result["methods"].items()
                },
                "common_count": len(result["common_observations"]),
                "common_median_pixels": result["common_median_pixels"],
            },
            indent=2,
        )
    )
