"""Development image consistency with predicted, not ground-truth calibration.

Correspondences are selected without candidate poses or intrinsics. All mutual
SIFT ratio matches are retained, including dynamic objects and false matches.
Consequently residuals are diagnostic evidence, not accuracy or DNF labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest, load_verified

from plumbline.image_evidence import rays_from_normalized_intrinsics, sampson_residuals


def analyze(run: Path, root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve existing evidence")
    manifest, poses, paths = load_verified(run, root)
    if manifest.get("calibration", {}).get("array") != "intrinsics_normalized":
        raise ValueError("run does not declare normalized calibration")
    with np.load(run / "prediction.npz", allow_pickle=False) as archive:
        calibration = archive["intrinsics_normalized"]
    if calibration.shape != (len(paths), 3, 3):
        raise ValueError("calibration/frame mismatch")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features, sizes = [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("cannot decode source frame")
        keypoints, descriptors = detector.detectAndCompute(image, None)
        features.append((np.asarray([p.pt for p in keypoints]).reshape(-1, 2), descriptors))
        sizes.append((image.shape[1], image.shape[0]))
    pairs = []
    for lag in (1, 10, 30):
        for a in range(0, len(paths) - lag, 5):
            b = a + lag
            xy_a, da = features[a]
            xy_b, db = features[b]
            accepted = []
            if da is not None and db is not None and min(len(da), len(db)) >= 2:
                forward = matcher.knnMatch(da, db, k=2)
                reverse = matcher.knnMatch(db, da, k=2)
                reverse_good = {
                    (m.queryIdx, m.trainIdx) for m, n in reverse if m.distance < 0.75 * n.distance
                }
                accepted = [
                    (m.queryIdx, m.trainIdx)
                    for m, n in forward
                    if m.distance < 0.75 * n.distance and (m.trainIdx, m.queryIdx) in reverse_good
                ]
            first = np.asarray([xy_a[i] for i, _ in accepted]).reshape(-1, 2)
            second = np.asarray([xy_b[j] for _, j in accepted]).reshape(-1, 2)
            rays_a = rays_from_normalized_intrinsics(first, calibration[a], sizes[a])
            rays_b = rays_from_normalized_intrinsics(second, calibration[b], sizes[b])
            errors = sampson_residuals(rays_a, rays_b, poses[a], poses[b])
            finite = errors[np.isfinite(errors)] if errors is not None else np.array([])
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "count": len(accepted),
                    "first_pixel_xy": first.tolist(),
                    "second_pixel_xy": second.tolist(),
                    "residuals": [float(e) if np.isfinite(e) else None for e in errors]
                    if errors is not None
                    else None,
                    "finite_residual_count": len(finite),
                    "median_finite_residual": float(np.median(finite)) if len(finite) else None,
                    "status": "zero_baseline"
                    if errors is None
                    else "no_matches"
                    if not len(accepted)
                    else "measured",
                }
            )
    result = {
        "schema": "davis_predicted_calibration_sampson_v1",
        "source_manifest_sha256": digest(run / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "opencv_version": cv2.__version__,
        "source_frames_verified": len(paths),
        "sift_nfeatures": 1500,
        "mutual_ratio": 0.75,
        "pair_selection": "lags 1,10,30; starts every 5 supplied frames",
        "calibration": "DAGE estimated normalized intrinsics, not independent calibration",
        "selection_uses_candidate": False,
        "static_mask": None,
        "units": "normalized camera coordinates, not pixels",
        "limitations": "dynamic objects and false matches retained; not ORE or a DNF label",
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.root, args.output)
    print(
        json.dumps(
            {"pairs": len(result["pairs"]), "matches": sum(p["count"] for p in result["pairs"])}
        )
    )
