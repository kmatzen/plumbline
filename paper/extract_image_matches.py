"""Candidate-blind SIFT matches on reconstructed native RealEstate10K frames.

Uses mutual nearest matches passing a 0.75 ratio test in both directions.
No estimated or reference poses are used to select correspondences. Calibration
comes from the dataset, so this pilot tests calibrated pose consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def extract(root: Path, export: dict) -> dict:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    clips = []
    for variant in export["variants"]:
        if variant["sigma"] != 0:
            continue
        clip_id = variant["sample_id"].split("/")[-1]
        directory = root / clip_id
        metadata_path = directory / (clip_id + ".txt")
        metadata = {}
        for line in metadata_path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 19:
                metadata[int(fields[0])] = [float(x) for x in fields[1:5]]
        descriptors = []
        normalized = []
        manifest = []
        for timestamp in variant["timestamps"]:
            paths = [directory / f"{timestamp}.{suffix}" for suffix in ("png", "jpg", "jpeg")]
            path = next(path for path in paths if path.exists())
            raw = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cannot decode {path}")
            points, desc = detector.detectAndCompute(image, None)
            h, w = image.shape
            fx, fy, cx, cy = metadata[timestamp]
            xy = np.asarray([p.pt for p in points]).reshape(-1, 2)
            normalized.append((xy / np.array([w, h]) - [cx, cy]) / [fx, fy])
            descriptors.append(desc)
            manifest.append(
                {
                    "timestamp": timestamp,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "shape": [h, w],
                    "features": len(points),
                }
            )
        pairs = []
        for a, b in combinations(range(len(descriptors)), 2):
            da, db = descriptors[a], descriptors[b]
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
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "count": len(accepted),
                    "first": [normalized[a][i].tolist() for i, _ in accepted],
                    "second": [normalized[b][j].tolist() for _, j in accepted],
                }
            )
        clips.append(
            {
                "sample_id": variant["sample_id"],
                "frames": manifest,
                "camera_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "pairs": pairs,
            }
        )
        print(f"Extracted {len(clips)}: {clip_id}", flush=True)
    return {
        "opencv_version": cv2.__version__,
        "sift_nfeatures": 1500,
        "ratio": 0.75,
        "mutual": True,
        "pose_filter": False,
        "calibration": "dataset intrinsics",
        "clips": clips,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.root, json.loads(args.export.read_text()))
    args.output.write_text(json.dumps(result))
