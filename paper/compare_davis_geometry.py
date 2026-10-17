"""Paired pose/calibration diagnostic on identical, previously selected matches.

Pixel distances avoid comparing normalized residuals with different focal scales.
The 2x2 pose/calibration swap is a diagnostic intervention, not a new tracker.
No human ranking, accuracy label, or DNF threshold follows from these values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from paper.build_reference_free_review import digest, load_verified
from PIL import Image

from plumbline.image_evidence import target_epipolar_lines


def symmetric_distances(first, second, ka, kb, sa, sb, pa, pb):
    forward = target_epipolar_lines(first, ka, kb, sa, sb, pa, pb)
    reverse = target_epipolar_lines(second, kb, ka, sb, sa, pb, pa)
    if forward is None or reverse is None:
        return None
    d1 = np.abs(np.sum(forward * np.c_[second, np.ones(len(second))], axis=1))
    d2 = np.abs(np.sum(reverse * np.c_[first, np.ones(len(first))], axis=1))
    return (d1 + d2) / 2


def compare(first_run: Path, second_run: Path, evidence_path: Path, root: Path, output: Path):
    if output.exists():
        raise ValueError("preserve existing comparisons")
    ma, pa, paths = load_verified(first_run, root)
    mb, pb, _ = load_verified(second_run, root)
    if ma["frames"] != mb["frames"] or ma["decoded_input_sha256"] != mb["decoded_input_sha256"]:
        raise ValueError("methods must use identical source frames and decoded inputs")
    evidence = json.loads(evidence_path.read_text())
    if evidence["source_manifest_sha256"] != digest(first_run / "manifest.json"):
        raise ValueError("matches do not belong to first run")
    intrinsics = []
    for run in (first_run, second_run):
        with np.load(run / "prediction.npz", allow_pickle=False) as archive:
            intrinsics.append(archive["intrinsics_normalized"])
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    rows = []
    for pair in evidence["pairs"]:
        a, b = pair["a"], pair["b"]
        first = np.asarray(pair["first_pixel_xy"]).reshape(-1, 2)
        second = np.asarray(pair["second_pixel_xy"]).reshape(-1, 2)
        for pose_id, poses in enumerate((pa, pb)):
            for calibration_id, k in enumerate(intrinsics):
                distances = symmetric_distances(
                    first, second, k[a], k[b], sizes[a], sizes[b], poses[a], poses[b]
                )
                finite = (
                    distances[np.isfinite(distances)] if distances is not None else np.array([])
                )
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "pose_source": pose_id,
                        "calibration_source": calibration_id,
                        "match_count": len(first),
                        "finite_count": len(finite),
                        "median_pixels": float(np.median(finite)) if len(finite) else None,
                        "status": "zero_baseline"
                        if distances is None
                        else "no_finite_measurements"
                        if not len(finite)
                        else "measured",
                    }
                )
    result = {
        "scope": "development geometric diagnostic, not method ranking",
        "sources": [
            {"run": str(run), "manifest_sha256": digest(run / "manifest.json")}
            for run in (first_run, second_run)
        ],
        "matches_sha256": digest(evidence_path),
        "script_sha256": digest(Path(__file__)),
        "metric": "mean of forward/backward point-to-epipolar-line distances in input pixels",
        "calibration_swap": "off-diagonal cells are interventions, not native method outputs",
        "frames_verified_per_method": len(paths),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first_run", "second_run", "evidence", "root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    compare(args.first_run, args.second_run, args.evidence, args.root, args.output)
