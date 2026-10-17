"""Native versus translation-sign control on fixed image correspondences."""

import argparse
import json
from pathlib import Path

import numpy as np
from paper.build_reference_free_review import digest, load_verified
from PIL import Image

from plumbline.cheirality import triangulation_diagnostics
from plumbline.image_evidence import rays_from_normalized_intrinsics, sampson_residuals


def probe(run: Path, evidence_run: Path, root: Path, output: Path):
    if output.exists():
        raise ValueError("preserve existing results")
    manifest, poses, paths = load_verified(run, root)
    match_manifest, _, _ = load_verified(evidence_run, root)
    if manifest["frames"] != match_manifest["frames"]:
        raise ValueError("source frames differ")
    evidence_path = evidence_run / "image-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence["source_manifest_sha256"] != digest(evidence_run / "manifest.json"):
        raise ValueError("evidence provenance mismatch")
    with np.load(run / "prediction.npz", allow_pickle=False) as archive:
        intrinsics = archive["intrinsics_normalized"]
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    rows = []
    for pair in evidence["pairs"]:
        a, b = pair["a"], pair["b"]
        first = rays_from_normalized_intrinsics(
            np.asarray(pair["first_pixel_xy"]).reshape(-1, 2), intrinsics[a], sizes[a]
        )
        second = rays_from_normalized_intrinsics(
            np.asarray(pair["second_pixel_xy"]).reshape(-1, 2), intrinsics[b], sizes[b]
        )
        native_error = sampson_residuals(first, second, poses[a], poses[b])
        for reverse in (False, True):
            p, q = poses[a].copy(), poses[b].copy()
            if reverse:
                p[:3, 3] *= -1
                q[:3, 3] *= -1
            result = triangulation_diagnostics(first, second, p, q)
            error = sampson_residuals(first, second, p, q)
            row = {"a": a, "b": b, "reverse_translation": reverse, "matches": len(first)}
            if result is None:
                row["status"] = "zero_baseline"
            else:
                resolved = result["resolved"]
                row.update(
                    {
                        "status": "measured",
                        "resolved": int(resolved.sum()),
                        "positive_both": int(result["positive_both"].sum()),
                        "negative_both": int(result["negative_both"].sum()),
                        "mixed_or_zero_depth": int(
                            (resolved & ~result["positive_both"] & ~result["negative_both"]).sum()
                        ),
                        "numerically_degenerate": int((~resolved).sum()),
                        "angle_degrees_quantiles_0_25_50_75_100": np.percentile(
                            result["ray_angle_degrees"], [0, 25, 50, 75, 100]
                        ).tolist()
                        if len(first)
                        else None,
                        "median_gap_baseline_units": float(
                            np.median(result["ray_gap_baseline_units"][resolved])
                        )
                        if resolved.any()
                        else None,
                        "epipolar_residual_unchanged": bool(
                            np.allclose(error, native_error, rtol=1e-10, atol=1e-12, equal_nan=True)
                        ),
                    }
                )
            rows.append(row)
    record = {
        "scope": "classical cheirality diagnostic and sign-control; not usability labels",
        "run": str(run),
        "manifest_sha256": digest(run / "manifest.json"),
        "matches_sha256": digest(evidence_path),
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/cheirality.py"),
        "calibration": "native predicted intrinsics; not known calibration",
        "limitations": [
            "All matches retained, including moving objects and mismatches.",
            "Numerical rank is not sufficient practical observability.",
            "Sign-reversed outputs are interventions, not additional tracker runs.",
            "Positive depths do not prove correct tracking.",
        ],
        "rows": rows,
    }
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "evidence_run", "root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    probe(args.run, args.evidence_run, args.root, args.output)
