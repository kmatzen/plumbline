"""Compare image-only geometric evidence with held-out reference pose metrics.

No reference pose participates in matching or the predicted image score.
Dataset intrinsics are supplied, so this is a calibrated-pose pilot baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from plumbline.image_evidence import sampson_residuals


def analyze(matches: dict, export: dict, metrics: dict) -> dict:
    variants = {v["sample_id"]: v for v in export["variants"] if v["sigma"] == 0}
    ground = {(r["sample_id"], r["method"]): r for r in metrics["rows"] if r["sigma"] == 0}
    rows = []
    for clip in matches["clips"]:
        variant = variants[clip["sample_id"]]
        assert [f["timestamp"] for f in clip["frames"]] == variant["timestamps"]
        systems = {
            method: np.asarray(prediction["poses"])
            for method, prediction in variant["predictions"].items()
        }
        systems["reference"] = np.asarray(variant["reference"])
        for method, poses in systems.items():
            pair_rows = []
            for pair in clip["pairs"]:
                if pair["count"] < 16:
                    continue
                residual = sampson_residuals(
                    np.asarray(pair["first"]),
                    np.asarray(pair["second"]),
                    poses[pair["a"]],
                    poses[pair["b"]],
                )
                if residual is None or not np.isfinite(residual).all():
                    continue
                pair_rows.append(
                    {
                        "a": pair["a"],
                        "b": pair["b"],
                        "matches": pair["count"],
                        "median_sampson": float(np.median(residual)),
                    }
                )
            rows.append(
                {
                    "sample_id": clip["sample_id"],
                    "method": method,
                    "supported_pairs": len(pair_rows),
                    "pairs": pair_rows,
                    "median_pair_residual": float(
                        np.median([r["median_sampson"] for r in pair_rows])
                    )
                    if len(pair_rows) >= 3
                    else None,
                }
            )
    comparisons = []
    for sample_id in variants:
        systems = {
            r["method"]: r
            for r in rows
            if r["sample_id"] == sample_id and r["method"] != "reference"
        }
        for a, b in combinations(sorted(systems), 2):
            left, right = systems[a], systems[b]
            # Compare only identical supported image pairs for both methods.
            lp = {(p["a"], p["b"]): p["median_sampson"] for p in left["pairs"]}
            rp = {(p["a"], p["b"]): p["median_sampson"] for p in right["pairs"]}
            common = sorted(lp.keys() & rp.keys())
            if len(common) < 3:
                continue
            image_delta = float(
                np.median([rp[k] for k in common]) - np.median([lp[k] for k in common])
            )
            auc_delta = ground[sample_id, a]["auc30"] - ground[sample_id, b]["auc30"]
            ate_delta = (
                ground[sample_id, b]["normalized_ate"] - ground[sample_id, a]["normalized_ate"]
            )
            if min(abs(image_delta), abs(auc_delta), abs(ate_delta)) <= 1e-8:
                continue
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "a": a,
                    "b": b,
                    "image_agrees_auc": image_delta * auc_delta > 0,
                    "image_agrees_ate": image_delta * ate_delta > 0,
                    "auc_ate_disagree": auc_delta * ate_delta < 0,
                }
            )
    ids = sorted({r["sample_id"] for r in comparisons})
    intervals = {}
    for key in ("image_agrees_auc", "image_agrees_ate"):
        if not ids:
            intervals[key] = {"fraction": None, "clip_bootstrap_ci95": None}
            continue
        groups = [[r[key] for r in comparisons if r["sample_id"] == s] for s in ids]
        successes = np.array([sum(g) for g in groups])
        counts = np.array([len(g) for g in groups])
        rng = np.random.default_rng(231)
        selected = rng.integers(0, len(ids), (10000, len(ids)))
        draws = successes[selected].sum(1) / counts[selected].sum(1)
        intervals[key] = {
            "fraction": float(successes.sum() / counts.sum()),
            "clip_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return {
        "calibration": "dataset intrinsics; prediction intrinsics not evaluated",
        "minimum_matches": 16,
        "minimum_pairs": 3,
        "rows": rows,
        "comparisons": comparisons,
        "decisive_comparisons": len(comparisons),
        "clips": len(ids),
        "agreement": intervals,
        "interpretation": "Agreement with reference metrics is not human preference validation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent / "results"
    matches = json.loads(args.matches.read_text())
    result = analyze(
        matches,
        json.loads((root / "realestate_pose_export.json").read_text()),
        json.loads((root / "realestate_analysis.json").read_text()),
    )
    result["matches_sha256"] = hashlib.sha256(args.matches.read_bytes()).hexdigest()
    manifest = {
        "matches_sha256": result["matches_sha256"],
        "extractor_sha256": hashlib.sha256(
            (root.parent / "extract_image_matches.py").read_bytes()
        ).hexdigest(),
        "opencv_version": matches["opencv_version"],
        "source_on_anima": "/data3/kmatzen/plumbline/research/image-evidence/matches.json",
        "clips": [
            {
                "sample_id": clip["sample_id"],
                "frames": clip["frames"],
                "camera_metadata_sha256": clip["camera_metadata_sha256"],
                "pair_match_counts": [p["count"] for p in clip["pairs"]],
            }
            for clip in matches["clips"]
        ],
    }
    (root / "image_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "image_evidence_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "comparisons"}}, indent=2)
    )
