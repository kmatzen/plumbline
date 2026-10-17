"""Analyze reconstructed real-data poses without assigning preference labels."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from plumbline.metrics.pose import auc, pairwise_pose_errors
from plumbline.tracking_review import trajectory_diagnostics, validate_trajectory_pair

ROOT = Path(__file__).parent


def analyze(source: dict) -> dict:
    rows = []
    failures = []
    for variant in source["variants"]:
        reference = np.asarray(variant["reference"])
        for method, prediction in variant["predictions"].items():
            poses = np.asarray(prediction["poses"])
            gate = validate_trajectory_pair(poses, reference)
            identity = {
                "sample_id": variant["sample_id"],
                "sigma": variant["sigma"],
                "method": method,
            }
            if not gate.ok:
                failures.append({**identity, "errors": gate.errors})
                continue
            diagnostic = trajectory_diagnostics(poses, reference)
            rotation, translation = pairwise_pose_errors(
                poses, reference, translation_antipodal=True
            )
            rows.append(
                {
                    **identity,
                    "auc30": auc(
                        np.maximum(rotation, translation), [30], mode="vggt_co3d_histogram"
                    )[30],
                    "normalized_ate": float(
                        np.sqrt(np.mean(diagnostic.translation_error_normalized**2))
                    ),
                    "translation_p95": float(
                        np.percentile(diagnostic.translation_error_normalized, 95)
                    ),
                    "rotation_mean": float(rotation.mean()),
                    "translation_direction_mean": float(translation.mean()),
                }
            )
    effects = {}
    for method in ("dust3r", "mast3r", "vggt"):
        native = {r["sample_id"]: r for r in rows if r["method"] == method and r["sigma"] == 0}
        blurred = {r["sample_id"]: r for r in rows if r["method"] == method and r["sigma"] == 1.5}
        ids = sorted(native.keys() & blurred.keys())
        difference = np.array([blurred[s]["auc30"] - native[s]["auc30"] for s in ids])
        rng = np.random.default_rng(109)
        bootstrap = difference[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(axis=1)
        effects[method] = {
            "n_paired_clips": len(ids),
            "native_auc30": float(np.mean([native[s]["auc30"] for s in ids])),
            "blurred_auc30": float(np.mean([blurred[s]["auc30"] for s in ids])),
            "mean_delta": float(difference.mean()),
            "paired_clip_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    # Each clip contributes three method pairs; keep ties explicit.
    comparisons = []
    for sample in sorted({r["sample_id"] for r in rows}):
        native = {r["method"]: r for r in rows if r["sample_id"] == sample and r["sigma"] == 0}
        for a, b in combinations(sorted(native), 2):
            auc_delta = native[a]["auc30"] - native[b]["auc30"]
            ate_delta = native[b]["normalized_ate"] - native[a]["normalized_ate"]
            comparisons.append(
                {
                    "sample_id": sample,
                    "a": a,
                    "b": b,
                    "auc_delta": auc_delta,
                    "ate_advantage": ate_delta,
                    "tie": abs(auc_delta) < 1e-8 or abs(ate_delta) < 1e-8,
                    "disagrees": auc_delta * ate_delta < 0,
                }
            )
    decisive = [r for r in comparisons if not r["tie"]]
    return {
        "source_provenance_limit": source["provenance_limit"],
        "rows": rows,
        "failures": failures,
        "blur_effects": effects,
        "comparisons": comparisons,
        "native_pair_count": len(comparisons),
        "decisive_pair_count": len(decisive),
        "disagreement_count": sum(r["disagrees"] for r in decisive),
        "interpretation": "Metric disagreement is not evidence of which metric or method is better.",
    }


if __name__ == "__main__":
    source = json.loads((ROOT / "results" / "realestate_pose_export.json").read_text())
    result = analyze(source)
    (ROOT / "results" / "realestate_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"rows", "comparisons"}},
            indent=2,
        )
    )
