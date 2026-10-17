"""Verify a fresh dense run and create reference-assisted diagnostic evidence.

This does not assign human or VLM usability labels. Source-image hashes are
retained but are only reverified when the source files are available separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from plumbline.tracking_review import write_tracking_review_artifact


def load_verified(run: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("run is not complete")
    prediction_path = run / "prediction.npz"
    if (
        hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        != manifest["prediction_file_sha256"]
    ):
        raise ValueError("prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["extrinsics"]
    with np.load(run / "reference.npz", allow_pickle=False) as archive:
        reference = archive["extrinsics"]
    if hashlib.sha256(reference.tobytes()).hexdigest() != manifest["reference_sha256"]:
        raise ValueError("reference hash mismatch")
    n = len(manifest["frames"])
    if n < 2 or prediction.shape != (n, 4, 4) or reference.shape != (n, 4, 4):
        raise ValueError("pose arrays do not match recorded frame count")
    timestamps = np.array([float(f["timestamp"]) for f in manifest["frames"]])
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError("timestamps must be finite and strictly increasing")
    return manifest, prediction, reference


def analyze(run: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("choose a new output directory")
    manifest, prediction, reference = load_verified(run)
    source_hash = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    artifact = write_tracking_review_artifact(
        prediction,
        reference,
        output,
        dataset="tum-dynamics-development",
        sequence=manifest["sequence"],
        system="dage",
        config_hash=source_hash[:16],
    )
    artifact_manifest = json.loads((artifact / "manifest.json").read_text())
    gate = artifact_manifest["gate"]
    timestamps = np.array([float(f["timestamp"]) for f in manifest["frames"]])
    result = {
        "source_manifest_sha256": source_hash,
        "source_run": str(run),
        "sequence": manifest["sequence"],
        "n_frames": len(timestamps),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "median_frame_interval_seconds": float(np.median(np.diff(timestamps))),
        "gate": gate,
        "diagnostic_unavailable_reason": artifact_manifest.get("diagnostic_unavailable_reason"),
        "artifact": str(artifact),
        "metrics": json.loads((artifact / "metrics.json").read_text()),
        "functional_verdict": "unreviewed",
        "evidence_scope": "reference-assisted; source-video usability has not been judged",
        "verification": "prediction bytes and reference array verified; source-frame hashes retained, not rechecked here",
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))
