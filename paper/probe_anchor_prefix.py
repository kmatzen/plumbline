"""Fixed-prefix sensitivity on retained TUM image tracks; no best-prefix choice."""

import argparse
import json
from pathlib import Path

import numpy as np
from paper.analyze_dense_run import load_verified
from paper.build_reference_free_review import digest
from paper.probe_davis_anchors import evaluate


def run(run_path, track_path, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    source = json.loads(track_path.read_text())
    manifest, prediction, reference = load_verified(run_path)
    if source["source_manifest_sha256"] != digest(run_path / "manifest.json"):
        raise ValueError("track source mismatch")
    if source["reference_file_sha256"] != digest(run_path / "reference.npz"):
        raise ValueError("reference calibration changed")
    size = np.array(manifest["shape"][1:3][::-1])
    k = np.load(run_path / "reference.npz")["intrinsics"].astype(float)
    k[:, :2, 2] += 0.5
    k[:, 0] /= size[0]
    k[:, 1] /= size[1]
    prefixes = [5, 11, 21, 31]
    cases = {}
    for count in prefixes:
        for name, poses in (("reference", reference), ("dage", prediction)):
            cases[f"{name}_{count}"] = [
                evaluate(t, poses, k, size, fit_count=count) for t in source["tracks"]
            ]
    rows = []
    for slot in range(len(source["tracks"])):
        for frame in range(max(prefixes), len(reference)):
            values = {
                name: anchors[slot]["rows"][frame]["distance_pixels"]
                for name, anchors in cases.items()
            }
            rows.append(
                {
                    "slot": slot,
                    "frame": frame,
                    "errors": values,
                    "common": all(v is not None for v in values.values()),
                }
            )
    common = [r for r in rows if r["common"]]
    result = {
        "status": "complete",
        "scope": "development prefix sensitivity, not threshold tuning or held-out validation",
        "track_source_sha256": digest(track_path),
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(Path(__file__).parent / "probe_davis_anchors.py"),
        "fit_counts": prefixes,
        "evaluation_start_frame": max(prefixes),
        "possible_observations": len(rows),
        "common_observations": len(common),
        "common_slots": sorted({r["slot"] for r in common}),
        "common_median_pixels": {
            name: float(np.median([r["errors"][name] for r in common])) if common else None
            for name in cases
        },
        "cases": cases,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "rows")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.run, args.tracks, args.output)
