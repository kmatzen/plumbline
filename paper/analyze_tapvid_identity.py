"""Sequence-level summaries; never treat correlated point frames as independent."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(result):
    names = sorted(result["sequences"])
    rng = np.random.default_rng(677)
    samples = rng.integers(0, len(names), (20000, len(names)))
    output = {}
    for gate in ("lk_available", "fb_1px", "ncc_0.8", "both"):
        counts = [result["sequences"][name]["counts"][gate] for name in names]
        for c in counts:
            assert c["accepted"] == c["accepted_correct"] + c["accepted_wrong"] + c["accepted_occluded"]
            assert c["visible"] == c["accepted_correct"] + c["accepted_wrong"] + c["rejected_visible"]
        rows = {}
        for metric, numerator, denominator in (
                ("accepted_precision", "accepted_correct", "accepted"),
                ("visible_correct_recall", "accepted_correct", "visible"),
                ("acceptance_fraction", "accepted", "planned")):
            values = np.array([c[numerator]/c[denominator] if c[denominator] else np.nan for c in counts])
            # Keep undefined values explicit. No CI silently drops zero-coverage clips.
            complete = bool(np.isfinite(values).all())
            rows[metric] = {"per_video": {name: float(v) if np.isfinite(v) else None for name, v in zip(names, values)},
                            "macro_mean": float(values.mean()) if complete else None,
                            "defined_videos": int(np.isfinite(values).sum()),
                            "clip_bootstrap_95_percentile": np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist() if complete else None}
        output[gate] = {"metrics": rows, "pooled_counts_descriptive_only": {k: sum(c[k] for c in counts) for k in counts[0]}}
    return {"videos": names, "video_count": len(names), "gates": output,
            "uncertainty": "20000 sequence-resampling bootstrap draws, seed 677; descriptive population-sampling uncertainty, not correction for model exposure or annotation error"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous summary")
    payload = args.source.read_bytes()
    result = summarize(json.loads(gzip.decompress(payload)))
    result["source_sha256"] = hashlib.sha256(payload).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
