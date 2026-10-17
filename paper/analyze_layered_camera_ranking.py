"""Exploratory camera-level median baseline; not a held-out selection result."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.analyze_camera_evidence_coverage import analyze as coverage_audit
from paper.stage_static_rgbd_control import digest


def median_bracket(slots):
    """All planned tracks retained; unresolved errors range over [0, infinity]."""
    if not slots:
        raise ValueError("nonempty planned roster required")
    lower, upper = [], []
    unresolved = 0
    for slot in slots:
        if slot["status"] == "numerical_bracket":
            lo, hi = slot["lower"], slot["upper"]
            if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError("invalid numerical bracket")
        else:
            lo, hi = 0., float("inf")
            unresolved += 1
        lower.append(lo)
        upper.append(hi)
    hi = float(np.median(upper))
    return dict(lower=float(np.median(lower)), upper=hi if np.isfinite(hi) else None,
                planned_tracks=len(slots), unresolved_tracks=unresolved)


def compare(original, changed):
    if original["upper"] is not None and changed["lower"] > original["upper"]:
        return "changed_higher_error"
    if changed["upper"] is not None and changed["upper"] < original["lower"]:
        return "changed_lower_error"
    return "not_separated"


def analyze(source, tracks):
    coverage = coverage_audit(source, tracks)
    data = json.loads(gzip.decompress(source.read_bytes()))
    indexed = {(c["seed"], c["change"]): c for c in data["cases"]}
    cases, counts = [], Counter()
    for case in coverage["cases"]:
        seed, name = case["seed"], case["change"]
        original = median_bracket(indexed[seed, "identity"]["slots"])
        changed = median_bracket(indexed[seed, name]["slots"])
        verdict = compare(original, changed)
        counts[verdict] += 1
        cases.append(dict(seed=seed, change=name, original=original, changed=changed,
                          verdict=verdict, tracks_with_changed_view_evidence=sum(
                              s["has_changed_view_evidence"] for s in case["slots"])))
    return dict(scope="exploratory median baseline on development camera controls, not functional DNF",
                protocol="median of all 12 planned track errors; unresolved scores bounded by [0,infinity], never dropped; identical retained observations per paired camera; coverage reported separately",
                limitations="aggregate chosen after track-level inspection; one shared scene/path; correlated seeds/changes; synthetic camera perturbations; not independent confirmation or novelty",
                source_sha256=digest(source), tracks_sha256=digest(tracks), script_sha256=digest(Path(__file__)),
                planned_camera_comparisons=len(cases), counts=dict(counts), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = analyze(args.source, args.tracks)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["counts"], indent=2))
