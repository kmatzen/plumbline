"""Separate absent changed-view evidence from geometric score ties."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_evidence_sensitivity import alternatives
from paper.probe_layered_observed_error import compare_brackets
from paper.render_layered_camera_control import camera_pose
from paper.stage_static_rgbd_control import digest


def classify(original, changed, changed_views):
    before, after = original["indices"], changed["indices"]
    if before != after or len(before) != len(set(before)):
        raise ValueError("paired scores require identical unique observation indices")
    supported = sorted(set(before) & set(changed_views))
    raw = compare_brackets(original, changed)
    if not changed_views:
        verdict = "no_camera_change"
    elif not supported:
        verdict = "no_changed_view_evidence"
    else:
        verdict = raw
    return dict(verdict=verdict, raw_bracket_comparison=raw,
                retained_observations=len(before), supported_changed_views=supported,
                has_changed_view_evidence=bool(supported))


def analyze(source, tracks_path):
    result = json.loads(gzip.decompress(source.read_bytes()))
    if result["source_sha256"] != digest(tracks_path):
        raise ValueError("track input hash mismatch")
    tracks = json.loads(gzip.decompress(tracks_path.read_bytes()))
    root = Path(__file__).parents[1]
    for data in (result, tracks):
        for name, expected in data["dependencies_sha256"].items():
            if digest(root/name) != expected:
                raise ValueError("dependency changed")
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    changes = {name: np.flatnonzero(np.max(np.abs(value-cameras), axis=(1, 2)) > 1e-10).tolist()
               for name, value in alternatives(cameras)}
    # Gauge is a representation control, not a physical perturbation.
    excluded = {"identity", "global_rigid_gauge"}
    expected = {(seed, name) for seed in range(10) for name in changes}
    indexed = {(c["seed"], c["change"]):c for c in result["cases"]}
    if set(indexed) != expected or len(indexed) != len(result["cases"]):
        raise ValueError("complete unique case roster required")
    cases, totals = [], Counter()
    for seed in range(10):
        base = indexed[seed, "identity"]["slots"]
        if [s["slot"] for s in base] != list(range(12)):
            raise ValueError("complete slot roster required")
        for name in changes:
            if name in excluded:
                continue
            candidate = indexed[seed, name]["slots"]
            if [s["slot"] for s in candidate] != list(range(12)):
                raise ValueError("complete candidate slot roster required")
            slots = [dict(slot=i, **classify(a, b, changes[name]))
                     for i, (a, b) in enumerate(zip(base, candidate, strict=True))]
            totals.update(s["verdict"] for s in slots)
            cases.append(dict(seed=seed, change=name, changed_views=changes[name], slots=slots))
    planned = sum(totals.values())
    return dict(scope="evidence coverage audit, not a DNF score or observability certificate",
                policy="identity/gauge excluded; paired identical observations; no changed-view observations reported separately from score ties; every planned physical comparison retained",
                limitations="one retained changed view is necessary but not sufficient evidence; correlated tracks/seeds; synthetic controls; no functional failure labels",
                source_sha256=digest(source), tracks_sha256=digest(tracks_path), script_sha256=digest(Path(__file__)),
                planned_comparisons=planned, counts=dict(totals),
                changed_view_evidence_fraction=1-totals["no_changed_view_evidence"]/planned,
                higher_error_fraction_of_planned=totals["changed_higher_error"]/planned,
                lower_error_fraction_of_planned=totals["changed_lower_error"]/planned,
                cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = analyze(args.source, args.tracks)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}, indent=2))
