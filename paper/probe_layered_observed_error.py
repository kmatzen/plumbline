"""Continuous geometric baseline with fixed RGB tracks, no oracle filtering."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_evidence_sensitivity import alternatives
from paper.probe_layered_minimum_error import minimum_box_error
from paper.render_layered_camera_control import K, camera_pose
from paper.stage_static_rgbd_control import digest


def compare_brackets(original, changed):
    if any(r["status"] != "numerical_bracket" for r in (original, changed)):
        return "unresolved"
    if changed["lower"] > original["upper"]:
        return "changed_higher_error"
    if changed["upper"] < original["lower"]:
        return "changed_lower_error"
    return "overlapping_brackets"


def assess(cameras, track):
    indices = [i for i, p in enumerate(track) if p is not None]
    if len(indices) < 2:
        return dict(status="insufficient_observations", indices=indices)
    xy = np.array([track[i] for i in indices])
    result = minimum_box_error(cameras[indices], xy)
    if result["status"] == "numerical_bracket":
        camera = (cameras[indices] @ np.r_[result["upper_point_world"], 1.])[:, :3]
        projected = camera @ K.T
        error = float(np.abs(projected[:, :2]/projected[:, 2:]-xy).max())
        if not (camera[:, 2] > 0).all() or error > result["upper"]+1e-5:
            raise AssertionError("upper witness failed independent reprojection")
        result["audited_upper_witness_error"] = error
    return dict(indices=indices, **result)


def run(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    root = Path(__file__).parents[1]
    for name, expected in data["dependencies_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("control dependency changed")
    if [c["seed"] for c in data["cases"]] != list(range(10)):
        raise ValueError("complete fixed seed roster required")
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    cases, totals = [], {}
    for case in data["cases"]:
        tracks = case["tracks"]
        if len(tracks) != 12 or any(len(t) != 24 for t in tracks):
            raise ValueError("complete track roster required")
        original = None
        for name, changed in alternatives(cameras):
            slots = [dict(slot=i, **assess(changed, t)) for i, t in enumerate(tracks)]
            if name == "identity":
                original = slots
            comparisons = [compare_brackets(a, b) for a, b in zip(original, slots)]
            if name == "global_rigid_gauge" and any(
                    c not in ("overlapping_brackets", "unresolved") for c in comparisons):
                raise AssertionError("gauge changed error brackets")
            counts = Counter(comparisons)
            totals.setdefault(name, Counter()).update(counts)
            cases.append(dict(seed=case["seed"], change=name, slots=slots,
                              comparisons=comparisons, counts=dict(counts)))
        print("completed seed", case["seed"], flush=True)
    return dict(scope="fixed RGB-track classical baseline on reference-camera controls; not DNF",
                protocol="all 10 seeds, 12 tracks, 14 camera variants; no oracle visibility/coordinate filtering; unchanged .01px bisection; all accepted frames",
                limitations="correlated scene/path; per-track comparisons not sequence failure rates; lower error need not mean better cameras; numerical bounds not formal certificates",
                source_sha256=digest(source),
                dependencies_sha256={str(p.relative_to(root)):digest(p) for p in
                    (Path(__file__), Path(__file__).with_name("probe_layered_minimum_error.py"))},
                totals={k:dict(v) for k,v in totals.items()}, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous result")
    result = run(args.source)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    print(json.dumps(result["totals"], indent=2))
