"""Paired exact/observed-track camera consistency on the rendered control."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_consistency_attribution import yaw_suffix
from paper.probe_joint_query_consistency import compare_queries
from paper.render_camera_control import K, anchors, camera_pose, project_anchors
from paper.stage_static_rgbd_control import digest


def tracks_for(case, arm):
    if arm not in {"observed", "exact_visible"}:
        raise ValueError("unknown track arm")
    start, _ = project_anchors(anchors(), 0, occluder=case["occluder"])
    tracks = [[start[slot].tolist()] + [None]*23 for slot in range(12)]
    seen = set()
    for row in case["rows"]:
        t, slot = row["frame"], row["slot"]
        if not 1 <= t <= 23 or not 0 <= slot < 12 or (t, slot) in seen:
            raise ValueError("invalid or duplicate observation")
        seen.add((t, slot))
        # The observed arm does not read visibility, reference xy, or error.
        if arm == "observed":
            tracks[slot][t] = row["predicted_xy"] if row["accepted"] else None
        else:
            tracks[slot][t] = row["reference_xy"] if row["visible"] else None
    if len(seen) != 276:
        raise ValueError("all planned observations required")
    return tracks


def run(source, output):
    if output.exists():
        raise ValueError("preserve prior result")
    data = json.loads(source.read_text())
    for name, expected in data["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("source implementation changed")
    if [(c["seed"], c["occluder"]) for c in data["cases"]] != [
        (seed, occ) for seed in range(10) for occ in (False, True)
    ]:
        raise ValueError("complete fixed source roster required")
    reference = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    results = []
    totals = {}
    for case in data["cases"]:
        for arm in ("observed", "exact_visible"):
            tracks = tracks_for(case, arm)
            for angle in (0, 1, 5):
                cameras = yaw_suffix(reference, angle)
                slots = [{"slot": slot, **compare_queries(cameras[:12], track[:12],
                          cameras[12:], track[12:], K, 2)} for slot, track in enumerate(tracks)]
                counts = Counter(s["status"] for s in slots)
                key = f"occluder={case['occluder']},arm={arm},yaw={angle}"
                totals.setdefault(key, Counter()).update(counts)
                results.append({"seed": case["seed"], "occluder": case["occluder"],
                                "track_arm": arm, "suffix_yaw_degrees": angle,
                                "slots": slots, "counts": dict(counts)})
    root = Path(__file__).parents[1]
    dependencies = [Path(__file__), Path(__file__).with_name("render_camera_control.py"),
                    Path(__file__).with_name("probe_consistency_attribution.py"),
                    Path(__file__).with_name("probe_joint_query_consistency.py"),
                    Path(__file__).with_name("probe_static_query_consistency.py"),
                    root / "src/plumbline/frustum_support.py"]
    result = {"scope": "controlled geometric sensitivity, not real-camera DNF or learned evaluation",
              "protocol": "all 20 rendered cases; fit frames 0..11; query 12..23; tolerance 2 pixels per coordinate; suffix yaw ramps 0/1/5 degrees; all 12 slots",
              "limitations": "exact-visible arm is an oracle diagnostic; same planar geometry across texture seeds; missing fit observations cause abstention; compatibility is not correctness",
              "source_sha256": digest(source),
              "dependencies_sha256": {str(p.relative_to(root)): digest(p) for p in dependencies},
              "totals": {k: dict(v) for k, v in totals.items()}, "cases": results}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    return result["totals"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
