"""Fixed first-to-target image matching on the complete layered-control roster."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.probe_layered_camera_control import score_tracks
from paper.probe_rendered_recovery import match_source
from paper.render_layered_camera_control import render_frame
from paper.stage_static_rgbd_control import digest


def direct_tracks(gray, queries):
    tracks = [[list(q)] for q in queries]
    records = []
    for t in range(1, len(gray)):
        matches = match_source(gray[0], gray[t], queries)
        if len(matches) != len(queries):
            raise ValueError("matcher changed query roster")
        for slot, row in enumerate(matches):
            tracks[slot].append(row["predicted_xy"] if row["accepted"] else None)
            records.append(dict(frame=t, slot=slot, **row))
    return tracks, records


def run(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    root = Path(__file__).parents[1]
    for name, expected in data["dependencies_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("source dependency changed")
    if [c["seed"] for c in data["cases"]] != list(range(10)):
        raise ValueError("complete seed roster required")
    cases, totals = [], Counter()
    for case in data["cases"]:
        frames = [render_frame(case["seed"], t)[0] for t in range(24)]
        if hashlib.sha256(np.stack(frames).tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("RGB changed")
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        queries = [t[0] for t in case["tracks"]]
        tracks, records = direct_tracks(gray, queries)
        counts = score_tracks(tracks, np.array(data["reference_xy"]), np.array(data["reference_visible"]))
        totals.update(counts)
        cases.append(dict(seed=case["seed"], tracks=tracks, matches=records,
                          rgb_sha256=case["rgb_sha256"], tracking=counts))
    dependencies = dict(data["dependencies_sha256"])
    for p in (Path(__file__), Path(__file__).with_name("probe_rendered_recovery.py"),
              Path(__file__).with_name("probe_tapvid_identity.py")):
        dependencies[str(p.relative_to(root))] = digest(p)
    return dict(scope="classical direct RGB matching control, not learned or camera-DNF validation",
                protocol="all fixed queries; first-to-target LK FB<=1 and 17x17 NCC>=.8; no reference/camera input; no temporal survival requirement",
                source_sha256=digest(source), dependencies_sha256=dependencies,
                tracking=dict(totals), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior results")
    result = run(args.source)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    print(json.dumps(result["tracking"], indent=2))
