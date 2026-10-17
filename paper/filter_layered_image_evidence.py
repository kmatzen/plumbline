"""Frozen NCC admission on existing RGB tracks; no camera-dependent selection."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.probe_layered_camera_control import score_tracks
from paper.probe_tapvid_identity import correlation, patch
from paper.render_layered_camera_control import render_frame
from paper.stage_static_rgbd_control import digest


def admit(gray, tracks):
    selected, scores = [], []
    for track in tracks:
        origin = patch(gray[0], track[0]) if track[0] is not None else None
        retained, nccs = [], []
        for frame, xy in enumerate(track):
            score = correlation(origin, patch(gray[frame], xy)) if xy is not None else None
            # The supplied initial query is not a predicted correspondence.
            keep = xy is not None and (frame == 0 or (score is not None and score >= .8))
            retained.append(xy if keep else None)
            nccs.append(score)
        selected.append(retained)
        scores.append(nccs)
    return selected, scores


def run(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    root = Path(__file__).parents[1]
    for name, expected in data["dependencies_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("control dependency changed")
    cases, totals = [], Counter()
    for case in data["cases"]:
        frames = [render_frame(case["seed"], t)[0] for t in range(24)]
        if hashlib.sha256(np.stack(frames).tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("rendered RGB mismatch")
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        tracks, scores = admit(gray, case["tracks"])
        # Reference fields are used only after image-only admission is complete.
        counts = score_tracks(tracks, np.array(data["reference_xy"]), np.array(data["reference_visible"]))
        totals.update(counts)
        cases.append(dict(seed=case["seed"], tracks=tracks, ncc=scores, tracking=counts,
                          rgb_sha256=case["rgb_sha256"]))
    dependencies = dict(data["dependencies_sha256"])
    for p in (Path(__file__), Path(__file__).with_name("probe_tapvid_identity.py")):
        dependencies[str(p.relative_to(root))] = digest(p)
    return dict(scope="fixed image-only admission baseline, not a new method or DNF",
                protocol="unchanged sequential LK coordinates; independent 17x17 source-patch NCC>=.8; initial queries retained; no camera or reference input to admission",
                source_sha256=digest(source), dependencies_sha256=dependencies,
                tracking=dict(totals), cases=cases)


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
    print(json.dumps(result["tracking"], indent=2))
