"""Image-only direct source matching; fixed FB/NCC gates, no oracle recovery."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from paper.probe_tapvid_identity import correlation, patch
from paper.render_camera_control import anchors, project_anchors, render_frame
from paper.stage_static_rgbd_control import digest


def match_source(source, target, queries):
    """Only two grayscale images and original 2D query positions enter."""
    queries = np.asarray(queries, dtype=np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    predicted, status, _ = cv2.calcOpticalFlowPyrLK(source, target, queries, None, **options)
    if predicted is None:
        return [{"accepted": False, "predicted_xy": None, "fb_pixels": None, "ncc": None}
                for _ in queries]
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(target, source, predicted, None, **options)
    rows = []
    for slot, query in enumerate(queries[:, 0]):
        xy = predicted[slot, 0]
        finite = bool(np.isfinite(xy).all())
        valid = finite and bool(status[slot, 0])
        fb = None
        if backward is not None and back_status[slot, 0] and np.isfinite(backward[slot]).all():
            fb = float(np.linalg.norm(backward[slot, 0]-query))
        ncc = correlation(patch(source, query), patch(target, xy)) if finite else None
        accepted = bool(valid and fb is not None and fb <= 1 and ncc is not None and ncc >= .8)
        rows.append({"accepted": accepted, "predicted_xy": xy.astype(float).tolist() if finite else None,
                     "fb_pixels": fb, "ncc": ncc})
    return rows


def run(source, output):
    if output.exists():
        raise ValueError("preserve prior results")
    previous = json.loads(source.read_text())
    for name, expected in previous["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("source implementation changed")
    cases, summaries = [], []
    cv2.setNumThreads(1)
    for case in previous["cases"]:
        frames = [render_frame(case["seed"], t, occluder=case["occluder"])[0] for t in range(24)]
        if hashlib.sha256(np.stack(frames).tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("rendered input changed")
        gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in frames]
        queries, _ = project_anchors(anchors(), 0, occluder=case["occluder"])
        # All predictions before reading any later reference values.
        rows = [{"frame": t, "slot": slot, **row} for t in range(1, 24)
                for slot, row in enumerate(match_source(gray[0], gray[t], queries))]
        for row, old in zip(rows, case["rows"], strict=True):
            if (row["frame"], row["slot"]) != (old["frame"], old["slot"]):
                raise ValueError("observation order mismatch")
            row.update(visible=old["visible"], reference_xy=old["reference_xy"],
                       sequential_accepted=old["accepted"])
            row["error_pixels"] = None if row["predicted_xy"] is None else float(np.linalg.norm(
                np.array(row["predicted_xy"])-old["reference_xy"]))
        cases.append({"seed": case["seed"], "occluder": case["occluder"],
                      "rgb_sha256": case["rgb_sha256"], "rows": rows})
    for occ in (False, True):
        rows = [r for c in cases if c["occluder"] == occ for r in c["rows"]]
        accepted = [r for r in rows if r["accepted"]]
        correct = [r for r in accepted if r["visible"] and r["error_pixels"] <= 2]
        summaries.append({"occluder": occ, "planned": len(rows), "accepted": len(accepted),
                          "correct_visible_within_2px": len(correct),
                          "accepted_occluded": sum(not r["visible"] for r in accepted),
                          "accepted_visible_over_2px": sum(r["visible"] and r["error_pixels"] > 2 for r in accepted),
                          "correct_previously_rejected": sum(not r["sequential_accepted"] for r in correct)})
    result = {"scope": "direct image matching control, not validated camera DNF or a learned contribution",
              "protocol": "same source pixels and original 12 queries; direct first-to-target LK; FB<=1px and 17x17 source-patch NCC>=0.8; no temporal survival requirement; all observations retained",
              "limitations": "fixed simple procedural scene; thresholds reused from prior TAPvid baseline; no real-video recovery validation",
              "previous_sha256": digest(source), "opencv_version": cv2.__version__,
              "source_sha256": {p.name: digest(p) for p in (Path(__file__),
                  Path(__file__).with_name("render_camera_control.py"), Path(__file__).with_name("probe_tapvid_identity.py"))},
              "summaries": summaries, "cases": cases}
    output.write_text(json.dumps(result, allow_nan=False)+"\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
