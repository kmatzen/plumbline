"""Fixed non-planar sensitivity control with separate observed/reference arms."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.probe_evidence_sensitivity import alternatives, joint_status
from paper.render_layered_camera_control import anchors, camera_pose, project_anchors, render_frame
from paper.stage_static_rgbd_control import digest


def track_rgb(frames, start):
    """RGB and initial queries only; no later reference coordinates/visibility."""
    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in frames]
    current = np.asarray(start, dtype=np.float32).reshape(-1, 1, 2)
    alive = np.ones(len(current), dtype=bool)
    tracks = [[p.tolist()] for p in current[:, 0]]
    for t in range(1, len(frames)):
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        predicted, status, _ = cv2.calcOpticalFlowPyrLK(gray[t-1], gray[t], current, None, **options)
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t-1], predicted, None, **options)
        fb = np.linalg.norm(backward[:, 0]-current[:, 0], axis=1)
        alive &= (status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
                  & np.isfinite(predicted).all(axis=(1, 2)) & (fb <= 1))
        for slot, track in enumerate(tracks):
            track.append(predicted[slot, 0].tolist() if alive[slot] else None)
        current = predicted
    return tracks


def score_tracks(tracks, reference, visibility):
    counts = Counter(planned=0, accepted=0, correct_visible_within_2px=0,
                     accepted_occluded=0, accepted_visible_over_2px=0)
    for slot, track in enumerate(tracks):
        for t in range(1, len(track)):
            counts["planned"] += 1
            if track[t] is None:
                continue
            counts["accepted"] += 1
            if not visibility[t, slot]:
                counts["accepted_occluded"] += 1
            elif np.linalg.norm(np.array(track[t])-reference[t, slot]) <= 2:
                counts["correct_visible_within_2px"] += 1
            else:
                counts["accepted_visible_over_2px"] += 1
    return dict(counts)


def run(output):
    if output.exists():
        raise ValueError("preserve previous results")
    points, surface_ids = anchors()
    references = [project_anchors(points, surface_ids, t) for t in range(24)]
    xy = np.stack([r[0] for r in references])
    visible = np.stack([r[1] for r in references])
    if not visible[0].all():
        raise ValueError("all fixed initial queries must be visible")
    exact = [[xy[t, s].tolist() if visible[t, s] else None for t in range(24)]
             for s in range(12)]
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    cases, totals, tracking = [], {}, Counter()
    for seed in range(10):
        frames = [render_frame(seed, t)[0] for t in range(24)]
        observed = track_rgb(frames, xy[0])
        scored = score_tracks(observed, xy, visible)
        tracking.update(scored)
        profiles = []
        for arm, tracks in (("observed", observed), ("exact_visible", exact)):
            original = None
            for change, changed in alternatives(cameras):
                slots = [{"slot": s, **joint_status(changed, track)} for s, track in enumerate(tracks)]
                statuses = [s["status"] for s in slots]
                if change == "identity":
                    original = statuses
                if change == "global_rigid_gauge" and statuses != original:
                    raise AssertionError("global gauge changed geometric verdicts")
                counts = Counter(statuses)
                totals.setdefault(f"{arm}/{change}", Counter()).update(counts)
                profiles.append(dict(arm=arm, change=change, slots=slots, counts=dict(counts)))
        cases.append(dict(seed=seed, rgb_sha256=hashlib.sha256(np.stack(frames).tobytes()).hexdigest(),
                          tracks=observed, tracking=scored, profiles=profiles))
    root = Path(__file__).parents[1]
    files = [Path(__file__), Path(__file__).with_name("render_layered_camera_control.py"),
             Path(__file__).with_name("render_camera_control.py"),
             Path(__file__).with_name("probe_evidence_sensitivity.py"),
             root/"src/plumbline/frustum_support.py"]
    result = dict(scope="three-depth reference-camera control, not real-method ranking or functional DNF",
                  protocol="seeds 0..9; all 12 geometry-fixed anchors; sequential LK FB<=1 without restart; 2px feasibility; unchanged 14 camera changes; prefix frames 0..11",
                  limitations="one geometry/path; seeds are correlated; exact arm is an oracle control only; finite camera changes are not functional failure labels",
                  versions=dict(numpy=np.__version__, opencv=cv2.__version__),
                  dependencies_sha256={str(p.relative_to(root)): digest(p) for p in files},
                  reference_xy=xy.tolist(), reference_visible=visible.tolist(), surface_ids=surface_ids.tolist(),
                  tracking=dict(tracking), totals={k:dict(v) for k,v in totals.items()}, cases=cases)
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    return {"tracking":result["tracking"], "totals":result["totals"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(run(parser.parse_args().output), indent=2))
