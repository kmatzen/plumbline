"""Fixed rendered LK diagnostic; exact correspondence labels, not camera DNF."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from paper.render_camera_control import anchors, project_anchors, render_frame
from paper.stage_static_rgbd_control import digest


def run(output):
    if output.exists():
        raise ValueError("preserve prior result")
    cases = []
    for seed in range(10):
        for occluder in (False, True):
            frames = [render_frame(seed, t, occluder=occluder)[0] for t in range(24)]
            gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in frames]
            start, visible = project_anchors(anchors(), 0, occluder=occluder)
            if not visible.all():
                raise ValueError("all fixed queries must be visible initially")
            current = start.astype(np.float32).reshape(-1, 1, 2)
            alive = np.ones(12, dtype=bool)
            rows = []
            for t in range(1, 24):
                predicted, status, _ = cv2.calcOpticalFlowPyrLK(
                    gray[t-1], gray[t], current, None, winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray[t], gray[t-1], predicted, None, winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                fb = np.linalg.norm(backward[:, 0]-current[:, 0], axis=1)
                alive &= (status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
                          & np.isfinite(predicted).all(axis=(1, 2)) & (fb <= 1))
                reference, visible = project_anchors(anchors(), t, occluder=occluder)
                for slot in range(12):
                    rows.append({"frame": t, "slot": slot, "accepted": bool(alive[slot]),
                                 "visible": bool(visible[slot]),
                                 "predicted_xy": predicted[slot, 0].tolist(),
                                 "reference_xy": reference[slot].tolist(),
                                 "error_pixels": float(np.linalg.norm(predicted[slot, 0]-reference[slot])),
                                 "fb_pixels": float(fb[slot])})
                current = predicted
            cases.append({"seed": seed, "occluder": occluder,
                          "rgb_sha256": hashlib.sha256(np.stack(frames).tobytes()).hexdigest(), "rows": rows})
    summaries = []
    for occluder in (False, True):
        rows = [r for c in cases if c["occluder"] == occluder for r in c["rows"]]
        accepted = [r for r in rows if r["accepted"]]
        correct = [r for r in accepted if r["visible"] and r["error_pixels"] <= 2]
        summaries.append({"occluder": occluder, "planned": len(rows), "accepted": len(accepted),
                          "correct_visible_within_2px": len(correct),
                          "accepted_occluded": sum(not r["visible"] for r in accepted),
                          "accepted_visible_over_2px": sum(r["visible"] and r["error_pixels"] > 2 for r in accepted)})
    result = {"scope": "rendered correspondence control only; no camera failure labels or real-data claims",
              "protocol": "seeds 0..9; clean and moving occluder; all 12 fixed anchors, 23 later frames; no restart; LK forward/backward <=1px; correctness <=2px",
              "limitations": "single textured plane; procedural appearance; no real camera method or VLM; correlated views and paired conditions",
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "source_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("render_camera_control.py"))},
              "summaries": summaries, "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(run(parser.parse_args().output), indent=2))
