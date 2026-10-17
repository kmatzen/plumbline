"""Development-only image checks for independent first-visible LK tracks."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def patch(gray, xy):
    x, y = map(float, xy)
    if not (8 <= x < gray.shape[1]-8 and 8 <= y < gray.shape[0]-8):
        return None
    return cv2.getRectSubPix(gray, (17, 17), (x, y)).astype(float)


def correlation(a, b):
    if a is None or b is None:
        return None
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 1e-8:
        return None
    return float(np.clip(np.sum(a*b)/norm, -1, 1))


def predict(gray, query_frame, query_xy):
    """Image-only candidate and checks: no later annotation or visibility inputs."""
    p = np.array(query_xy, dtype=np.float32).reshape(1, 1, 2)
    source = patch(gray[query_frame], query_xy)
    records = []
    alive = True
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    for t in range(query_frame+1, len(gray)):
        row = {"frame": t, "xy": None, "forward_backward_pixels": None, "source_patch_ncc": None}
        if alive:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray[t-1], gray[t], p, None, **options)
            alive = bool(nxt is not None and status[0, 0] and np.isfinite(nxt).all())
            if alive:
                x, y = nxt.reshape(2)
                alive = bool(0 <= x < gray[t].shape[1] and 0 <= y < gray[t].shape[0])
            if alive:
                back, ok, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t-1], nxt, None, **options)
                if back is not None and ok[0, 0] and np.isfinite(back).all():
                    row["forward_backward_pixels"] = float(np.linalg.norm(back-p))
                row["xy"] = nxt.reshape(2).astype(float).tolist()
                row["source_patch_ncc"] = correlation(source, patch(gray[t], row["xy"]))
                p = nxt
        records.append(row)
    return records


def decisions(row):
    present = row["xy"] is not None
    fb = present and row["forward_backward_pixels"] is not None and row["forward_backward_pixels"] <= 1
    ncc = present and row["source_patch_ncc"] is not None and row["source_patch_ncc"] >= .8
    return {"lk_available": present, "fb_1px": fb, "ncc_0.8": ncc, "both": fb and ncc}


def score(records, normalized_points, occluded, width, height):
    counts = {name: dict(accepted_correct=0, accepted_wrong=0, accepted_occluded=0,
                         rejected_visible=0, accepted=0, planned=0, visible=0) for name in
              ("lk_available", "fb_1px", "ncc_0.8", "both")}
    for row in records:
        t = row["frame"]
        visible = not bool(occluded[t])
        error = None if row["xy"] is None else float(np.linalg.norm(
            (np.array(row["xy"])/[width, height] - normalized_points[t]) * 256))
        row["scoring_only"] = {"visible": visible, "error_at_256_pixels": error}
        for name, accept in decisions(row).items():
            c = counts[name]
            c["planned"] += 1
            c["visible"] += int(visible)
            c["accepted"] += int(accept)
            c["accepted_correct"] += int(accept and visible and error < 4)
            c["accepted_wrong"] += int(accept and visible and error >= 4)
            c["accepted_occluded"] += int(accept and not visible)
            c["rejected_visible"] += int(not accept and visible)
    return counts


def run(root, output, split="development"):
    if split not in {"development", "held_out"}:
        raise ValueError("explicit valid split required")
    if output.exists():
        raise ValueError("preserve previous output")
    manifest = json.loads((root / "manifest.json").read_text())
    results = {}
    cv2.setNumThreads(1)
    for name, entry in manifest["sequences"].items():
        if entry["split"] != split:
            continue  # Never open arrays outside the explicitly requested split.
        path = root / entry["file"]
        if digest(path) != entry["file_sha256"]:
            raise ValueError("changed input archive")
        with np.load(path, allow_pickle=False) as a:
            video, points, occ = a["video"], a["points"], a["occluded"]
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
        height, width = video.shape[1:3]
        del video
        tracks = []
        for index in range(len(points)):
            visible = np.flatnonzero(~occ[index])
            if not len(visible):
                tracks.append({"track": index, "status": "no_visible_query"})
                continue
            first = int(visible[0])
            query = points[index, first] * [width, height]
            records = predict(gray, first, query)
            # Scoring happens only after all predictions/checks for this track exist.
            counts = score(records, points[index], occ[index], width, height)
            tracks.append({"track": index, "query_frame": first, "query_xy": query.tolist(),
                           "counts": counts, "records": records})
        totals = {}
        for track in tracks:
            for gate, counts in track.get("counts", {}).items():
                dest = totals.setdefault(gate, {key: 0 for key in counts})
                for key, value in counts.items():
                    dest[key] += value
        results[name] = {"file_sha256": entry["file_sha256"], "tracks": tracks, "counts": totals}
        print(name, totals, flush=True)
    result = {"scope": "correspondence checks only; not camera DNF or full TAP-Vid benchmark",
              "split": split,
              "source_manifest_sha256": digest(root / "manifest.json"),
              "script_sha256": digest(Path(__file__)), "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "sequences": results,
              "protocol": "native resolution; one independent LK track per first-visible query; no restart; FB <=1 native px; source NCC >=0.8 on 17x17 gray patch; correctness visible and error <4 px after 256x256 coordinate scaling",
              "limitations": "no staticness labels; fixed patch lacks scale/rotation invariance; first-visible annotation initializes query; accepted occluded positions counted as errors; no threshold search"}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(json.dumps(result, allow_nan=False).encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "held_out"), default="development")
    args = parser.parse_args()
    run(args.root, args.output, args.split)
