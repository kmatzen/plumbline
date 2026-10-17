"""Development-only correspondence supervision; never camera-DNF labels."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def partitions(names):
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("exactly ten distinct development videos required")
    ordered = sorted(names, key=lambda name: hashlib.sha256(("plumbline-track-identity-v1:"+name).encode()).hexdigest())
    return {n: "train" if i < 6 else "validation" if i < 8 else "calibration" for i, n in enumerate(ordered)}


def target_label(candidate_xy, reference_xy, occluded):
    if occluded:
        return "occluded"
    error = float(np.linalg.norm(np.array(candidate_xy) - reference_xy))
    if error <= 2:
        return "close"
    if error >= 8:
        return "far"
    return None  # Annotation precision and the 2–8px margin remain explicit.


def crop(image, xy):
    x, y = np.rint(xy).astype(int)
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        raise ValueError("rounded display point outside image")
    padded = cv2.copyMakeBorder(image, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return padded[y:y+161, x:x+161]


def render(first, second, source_xy, candidate_xy):
    """Only visible images and proposed coordinates enter rendering, never targets."""
    canvas = np.full((383, 544, 3), 240, np.uint8)
    for offset, title, frame, xy in ((0, "A: start", first, source_xy), (280, "B: candidate", second, candidate_xy)):
        cv2.putText(canvas, title, (offset+8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
        # Context preserves aspect ratio and is letterboxed, not stretched.
        h, w = frame.shape[:2]
        scale = min(264/w, 160/h)
        resized = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        y = 32+(160-resized.shape[0])//2
        canvas[y:y+resized.shape[0], offset:offset+resized.shape[1]] = resized
        cv2.circle(canvas, (offset+round(xy[0]*scale), y+round(xy[1]*scale)), 4, (0, 255, 255), 1, cv2.LINE_AA)
        canvas[214:375, offset+51:offset+212] = crop(frame, xy)
        cv2.circle(canvas, (offset+131, 294), 7, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


PROMPT = ('Compare the marked detail in A with the candidate marked in B. The top row shows context; '
          'the bottom row shows native-pixel crops. Is the candidate close to the same detail, far from it, '
          'or is the original detail occluded? Reply with one label: close, far, occluded, or uncertain. '
          'Close means within 2 native pixels; far means at least 8 native pixels. Use uncertain when the images do not support a decision. '
          'Do not judge camera accuracy or whether the object is moving.')


def run(root, source, output):
    if output.exists():
        raise ValueError("preserve previous corpus")
    staging = json.loads((root / "manifest.json").read_text())
    predictions = json.loads(gzip.decompress(source.read_bytes()))
    if predictions["source_manifest_sha256"] != digest(root / "manifest.json"):
        raise ValueError("source mismatch")
    names = [n for n, v in staging["sequences"].items() if v["split"] == "development"]
    if set(predictions["sequences"]) != set(names):
        raise ValueError("development-only prediction set required")
    split = partitions(names)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    records = []
    for name in names:
        entry = staging["sequences"][name]
        path = root / entry["file"]
        if digest(path) != entry["file_sha256"]:
            raise ValueError("changed video/annotations")
        with np.load(path, allow_pickle=False) as a:
            video, points, occ = a["video"], a["points"], a["occluded"]
        h, w = video.shape[1:3]
        for track in predictions["sequences"][name]["tracks"]:
            first = track["query_frame"]
            candidates = {r["frame"]: r for r in track["records"]}
            for lag in (5, 15, 30, 60):
                frame = first+lag
                identity = hashlib.sha256(f'{name}:{track["track"]}:{frame}'.encode()).hexdigest()[:20]
                record = {"id": identity, "sequence": name, "partition": split[name], "track": track["track"],
                          "query_frame": first, "frame": frame, "lag": lag, "label": None}
                row = candidates.get(frame)
                if frame >= len(video):
                    record["status"] = "beyond_video"
                elif row is None or row["xy"] is None:
                    record["status"] = "candidate_unavailable"
                else:
                    try:
                        panel = render(video[first], video[frame], track["query_xy"], row["xy"])
                    except ValueError:
                        record["status"] = "rounded_display_point_outside"
                        records.append(record)
                        continue
                    target = points[track["track"], frame] * [w, h]
                    label = target_label(row["xy"], target, bool(occ[track["track"], frame]))
                    image_path = output / "images" / (identity+".png")
                    if not cv2.imwrite(str(image_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("image write failed")
                    record.update(status="labeled" if label else "borderline_unlabeled", label=label,
                                  image=str(image_path.relative_to(output)), image_sha256=digest(image_path),
                                  source_xy=track["query_xy"], candidate_xy=row["xy"])
                records.append(record)
        print(name, split[name], flush=True)
    for partition in ("train", "validation", "calibration"):
        examples = [{"id": r["id"], "image": r["image"], "prompt": PROMPT, "target": r["label"]}
                    for r in records if r["partition"] == partition and r["status"] == "labeled"]
        (output / (partition+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in examples))
    manifest = {"scope": "candidate correspondence supervision only; not staticness, camera quality, or DNF",
                "source_manifest_sha256": digest(root / "manifest.json"), "predictions_sha256": digest(source),
                "script_sha256": digest(Path(__file__)), "opencv_version": cv2.__version__,
                "partition_policy": "six train, two validation, two calibration videos from development only; existing 20 held-out videos never opened",
                "label_policy": "reviewed point annotations; close <=2 native Euclidean pixels, far >=8, occluded separate; 2–8 margin unlabeled; no synthetic rationale or VLM teacher label",
                "selection": "lags 5/15/30/60 from first-visible query for every development track; missing and borderline cases retained in manifest",
                "limitations": "sparse frames; imperfect annotations; natural LK errors only; tiny video-level training set; no exact identity/staticness claim; validation/calibration are not fresh external tests",
                "prompt": PROMPT, "partitions": split,
                "exports_sha256": {p: digest(output / (p+".jsonl")) for p in ("train", "validation", "calibration")},
                "counts": {p: dict(Counter((r["label"] or r["status"]) for r in records if r["partition"] == p)) for p in ("train", "validation", "calibration")},
                "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(manifest["counts"], flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.source, args.output)
