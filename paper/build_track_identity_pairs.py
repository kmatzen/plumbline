"""Matched-size RGB crop pairs for reviewing track identity, without camera scores."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper.build_source_crops import centered_crop
from paper.stage_static_rgbd_control import digest


def pair_image(first, second):
    """Equal native-resolution panels; the marker rings do not cover the center."""
    if first.shape != (161, 161, 3) or second.shape != first.shape:
        raise ValueError("two native 161-pixel color crops required")
    panel = np.full((193, 338, 3), 240, dtype=np.uint8)
    for x, label, crop in ((0, "A", first), (177, "B", second)):
        panel[32:, x:x+161] = crop
        cv2.putText(panel, label, (x+75, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(panel, (x+80, 112), 7, (255, 255, 0), 1, cv2.LINE_AA)
    return panel


def build(root, tracks_path, output):
    if output.exists():
        raise ValueError("preserve previous pairs")
    source = root / "staging-manifest.json"
    manifest = json.loads(source.read_text())
    tracks = json.loads(tracks_path.read_text())
    if tracks["source_manifest_sha256"] != digest(source) or len(tracks["tracks"]) != 12:
        raise ValueError("track source or slot count mismatch")
    images = {}
    for frame in (1, 89):
        info = manifest["frames"][frame]
        path = root / info["path"]
        if digest(path) != info["file_sha256"]:
            raise ValueError("source RGB changed")
        images[frame] = cv2.imread(str(path))
        if images[frame] is None or images[frame].shape != (480, 640, 3):
            raise ValueError("invalid native RGB image")
    output.mkdir(parents=True)
    items = []
    for slot, track in enumerate(tracks["tracks"]):
        if len(track) != 90:
            raise ValueError("expected all original frames")
        missing = [i for i in (1, 89) if track[i]["xy"] is None]
        if missing:
            items.append({"slot": slot, "status": "unavailable", "missing_frames": missing})
            continue
        crops = [centered_crop(images[i], track[i]["xy"]) for i in (1, 89)]
        # Same-image duplicates are only positive visual controls. Temporal pairs
        # have no correctness label, even if a tracker retained both endpoints.
        for kind, right in (("temporal", crops[1][0]), ("duplicate_control", crops[0][0])):
            path = output / f"slot-{slot}-{kind}.png"
            if not cv2.imwrite(str(path), pair_image(crops[0][0], right)):
                raise RuntimeError("failed to save pair")
            items.append({"slot": slot, "status": "ready", "kind": kind,
                          "frames": [1, 89] if kind == "temporal" else [1, 1],
                          "rounded_pixels": [crops[0][1], crops[1][1] if kind == "temporal" else crops[0][1]],
                          "image": path.name, "image_sha256": digest(path),
                          "label": "identical_image_detail" if kind == "duplicate_control" else None})
    record = {"scope": "track-identity pilot inputs; temporal pairs unlabeled, not DNF evidence",
              "policy": "all 12 slots retained; frame 1 to 89; missing endpoints unavailable; no camera/depth-based filtering",
              "display": "equal native-resolution 161x161 crops; cyan ring at rounded target; neutral padding",
              "limitation": "duplicate controls cover only slots with both endpoints; no negative controls or independent labels",
              "source_manifest_sha256": digest(source), "tracks_sha256": digest(tracks_path),
              "script_sha256": digest(Path(__file__)),
              "crop_script_sha256": digest(Path(__file__).with_name("build_source_crops.py")), "items": items}
    (output / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = build(args.root, args.tracks, args.output)
    print("Retained", len(result["items"]), "pair/control/missing records")
