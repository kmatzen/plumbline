"""All frame-zero grid targets, without later track/pose/depth selection."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper.stage_rgbd_validation import sha


def centered_crop(image, xy, radius=80):
    """Odd-width native-pixel crop, with the rounded target exactly central.

    Outside-source pixels are neutral gray, not replicated scene content.
    """
    x, y = np.rint(xy).astype(int)
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("target outside source")
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius,
                               cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return padded[y:y + 2 * radius + 1, x:x + 2 * radius + 1], [int(x), int(y)]


def build(run, source, tracks_path, output):
    if output.exists():
        raise ValueError("preserve existing crops")
    manifest = json.loads((run / "manifest.json").read_text())
    tracks = json.loads(tracks_path.read_text())
    if tracks["source_manifest_sha256"] != sha(run / "manifest.json"):
        raise ValueError("track source mismatch")
    frame = manifest["frames"][0]
    path = source / frame["path"]
    if sha(path) != frame["file_sha256"]:
        raise ValueError("source image mismatch")
    image = cv2.imread(str(path))
    output.mkdir(parents=True)
    items = []
    for slot, track in enumerate(tracks["tracks"]):
        xy = track[0]["xy"]
        if xy is None:
            items.append({"slot": slot, "status": "no_initial_target"})
            continue
        crop, pixel = centered_crop(image, xy)
        destination = output / f"slot-{slot}.png"
        if not cv2.imwrite(str(destination), crop):
            raise RuntimeError("cannot save crop")
        items.append({"slot": slot, "status": "ready", "xy": xy,
                      "rounded_pixel": pixel, "image": destination.name,
                      "image_sha256": sha(destination)})
    record = {
        "scope": "exploratory caption diagnostic, not held-out validation or DNF labels",
        "selection": "all frame-zero targets in original grid order; no later observation selection",
        "crop_policy": "161x161 native pixels centered at rounded target; gray padding outside source; no marker",
        "source_manifest_sha256": sha(run / "manifest.json"),
        "source_image_sha256": sha(path), "tracks_sha256": sha(tracks_path),
        "script_sha256": sha(Path(__file__)), "items": items,
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.run, args.source, args.tracks, args.output)
