"""Build the fixed 12-slot RGB evidence without requiring depth or reference poses."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper.probe_davis_anchors import grid_features, track_feature
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest


def run(root, output):
    if output.exists():
        raise ValueError("preserve previous tracks")
    images, frames, source_hash = load_rgb(root)
    del images
    # Match the original control's decoder exactly, rather than introducing a
    # different RGB-to-grayscale rounding convention during replication.
    gray = [cv2.imread(str(root / frame["path"]), cv2.IMREAD_GRAYSCALE) for frame in frames]
    if any(image is None or image.shape != (480, 640) for image in gray):
        raise ValueError("invalid native grayscale input")
    masks = [np.zeros_like(image) for image in gray]
    tracks = [track_feature(gray, masks, feature) for feature in grid_features(gray[0], masks[0])]
    if len(tracks) != 12 or any(len(t) != 90 for t in tracks):
        raise ValueError("retain all slots and frames")
    result = {"scope": "RGB-only fixed-grid LK observations; not staticness or camera-quality labels",
              "source_manifest_sha256": source_hash, "frames": frames,
              "script_sha256": digest(Path(__file__)), "opencv_version": cv2.__version__,
              "tracker_script_sha256": digest(Path(__file__).with_name("probe_davis_anchors.py")),
              "loader_script_sha256": digest(Path(__file__).with_name("run_static_dage.py")),
              "tracks": tracks}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print("Retained", len(run(args.root, args.output)["tracks"]), "tracks")
