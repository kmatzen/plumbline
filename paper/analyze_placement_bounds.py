"""Report sensitivity curves with all planned evidence, without selecting a cutoff."""

import argparse
import json
from pathlib import Path

from paper.build_reference_free_review import digest

from plumbline.placement_bounds import paired_bounds


def analyze(root, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    cases = []
    for clip in ("bear", "bmx-bumps", "boat"):
        for tracker, path in (
            ("LK", root / f"davis/grid-anchors-v1/{clip}.json"),
            ("CoTracker3", root / f"davis/cotracker-{clip}-v1/anchor-analysis.json"),
        ):
            data = json.loads(path.read_text())
            values = [[], []]
            for slot, track in enumerate(data["tracks"]):
                for frame in range(11, len(track)):
                    for index, method in enumerate(data["methods"]):
                        error = method["anchors"][slot]["rows"][frame]["distance_pixels"]
                        values[index].append(
                            error if track[frame].get("annotation_label") == 0 else None
                        )
            cases.append((clip, tracker, path, values, ["DAGE", "DA3"]))
    path = root / "dense/tum-anchor-xyz-v1.json"
    data = json.loads(path.read_text())
    for right in ("dage", "reference_later_yaw_5deg"):
        values = [
            [r["distance_pixels"] for a in data["methods"][name]["anchors"] for r in a["rows"][11:]]
            for name in ("reference", right)
        ]
        cases.append(("walking-xyz", "LK", path, values, ["reference", right]))
    result = {
        "status": "complete",
        "scope": "conditional diagnostic bounds, not human usability, confidence intervals, or DNF",
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/placement_bounds.py"),
        "tolerance_policy": "fixed descriptive sweep 1,2,4,8,16 source pixels; no operating cutoff selected",
        "cases": [],
    }
    for clip, tracker, path, values, names in cases:
        result["cases"].append(
            {
                "clip": clip,
                "image_tracker": tracker,
                "methods": names,
                "source_sha256": digest(path),
                "source": str(path),
                "bounds": [paired_bounds(*values, t) for t in (1, 2, 4, 8, 16)],
            }
        )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case in result["cases"]:
        print(
            case["clip"],
            case["image_tracker"],
            case["methods"],
            [(r["tolerance_pixels"], r["ordering"]) for r in case["bounds"]],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
