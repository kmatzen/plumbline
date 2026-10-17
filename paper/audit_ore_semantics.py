"""Compare the external authors' kernel with our explicitly ORE-style prototype.

Synthetic cases expose policy differences; they do not validate either metric
against human usability. External code stays outside the repository.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from paper.build_reference_free_review import digest

from plumbline.object_reprojection import score_static_track


def cases():
    k = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]], dtype=float)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, 1, 2]
    boxes = np.array([[100, 100, 100, 100], [80, 100, 80, 100], [60, 100, 60, 100]], dtype=float)
    yield "exact_center_ray", poses.copy(), k, boxes.copy()
    reversed_poses = poses.copy()
    reversed_poses[:, 0, 3] *= -1
    yield "reversed_translation_normalization", reversed_poses, k, boxes.copy()
    yield "off_axis_radial_vs_axial_depth", poses.copy(), k, boxes + np.array([60, 0, 60, 0])
    border_poses = np.tile(np.eye(4), (3, 1, 1))
    border_poses[:, 1, 3] = [0, 2, 4]
    yield "image_border_suppression", border_poses, k, np.tile([0, 90, 20, 110], (3, 1))
    behind = np.tile(np.eye(4), (3, 1, 1))
    behind[1:, 2, 3] = 20
    yield "behind_camera_projection", behind, k, np.tile([100, 100, 100, 100], (3, 1))
    extreme = np.tile(np.eye(4), (3, 1, 1))
    extreme[1:, 0, 3] = [100, 200]
    yield "large_error_clipping", extreme, k, np.tile([100, 100, 100, 100], (3, 1))


def audit(source, output):
    if output.exists():
        raise ValueError("preserve existing audits")
    sys.path.insert(0, str(source.resolve()))
    upstream = importlib.import_module("eval_utils")
    if Path(upstream.__file__).resolve() != (source / "eval_utils.py").resolve():
        raise ValueError("unexpected imported implementation")
    rows = []
    grid = np.array([1.0, 5.0, 10.0])
    for name, poses, k, boxes in cases():
        camera = np.eye(4)
        camera[:3, :3] = k
        objective = [
            float(
                upstream.calc_metric_for_scale(
                    np.array([depth]),
                    dict(enumerate(boxes)),
                    camera,
                    np.linalg.inv(poses),
                    None,
                    None,
                    0,
                    metric="PointDis",
                    img_width=200,
                    img_height=200,
                    local_only=True,
                )
            )
            for depth in grid
        ]
        ours = score_static_track(poses, k, boxes, (200, 200), grid, static_verified=True)
        rows.append(
            {
                "case": name,
                "upstream_scores_by_depth": objective,
                "upstream_minimum": min(objective),
                "prototype": ours,
                "world_from_camera": poses.tolist(),
                "intrinsics": k.tolist(),
                "boxes_xyxy": boxes.tolist(),
            }
        )
    result = {
        "scope": "synthetic implementation-policy audit; not a full ORE reproduction or quality benchmark",
        "upstream_eval_utils_sha256": digest(source / "eval_utils.py"),
        "prototype_sha256": digest(
            Path(__file__).parents[1] / "src/plumbline/object_reprojection.py"
        ),
        "script_sha256": digest(Path(__file__)),
        "depth_values": grid.tolist(),
        "depth_note": "the same numbers mean ray distance upstream and camera-axis depth in the prototype",
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.source, args.output)
