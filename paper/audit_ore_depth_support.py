"""Audit selected depths on the official sample without altering its metric.

Positive-depth-constrained minima are reported as a separate diagnostic variant,
not as published ORE. This one supplied sequence cannot estimate general failure
rates or validate functional-DNF labels.
"""

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from paper.run_ore_demo import MODELS, digest

CLIP = "1a3273d3-f6a7-4ff3-8339-c5eeca5b6999"


def camera_depths(poses_world_to_camera, anchor_index, point_on_ray):
    world = np.linalg.solve(poses_world_to_camera[anchor_index], np.r_[point_on_ray, 1.0])
    return (poses_world_to_camera @ world)[:, 2]


def choose_two_grid_minimum(scores):
    """Match the released demo: first minimum within each grid; grid 2 on tie."""
    indices = [int(np.argmin(values)) for values in scores]
    grid = 0 if scores[0][indices[0]] < scores[1][indices[1]] else 1
    return grid, indices[grid]


def positive_depth_interval(slopes, offsets):
    """Intersect strict affine constraints a*d+b>0 with d>0 (float64)."""
    lower, upper = 0.0, np.inf
    for a, b in zip(slopes, offsets, strict=True):
        if a > 0:
            lower = max(lower, -b / a)
        elif a < 0:
            upper = min(upper, -b / a)
        elif b <= 0:
            return {"feasible": False, "lower_exclusive": None, "upper_exclusive": None}
    return {
        "feasible": bool(lower < upper),
        "lower_exclusive": float(lower),
        "upper_exclusive": float(upper) if np.isfinite(upper) else None,
    }


def audit(source, demo_path, output, box_support_enabled=False):
    source, demo_path, output = source.resolve(), demo_path.resolve(), output.resolve()
    if output.exists():
        raise ValueError("preserve previous audits")
    demo = json.loads(demo_path.read_text())
    if demo["status"] != "complete":
        raise ValueError("completed demo required")
    for name, expected in demo["source_files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"upstream source changed: {name}")
    record = {
        "status": "running",
        "scope": "positive-depth diagnostic on official sample poses; not full benchmark or DNF validation",
        "demo_manifest_sha256": digest(demo_path),
        "script_sha256": digest(Path(__file__)),
        "constraint": "strictly positive camera z at every annotated frame; no new score threshold",
        "methods": [],
        "box_support_enabled": box_support_enabled,
    }

    def save():
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    cwd, paths = Path.cwd(), sys.path[:]
    alias_exists, alias = "int" in np.__dict__, np.__dict__.get("int")
    np.int = int
    try:
        os.chdir(source)
        sys.path.insert(0, str(source))
        evaluator = importlib.import_module("ego4d_evaluation")
        utils = importlib.import_module("eval_utils")
        if Path(utils.__file__).resolve() != source / "eval_utils.py":
            raise ValueError("unexpected upstream import")
        dataset = json.loads((source / "egotracks_with_additional_annotation.json").read_text())
        matches = [
            (i, j)
            for i, v in enumerate(dataset["videos"])
            for j, c in enumerate(v["clips"])
            if c["source_clip_uid"] == CLIP
        ]
        if len(matches) != 1:
            raise ValueError("ambiguous sample clip")
        video_id, clip_id = matches[0]
        queries = dataset["videos"][video_id]["clips"][clip_id]["annotations"][0]["query_sets"]
        camera_type = json.loads((source / "video_meta.json").read_text())[CLIP]
        calib = np.loadtxt(source / f"calib/{camera_type}.txt")
        camera = np.eye(4)
        camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2] = calib[:4]
        grids = [np.mgrid[slice(0.1, 3.0, 0.05)], np.mgrid[slice(1.0, 200.0, 2.0)]]
        record["grids"] = [grid.tolist() for grid in grids]
        for model in MODELS:
            poses = utils.get_pose(CLIP, model, dataset="ego4d", world2cam=True, scenario="kitchen")
            tracks = []
            original = next(m for m in demo["methods"] if m["model"] == model)
            for track_id, query in queries.items():
                if (
                    "attributes_occurrence" not in query
                    or evaluator.check_moving(query["attributes_occurrence"])
                    or "lt_track" not in query
                ):
                    continue
                boxes, anchor = evaluator.extract_tracklet(
                    dataset, video_id, clip_id, track_id, calib=calib
                )
                indices = np.array(sorted(boxes))
                center = (boxes[anchor][:2] + boxes[anchor][2:]) / 2
                offset = camera_depths(poses, anchor, np.zeros(3))[indices]
                unit_point = utils.convert_from_uvd(center[0], center[1], 1.0, camera)
                slope = camera_depths(poses, anchor, unit_point)[indices] - offset
                interval = positive_depth_interval(slope, offset)
                scores, depths = [], []
                for grid in grids:
                    values, zs = [], []
                    for distance in grid:
                        values.append(
                            float(
                                utils.calc_metric_for_scale(
                                    np.array([distance]),
                                    boxes,
                                    camera,
                                    poses,
                                    None,
                                    None,
                                    anchor,
                                    metric="PointDis",
                                    local_only=True,
                                )
                            )
                        )
                        point = utils.convert_from_uvd(center[0], center[1], distance, camera)
                        zs.append(camera_depths(poses, anchor, point)[indices])
                    scores.append(np.asarray(values))
                    depths.append(np.asarray(zs))
                grid_id, index = choose_two_grid_minimum(scores)
                value = float(scores[grid_id][index])
                official = original["log_metric"][f"{CLIP}_{track_id}"]
                if not np.isclose(value, official["PointDis"], rtol=1e-10, atol=1e-12):
                    raise ValueError("recomputed official score differs")
                selected = depths[grid_id][index]
                valid = [(z > 0).all(axis=1) for z in depths]
                constrained = [
                    np.where(ok, score, np.inf) for ok, score in zip(valid, scores, strict=True)
                ]
                cg, ci = choose_two_grid_minimum(constrained)
                constrained_value = float(constrained[cg][ci])
                tracks.append(
                    {
                        "track_id": track_id,
                        "annotated_frames": len(indices),
                        "official_score": value,
                        "official_depth": float(grids[grid_id][index]),
                        "official_nonpositive_frames": int((selected <= 0).sum()),
                        "official_min_z": float(selected.min()),
                        "official_grid_endpoint": index in (0, len(grids[grid_id]) - 1),
                        "positive_depth_candidate_count": sum(int(v.sum()) for v in valid),
                        "positive_depth_interval": interval,
                        "constrained_score": constrained_value
                        if np.isfinite(constrained_value)
                        else None,
                        "constrained_depth": float(grids[cg][ci])
                        if np.isfinite(constrained_value)
                        else None,
                        "grid_scores": [s.tolist() for s in scores],
                        "grid_nonpositive_frame_counts": [
                            (z <= 0).sum(axis=1).tolist() for z in depths
                        ],
                    }
                )
                if box_support_enabled:
                    from plumbline.frustum_support import box_support

                    observed_boxes = np.array([boxes[i] for i in indices])
                    invalid = ~np.isfinite(observed_boxes).all(axis=1) | (
                        observed_boxes[:, :2] > observed_boxes[:, 2:]
                    ).any(axis=1)
                    tracks[-1]["invalid_box_frames"] = indices[invalid].tolist()
                    tracks[-1]["invalid_boxes_xyxy"] = observed_boxes[invalid].tolist()
                    if invalid.any():
                        tracks[-1]["box_support"] = {
                            mode: {"status": "invalid_boxes", "invalid_count": int(invalid.sum())}
                            for mode in ("anchor_only", "all_observations")
                        }
                    else:
                        tracks[-1]["box_support"] = {
                            "anchor_only": box_support(
                                poses[indices], camera[:3, :3], observed_boxes, anchor_only=True
                            ),
                            "all_observations": box_support(
                                poses[indices], camera[:3, :3], observed_boxes
                            ),
                        }
                    record["box_support_kernel_sha256"] = digest(
                        Path(__file__).parents[1] / "src/plumbline/frustum_support.py"
                    )
            record["methods"].append(
                {
                    "model": model,
                    "tracks": tracks,
                    "official_mean": float(np.mean([t["official_score"] for t in tracks])),
                    "constrained_mean": float(np.mean([t["constrained_score"] for t in tracks]))
                    if all(t["constrained_score"] is not None for t in tracks)
                    else None,
                }
            )
            save()
            print(model, "audited", flush=True)
        record["status"] = "complete"
    except Exception as error:
        record.update(
            {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        os.chdir(cwd)
        sys.path = paths
        if alias_exists:
            np.int = alias
        else:
            delattr(np, "int")
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "demo", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--box-support", action="store_true")
    args = parser.parse_args()
    audit(args.source, args.demo, args.output, args.box_support)
