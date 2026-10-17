"""Controlled camera/track interventions; consistency is not causal attribution."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.probe_joint_query_consistency import compare_queries
from paper.stage_static_rgbd_control import digest


CONDITIONS = ("clean", "camera_yaw_1", "camera_yaw_5", "bad_track_1",
              "bad_track_3", "coherent_bad_tracks", "missing_queries")
K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])


def project(cameras, points):
    xyz = np.einsum("tij,nj->tni", cameras[:, :3], np.c_[points, np.ones(len(points))])
    if np.any(xyz[:, :, 2] <= 0):
        raise ValueError("control points must remain in front of cameras")
    pixels = np.einsum("ij,tnj->tni", K, xyz)
    return pixels[:, :, :2] / pixels[:, :, 2:]


def scene(seed):
    rng = np.random.default_rng(seed)
    cameras = np.repeat(np.eye(4)[None], 24, axis=0)
    cameras[:, 0, 3] = -np.linspace(0, .8, 24)
    points = np.column_stack([rng.uniform(-.5, .5, 12), rng.uniform(-.5, .5, 12),
                              rng.uniform(3, 6, 12)])
    return cameras, points, rng.uniform(-.25, .25, (24, 12, 2))


def yaw_suffix(cameras, degrees):
    poses = np.linalg.inv(cameras)
    angles = np.r_[np.zeros(12), np.linspace(degrees / 12, degrees, 12)]
    poses[:, :3, :3] = poses[:, :3, :3] @ Rotation.from_euler("y", angles[:, None], degrees=True).as_matrix()
    return np.linalg.inv(poses)


def make_case(seed, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown intervention")
    cameras, points, noise = scene(seed)
    observed = project(cameras, points) + noise
    candidate = cameras.copy()
    if condition.startswith("camera_yaw_"):
        candidate = yaw_suffix(cameras, float(condition.rsplit("_", 1)[1]))
    elif condition.startswith("bad_track_"):
        count = int(condition.rsplit("_", 1)[1])
        # A deterministic suffix jump; retained even if the checker misses it.
        observed[12:, :count, 1] += 16
    elif condition == "coherent_bad_tracks":
        # Correct camera estimates, but a shared erroneous image-coordinate warp.
        observed[12:] = project(yaw_suffix(cameras, 5), points)[12:] + noise[12:]
    tracks = [[None if condition == "missing_queries" and t >= 12 else observed[t, n].tolist()
               for t in range(24)] for n in range(12)]
    return candidate, tracks


def assess(cameras, tracks, tolerance):
    """No intervention name, reference poses, or scene points enter this call."""
    if cameras.shape != (24, 4, 4) or len(tracks) != 12 or any(len(t) != 24 for t in tracks):
        raise ValueError("24 cameras and twelve 24-frame tracks required")
    slots = [dict(slot=i, **compare_queries(cameras[:12], t[:12], cameras[12:], t[12:], K, tolerance))
             for i, t in enumerate(tracks)]
    counts = Counter(s["status"] for s in slots)
    contradicted = counts["joint_contradiction"]
    compatible = counts["jointly_compatible"]
    unknown = len(slots) - contradicted - compatible
    return {"counts": dict(counts), "slots": slots,
            "contradicted_fraction_of_all_slots": contradicted / len(slots),
            "unresolved_fraction_of_all_slots": unknown / len(slots),
            "scope": "inconsistency counts only; no causal camera-failure verdict"}


def run(output):
    if output.exists():
        raise ValueError("preserve prior experiment")
    cases = []
    for seed in range(10):
        for condition in CONDITIONS:
            cameras, tracks = make_case(seed, condition)
            cases.append({"seed": seed, "intervention": condition,
                          "candidate_world_to_camera": cameras.tolist(), "tracks_xy": tracks,
                          "evaluations": [{"tolerance_pixels": tolerance, **assess(cameras, tracks, tolerance)}
                                          for tolerance in (1, 2)]})
    result = {"scope": "synthetic intervention sensitivity, not real-data accuracy or DNF validation",
              "protocol": "10 seeds; 24 frames; fit first 12; all next 12 queries; 12 points at depths 3–6; bounded 0.25-pixel noise; no tuning",
              "limitations": "simple translation and ideal pinhole points; no rendering, occlusion, learned tracker, or human judgments",
              "script_sha256": digest(Path(__file__)),
              "dependencies_sha256": {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in (
                  Path(__file__).with_name("probe_joint_query_consistency.py"),
                  Path(__file__).with_name("probe_static_query_consistency.py"),
                  Path(__file__).parents[1] / "src/plumbline/frustum_support.py")},
              "cases": cases}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False) + "\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    for condition in CONDITIONS:
        for tolerance in (1, 2):
            counts = Counter()
            for case in result["cases"]:
                if case["intervention"] == condition:
                    counts.update(next(e["counts"] for e in case["evaluations"] if e["tolerance_pixels"] == tolerance))
            print(condition, tolerance, dict(counts))
