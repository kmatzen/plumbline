"""Fixed candidate-change profile: diagnostic sensitivity, not a DNF score."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.probe_rendered_consistency import tracks_for
from paper.render_camera_control import K, camera_pose
from paper.stage_static_rgbd_control import digest
from plumbline.frustum_support import box_support


def alternatives(cameras):
    """Fixed local-axis changes preserve the first twelve cameras."""
    poses = np.linalg.inv(cameras)
    if poses.shape != (24, 4, 4):
        raise ValueError("24 cameras required")
    yield "identity", cameras.copy()
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [17, -23, 31], degrees=True).as_matrix()
    gauge[:3, 3] = [3, -2, 1]
    yield "global_rigid_gauge", cameras @ np.linalg.inv(gauge)
    ramp = np.r_[np.zeros(12), np.arange(1, 13)/12]
    baseline = np.linalg.norm(poses[:, :3, 3]-poses[0, :3, 3], axis=1).max()
    for axis, name in enumerate("xyz"):
        for sign in (-1, 1):
            changed = poses.copy()
            changed[:, :3, :3] = poses[:, :3, :3] @ Rotation.from_euler(
                name, (sign*5*ramp)[:, None], degrees=True).as_matrix()
            yield f"rotation_{name}_{sign*5:+d}deg", np.linalg.inv(changed)
            changed = poses.copy()
            # A camera-local displacement scaled by the candidate baseline;
            # no reference translations or scene depth enter this operator.
            changed[:, :3, 3] += sign*.25*baseline*ramp[:, None]*poses[:, :3, axis]
            yield f"translation_{name}_{sign:+d}_quarter_baseline", np.linalg.inv(changed)


def joint_status(cameras, track, *, tolerance=2):
    """Two feasibility solves; equivalent to the existing joint-status branch."""
    observed_fit = [i for i in range(12) if track[i] is not None]
    observed_query = [i for i in range(12, 24) if track[i] is not None]
    record = {"fitting_indices": observed_fit, "query_indices": observed_query}
    if len(observed_fit) < 2:
        return {**record, "status": "insufficient_fitting_observations"}

    def support(indices):
        xy = np.array([track[i] for i in indices])
        return box_support(cameras[indices], K, np.c_[xy-tolerance, xy+tolerance])

    early = support(observed_fit)
    if early["status"] != "positive_witness":
        return {**record, "status": "fitting_geometry_not_established", "early": early}
    if not observed_query:
        return {**record, "status": "missing_query_observations", "early": early}
    combined = support(observed_fit+observed_query)
    status = {"positive_witness": "jointly_compatible", "solver_infeasible": "joint_contradiction"}.get(
        combined["status"], "numerically_unresolved")
    return {**record, "status": status, "early": early, "combined": combined}


def run(sources, output):
    if output.exists():
        raise ValueError("preserve prior results")
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    results, totals = [], {}
    for source in sources:
        data = json.loads(source.read_text())
        for name, expected in data["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError("source code mismatch")
        if [(c["seed"], c["occluder"]) for c in data["cases"]] != [
            (s, o) for s in range(10) for o in (False, True)
        ]:
            raise ValueError("complete fixed case roster required")
        for case in data["cases"]:
            tracks = tracks_for(case, "observed")
            baseline_status = None
            for name, changed in alternatives(cameras):
                slots = [{"slot": i, **joint_status(changed, track)} for i, track in enumerate(tracks)]
                statuses = [s["status"] for s in slots]
                if name == "identity":
                    baseline_status = statuses
                counts = Counter(statuses)
                transitions = Counter(f"{before} -> {after}" for before, after in zip(baseline_status, statuses))
                key = f"{source.stem},occluder={case['occluder']},{name}"
                totals.setdefault(key, Counter()).update(counts)
                results.append({"source": source.name, "seed": case["seed"], "occluder": case["occluder"],
                                "change": name, "slots": slots, "counts": dict(counts), "transitions": dict(transitions)})
    root = Path(__file__).parents[1]
    dependencies = [Path(__file__), Path(__file__).with_name("probe_rendered_consistency.py"),
                    Path(__file__).with_name("render_camera_control.py"), root/"src/plumbline/frustum_support.py"]
    result = {"scope": "rendered reference-camera sensitivity profile, not real-data performance or a DNF threshold",
              "protocol": "all 40 source cases and all slots; identity, rigid global gauge, signed 5-degree local rotations and signed quarter-baseline local translations; unchanged images/tracks/calibration; 2px coordinate tolerance",
              "limitations": "single scene geometry; finite changes do not span all failures; perturbations are not functional labels; correlated texture seeds; positive witness does not establish accuracy",
              "source_sha256": {str(p): digest(p) for p in sources},
              "dependencies_sha256": {str(p.relative_to(root)): digest(p) for p in dependencies},
              "totals": {k: dict(v) for k, v in totals.items()}, "cases": results}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    return result["totals"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequential", type=Path)
    parser.add_argument("direct", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run([args.sequential, args.direct], args.output), indent=2))
