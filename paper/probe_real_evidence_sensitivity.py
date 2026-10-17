"""Sensitivity of saved real-video camera estimates; no reference-pose scoring."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.probe_static_point_support import load_prediction
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest
from plumbline.frustum_support import box_support

K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])


def alternatives(poses, end):
    yield "identity", np.linalg.inv(poses)
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [17, -23, 31], degrees=True).as_matrix()
    gauge[:3, 3] = [3, -2, 1]
    yield "global_rigid_gauge", np.linalg.inv(gauge @ poses)
    ramp = np.r_[np.zeros(end+1), np.arange(1, len(poses)-end)/(len(poses)-end-1)]
    baseline = np.linalg.norm(poses[:, :3, 3]-poses[0, :3, 3], axis=1).max()
    for axis, name in enumerate("xyz"):
        for sign in (-1, 1):
            changed = poses.copy()
            changed[:, :3, :3] = poses[:, :3, :3] @ Rotation.from_euler(
                name, (sign*5*ramp)[:, None], degrees=True).as_matrix()
            yield f"rotation_{name}_{sign*5:+d}deg", np.linalg.inv(changed)
            changed = poses.copy()
            changed[:, :3, 3] += sign*.25*baseline*ramp[:, None]*poses[:, :3, axis]
            yield f"translation_{name}_{sign:+d}_quarter_baseline", np.linalg.inv(changed)


def assess(cameras, track, end, tolerance):
    fit = [i for i in range(1, end+1) if track[i]["xy"] is not None]
    query = [i for i in range(end+1, 90) if track[i]["xy"] is not None]
    record = {"fitting_indices": fit, "query_indices": query}
    if len(fit) < 2:
        return {**record, "status": "insufficient_fitting_observations"}

    def solve(indices):
        xy = np.array([track[i]["xy"] for i in indices])
        return box_support(cameras[indices], K, np.c_[xy-tolerance, xy+tolerance])

    early = solve(fit)
    if early["status"] != "positive_witness":
        return {**record, "status": "fitting_geometry_not_established", "early": early}
    if not query:
        return {**record, "status": "missing_query_observations", "early": early}
    combined = solve(fit+query)
    status = {"positive_witness": "jointly_compatible", "solver_infeasible": "joint_contradiction"}.get(
        combined["status"], "numerically_unresolved")
    return {**record, "status": status, "early": early, "combined": combined}


def run(root, control_path, predictions, output):
    if output.exists():
        raise ValueError("preserve prior results")
    control = json.loads(control_path.read_text())
    images, frames, source_hash = load_rgb(root)
    if control["source_manifest_sha256"] != source_hash or len(control["tracks"]) != 12:
        raise ValueError("track source mismatch")
    input_hash = hashlib.sha256(images.tobytes()).hexdigest()
    del images
    cases, provenance = [], {}
    for name, path, expected in (("dage", predictions/"dage-v1", "DAGE"),
                                 ("da3", predictions/"da3-v1", "DA3-LARGE-1.1")):
        poses, record = load_prediction(path, source_hash, frames, input_hash)
        if record["model"] != expected:
            raise ValueError("model label mismatch")
        provenance[name] = record
        for end in (10, 30, 60):
            for tolerance in (1, 2, 4, 8):
                base = None
                for change, cameras in alternatives(poses, end):
                    slots = [{"slot": i, **assess(cameras, track, end, tolerance)}
                             for i, track in enumerate(control["tracks"])]
                    statuses = [s["status"] for s in slots]
                    if change == "identity":
                        base = statuses
                    transitions = Counter(f"{a} -> {b}" for a, b in zip(base, statuses))
                    cases.append({"model": name, "fit_end": end, "tolerance_pixels": tolerance,
                                  "change": change, "slots": slots,
                                  "counts": dict(Counter(statuses)), "transitions": dict(transitions)})
    repo = Path(__file__).parents[1]
    dependencies = [Path(__file__), Path(__file__).with_name("probe_static_point_support.py"),
                    Path(__file__).with_name("run_static_dage.py"), repo/"src/plumbline/frustum_support.py"]
    result = {"scope": "real-video evidence sensitivity on saved full-clip estimates, not accuracy or DNF validation",
              "inputs": "decoded RGB reverified; fixed RGB-derived tracks; estimated cameras; published K; no reference-pose or depth values used",
              "limitations": "tracks/staticness/calibration not independently certified; full-clip model inference; changes can correct an unknown error; finite perturbations are not functional labels",
              "protocol": "all 12 tracks, fit ends 10/30/60, tolerances 1/2/4/8; fixed signed five-degree rotations and quarter-baseline translations; identity and rigid gauge controls; available prefix observations",
              "source_manifest_sha256": source_hash, "tracks_sha256": digest(control_path),
              "predictions": provenance,
              "dependencies_sha256": {str(p.relative_to(repo)): digest(p) for p in dependencies}, "cases": cases}
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.control, args.predictions, args.output)
    for c in result["cases"]:
        if c["change"] == "identity":
            print(c["model"], c["fit_end"], c["tolerance_pixels"], c["counts"])
