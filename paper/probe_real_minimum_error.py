"""Continuous classical fitting error on saved real-video camera estimates."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from paper.analyze_layered_camera_ranking import median_bracket
from paper.probe_real_evidence_sensitivity import K
from paper.probe_static_point_support import load_prediction
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest
from plumbline.frustum_support import box_support


def minimum_error(cameras, calibration, xy):
    """Same fixed [0,64], .01-pixel search, with explicit native calibration."""
    xy = np.asarray(xy, dtype=float)
    calls = []
    def solve(tolerance):
        answer = box_support(cameras, calibration, np.c_[xy-tolerance, xy+tolerance])
        calls.append(dict(tolerance=tolerance, result=answer))
        return answer
    lower, upper = 0., 64.
    high = solve(upper)
    if high["status"] != "positive_witness":
        return dict(status="upper_bound_not_established", solves=calls)
    while upper-lower > .01:
        middle = (lower+upper)/2
        answer = solve(middle)
        if answer["status"] == "positive_witness":
            upper, high = middle, answer
        elif answer["status"] == "solver_infeasible":
            lower = middle
        else:
            return dict(status="numerically_unresolved", lower=lower, upper=upper, solves=calls)
    camera = (cameras @ np.r_[high["point_world"], 1.])[:, :3]
    projected = camera @ calibration.T
    error = float(np.abs(projected[:, :2]/projected[:, 2:]-xy).max())
    if not (camera[:, 2] > 0).all() or error > upper+1e-5:
        raise AssertionError("upper witness failed independent reprojection")
    return dict(status="numerical_bracket", lower=lower, upper=upper,
                audited_upper_witness_error=error, solves=calls)


def run(root, control_path, predictions):
    control = json.loads(control_path.read_text())
    images, frames, source_hash = load_rgb(root)
    if control["source_manifest_sha256"] != source_hash or len(control["tracks"]) != 12:
        raise ValueError("track source mismatch")
    input_hash = hashlib.sha256(images.tobytes()).hexdigest()
    del images
    cases = []
    for name, expected in (("dage", "DAGE"), ("da3", "DA3-LARGE-1.1")):
        poses, provenance = load_prediction(predictions/f"{name}-v1", source_hash, frames, input_hash)
        if provenance["model"] != expected:
            raise ValueError("model identity mismatch")
        cameras = np.linalg.inv(poses)
        slots = []
        for slot, track in enumerate(control["tracks"]):
            indices = [i for i in range(1, 90) if track[i]["xy"] is not None]
            result = (minimum_error(cameras[indices], K, np.array([track[i]["xy"] for i in indices]))
                      if len(indices) >= 2 else dict(status="insufficient_observations"))
            slots.append(dict(slot=slot, indices=indices, **result))
        cases.append(dict(model=name, provenance=provenance, slots=slots, median_error=median_bracket(slots)))
    repo = Path(__file__).parents[1]
    paths = [Path(__file__), Path(__file__).with_name("analyze_layered_camera_ranking.py"),
             Path(__file__).with_name("probe_real_evidence_sensitivity.py"),
             Path(__file__).with_name("probe_static_point_support.py"),
             Path(__file__).with_name("run_static_dage.py"), repo/"src/plumbline/frustum_support.py"]
    return dict(scope="two-method development RGB geometric baseline; not ground-truth-free accuracy certification or DNF",
                protocol="unchanged RGB tracks and published calibration; frames 1..89; all 12 tracks; native-pixel [0,64] bisection to .01; unresolved slots retained in median bounds",
                limitations="full-clip predictions; unverified staticness/identity/calibration; development comparison; no reference poses or depth values used for score",
                source_manifest_sha256=source_hash, tracks_sha256=digest(control_path),
                calibration=K.tolist(), dependencies_sha256={str(p.relative_to(repo)):digest(p) for p in paths}, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior results")
    report = run(args.root, args.control, args.predictions)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(report, allow_nan=False)+"\n").encode())
    print(json.dumps([{k:v for k,v in c.items() if k != "slots"} for c in report["cases"]], indent=2))
