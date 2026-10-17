"""Classical bounded-reprojection support on the retained static control."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_static_rgbd_control import reference_cases
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest, nearest, parse_index
from plumbline.conventions import assert_valid_extrinsics
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.frustum_support import box_support


def support(track, poses, calibration, tolerance):
    if len(track) != 90 or len(poses) != 90:
        raise ValueError("expected original 90-frame sample")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive pixel tolerance required")
    fit = list(range(1, 11))
    if any(track[i]["xy"] is None for i in fit):
        return {"status": "insufficient_early_observations"}
    indices = [i for i in range(1, 90) if track[i]["xy"] is not None]

    def solve(frames):
        xy = np.array([track[i]["xy"] for i in frames])
        boxes = np.c_[xy - tolerance, xy + tolerance]
        return box_support(np.linalg.inv(np.stack([poses[i] for i in frames])), calibration, boxes)

    return {"status": "checked", "fit_original_frames": fit,
            "observed_original_frames": indices,
            "unavailable_reference_frames": [0],
            "missing_track_frames": [i for i in range(1, 90) if track[i]["xy"] is None],
            "later_observed_frames": sum(i >= 11 for i in indices),
            "early": solve(fit), "all_observed": solve(indices)}


def load_prediction(run_path, source_hash, frames, input_hash):
    manifest = json.loads((run_path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("prediction run is incomplete")
    if (manifest["source_manifest_sha256"] != source_hash or manifest["frames"] != frames
            or manifest["input_sha256"] != input_hash):
        raise ValueError("prediction inputs differ from control")
    if manifest["runner_sha256"] != digest(run_path / "executed-runner.py"):
        raise ValueError("prediction runner mismatch")
    path = run_path / "prediction.npz"
    if digest(path) != manifest["prediction_file_sha256"]:
        raise ValueError("prediction bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        poses = archive["extrinsics"]
    if poses.shape != (90, 4, 4):
        raise ValueError("prediction frame count mismatch")
    assert_valid_extrinsics(poses)
    return poses, {"manifest_sha256": digest(run_path / "manifest.json"),
                   "prediction_file_sha256": digest(path),
                   "verification": "source and decoded RGB reverified; all 90 rigid poses checked; no alignment used for point support"}


def run(root, control_path, output, prediction_run=None):
    if output.exists():
        raise ValueError("preserve previous analyses")
    manifest_path = root / "staging-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    control = json.loads(control_path.read_text())
    if control["source_manifest_sha256"] != digest(manifest_path):
        raise ValueError("control source mismatch")
    reference_path = root / "groundtruth.txt"
    if digest(reference_path) != manifest["indices"]["groundtruth.txt"]:
        raise ValueError("reference index mismatch")
    index = parse_index(reference_path.read_bytes(), 8)
    poses = []
    for row in manifest["frames"]:
        expected = nearest(row["timestamp"], index)
        if row["reference"] != expected:
            raise ValueError("reference association mismatch")
        poses.append(tum_pose_to_matrix(*map(float, expected["matches"][0]["values"]))
                     if expected["status"] == "associated" else None)
    cases = reference_cases(poses)
    prediction_record = None
    if prediction_run is not None:
        import hashlib
        from plumbline.metrics.pose import trajectory_ate_rmse_sim3, trajectory_rpe_rmse_sim3
        images, frames, source_hash = load_rgb(root)
        predicted, prediction_record = load_prediction(
            prediction_run, source_hash, frames, hashlib.sha256(images.tobytes()).hexdigest())
        del images
        cases["dage"] = predicted
        reference = np.stack(poses[1:])
        translation, rotation = trajectory_rpe_rmse_sim3(predicted[1:], reference)
        prediction_record["reference_metrics"] = {
            "scope": "89 associated reference poses, original frames 1–89; nearest-timestamp reference; not functional labels",
            "ate_rmse_sim3_m": trajectory_ate_rmse_sim3(predicted[1:], reference),
            "rpe_translation_rmse_sim3_m": translation,
            "rpe_rotation_rmse_deg": rotation,
        }
    calibration = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    results = []
    if len(control["tracks"]) != 12:
        raise ValueError("retain all 12 original slots")
    for name, cameras in cases.items():
        for tolerance in (1, 2, 4, 8):
            anchors = [support(t, cameras, calibration, tolerance) for t in control["tracks"]]
            counts = {phase: dict(Counter(a[phase]["status"] if a["status"] == "checked" else a["status"] for a in anchors))
                      for phase in ("early", "all_observed")}
            results.append({"method": name, "tolerance_pixels": tolerance, "anchors": anchors, "counts": counts})
    result = {
        "scope": "joint static-point consistency; not held-out prediction, novel triangulation, or camera DNF",
        "policy": "unchanged 1,2,4,8 pixel axis-wise tolerance grid; no tolerance selected as a validity threshold",
        "assumptions": "correct static-point tracks, calibration, and timing; solver does not establish these",
        "control_sha256": digest(control_path), "source_manifest_sha256": digest(manifest_path),
        "script_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(Path(__file__).parents[1] / "src/plumbline/frustum_support.py"),
        "camera_variant_script_sha256": digest(Path(__file__).with_name("probe_static_rgbd_control.py")),
        "planned_later_observations": 948, "cases": results,
        "prediction": prediction_record,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prediction-run", type=Path)
    args = parser.parse_args()
    for row in run(args.root, args.control, args.output, args.prediction_run)["cases"]:
        print(row["method"], row["tolerance_pixels"], row["counts"])
