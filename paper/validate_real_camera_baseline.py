"""Reference-pose validation kept separate from reference-free camera scoring."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.probe_static_point_support import load_prediction
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest, nearest, parse_index
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix
from plumbline.metrics.pose import trajectory_ate_rmse_sim3, trajectory_rpe_rmse_sim3


def reference_poses(manifest, index):
    poses, associations = [], []
    for row in manifest["frames"]:
        association = nearest(row["timestamp"], index)
        if association != row["reference"]:
            raise ValueError("saved reference association mismatch")
        associations.append(association)
        poses.append(tum_pose_to_matrix(*map(float, association["matches"][0]["values"]))
                     if association["status"] == "associated" else None)
    return poses, associations


def run(root, predictions, score_path):
    scores = json.loads(gzip.decompress(score_path.read_bytes()))
    manifest = json.loads((root/"staging-manifest.json").read_text())
    reference_path = root/"groundtruth.txt"
    if digest(reference_path) != manifest["indices"]["groundtruth.txt"]:
        raise ValueError("reference file changed")
    poses, associations = reference_poses(manifest, parse_index(reference_path.read_bytes(), 8))
    # Same original frame roster as the geometric score, no metric-based removal.
    indices = list(range(1, 90))
    if len(poses) != 90 or any(poses[i] is None for i in indices):
        raise ValueError("all 89 scored frames need unambiguous reference associations")
    reference = np.stack([poses[i] for i in indices])
    images, frames, source_hash = load_rgb(root)
    if scores["source_manifest_sha256"] != source_hash:
        raise ValueError("score source mismatch")
    image_hash = hashlib.sha256(images.tobytes()).hexdigest()
    del images
    rows = []
    for item in scores["cases"]:
        name = item["model"]
        if name not in ("dage", "da3"):
            raise ValueError("unexpected camera method")
        predicted, provenance = load_prediction(predictions/f"{name}-v1", source_hash, frames, image_hash)
        if provenance != item["provenance"]:
            raise ValueError("scored prediction provenance changed")
        translation, rotation = trajectory_rpe_rmse_sim3(predicted[indices], reference)
        rows.append(dict(model=name, geometric_median_pixels=item["median_error"],
                         ate_rmse_sim3_m=trajectory_ate_rmse_sim3(predicted[indices], reference),
                         rpe_translation_rmse_sim3_m=translation, rpe_rotation_rmse_deg=rotation))
    return dict(scope="reference-based development validation of previously frozen RGB-only scores; not functional DNF",
                policy="same 89 original frames for all metrics; Sim3 alignment; adjacent selected-frame RPE; nearest reference within 20ms; no frame/method selection by outcome",
                score_sha256=digest(score_path), source_manifest_sha256=source_hash,
                reference_sha256=digest(reference_path), script_sha256=digest(Path(__file__)),
                scored_original_indices=indices, reference_association_counts=dict(Counter(a["status"] for a in associations)),
                maximum_scored_timestamp_delta_seconds=max(abs(associations[i]["delta_seconds"]) for i in indices),
                limitations="two development methods/clips; nearest timestamp rather than exact synchronization; pose errors are not human usability labels; score already computed without reference",
                methods=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "predictions", "score", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.predictions, args.score)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["methods"], indent=2))
