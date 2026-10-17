"""Describe fitting-ray geometry; numerical conditioning is not a DNF label."""

import argparse
import json
from pathlib import Path

import numpy as np
from paper.analyze_dense_run import load_verified
from paper.build_reference_free_review import digest


def analyze(run, probe, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    manifest, prediction, reference = load_verified(run)
    record = json.loads(probe.read_text())
    if record["source_manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("probe source mismatch")
    with np.load(run / "reference.npz") as data:
        k = data["intrinsics"].astype(float)
    result = {
        "status": "complete",
        "sequence": manifest["sequence"],
        "scope": "descriptive fitting geometry, not uncertainty calibration or camera failure",
        "probe_sha256": digest(probe),
        "script_sha256": digest(Path(__file__)),
        "methods": {},
    }
    for name, poses in (("reference", reference), ("dage", prediction)):
        anchors = []
        for slot, track in enumerate(record["tracks"]):
            xy = [r["xy"] for r in track[:11]]
            row = {
                "slot": slot,
                "fit_status": record["methods"][name]["anchors"][slot]["fit_status"],
            }
            if any(p is None for p in xy):
                row["geometry"] = None
            else:
                rays = np.linalg.solve(k[:11], np.c_[xy, np.ones(11)][..., None])[..., 0]
                rays = np.einsum("nij,nj->ni", poses[:11, :3, :3], rays)
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                matrix = (np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]).sum(axis=0)
                eigen = np.linalg.eigvalsh(matrix)
                angle = np.degrees(
                    np.arctan2(np.linalg.norm(np.cross(rays[0], rays), axis=1), rays @ rays[0])
                )
                row["geometry"] = {
                    "eigenvalues": eigen.tolist(),
                    "smallest_to_largest_eigenvalue": float(eigen[0] / eigen[-1]),
                    "maximum_world_ray_angle_from_first_degrees": float(angle.max()),
                }
            anchors.append(row)
        displacement = np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)
        result["methods"][name] = {
            "maximum_camera_center_displacement": float(displacement.max()),
            "translation_units": "reference dataset meters"
            if name == "reference"
            else "unscaled model units; do not compare to reference meters",
            "anchors": anchors,
        }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, method in result["methods"].items():
        angles = [
            r["geometry"]["maximum_world_ray_angle_from_first_degrees"]
            for r in method["anchors"]
            if r["geometry"]
        ]
        print(
            name,
            "displacement",
            method["maximum_camera_center_displacement"],
            method["translation_units"],
            "median max angle",
            np.median(angles),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "probe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.probe, args.output)
