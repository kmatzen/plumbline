"""Recover pose-only comparison data using the cache-producing loader revision.

Run on Anima in the original environment. The result records reconstructed
timestamps and the legacy input fingerprint. It cannot certify historical frame
identity because that fingerprint covers only the first 1KB of image data.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from plumbline.datasets.realestate10k_pose import RealEstate10KPoseEvalLoader
from plumbline.runner import _sample_input_fingerprint


def export(root: Path, cache: Path) -> dict:
    variants = []
    missing = []
    for sigma in (0.0, 1.5):
        dataset = RealEstate10KPoseEvalLoader(root=root, num_frames=10, seed=0, lowpass_sigma=sigma)
        for sample in dataset:
            fingerprint = _sample_input_fingerprint(sample)
            # This is the old cache's naming rule, before sanitize_id changed.
            filename = sample.sample_id.replace("/", "_") + "_" + fingerprint + ".npz"
            predictions = {}
            for model in ("dust3r", "mast3r", "vggt"):
                matches = sorted((cache / model).glob("*/realestate10k-pose-eval/" + filename))
                if len(matches) != 1:
                    missing.append(
                        {
                            "sample": sample.sample_id,
                            "sigma": sigma,
                            "model": model,
                            "matches": len(matches),
                        }
                    )
                    continue
                path = matches[0]
                with np.load(path, allow_pickle=False) as archive:
                    poses = archive["extrinsics"]
                    metadata = json.loads(str(archive["_meta_json"]))
                predictions[model] = {
                    "poses": poses.tolist(),
                    "metadata": metadata,
                    "path": str(path.relative_to(cache)),
                    "pose_sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
                }
            variants.append(
                {
                    "sample_id": sample.sample_id,
                    "sigma": sigma,
                    "timestamps": list(sample.metadata["timestamps"]),
                    "legacy_input_fingerprint": fingerprint,
                    "reference": sample.extrinsics_gt.tolist(),
                    "predictions": predictions,
                }
            )
    loader_file = Path(inspect.getfile(RealEstate10KPoseEvalLoader))
    return {
        "root": str(root),
        "seed": 0,
        "num_frames": 10,
        "loader_file": str(loader_file),
        "loader_sha256": hashlib.sha256(loader_file.read_bytes()).hexdigest(),
        "provenance_limit": "Legacy cache keys do not hash all frames or record timestamps.",
        "variants": variants,
        "missing": missing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.cache)))
