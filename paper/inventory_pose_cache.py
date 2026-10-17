"""Read camera-cache provenance without loading large depth or point arrays.

Run on the host holding predictions. Output is an inventory, not an evaluation:
matching filenames do not establish matching frame selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inventory(root: Path) -> dict:
    methods = {}
    names = []
    for method in ("dust3r", "mast3r", "vggt"):
        rows = []
        for path in sorted((root / method).glob("*/realestate10k-pose-eval/*.npz")):
            with np.load(path, allow_pickle=False) as data:
                poses = data["extrinsics"]
                metadata = json.loads(str(data["_meta_json"]))
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "filename": path.name,
                    "configuration_hash": path.parent.parent.name,
                    "pose_shape": list(poses.shape),
                    "pose_dtype": str(poses.dtype),
                    "pose_sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
                    "all_finite": bool(np.isfinite(poses).all()),
                    "metadata": metadata,
                    "has_frame_timestamps": "timestamps" in metadata,
                }
            )
        methods[method] = rows
        names.append({row["filename"] for row in rows})
    common = sorted(set.intersection(*names))
    return {
        "source_root": str(root),
        "counts": {name: len(rows) for name, rows in methods.items()},
        "common_filename_count": len(common),
        "common_filenames": common,
        "interpretation": "Candidate overlap only. Frame correspondence is unverified.",
        "methods": methods,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))
