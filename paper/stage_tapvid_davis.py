"""Stage the pinned official TAP-Vid DAVIS release without generic pickle loading."""

import argparse
import hashlib
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
try:
    from numpy._core.multiarray import _reconstruct, scalar
except ImportError:  # NumPy 1.x on the staging host.
    from numpy.core.multiarray import _reconstruct, scalar

SOURCE_URL = "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip"
SOURCE_SHA256 = "60eb6239c57a9877900c269d14637c2b6eece8fa280f59e4592324c6392031b6"
SOURCE_BYTES = 1668710491
UPSTREAM_COMMIT = "730cda1c730877cfedbe01bf87fb1cadb78a565d"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    """Permit only the NumPy constructors needed by this verified release.

    This is not a general-purpose sandbox for arbitrary untrusted pickle files.
    The archive digest is checked before the loader is used.
    """

    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
                   ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
                   ("numpy.core.multiarray", "scalar"): scalar,
                   ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
                   ("numpy._core.multiarray", "scalar"): scalar}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"unsupported pickle global: {module}.{name}")
        return allowed[module, name]


def assign_splits(names):
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("distinct sequence names required")
    ordered = sorted(names, key=lambda name: hashlib.sha256(
        ("plumbline-track-identity-v1:" + name).encode()).hexdigest())
    ndev = len(names) // 3
    return {name: "development" if i < ndev else "held_out" for i, name in enumerate(ordered)}


def validate_sequence(name, data):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("unsafe sequence name")
    if not isinstance(data, dict) or set(data) != {"video", "points", "occluded"}:
        raise ValueError("unexpected sequence fields")
    video, points, occ = (data[k] for k in ("video", "points", "occluded"))
    if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("RGB uint8 video required")
    if points.ndim != 3 or points.shape[1:] != (len(video), 2) or points.dtype.kind != "f":
        raise ValueError("floating N,T,2 normalized xy points required")
    if occ.dtype != np.bool_ or occ.shape != points.shape[:2]:
        raise ValueError("boolean N,T occlusion mask required")
    if not np.isfinite(points).all():
        raise ValueError("nonfinite annotation coordinates")
    outside = np.any((points < 0) | (points > 1), axis=-1)
    return {"video_shape": list(video.shape), "points_shape": list(points.shape),
            "points_dtype": str(points.dtype), "visible_observations": int((~occ).sum()),
            "occluded_observations": int(occ.sum()),
            "all_occluded_tracks": int(occ.all(axis=1).sum()),
            "visible_outside_unit_square": int((outside & ~occ).sum()),
            "decoded_array_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                     for key, value in data.items()}}


def run(archive, output):
    if output.exists():
        raise ValueError("preserve previous staging output")
    if archive.stat().st_size != SOURCE_BYTES or digest(archive) != SOURCE_SHA256:
        raise ValueError("archive differs from pinned official release")
    with zipfile.ZipFile(archive) as z:
        with z.open("tapvid_davis/tapvid_davis.pkl") as stream:
            data = ArrayUnpickler(stream).load()
            if stream.read(1):
                raise ValueError("trailing pickle data")
        docs = {name: z.read("tapvid_davis/" + name) for name in ("README.md", "SOURCES.md")}
    if not isinstance(data, dict) or len(data) != 30:
        raise ValueError("expected all thirty DAVIS sequences")
    splits = assign_splits(list(data))
    entries = {name: {"split": splits[name], **validate_sequence(name, item)} for name, item in data.items()}
    output.mkdir(parents=True)
    for name, item in data.items():
        path = output / (name + ".npz")
        np.savez(path, **item)
        entries[name]["file"] = path.name
        entries[name]["file_sha256"] = digest(path)
        print(name, entries[name]["video_shape"], splits[name], flush=True)
    for name, content in docs.items():
        (output / name).write_bytes(content)
    manifest = {"source_url": SOURCE_URL, "archive_sha256": SOURCE_SHA256,
                "archive_bytes": SOURCE_BYTES, "upstream_documentation_commit": UPSTREAM_COMMIT,
                "script_sha256": digest(Path(__file__)), "numpy_version": np.__version__,
                "coordinate_policy": "original normalized xy retained; raster xy = points * [width,height], not [width-1,height-1]",
                "annotation_scope": "human-reviewed, optical-flow-assisted point positions and visibility; not staticness, camera ground truth, or DNF labels",
                "split_policy": "sort SHA256(plumbline-track-identity-v1: + name); first ten development; remaining twenty held out; no metrics examined before split",
                "training_policy": "no training or threshold selection on held-out sequences; pretrained model exposure unknown",
                "documents_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in docs.items()},
                "sequences": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
