"""Stage depth evidence for an existing RGB manifest without changing that sample.

Depth and reference poses are validation-only inputs, never evaluator inputs.
Nearest timestamp association is approximate; retain offsets and missing matches.
"""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def associate(timestamp, index, max_delta=0.02):
    """Nearest depth; ties prefer earlier time. No unreported RGB exclusions."""
    if not index:
        return None
    stamp, path = min(index, key=lambda row: (abs(row[0] - timestamp), row[0]))
    delta = stamp - timestamp
    return {
        "timestamp": stamp,
        "path": path,
        "delta_seconds": delta,
        "within_tolerance": abs(delta) <= max_delta,
    }


def stage(archive, manifest_path, output):
    if output.exists():
        raise ValueError("preserve previous staging attempts")
    manifest = json.loads(manifest_path.read_text())
    sequence = manifest["sequence"]
    if PurePosixPath(sequence).name != sequence:
        raise ValueError("invalid sequence name")
    output.mkdir(parents=True)
    rows = []
    with tarfile.open(archive, "r:gz") as tar:

        def read(name):
            member = tar.getmember(name)
            if not member.isfile():
                raise ValueError("expected ordinary archive file")
            return tar.extractfile(member).read()

        depth_index_bytes = read(f"{sequence}/depth.txt")
        index = []
        for line in depth_index_bytes.decode().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stamp, path = line.split()
            if PurePosixPath(path).parts[:1] != ("depth",) or ".." in PurePosixPath(path).parts:
                raise ValueError("unsafe depth path")
            index.append((float(stamp), path))
        if len({stamp for stamp, _ in index}) != len(index):
            raise ValueError("duplicate depth timestamps")
        (output / "depth.txt").write_bytes(depth_index_bytes)
        for frame, source in enumerate(manifest["frames"]):
            rgb = read(source["path"])
            if hashlib.sha256(rgb).hexdigest() != source["file_sha256"]:
                raise ValueError("archive RGB differs from experiment source")
            match = associate(float(source["timestamp"]), index)
            row = {
                "frame": frame,
                "rgb_path": source["path"],
                "rgb_timestamp": source["timestamp"],
                "rgb_sha256": source["file_sha256"],
                "nearest_depth": match,
                "status": "no_depth_within_20ms",
            }
            if match and match["within_tolerance"]:
                payload = read(f"{sequence}/{match['path']}")
                destination = output / match["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(payload)
                row.update(status="staged", depth_sha256=hashlib.sha256(payload).hexdigest())
            rows.append(row)
    selected = [r["nearest_depth"]["path"] for r in rows if r["status"] == "staged"]
    result = {
        "scope": "reference-assisted validation evidence only; not evaluator input or staticness labels",
        "sequence": sequence,
        "archive_sha256": sha(archive),
        "source_manifest_sha256": sha(manifest_path),
        "script_sha256": sha(Path(__file__)),
        "depth_index_sha256": sha(output / "depth.txt"),
        "policy": "nearest timestamp within 20 ms, earlier timestamp on ties; missing RGB-depth matches retained",
        "depth_scale_divisor": 5000,
        "zero_depth": "missing",
        "format_source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
        "reused_depth_frames": {p: n for p, n in Counter(selected).items() if n > 1},
        "rows": rows,
    }
    (output / "depth-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = stage(args.archive, args.manifest, args.output)
    print(dict(Counter(row["status"] for row in result["rows"])))
