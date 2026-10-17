"""Stage a fixed RGB sample with validation-only depth/reference associations."""

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

SEQUENCE = "rgbd_dataset_freiburg3_structure_texture_near"
SEQUENCES = (SEQUENCE, "rgbd_dataset_freiburg3_long_office_household")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_index(payload, fields):
    rows = []
    for line_number, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        if len(values) != fields or not math.isfinite(float(values[0])):
            raise ValueError("invalid timestamp index")
        rows.append({"timestamp": float(values[0]), "values": values[1:], "line": line_number})
    return rows


def nearest(timestamp, rows):
    if not rows:
        return {"status": "empty_index", "matches": []}
    best = min(rows, key=lambda r: (abs(r["timestamp"] - timestamp), r["timestamp"]))
    matches = [r for r in rows if r["timestamp"] == best["timestamp"]]
    delta = best["timestamp"] - timestamp
    status = "outside_20ms" if abs(delta) > .02 else (
        "ambiguous_timestamp" if len(matches) > 1 else "associated"
    )
    return {"status": status, "delta_seconds": delta, "matches": matches}


def safe_image_path(name, kind):
    path = PurePosixPath(name)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != kind or ".." in path.parts:
        raise ValueError("unsafe image path")
    return path


def stage(archive, output, *, sequence=SEQUENCE):
    if sequence not in SEQUENCES:
        raise ValueError("sequence must be an explicitly supported Freiburg-3 control")
    if output.exists():
        raise ValueError("preserve existing staging attempts")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("duplicate archive member")

        def read(relative):
            member = tar.getmember(f"{sequence}/{relative}")
            if not member.isfile():
                raise ValueError("expected ordinary archive file")
            return tar.extractfile(member).read()

        indices = {name: read(name) for name in ("rgb.txt", "depth.txt", "groundtruth.txt")}
        rgb = sorted(parse_index(indices["rgb.txt"], 2), key=lambda r: r["timestamp"])
        if len({r["timestamp"] for r in rgb}) != len(rgb):
            raise ValueError("ambiguous RGB chronology")
        selected = rgb[::3][:90]
        if len(selected) != 90:
            raise ValueError("need 90 RGB frames at stride three")
        depth = parse_index(indices["depth.txt"], 2)
        reference = parse_index(indices["groundtruth.txt"], 8)
        output.mkdir(parents=True)
        for name, payload in indices.items():
            (output / name).write_bytes(payload)
        frames = []
        for index, row in enumerate(selected):
            relative = safe_image_path(row["values"][0], "rgb")
            destination = output / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(read(relative))
            depth_match = nearest(row["timestamp"], depth)
            if depth_match["status"] == "associated":
                depth_path = safe_image_path(depth_match["matches"][0]["values"][0], "depth")
                destination_depth = output / depth_path
                destination_depth.parent.mkdir(exist_ok=True)
                if not destination_depth.exists():
                    destination_depth.write_bytes(read(depth_path))
                depth_match.update(path=str(depth_path), file_sha256=digest(destination_depth))
            frames.append({"frame": index, "rgb_index": index * 3,
                           "path": str(relative), "timestamp": row["timestamp"],
                           "file_sha256": digest(destination), "depth": depth_match,
                           "reference": nearest(row["timestamp"], reference)})
    record = {
        "scope": "static-scene development control; reference and depth are validation-only",
        "sequence": sequence, "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size, "script_sha256": digest(Path(__file__)),
        "selection": "first 90 chronological RGB frames at stride 3, before depth/reference association",
        "association": "nearest timestamp within 20 ms; earlier timestamp on ties; duplicate selected timestamp is ambiguous; no interpolation",
        "indices": {name: digest(output / name) for name in indices},
        "association_counts": {kind: dict(Counter(r[kind]["status"] for r in frames))
                               for kind in ("depth", "reference")},
        "frames": frames,
    }
    (output / "staging-manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", choices=SEQUENCES, default=SEQUENCE)
    args = parser.parse_args()
    print(stage(args.archive, args.output, sequence=args.sequence)["association_counts"])
