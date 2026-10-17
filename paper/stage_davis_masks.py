"""Stage training-run instance annotations separately from reference-free inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage(archive: Path, manifest_path: Path, output: Path):
    if output.exists():
        raise ValueError("preserve existing annotation staging")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("run must be complete before linking annotations")
    indices = [f["frame_index"] for f in manifest["frames"]]
    if not indices or indices != sorted(set(indices)):
        raise ValueError("need unique ordered frames")
    sequence = manifest["sequence"]
    if manifest["split"] != "train" or not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise ValueError("training sequence required")
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("duplicate archive entries")
        train = source.read("DAVIS/ImageSets/2017/train.txt").decode().split()
        if sequence not in train:
            raise ValueError("not in official training split")
        entries = []
        for frame in manifest["frames"]:
            original = PurePosixPath(frame["path"])
            if str(original.parent) != f"DAVIS/JPEGImages/480p/{sequence}":
                raise ValueError("unexpected frame path")
            if not re.fullmatch(r"[0-9]+\.jpg", original.name):
                raise ValueError("unexpected frame filename")
            if int(original.stem) != frame["frame_index"]:
                raise ValueError("frame index mismatch")
            if hashlib.sha256(source.read(str(original))).hexdigest() != frame["file_sha256"]:
                raise ValueError("archive image does not match run")
            name = f"DAVIS/Annotations/480p/{sequence}/{original.stem}.png"
            entries.append((frame["frame_index"], original.stem + ".png", source.read(name)))
    output.mkdir(parents=True)
    rows = []
    for index, name, raw in entries:
        (output / name).write_bytes(raw)
        rows.append({"frame_index": index, "path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    result = {
        "scope": "annotation-assisted development ablation; not static-motion truth",
        "sequence": sequence,
        "archive_sha256": digest(archive),
        "source_manifest_sha256": digest(manifest_path),
        "frames": rows,
    }
    (output / "masks.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    stage(args.archive, args.manifest, args.output)
