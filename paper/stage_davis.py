"""Stage only DAVIS 2017 frames and split lists for reference-free evaluation.

Segmentation annotations are deliberately not extracted. The inventory contains
frame indices, not invented timestamps, intrinsics, or camera poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(archive: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists; preserve staged datasets")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        splits = {}
        split_files = {}
        for split in ("train", "val"):
            name = f"DAVIS/ImageSets/2017/{split}.txt"
            raw = source.read(name)
            sequences = raw.decode().split()
            if not sequences or len(sequences) != len(set(sequences)):
                raise ValueError("empty or duplicate split sequences")
            if any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in sequences):
                raise ValueError("unsafe sequence name")
            splits[split] = sequences
            split_files[name] = raw
        if set(splits["train"]) & set(splits["val"]):
            raise ValueError("train/validation overlap")
        selected = {}
        for split, sequences in splits.items():
            for sequence in sequences:
                prefix = f"DAVIS/JPEGImages/480p/{sequence}/"
                paths = sorted(n for n in names if n.startswith(prefix) and n.endswith(".jpg"))
                if not paths:
                    raise ValueError(f"no frames for {sequence}")
                for name in paths:
                    relative = name.removeprefix(prefix)
                    if not re.fullmatch(r"[0-9]+\.jpg", relative):
                        raise ValueError("unexpected frame path")
                    info = source.getinfo(name)
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("symlink frame is not supported")
                indices = [int(PurePosixPath(p).stem) for p in paths]
                if indices != sorted(set(indices)):
                    raise ValueError("ambiguous numeric frame order")
                selected[sequence] = (split, paths, indices)
        output.mkdir(parents=True)
        for name, raw in split_files.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        rows = []
        for sequence, (split, paths, indices) in selected.items():
            frames = []
            for name, index in zip(paths, indices, strict=True):
                raw = source.read(name)  # ZipFile verifies each extracted entry's CRC.
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                frames.append(
                    {
                        "frame_index": index,
                        "path": name,
                        "file_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            rows.append({"sequence": sequence, "split": split, "frames": frames})
    inventory = {
        "dataset": "DAVIS-2017-trainval-480p",
        "archive_sha256": digest_file(archive),
        "split_file_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in split_files.items()
        },
        "camera_ground_truth": None,
        "intrinsics": None,
        "capture_timestamps": None,
        "segmentation_annotations_extracted": False,
        "sequences": rows,
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inventory = stage(args.archive, args.output)
    print(
        json.dumps(
            {
                "archive_sha256": inventory["archive_sha256"],
                "sequences": len(inventory["sequences"]),
                "frames": sum(len(s["frames"]) for s in inventory["sequences"]),
            },
            indent=2,
        )
    )
