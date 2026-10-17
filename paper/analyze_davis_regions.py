"""Partition existing matches by DAVIS instances, without labeling camera failures.

Official encoding: zero background, 255 void, positive non-255 instance IDs.
https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/davis.py
Background does not mean static; labeled objects do not necessarily move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from paper.build_reference_free_review import digest
from PIL import Image


def labels_at(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    if mask.ndim != 2 or not np.isfinite(xy).all():
        raise ValueError("need 2D mask and finite points")
    pixels = np.floor(xy + 0.5).astype(int)
    valid = (
        (pixels >= 0).all(axis=1) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] < mask.shape[0])
    )
    values = np.full(len(points), 255, dtype=int)
    values[valid] = mask[pixels[valid, 1], pixels[valid, 0]]
    return values


def analyze(run: Path, evidence: Path, annotations: Path, output: Path):
    if output.exists():
        raise ValueError("preserve existing analyses")
    manifest = json.loads((run / "manifest.json").read_text())
    masks = json.loads((annotations / "masks.json").read_text())
    data = json.loads(evidence.read_text())
    source_hash = digest(run / "manifest.json")
    if (
        masks["source_manifest_sha256"] != source_hash
        or data["source_manifest_sha256"] != source_hash
    ):
        raise ValueError("evidence from different runs")
    if [f["frame_index"] for f in masks["frames"]] != [
        f["frame_index"] for f in manifest["frames"]
    ]:
        raise ValueError("mask/frame ordering mismatch")
    arrays = []
    for row in masks["frames"]:
        path = (annotations / row["path"]).resolve()
        if not path.is_relative_to(annotations.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("mask hash/path mismatch")
        with Image.open(path) as image:
            mask = np.asarray(image)
        if mask.shape != tuple(manifest["shape"][1:3]):
            raise ValueError("mask/image shape mismatch")
        arrays.append(mask)
    rows = []
    for pair in data["pairs"]:
        a = labels_at(arrays[pair["a"]], pair["first_pixel_xy"])
        b = labels_at(arrays[pair["b"]], pair["second_pixel_xy"])
        known = (a != 255) & (b != 255)
        groups = {
            "both_background": known & (a == 0) & (b == 0),
            "same_annotated_object": known & (a > 0) & (a == b),
            "cross_region": known & (a != b),
            "void_or_outside": ~known,
        }
        errors = np.asarray(
            pair["residuals"] if pair["residuals"] is not None else [None] * pair["count"],
            dtype=float,
        )
        assert sum(int(v.sum()) for v in groups.values()) == pair["count"]
        summary = {}
        for group, selected in groups.items():
            finite = errors[selected & np.isfinite(errors)]
            summary[group] = {
                "count": int(selected.sum()),
                "finite_count": len(finite),
                "median": float(np.median(finite)) if len(finite) else None,
            }
        rows.append({"a": pair["a"], "b": pair["b"], "groups": summary})
    result = {
        "scope": "annotation-assisted region analysis, not automatic motion masks or DNF",
        "source_manifest_sha256": source_hash,
        "evidence_sha256": digest(evidence),
        "masks_manifest_sha256": digest(annotations / "masks.json"),
        "script_sha256": digest(Path(__file__)),
        "pairs": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "evidence", "annotations", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.evidence, args.annotations, args.output)
