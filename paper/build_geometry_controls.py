"""Construct point/line reading controls, not valid camera trajectories or DNF labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper.build_reference_free_review import digest, load_verified
from PIL import Image

from plumbline.image_evidence import target_epipolar_lines


def build(run: Path, root: Path, evidence: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve existing controls")
    _, poses, paths = load_verified(run, root)
    data = json.loads(evidence.read_text())
    if data["source_manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("evidence/manifest mismatch")
    pair = next(p for p in data["pairs"] if p["a"] == 0 and p["b"] == 30)
    first, second = np.array(pair["first_pixel_xy"]), np.array(pair["second_pixel_xy"])
    selected = np.linspace(0, len(first) - 1, 6, dtype=int)
    if len(set(selected)) != 6:
        raise ValueError("need six distinct matches")
    with np.load(run / "prediction.npz", allow_pickle=False) as archive:
        k = archive["intrinsics_normalized"]
    with Image.open(paths[0]) as source:
        first_size = source.size
    with Image.open(paths[30]) as source:
        target = np.asarray(source.convert("RGB"))
        second_size = source.size
    lines = target_epipolar_lines(
        first[selected], k[0], k[30], first_size, second_size, poses[0], poses[30]
    )
    if lines is None or not np.isfinite(lines).all():
        raise ValueError("need defined lines")
    output.mkdir(parents=True)
    cases = []
    for name, distance in (("control_on", 0), ("control_off", 50)):
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
        distances = []
        for index, (ax, point, line) in enumerate(
            zip(axes.flat, second[selected], lines, strict=True)
        ):
            x, y = point
            a, b, _ = line
            # Same orientation, but shift to an exact, known point-line distance.
            c = -a * x - b * y + distance
            distances.append(float(abs(a * x + b * y + c)))
            ax.imshow(target)
            if abs(b) > abs(a):
                xs = np.array([x - 100, x + 100])
                ax.plot(xs, -(a * xs + c) / b, color="cyan", linewidth=2)
            else:
                ys = np.array([y - 100, y + 100])
                ax.plot(-(b * ys + c) / a, ys, color="cyan", linewidth=2)
            ax.scatter([x], [y], edgecolors="red", facecolors="none", s=180, linewidths=2)
            ax.set_xlim(x - 90, x + 90)
            ax.set_ylim(y + 90, y - 90)
            ax.set_title(str(index + 1))
            ax.axis("off")
        path = output / f"{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        cases.append(
            {
                "name": name,
                "image_sha256": digest(path),
                "point_line_distance_pixels": distances,
                "expected_visual_relation": "consistent" if distance == 0 else "inconsistent",
            }
        )
    result = {
        "scope": "constructed visual grounding controls, not camera usability",
        "manifest_sha256": digest(run / "manifest.json"),
        "evidence_sha256": digest(evidence),
        "renderer_sha256": digest(Path(__file__)),
        "match_indices": selected.tolist(),
        "cases": cases,
    }
    (output / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "root", "evidence", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.run, args.root, args.evidence, args.output)
