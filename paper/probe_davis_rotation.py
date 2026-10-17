"""Development rotation sensitivity and epipolar overlays on fixed image matches.

Not a preference benchmark: the original estimate is not ground truth, and
perturbation magnitude does not define human usability. No labels are generated.
"""

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
from scipy.spatial.transform import Rotation

from plumbline.image_evidence import (
    rays_from_normalized_intrinsics,
    sampson_residuals,
    target_epipolar_lines,
)


def perturb(pose: np.ndarray, angle: float, axis: int) -> np.ndarray:
    result = pose.copy()
    vector = np.zeros(3)
    vector[axis] = np.deg2rad(angle)
    result[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(vector).as_matrix()
    return result


def probe(run: Path, root: Path, evidence: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve previous probes")
    manifest, poses, paths = load_verified(run, root)
    data = json.loads(evidence.read_text())
    if data["source_manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("evidence belongs to another run")
    with np.load(run / "prediction.npz", allow_pickle=False) as archive:
        intrinsics = archive["intrinsics_normalized"]
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    rows = []
    for pair in data["pairs"]:
        a, b = pair["a"], pair["b"]
        first = np.asarray(pair["first_pixel_xy"]).reshape(-1, 2)
        second = np.asarray(pair["second_pixel_xy"]).reshape(-1, 2)
        first_rays = rays_from_normalized_intrinsics(first, intrinsics[a], sizes[a])
        second_rays = rays_from_normalized_intrinsics(second, intrinsics[b], sizes[b])
        for axis in range(3):
            for angle in (-15, -5, -1, 0, 1, 5, 15):
                errors = sampson_residuals(
                    first_rays, second_rays, poses[a], perturb(poses[b], angle, axis)
                )
                finite = errors[np.isfinite(errors)] if errors is not None else np.array([])
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "axis": axis,
                        "angle_degrees": angle,
                        "finite_count": len(finite),
                        "median": float(np.median(finite)) if len(finite) else None,
                    }
                )
    # Fixed pair and image-order subsampling, never residual-based cherry-picking.
    pair = next(p for p in data["pairs"] if p["a"] == 0 and p["b"] == 30)
    a, b = pair["a"], pair["b"]
    first = np.asarray(pair["first_pixel_xy"]).reshape(-1, 2)
    second = np.asarray(pair["second_pixel_xy"]).reshape(-1, 2)
    if len(first) < 6:
        raise ValueError("fixed overlay pair requires six matches")
    chosen = np.linspace(0, len(first) - 1, 6, dtype=int)
    colors = plt.get_cmap("tab10")(np.arange(6))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for row, angle in enumerate((0, 15)):
        candidate = perturb(poses[b], angle, 1)
        lines = target_epipolar_lines(
            first[chosen], intrinsics[a], intrinsics[b], sizes[a], sizes[b], poses[a], candidate
        )
        for column, index in enumerate((a, b)):
            ax = axes[row, column]
            with Image.open(paths[index]) as image:
                ax.imshow(image.convert("RGB"))
            xy = first[chosen] if column == 0 else second[chosen]
            ax.scatter(
                xy[:, 0], xy[:, 1], edgecolors=colors, s=70, marker="o", facecolors="none", linewidths=2
            )
            for label, point in enumerate(xy):
                ax.text(
                    *point,
                    str(label + 1),
                    color="white",
                    fontsize=10,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
                )
            ax.set_title(
                f"Frame {manifest['frames'][index]['frame_index']} | "
                + ("original estimate" if angle == 0 else "target rotation +15 deg local Y")
            )
            ax.set_xlim(-0.5, sizes[index][0] - 0.5)
            ax.set_ylim(sizes[index][1] - 0.5, -0.5)
            ax.axis("off")
        if lines is not None:
            ax = axes[row, 1]
            for line, color in zip(lines, colors, strict=True):
                x, y, c = line
                if not np.isfinite(line).all():
                    continue
                if abs(y) > abs(x):
                    xs = np.array([0, sizes[b][0] - 1])
                    ax.plot(xs, -(x * xs + c) / y, color=color, linewidth=1.5)
                else:
                    ys = np.array([0, sizes[b][1] - 1])
                    ax.plot(-(y * ys + c) / x, ys, color=color, linewidth=1.5)
    fig.suptitle(
        "Circles: image matches. Same-color lines: candidate epipolar constraints.\n"
        "Static correct matches should lie on their lines. Moving objects and false matches may not.\n"
        "Estimated calibration; no ground-truth trajectory or usability label."
    )
    output.mkdir(parents=True)
    fig.savefig(output / "overlay.png", dpi=130)
    plt.close(fig)
    result = {
        "scope": "development sensitivity, not preference accuracy",
        "manifest_sha256": digest(run / "manifest.json"),
        "evidence_sha256": digest(evidence),
        "script_sha256": digest(Path(__file__)),
        "overlay_sha256": digest(output / "overlay.png"),
        "overlay_match_indices": chosen.tolist(),
        "rows": rows,
    }
    (output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "root", "evidence", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = probe(args.run, args.root, args.evidence, args.output)
    print(json.dumps({"rows": len(result["rows"])}))
