"""Hash-verified source/trajectory contact sheet without reference camera poses.

This development evidence does not establish camera accuracy or a DNF verdict.
Frame indices, not seconds, label motion. Positions retain predicted units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plumbline.conventions import assert_valid_extrinsics


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified(run: Path, root: Path) -> tuple[dict, np.ndarray, list[Path]]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("run must be complete")
    prediction_path = run / "prediction.npz"
    if digest(prediction_path) != manifest["prediction_file_sha256"]:
        raise ValueError("prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as archive:
        poses = archive["extrinsics"]
    frames = manifest["frames"]
    if poses.shape != (len(frames), 4, 4) or len(frames) < 2:
        raise ValueError("pose/frame count mismatch")
    assert_valid_extrinsics(poses)
    indices = [f["frame_index"] for f in frames]
    if indices != sorted(set(indices)):
        raise ValueError("frame indices must be unique and increasing")
    paths = []
    for frame in frames:
        path = (root / frame["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != frame["file_sha256"]:
            raise ValueError("source frame path or hash mismatch")
        paths.append(path)
    return manifest, poses, paths


def build(run: Path, root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve prior artifacts; choose a new output directory")
    manifest, poses, paths = load_verified(run, root)
    selected = np.unique(np.linspace(0, len(paths) - 1, min(6, len(paths)), dtype=int))
    positions = poses[:, :3, 3]
    indices = np.array([f["frame_index"] for f in manifest["frames"]])
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(paths)))
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    grid = fig.add_gridspec(3, 6)
    for column, index in enumerate(selected):
        ax = fig.add_subplot(grid[0, column])
        with Image.open(paths[index]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"Frame {indices[index]}")
        ax.axis("off")
    for column, (a, b) in enumerate(((0, 1), (0, 2), (2, 1))):
        ax = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        ax.plot(positions[:, a], positions[:, b], color="0.75", linewidth=1)
        ax.scatter(positions[:, a], positions[:, b], c=colors, s=10)
        span = max(float(np.ptp(positions[:, [a, b]], axis=0).max()), 1e-6)
        for index in selected:
            p = positions[index]
            # Camera +Z optical direction, not velocity; glyph length is arbitrary.
            direction = poses[index, :3, 2]
            ax.arrow(
                p[a],
                p[b],
                direction[a] * span * 0.12,
                direction[b] * span * 0.12,
                color=colors[index],
                head_width=span * 0.02,
                length_includes_head=True,
            )
            ax.annotate(str(indices[index]), (p[a], p[b]), fontsize=8)
        ax.set_xlabel(f"{'XYZ'[a]} (predicted units)")
        ax.set_ylabel(f"{'XYZ'[b]} (predicted units)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    relative = poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    for column, values, label in (
        (0, step, "Translation step (predicted units)"),
        (3, angles, "Rotation step (degrees)"),
    ):
        ax = fig.add_subplot(grid[2, column : column + 3])
        ax.plot(indices[1:], values)
        ax.set_xlabel("Ending frame index; capture timing unknown")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Source frames and estimated camera motion — no reference poses\n"
        "Arrows: camera +Z direction (arbitrary glyph length). "
        "No alignment, error metric, or usability verdict."
    )
    output.mkdir(parents=True)
    fig.savefig(output / "review.png", dpi=140)
    plt.close(fig)
    result = {
        "renderer": "reference_free_contact_sheet_v1",
        "renderer_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(run / "manifest.json"),
        "prediction_file_sha256": manifest["prediction_file_sha256"],
        "review_sha256": digest(output / "review.png"),
        "source_frames_reverified": len(paths),
        "selected_frame_indices": indices[selected].tolist(),
        "functional_verdict": "unreviewed",
        "limitations": [
            "sparse source views",
            "no reprojection evidence",
            "unknown capture timing",
            "predicted units are not independently calibrated",
            "not an accuracy test",
        ],
    }
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.root, args.output), indent=2))
