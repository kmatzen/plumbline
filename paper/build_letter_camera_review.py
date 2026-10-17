"""One author-review example with actual estimated cameras, not a blinded study."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.build_anchor_preview import fit_point
from paper.probe_static_point_support import load_prediction
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest


def project_marker(point, pose):
    # Saved rotations can be slightly nonorthonormal; do not replace inverse
    # by transpose when rendering their actual camera transformation.
    camera = np.linalg.solve(np.asarray(pose, dtype=float), np.r_[point, 1.])[:3]
    if camera[2] <= 0:
        return None
    pixel = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]]) @ camera
    return pixel[:2]/pixel[2]


def build(root, tracks_path, predictions, output):
    if output.exists():
        raise ValueError("preserve existing reviews")
    images, frames, source_hash = load_rgb(root)
    control = json.loads(tracks_path.read_text())
    if control["source_manifest_sha256"] != source_hash:
        raise ValueError("track source mismatch")
    track = control["tracks"][6]
    fit_indices = list(range(1, 11))
    shown = (1, 45, 89)
    if any(track[i]["xy"] is None for i in fit_indices+list(shown)):
        raise ValueError("selected teaching detail unavailable")
    size = np.array([640, 480])
    k = np.array([[535.4/640, 0, (320.1+.5)/640],
                  [0, 539.2/480, (247.6+.5)/480], [0, 0, 1.]])
    input_hash = hashlib.sha256(images.tobytes()).hexdigest()
    # Fixed opaque order for author comprehension only, not random study allocation.
    mappings = (("A", "da3-v1", "DA3-LARGE-1.1"), ("B", "dage-v1", "DAGE"))
    records = []
    for label, directory, model in mappings:
        poses, provenance = load_prediction(predictions/directory, source_hash, frames, input_hash)
        if provenance["model"] != model:
            raise ValueError("model mismatch")
        point = fit_point(np.array([track[i]["xy"] for i in fit_indices]),
                          poses[fit_indices], np.repeat(k[None], 10, axis=0), size)
        if point is None:
            raise ValueError("early point fit is degenerate; no replacement point allowed")
        projected = [project_marker(point, pose) for pose in poses]
        if any(projected[i] is None for i in fit_indices):
            raise ValueError("nonpositive early depth")
        records.append({"display": label, "model": model, "provenance": provenance,
                        "point_world": point.tolist(),
                        "projected_xy": [p.tolist() if p is not None else None for p in projected]})
    output.mkdir(parents=True)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for col, frame in enumerate(shown):
        xy = np.array(track[frame]["xy"])
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(images[frame])
            ax.set_xlim(xy[0]-58, xy[0]+58)
            ax.set_ylim(xy[1]+58, xy[1]-58)
            ax.set_xticks([])
            ax.set_yticks([])
            if row:
                predicted = records[row-1]["projected_xy"][frame]
                if predicted is None:
                    ax.text(.5, .5, "No projection", transform=ax.transAxes, ha="center", color="red")
                else:
                    ax.plot(*predicted, "+", color="#00ffff", markersize=14, markeredgewidth=2)
                    if np.max(np.abs(np.array(predicted)-xy))>58:
                        ax.text(.5, .08, "Marker outside crop", transform=ax.transAxes,
                                ha="center", color="white", bbox={"facecolor":"black","alpha":.7})
        axes[0, col].set_title(f"Frame {frame}")
    for ax, label in zip(axes[:, 0], ("Image without marker", "A", "B")):
        ax.set_ylabel(label, fontsize=13)
    fig.suptitle("Does the cyan + stay on the same part of the letter e?", fontsize=16)
    fig.text(.5, .025, "Read each row left to right. A and B use actual estimated cameras.\n"
             "Each 3D marker is fitted in frames 1-10, then held fixed. Crops follow an image tracker.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(output/"review.png", dpi=130)
    plt.close(fig)
    result = {"scope": "author comprehension example, not independent validation or whole-trajectory DNF",
              "selection": "previously shown printed-e detail, slot 6; frames 1/45/89 fixed before this rendering; no residual-based method selection",
              "question": "Which rows keep the cyan cross on the same printed detail: A, B, both, neither, or cannot tell?",
              "limitations": "only one selected detail and three displayed frames; image tracking, calibration, point fitting, and camera errors can all affect placement; models inferred from all 90 frames",
              "source_sha256": source_hash, "tracks_sha256": digest(tracks_path),
              "fit_indices": fit_indices, "shown_frames": list(shown), "records": records,
              "script_sha256": digest(Path(__file__)),
              "dependencies_sha256": {p.name: digest(p) for p in (Path(__file__).with_name("build_anchor_preview.py"),
                  Path(__file__).with_name("probe_static_point_support.py"), Path(__file__).with_name("run_static_dage.py"))},
              "image_sha256": digest(output/"review.png")}
    (output/"manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tracks", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.root, args.tracks, args.predictions, args.output)
