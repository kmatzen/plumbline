"""Depth-anchored drift illustrations, not blinded human validation."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from paper.probe_static_rotation_response import perturb
from paper.stage_static_rgbd_control import digest
from plumbline.datasets.tum_dynamics import tum_pose_to_matrix


def project(point, camera):
    xyz = np.linalg.inv(camera) @ np.r_[point, 1.0]
    if not np.isfinite(xyz).all() or xyz[2] <= 0:
        return None
    k = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
    pixel = k @ xyz[:3]
    return (pixel[:2] / pixel[2]).tolist()


def render(root, control_path, output):
    if output.exists():
        raise ValueError("preserve previous illustrations")
    manifest_path = root / "staging-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    control = json.loads(control_path.read_text())
    if control["source_manifest_sha256"] != digest(manifest_path):
        raise ValueError("source mismatch")
    poses = [None] + [tum_pose_to_matrix(*map(float, r["reference"]["matches"][0]["values"]))
                      for r in manifest["frames"][1:]]
    drift = perturb(poses, 5, "ramp")
    frames = (1, 45, 89)
    images = []
    for frame in frames:
        row = manifest["frames"][frame]
        path = root / row["path"]
        if digest(path) != row["file_sha256"]:
            raise ValueError("RGB changed")
        images.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    output.mkdir(parents=True)
    records = []
    for slot, track in enumerate(control["tracks"]):
        measured = control["depth_tracks"][slot]["rows"][1]
        if measured["status"] != "measured" or any(track[i]["xy"] is None for i in range(1, 90)):
            continue
        point = measured["world_point_m"]
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        rows = []
        for col, (frame, image) in enumerate(zip(frames, images, strict=True)):
            xy = track[frame]["xy"]
            reference_xy = project(point, poses[frame])
            drift_xy = project(point, drift[frame])
            rows.append({"frame": frame, "tracked_xy": xy,
                         "reference_xy": reference_xy, "drift_xy": drift_xy})
            for row in range(4):
                ax = axes[row, col]
                ax.imshow(image)
                if row != 1:
                    ax.add_patch(Circle(xy, 7 if row == 0 else 4, fill=False,
                                        color="#ffdf00", linewidth=1.5))
                if row >= 1:
                    ax.set_xlim(xy[0] - 65, xy[0] + 65)
                    ax.set_ylim(xy[1] + 65, xy[1] - 65)
                if row >= 2:
                    predicted = reference_xy if row == 2 else drift_xy
                    if predicted is not None:
                        ax.plot(*predicted, marker="+", color="#00ffff", markersize=13, markeredgewidth=2)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, col].set_title(f"Frame {frame}" + (" — starting point" if frame == 1 else ""))
        for ax, label in zip(axes[:, 0], ("Scene / target", "Same crop, unmarked", "Reference camera", "Gradual 5° yaw error")):
            ax.set_ylabel(label, fontsize=11)
        fig.suptitle(f"Does the cyan + stay on the starting surface detail?  •  Slot {slot}\n"
                     "Yellow ring: image tracker (may be wrong). Cyan +: one fixed 3D point, never refitted.", fontsize=12)
        fig.text(.5, .015, "Teaching diagnostic: point anchored with frame-1 sensor depth and reference pose.\n"
                 "Not a reference-free score, blinded comparison, or human failure label. Crops follow the tracker.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .055, 1, .95))
        path = output / f"slot-{slot}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        records.append({"slot": slot, "point_world_m": point, "frames": rows,
                        "image": path.name, "image_sha256": digest(path)})
    result = {"scope": "unblinded depth-anchored teaching illustrations; not evaluation labels",
              "selection": "all slots observed in every frame 1–89 with measured depth at frame 1; no residual-based selection",
              "source_manifest_sha256": digest(manifest_path), "control_sha256": digest(control_path),
              "script_sha256": digest(Path(__file__)), "items": records}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print("Rendered", len(render(args.root, args.control, args.output)["items"]), "illustrations")
