"""Render exploratory RGB/depth witnesses, not blinded validation examples."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from paper.stage_rgbd_validation import sha


def render(run, source, tracks_path, audit_path, depth_root, output):
    if output.exists():
        raise ValueError("preserve previous witness packets")
    manifest = json.loads((run / "manifest.json").read_text())
    tracks = json.loads(tracks_path.read_text())
    audit = json.loads(audit_path.read_text())
    depth_manifest = json.loads((depth_root / "depth-manifest.json").read_text())
    assert audit["source_manifest_sha256"] == sha(run / "manifest.json")
    assert audit["tracks_sha256"] == sha(tracks_path)
    assert audit["depth_manifest_sha256"] == sha(depth_root / "depth-manifest.json")
    output.mkdir(parents=True)
    items = []
    for track in audit["tracks"]:
        witness = track["observed_spread"]["diameter"]
        if witness is None:
            continue
        slot = track["slot"]
        frames = [0, *witness["frames"]]
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        for col, frame in enumerate(frames):
            info = manifest["frames"][frame]
            path = source / info["path"]
            assert sha(path) == info["file_sha256"]
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            xy = tracks["tracks"][slot][frame]["xy"]
            assert xy is not None
            x, y = xy
            for row in (0, 1):
                ax = axes[row, col]
                ax.imshow(rgb)
                ax.add_patch(Circle((x, y), 6, fill=False, color="cyan", linewidth=1))
                if row == 1:
                    ax.set_xlim(max(0, x - 45), min(639, x + 45))
                    ax.set_ylim(min(479, y + 45), max(0, y - 45))
                ax.axis("off")
            axes[0, col].set_title(f"Frame {frame} ({'starting target' if col == 0 else 'diameter endpoint'})")
            row = depth_manifest["rows"][frame]
            ax = axes[2, col]
            if row["status"] == "staged":
                depth_path = depth_root / row["nearest_depth"]["path"]
                assert sha(depth_path) == row["depth_sha256"]
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(float) / 5000
                depth[depth == 0] = np.nan
                im = ax.imshow(depth, vmin=0, vmax=10, cmap="viridis")
                ax.add_patch(Circle((x, y), 6, fill=False, color="red", linewidth=1))
                ax.set_xlim(max(0, x - 45), min(639, x + 45))
                ax.set_ylim(min(479, y + 45), max(0, y - 45))
                fig.colorbar(im, ax=ax, label="Depth (m), fixed 0–10 scale", fraction=.04)
                ax.set_title(f"Depth crop; offset {row['nearest_depth']['delta_seconds'] * 1000:.3f} ms")
            else:
                ax.text(.5, .5, "No depth within 20 ms", ha="center", transform=ax.transAxes)
            ax.axis("off")
        fig.suptitle(f"Slot {slot}: largest measured separation {witness['distance_m']:.3f} m\nExploratory selection by 3D spread; not a camera-failure label")
        fig.tight_layout()
        path = output / f"slot-{slot}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        items.append({"slot": slot, "frames": frames, "image": path.name, "sha256": sha(path)})
    record = {
        "scope": "exploratory source inspection by assistant; not human validation",
        "selection": "all slots with a measured diameter; starting frame plus both diameter endpoints",
        "audit_sha256": sha(audit_path),
        "script_sha256": sha(Path(__file__)),
        "items": items,
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "tracks", "audit", "depth_root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    render(args.run, args.source, args.tracks, args.audit, args.depth_root, args.output)
