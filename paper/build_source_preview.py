"""Create a source-only video preview from verified dense-run input frames.

Preview playback uses the median sampling interval, not exact original timing.
Retained timestamps are authoritative. This is not candidate-tracking evidence
by itself and cannot establish a functional DNF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def build(manifest_path: Path, root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("choose a new output directory")
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["frames"]
    timestamps = np.array([float(f["timestamp"]) for f in frames])
    if len(frames) < 2 or not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError("need at least two ordered finite timestamps")
    paths = []
    for frame in frames:
        path = (root / frame["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("frame path outside dataset root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame["file_sha256"]:
            raise ValueError("source image hash mismatch")
        paths.append(path)
    output.mkdir(parents=True)
    staged = output / "frames"
    staged.mkdir()
    for index, path in enumerate(paths):
        if path.suffix.lower() != ".png":
            raise ValueError("this source preview expects original PNG frames")
        shutil.copyfile(path, staged / f"{index:06d}.png")
    interval = float(np.median(np.diff(timestamps)))
    command = [
        "ffmpeg",
        "-nostdin",
        "-n",
        "-loglevel",
        "error",
        "-framerate",
        str(1 / interval),
        "-i",
        str(staged / "%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "source.mp4"),
    ]
    subprocess.run(command, check=True)
    result = {
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "preview_sha256": hashlib.sha256((output / "source.mp4").read_bytes()).hexdigest(),
        "n_frames": len(frames),
        "source_timestamps": timestamps.tolist(),
        "preview_interval_seconds": interval,
        "maximum_timestamp_displacement_seconds": float(
            np.max(np.abs(timestamps - timestamps[0] - np.arange(len(frames)) * interval))
        ),
        "timing_note": "constant-rate preview; source timestamps, not preview timing, define temporal measurements",
        "scope": "source-only evidence, no tracker output or usability label",
        "command": command,
    }
    (output / "preview.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.manifest, args.root, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"source_timestamps", "command"}}, indent=2
        )
    )
