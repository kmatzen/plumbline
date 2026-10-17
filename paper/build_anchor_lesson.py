"""Teaching display with explicitly simulated overlays, not tracker results."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest, load_verified


def demonstration_offset(frame: int) -> np.ndarray:
    """Explicitly chosen teaching offset: zero initially, then drift to 50px."""
    return np.array([min(50.0, max(0.0, (frame - 10) * 1.5)), 0.0])


def build(run: Path, root: Path, preview: Path, output: Path):
    if output.exists():
        raise ValueError("preserve previous lessons")
    manifest, _, paths = load_verified(run, root)
    record = json.loads(preview.read_text())
    if record["source_manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("preview source mismatch")
    rows = record["rows"]
    if len(rows) != len(paths) or [r["frame"] for r in rows] != list(range(len(paths))):
        raise ValueError("preview frame alignment mismatch")
    output.mkdir(parents=True)
    frames = output / "frames"
    frames.mkdir()
    evidence = []
    for index, (path, row) in enumerate(zip(paths, rows, strict=True)):
        source = cv2.imread(str(path))
        height, width = source.shape[:2]
        observed = row["observed_xy"]
        canvas = np.full((height + 150, width * 2, 3), 25, dtype=np.uint8)
        descriptions = ["ATTACHED (simulated)", "DRIFTING (simulated)"]
        offset = demonstration_offset(index)
        for panel in range(2):
            origin = panel * width
            canvas[100 : 100 + height, origin : origin + width] = source
            cv2.putText(
                canvas,
                descriptions[panel],
                (origin + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Watch the wall, not the bear.",
                (origin + 20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if observed is not None:
                xy = np.array(observed, dtype=float)
                predicted = xy + (offset if panel else np.zeros(2))
                dot = tuple(np.rint(xy + np.array([origin, 100])).astype(int))
                ring = tuple(np.rint(predicted + np.array([origin, 100])).astype(int))
                cv2.circle(canvas, dot, 5, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, ring, 12, (0, 140, 255), 2, cv2.LINE_AA)
                if panel:
                    cv2.line(canvas, dot, ring, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(
                    canvas,
                    "Track missing: cannot compare",
                    (origin + 20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            canvas,
            "Cyan dot: image detail. Orange ring: marker. Look for the ring sliding away from that detail.",
            (20, height + 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frames / f"{index:06d}.png"), canvas)
        evidence.append(
            {
                "frame": index,
                "observation_available": observed is not None,
                "left_offset_xy": [0.0, 0.0],
                "right_offset_xy": offset.tolist(),
            }
        )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-framerate",
            "5",
            "-i",
            str(frames / "%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "lesson.mp4"),
        ],
        check=True,
    )
    shutil.copyfile(frames / f"{min(40, len(rows) - 1):06d}.png", output / "example.png")
    (output / "index.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>What to look for</title>
<style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:20px/1.5 system-ui;background:#faf9f5;color:#17232b}video,img{width:100%}p{max-width:1000px}</style>
<h1>Watch whether the marker stays on the wall.</h1>
<p>The cyan dot follows a detail in the stone wall. The orange ring represents a marker placed on that detail.
On the left, they stay together. On the right, the ring gradually slides away.</p>
<video controls muted loop playsinline src="lesson.mp4" poster="example.png"></video>
<p><strong>These are simulated examples, not results from two methods.</strong>
We placed the left ring on the image track and deliberately moved the right ring away.
This explains the display; it does not establish a threshold for acceptable tracking.</p>
<h2>What can go wrong with the evidence?</h2>
<p>First check that the cyan dot stays on the same physical detail. If it slips, disappears,
or follows something that moves by itself, the comparison cannot tell us whether the camera is wrong.
The dot is an automated image track, not ground truth.</p>
<p>For real results, the ring comes from a fixed 3D point and estimated cameras.
Separation can come from camera error, calibration error, the point fit, or image-tracking error.
One marker cannot establish the quality of an entire trajectory.</p>
<p>No answers are requested on this page. Playback is slowed to 5 displayed frames per second;
the source capture timing is unknown.</p></html>""")
    result = {
        "purpose": "display instruction only; synthetic overlays are not method outputs or human labels",
        "source_manifest_sha256": digest(run / "manifest.json"),
        "source_sequence": manifest["sequence"],
        "preview_sha256": digest(preview),
        "script_sha256": digest(Path(__file__)),
        "video_sha256": digest(output / "lesson.mp4"),
        "display_fps": 5,
        "capture_timing": None,
        "offsets": evidence,
    }
    (output / "lesson.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "root", "preview", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.run, args.root, args.preview, args.output)
