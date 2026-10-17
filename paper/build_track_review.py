"""Source-only author-review pilot; no camera scores or pose-based selection."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest


def select_slots(tracks, count=3):
    """First nonempty grid slots, with no access to cameras or later survival."""
    return [i for i, track in enumerate(tracks) if track[0]["xy"] is not None][:count]


def build(run, source, tracks_path, output):
    if output.exists():
        raise ValueError("preserve previous review packets")
    manifest = json.loads((run / "manifest.json").read_text())
    record = json.loads(tracks_path.read_text())
    if record["source_manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("track provenance mismatch")
    images = []
    for frame in manifest["frames"]:
        path = source / frame["path"]
        if digest(path) != frame["file_sha256"]:
            raise ValueError("source image hash mismatch")
        images.append(cv2.imread(str(path)))
    slots = select_slots(record["tracks"])
    output.mkdir(parents=True)
    items = []
    for number, slot in enumerate(slots, 1):
        track = record["tracks"][slot]
        frame_dir = output / f"frames-{number}"
        frame_dir.mkdir()
        previous = np.array(track[0]["xy"])
        for frame, (source_image, observation) in enumerate(zip(images, track, strict=True)):
            height, width = source_image.shape[:2]
            canvas = np.full((height + 100, width + 320, 3), 24, np.uint8)
            canvas[70 : 70 + height, :width] = source_image
            xy = observation["xy"]
            center = np.array(xy) if xy is not None else previous
            # Center the zoom on the measured position, retaining the source view
            # so a moving crop cannot conceal a jump to a different object.
            x, y = np.rint(center).astype(int)
            padded = cv2.copyMakeBorder(source_image, 40, 40, 40, 40, cv2.BORDER_REPLICATE)
            crop = padded[y : y + 80, x : x + 80]
            canvas[70:390, width : width + 320] = cv2.resize(crop, (320, 320))
            if xy is not None:
                cv2.circle(canvas, (x, y + 70), 7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (width + 160, 230), 12, (255, 255, 0), 2, cv2.LINE_AA)
                previous = center
            status = (
                "Follow the cyan circle's physical detail."
                if xy is not None
                else "Track lost. No position is being claimed."
            )
            cv2.putText(
                canvas,
                f"Example {number} | frame {frame}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                status,
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Source video + 4x detail crop. Playback: 5 frames/s.",
                (15, height + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(frame_dir / f"{frame:06d}.png"), canvas)
        video = output / f"example-{number}.mp4"
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
                str(frame_dir / "%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        items.append(
            {"example": number, "slot": slot, "video": video.name, "sha256": digest(video)}
        )
    cards = "\n".join(
        f'<h2>Example {i["example"]}</h2><video controls muted playsinline preload="metadata" src="{i["video"]}"></video>'
        for i in items
    )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Does the circle follow the same detail?</title><style>body{font:19px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}video{width:100%}li{margin:12px 0}</style>
<h1>Does the circle follow the same physical detail?</h1>
<p>These are image tracks, not camera estimates. You are not being asked to judge a camera path.</p>
<ol><li>Pause at the start. Find the small detail inside the cyan circle. The larger view on the right shows that same area.</li>
<li>Play the video. Does the circle keep following that original detail, or slip onto a different one? Movement across the screen is normal.</li>
<li>If the circle disappears and the display says “Track lost,” no position is being claimed after that point. Judge only the part where it is present.</li></ol>
<p>For each example, reply <strong>same detail</strong>, <strong>slips</strong>, or <strong>cannot tell</strong>.
If it slips, give an approximate frame number. One answer is enough; you do not need to finish all three.</p>
<p>Separately, if clear: is that detail on a fixed part of the room, on something moving independently, or can you not tell?
Do not use motion across the screen alone to decide: the camera is moving too.</p>
<p>This is a test of the review display. “Cannot tell” is useful. Your answers will be treated as project-author development feedback, not independent validation.</p>
"""
        + cards
        + "<p>Playback is slowed to 5 displayed frames per second. The examples are the first three nonempty first-frame grid cells; selection uses no camera estimates or later tracking outcomes.</p></html>"
    )
    (output / "index.html").write_text(page)
    (output / "study.json").write_text(
        json.dumps(
            {
                "purpose": "source-track author-review pilot; not camera DNF",
                "script_sha256": digest(Path(__file__)),
                "tracks_sha256": digest(tracks_path),
                "items": items,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "tracks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.run, args.source, args.tracks, args.output)
