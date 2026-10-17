"""Source-only author-review pilot; no camera scores or pose-based selection."""

import argparse
import json
import shutil
import subprocess
from html import escape
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest


def select_slots(tracks, count=3):
    """First nonempty grid slots, with no access to cameras or later survival."""
    return [i for i, track in enumerate(tracks) if track[0]["xy"] is not None][:count]


def render_review(items):
    """Identity-first review; skipped and unavailable evidence are not failures."""
    cards = []
    for item in items:
        number = int(item["example"])
        video = escape(item["video"], quote=True)
        cards.append(f'''<section data-example="{number}">
<h2>Example {number}</h2>
<h3>1. Find one detail in the starting image</h3>
<p>Look inside the cyan ring. Can you name the exact detail, such as a window corner?
“The wall” is not specific enough. If you cannot identify a detail, stop here for this example.</p>
<img src="frames-{number}/000000.png" alt="Starting image for example {number}, with a cyan target ring and enlarged crop">
<label>Can you identify the detail?
<select class="identity"><option value="unanswered">Not answered</option>
<option value="identifiable">Yes, I can name it</option>
<option value="unidentifiable">No, I cannot identify it</option></select></label>
<label>Name the detail <input class="detail" placeholder="For example: the lower-left corner of the window"></label>
<fieldset disabled><legend>2. Watch that same detail</legend>
<p>Keep the starting image above for comparison. Watch the detail you named, not just the ring.
Does the ring stay on it? Both may move across the screen together; that is not slipping.</p>
<video controls muted playsinline preload="metadata" src="{video}"></video>
<p><strong>Stays:</strong> the ring follows the original detail.
<strong>Slips:</strong> the original detail is still visible, but the ring moves onto a different detail.
<strong>Cannot tell:</strong> blur, obstruction, or the display prevents a decision.</p>
<p>If the ring disappears with “Track lost,” judge only the visible part. A missing ring is not evidence of slipping.</p>
<label>Your answer <select class="tracking"><option value="unanswered">Not answered</option>
<option value="same_detail">Stays on the same detail</option><option value="slips">Slips onto a different detail</option>
<option value="cannot_tell">Cannot tell</option></select></label>
<label>Frame where it first slips, if visible <input class="frame" type="number" min="0" step="1"></label>
</fieldset></section>''')
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Check one image track</title>
<style>body{font:18px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 20px}
img,video{width:100%;height:auto}section{border-top:2px solid #aaa;margin:36px 0;padding-top:12px}
label{display:block;margin:16px 0}input,select,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}
.detail{display:block;width:100%}fieldset{margin-top:24px}fieldset:disabled{opacity:.5}</style>
<h1>Check one image track</h1>
<p>First identify a detail. Then check whether the cyan ring follows it.
You do not need to judge a camera path, a 3D reconstruction, or which method is better.</p>
<p>One example is enough. “Cannot identify it” is a useful answer.
These are project-author development checks, not independent validation or camera-failure labels.</p>
"""
        + "\n".join(cards)
        + '<script type="application/json" id="evidence">'
        + json.dumps(items).replace("<", "\\u003c")
        + "</script>"
        + """
<button id="download">Download my answers</button><p id="status" role="status">Answers are not sent anywhere or saved automatically. Download them before closing this page.</p>
<script>
const sections = [...document.querySelectorAll('section')];
for (const section of sections) {
  const update = () => {
    const ready = section.querySelector('.identity').value === 'identifiable'
      && section.querySelector('.detail').value.trim().length > 0;
    section.querySelector('fieldset').disabled = !ready;
    if (!ready) section.querySelector('video').pause();
  };
  section.querySelector('.identity').addEventListener('change', update);
  section.querySelector('.detail').addEventListener('input', update);
}
document.querySelector('#download').addEventListener('click', () => {
  const answers = sections.map(section => {
    const identity = section.querySelector('.identity').value;
    const detail = section.querySelector('.detail').value.trim();
    const ready = identity === 'identifiable' && detail.length > 0;
    const tracking = ready ? section.querySelector('.tracking').value : 'not_assessed';
    const frame = section.querySelector('.frame');
    return {example: Number(section.dataset.example), identity, detail,
      tracking, first_slip_frame: tracking === 'slips' && frame.value !== '' && frame.checkValidity()
        ? Number(frame.value) : null};
  });
  const data = {schema: 'source-track-review-v2', purpose: 'author development feedback, not camera DNF',
    evidence: JSON.parse(document.querySelector('#evidence').textContent), answers};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'track-review-answers.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  document.querySelector('#status').textContent = 'Download requested. Check your downloads; no answers were sent to a server.';
});
</script></html>"""
    )


def revise_packet(previous, output):
    """Revise only presentation, retaining exact videos and source-packet identity."""
    if output.exists():
        raise ValueError("preserve previous review packets")
    study = json.loads((previous / "study.json").read_text())
    for item in study["items"]:
        if digest(previous / item["video"]) != item["sha256"]:
            raise ValueError("review video hash mismatch")
        if not (previous / f"frames-{int(item['example'])}/000000.png").is_file():
            raise ValueError("missing starting image")
    output.mkdir(parents=True)
    for item in study["items"]:
        shutil.copyfile(previous / item["video"], output / item["video"])
        relative = Path(f"frames-{int(item['example'])}/000000.png")
        (output / relative.parent).mkdir()
        shutil.copyfile(previous / relative, output / relative)
        item["starting_image_sha256"] = digest(output / relative)
    study["previous_study_sha256"] = digest(previous / "study.json")
    study["presentation_script_sha256"] = digest(Path(__file__))
    study["presentation_version"] = "identity-first-v2"
    (output / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    (output / "index.html").write_text(render_review(study["items"]))


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
    page = render_review(items)
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
