#!/usr/bin/env python
"""Stage a RealEstate10K pose-eval subset by scraping frames from YouTube.

RealEstate10K ships only per-clip camera ``.txt`` files (URL + per-frame
intrinsics+pose); the RGB frames must be cut from the source YouTube video at
each microsecond timestamp. This script does exactly that, disk-carefully:

  1. read the official test ``.txt`` files (from the 720 MB metadata tar),
  2. for each clip download the video at <=480p (video-only, no audio) to a
     scratch file, extract ``--frames-per-clip`` evenly-spaced frames with
     ffmpeg, then **delete the video immediately**,
  3. keep the clip only if >= ``--min-frames`` frames were extracted, writing
     ``<out>/<clip_id>/<clip_id>.txt`` + ``<out>/<clip_id>/<timestamp>.jpg``
     in the exact layout ``RealEstate10KPoseEvalLoader`` expects.

It is deliberately resumable (skips clips already staged), fault-tolerant (a
dead/blocked video just drops the clip), and bounded — it stops at
``--target`` usable clips or when free disk falls below ``--min-free-gb``.

YouTube link-rot + bot-blocking means the usable yield is well under 100%;
run a small ``--target`` pilot first and read the printed success rate before
committing to a big scrape. ``--cookies-from-browser`` / ``--cookies`` are
forwarded to yt-dlp if bot-blocking bites.

Example (pilot)::

    python scripts/stage_realestate10k.py \
        --meta ~/data/re10k_meta/RealEstate10K/test \
        --out  ~/data/realestate10k \
        --target 20
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_timestamps(txt_path: Path) -> tuple[str, list[int]]:
    """Return ``(youtube_url, [timestamps_microseconds])`` from a clip .txt."""
    url = ""
    timestamps: list[int] = []
    for raw in txt_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("http"):
            url = line
            continue
        parts = line.split()
        if len(parts) >= 19:
            timestamps.append(int(parts[0]))
    return url, timestamps


def evenly_spaced(items: list[int], k: int) -> list[int]:
    """Pick k evenly-spaced elements (inclusive of endpoints) from a list."""
    if k >= len(items):
        return items
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download_video(url: str, dest: Path, ytdlp_extra: list[str]) -> bool:
    """Download <=480p video-only to ``dest``. Returns True on success."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=480]/b[height<=480]/worst",
        "--no-playlist", "--no-warnings", "--quiet",
        "--retries", "2", "--socket-timeout", "20",
        "-o", str(dest),
        *ytdlp_extra,
        url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def extract_frame(video: Path, sec: float, out: Path) -> bool:
    """Extract a single frame at ``sec`` seconds (accurate input seek)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{sec:.6f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def stage_clip(
    txt: Path, out_root: Path, scratch: Path, *,
    frames_per_clip: int, min_frames: int, ytdlp_extra: list[str],
) -> int:
    """Stage one clip. Returns number of frames extracted (0 = dropped)."""
    clip_id = txt.stem
    clip_dir = out_root / clip_id
    if (clip_dir / f"{clip_id}.txt").exists():
        existing = len(list(clip_dir.glob("*.jpg")))
        if existing >= min_frames:
            return existing  # already staged
        shutil.rmtree(clip_dir, ignore_errors=True)

    url, timestamps = parse_timestamps(txt)
    if not url or len(timestamps) < min_frames:
        return 0
    chosen = evenly_spaced(timestamps, frames_per_clip)

    video = scratch / f"{clip_id}.mp4"
    video.unlink(missing_ok=True)
    if not download_video(url, video, ytdlp_extra):
        video.unlink(missing_ok=True)
        return 0

    clip_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for ts in chosen:
        if extract_frame(video, ts / 1_000_000.0, clip_dir / f"{ts}.jpg"):
            n_ok += 1
    video.unlink(missing_ok=True)

    if n_ok < min_frames:
        shutil.rmtree(clip_dir, ignore_errors=True)
        return 0
    shutil.copyfile(txt, clip_dir / f"{clip_id}.txt")
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, required=True, help="dir of test .txt files")
    ap.add_argument("--out", type=Path, required=True, help="output dataset root")
    ap.add_argument("--target", type=int, default=20, help="usable clips to stage")
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = all clips")
    ap.add_argument("--cookies", type=str, default=None)
    ap.add_argument("--cookies-from-browser", type=str, default=None)
    args = ap.parse_args()

    ytdlp_extra: list[str] = []
    if args.cookies:
        ytdlp_extra += ["--cookies", args.cookies]
    if args.cookies_from_browser:
        ytdlp_extra += ["--cookies-from-browser", args.cookies_from_browser]

    txts = sorted(args.meta.glob("*.txt"))
    random.Random(args.seed).shuffle(txts)
    if args.max_attempts:
        txts = txts[: args.max_attempts]

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="re10k_scratch_"))
    staged = attempts = 0
    try:
        for txt in txts:
            if staged >= args.target:
                break
            if free_gb(args.out) < args.min_free_gb:
                print(f"[stop] free disk {free_gb(args.out):.1f} GB < {args.min_free_gb}")
                break
            attempts += 1
            n = stage_clip(
                txt, args.out, scratch,
                frames_per_clip=args.frames_per_clip,
                min_frames=args.min_frames,
                ytdlp_extra=ytdlp_extra,
            )
            if n:
                staged += 1
                flag = "OK "
            else:
                flag = "-- "
            rate = staged / attempts if attempts else 0.0
            print(
                f"[{flag}] {staged}/{args.target} staged | "
                f"attempt {attempts} ({rate:.0%} hit) | {txt.stem} n={n} | "
                f"free {free_gb(args.out):.1f}GB",
                flush=True,
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nDONE: {staged} clips staged from {attempts} attempts "
          f"({staged / attempts:.0%} hit rate)" if attempts else "DONE: no attempts")
    return 0 if staged else 1


if __name__ == "__main__":
    raise SystemExit(main())
