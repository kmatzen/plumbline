#!/usr/bin/env python3
"""Selectively fetch KITTI raw frames for the Eigen-benchmark 652 split.

Populates ``$KITTI_ROOT/raw/<date>/<drive>_sync/...`` with only the
files actually evaluated by plumbline's KITTI reproductions. Avoids
the ~65 GB "download every raw drive fully" footprint that was a
pain point on earlier GPU sessions.

Bandwidth: ~8.5 GB (28 drive zips at ~300 MB each). On-disk: ~700 MB
after pruning (a file per listed frame) + a few MB for calib/timestamps.

Downloads are resumable; re-running the script skips drives whose
listed frames are already present. Safe to interrupt and restart.

Usage
-----

    scripts/fetch_kitti.py --kitti-root ~/data/kitti

Use ``--camera image_02`` (default) to fetch only the left camera's
frames, ``--camera both`` to fetch both rectified cameras. The
committed 652-sample list only references ``image_02``.

The fetcher does NOT download ``data_depth_annotated.zip`` (~14 GB,
single bundle covering all drives). Fetch that separately::

    curl -L -o data_depth_annotated.zip \\
        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
    unzip -d $KITTI_ROOT/depth_annotated data_depth_annotated.zip
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict

S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def parse_sample_list(path: pathlib.Path) -> list[tuple[str, str, str, str]]:
    """Return a list of (date, drive_sync, frame_id_zeropadded, camera)."""
    entries = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{path}: cannot parse '{line}'")
        drive_token, frame_token, cam_token = parts[:3]
        drive = drive_token.split("/")[-1]
        if not drive.endswith("_sync"):
            raise ValueError(f"{path}: drive token does not end in _sync: {drive_token!r}")
        # "2011_09_26_drive_0002_sync" -> "2011_09_26"
        date = drive[:10]
        frame = frame_token.zfill(10)
        if cam_token.lower() == "l":
            cam = "image_02"
        elif cam_token.lower() == "r":
            cam = "image_03"
        elif cam_token.startswith("image_0"):
            cam = cam_token
        else:
            raise ValueError(f"{path}: unknown camera token {cam_token!r}")
        entries.append((date, drive, frame, cam))
    return entries


def drive_zip_url(drive_sync: str) -> str:
    """Mirror URL for ``<date>_drive_XXXX_sync.zip`` on KITTI's S3."""
    base = drive_sync[: -len("_sync")]
    return f"{S3_BASE}/{base}/{drive_sync}.zip"


def calib_zip_url(date: str) -> str:
    return f"{S3_BASE}/{date}_calib.zip"


def download_to(url: str, dest: pathlib.Path) -> None:
    """Download ``url`` to ``dest`` with a visible progress indicator.

    Atomic: writes to ``dest.part`` first, renames on success.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r    {done/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)",
                          end="", file=sys.stderr)
        print(file=sys.stderr)
    tmp.rename(dest)


def extract_selected(zip_path: pathlib.Path, out_root: pathlib.Path,
                     wanted_prefixes: tuple[str, ...]) -> int:
    """Extract only members whose name starts with any wanted prefix.

    Returns the number of files extracted. Also extracts two-level
    auxiliary files at the top of the drive tree (calib-like, oxts
    metadata etc.) — small, cheap, keeps loaders happy.
    """
    n = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # Skip the leading date dir — the zip's root is
            # "<date>/<drive>_sync/..." whereas we already organise
            # by date. Keep the path verbatim; out_root is $KITTI_ROOT/raw.
            if any(name.startswith(p) for p in wanted_prefixes):
                target = out_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=pathlib.Path,
                        default=os.environ.get("KITTI_ROOT"),
                        help="Root dir; defaults to $KITTI_ROOT.")
    parser.add_argument("--sample-list", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent /
                                "reproductions" / "kitti_eigen_benchmark_652.txt",
                        help="Sample list (default: committed Eigen-652 list).")
    parser.add_argument("--camera", choices=("image_02", "image_03", "both"),
                        default="image_02",
                        help="Rectified-camera stream(s) to extract.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without fetching anything.")
    args = parser.parse_args()

    if args.kitti_root is None:
        parser.error("Set --kitti-root or export $KITTI_ROOT.")

    entries = parse_sample_list(args.sample_list)
    print(f"parsed {len(entries)} frames from {args.sample_list}", file=sys.stderr)

    # Group by drive; collect all listed frames across cameras.
    per_drive: dict[tuple[str, str], set[str]] = defaultdict(set)
    for date, drive, frame, cam in entries:
        per_drive[(date, drive)].add(frame)
    dates = sorted({d for d, _ in per_drive.keys()})
    print(f"  {len(per_drive)} drives across {len(dates)} dates", file=sys.stderr)

    # Which cameras to extract.
    cams = ("image_02", "image_03") if args.camera == "both" else (args.camera,)

    if args.dry_run:
        print(f"\n[dry-run] would fetch {len(dates)} calib zips and {len(per_drive)} drive zips.")
        for (date, drive), frames in sorted(per_drive.items()):
            print(f"  {drive:40s}  {len(frames)} frame(s)  cams={cams}")
        return 0

    raw_root = args.kitti_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # Calibration (one zip per date, ~4 KB each).
    for date in dates:
        calib_marker = raw_root / date / "calib_cam_to_cam.txt"
        if calib_marker.exists():
            continue
        with tempfile.TemporaryDirectory() as td:
            zpath = pathlib.Path(td) / f"{date}_calib.zip"
            download_to(calib_zip_url(date), zpath)
            # Archive root is "<date>/", so extracting into raw_root places files
            # at raw_root/<date>/*.
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(raw_root)
        print(f"  [{date}] calib installed", file=sys.stderr)

    # Drive zips: per-drive selective extract + delete.
    for (date, drive), frames in sorted(per_drive.items()):
        # Already-complete check: every listed frame present under every wanted cam.
        drive_dir = raw_root / date / drive
        missing = False
        for cam in cams:
            for frame in frames:
                if not (drive_dir / cam / "data" / f"{frame}.png").exists():
                    missing = True
                    break
            if missing:
                break
        if not missing:
            print(f"  [{drive}] already complete ({len(frames)} frame(s))", file=sys.stderr)
            continue

        # Prefixes inside the drive zip that we want to keep.
        # Archive layout: <date>/<drive_sync>/<cam>/data/<frame>.png etc.
        # Keep image_0N/timestamps.txt (cheap, loader-useful) + data/<frame>.png for listed frames.
        wanted_prefixes = []
        for cam in cams:
            wanted_prefixes.append(f"{date}/{drive}/{cam}/timestamps.txt")
            for frame in frames:
                wanted_prefixes.append(f"{date}/{drive}/{cam}/data/{frame}.png")
        wanted_prefixes = tuple(wanted_prefixes)

        with tempfile.TemporaryDirectory() as td:
            zpath = pathlib.Path(td) / f"{drive}.zip"
            download_to(drive_zip_url(drive), zpath)
            n = extract_selected(zpath, raw_root, wanted_prefixes)
        print(f"  [{drive}] extracted {n} file(s)", file=sys.stderr)

    print("\ndone.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
