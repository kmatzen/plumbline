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
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict

# Minimum free space we require on the drive zip's temp directory before
# attempting a download. Each drive zip is ~320 MB; aria2 with 16 parallel
# splits transiently writes more, and macOS Spotlight / Time Machine can
# reclaim freshly-freed space concurrently. 1 GB is a conservative safety
# margin that has cleared the disk-full failure mode in practice.
_MIN_FREE_BYTES_PER_DRIVE: int = 1 * 1024 * 1024 * 1024

# Drive zips above this size get extracted via HTTP range requests
# (remotezip) instead of being downloaded fully. A handful of KITTI
# drives — notably the long 2011_10_03 residential sequences — exceed
# 10 GB each, and the Eigen-benchmark only needs 24 frames (~24 MB
# extracted) per drive, so a sparse-range fetch is dramatically cheaper
# than downloading + extracting the full zip. Threshold 2 GB catches
# the whales without hitting every normal 300 MB drive.
_WHALE_ZIP_THRESHOLD_BYTES: int = 2 * 1024 * 1024 * 1024

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

    Atomic: writes to ``dest.part`` first, renames on success. Prefers
    ``aria2c`` (16 parallel connections per server) when available —
    each KITTI drive zip is ~300 MB and a single TCP stream from
    eu-central-1 to a residential US line caps at ~1.3 MB/s; aria2's
    parallel chunks reach 5-10x that. Falls back to urllib when aria2
    isn't installed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}", file=sys.stderr)
    if shutil.which("aria2c"):
        _download_with_aria2(url, tmp)
    else:
        _download_with_urllib(url, tmp)
    tmp.rename(dest)


def _download_with_aria2(url: str, tmp: pathlib.Path) -> None:
    """Run aria2c with parallel range requests; let it print its own progress."""
    subprocess.run(
        [
            "aria2c",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--console-log-level=warn",
            "--summary-interval=5",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--dir", str(tmp.parent),
            "--out", tmp.name,
            url,
        ],
        check=True,
    )


def _head_content_length(url: str) -> int | None:
    """HEAD the URL and return Content-Length (or None if unavailable)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _extract_via_remotezip(url: str, wanted_names: list[str], out_root: pathlib.Path) -> int:
    """Stream-extract only the listed zip entries via HTTP range requests.

    Pulls individual files' bytes directly from the remote zip without
    downloading the whole archive. Requires the ``remotezip`` package
    and a server that supports HTTP Range (KITTI's S3 mirror does).
    Returns the number of entries extracted.
    """
    try:
        from remotezip import RemoteZip  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "remotezip is needed for whale-drive extraction. Install with "
            "`uv pip install remotezip` or pass smaller drives only."
        ) from exc

    n = 0
    with RemoteZip(url) as zf:
        names_in_zip = {info.filename for info in zf.infolist()}
        for name in wanted_names:
            if name not in names_in_zip:
                print(f"    skipping (not in zip): {name}", file=sys.stderr)
                continue
            zf.extract(name, path=out_root)
            n += 1
    return n


def _download_with_urllib(url: str, tmp: pathlib.Path) -> None:
    """Single-stream urllib fallback — slower but no extra dependency."""
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
                    print(
                        f"\r    {done/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)",
                        end="",
                        file=sys.stderr,
                    )
        print(file=sys.stderr)


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
    # Continue-on-failure: a single drive failing (disk-full, network, etc.)
    # shouldn't abort the whole run. We log the failure and move on; the
    # fetcher is idempotent so a re-run picks up where this one left off.
    failed: list[tuple[str, str, Exception]] = []
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

        try:
            zip_url = drive_zip_url(drive)
            # Route whale drives (>2 GB) through remotezip HTTP range
            # extraction — downloading the full zip just to throw away 99%
            # of it is wasteful + doesn't fit on small disks. Normal
            # drives (~300 MB) use the aria2/urllib download-then-extract
            # path because that's faster for entries we keep mostly-all-of.
            size = _head_content_length(zip_url)
            if size is not None and size > _WHALE_ZIP_THRESHOLD_BYTES:
                print(
                    f"  [{drive}] {size/1e9:.1f} GB zip — using remotezip range extraction",
                    file=sys.stderr,
                )
                n = _extract_via_remotezip(zip_url, list(wanted_prefixes), raw_root)
            else:
                with tempfile.TemporaryDirectory() as td:
                    td_path = pathlib.Path(td)
                    free = shutil.disk_usage(td_path).free
                    if free < _MIN_FREE_BYTES_PER_DRIVE:
                        raise OSError(
                            f"only {free/1e9:.2f} GB free at {td_path}; "
                            f"need >= {_MIN_FREE_BYTES_PER_DRIVE/1e9:.1f} GB per drive"
                        )
                    zpath = td_path / f"{drive}.zip"
                    download_to(zip_url, zpath)
                    n = extract_selected(zpath, raw_root, wanted_prefixes)
            print(f"  [{drive}] extracted {n} file(s)", file=sys.stderr)
        except (OSError, subprocess.CalledProcessError) as exc:
            # Record and continue — partial progress is still progress.
            # Re-running is idempotent: already-complete drives are skipped
            # and this drive will be re-attempted next run.
            failed.append((date, drive, exc))
            print(f"  [{drive}] FAILED: {exc}", file=sys.stderr)

    print(file=sys.stderr)
    if failed:
        print(
            f"done with {len(failed)} failure(s); re-run this script to retry:",
            file=sys.stderr,
        )
        for date, drive, exc in failed:
            print(f"  {drive}: {type(exc).__name__}", file=sys.stderr)
        return 2
    print("done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
