#!/usr/bin/env python
"""Stage the TUM-RGBD freiburg3 *dynamic* sequences for video-pose eval.

These are the eight sequences MonST3R / DAGE Table 4 use for the
"TUM-Dynamics" pose column (DAGE arXiv:2603.03744). The data is public (no
ToS): each sequence is a ``.tgz`` from the TUM Computer Vision group.

The DAGE/MonST3R pose eval only uses **90 frames per sequence** (first 90 of the
associated rgb↔groundtruth stream, at stride 3). So rather than extract the full
~800-frame archive (~0.5 GB each, depth included), this script downloads each
``.tgz`` and extracts **only** ``rgb.txt`` + ``groundtruth.txt`` + the 90
selected rgb frames — ~30 MB/sequence, ~250 MB total (disk-friendly for tight
hosts). The frame selection reuses the loader's own
``select_tum_frames`` so staging and eval agree exactly.

Already-downloaded full sequences (an ``rgb.txt`` already on disk) are *pruned*
in place to the same 90 frames, reclaiming space.

Example::

    python scripts/stage_tum_dynamics.py --out ~/data/tum_dynamics

Then point the loader at it: ``--data-root ~/data/tum_dynamics`` or
``export TUM_ROOT=~/data/tum_dynamics``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    from plumbline.datasets.tum_dynamics import TUM_DYNAMIC_SEQUENCES, select_tum_frames
except ImportError:  # allow running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from plumbline.datasets.tum_dynamics import TUM_DYNAMIC_SEQUENCES, select_tum_frames

# Mirrors MonST3R's data/download_tum_dynamics.sh.
_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3"


def _download(url: str, dest: Path) -> None:
    """Download ``url`` → ``dest`` (prefer wget/curl, fall back to urllib)."""
    for tool, cmd in (
        ("wget", ["wget", "-q", "--show-progress", "-O", str(dest), url]),
        ("curl", ["curl", "-fL", "--progress-bar", "-o", str(dest), url]),
    ):
        try:
            subprocess.run(cmd, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{tool} failed downloading {url}: {exc}") from exc
    urllib.request.urlretrieve(url, dest)


def _prune_to_selected(seq_dir: Path) -> int:
    """Delete every rgb frame not in the 90-frame eval selection. Returns #kept."""
    sel_rgb, _gt = select_tum_frames(seq_dir / "rgb.txt", seq_dir / "groundtruth.txt")
    keep = {Path(p).name for p in sel_rgb}
    rgb_dir = seq_dir / "rgb"
    for img in rgb_dir.glob("*"):
        if img.name not in keep:
            img.unlink()
    return len(keep)


def _stage_selective(seq: str, out: Path, keep_tgz: bool) -> int:
    """Download one sequence and extract only txts + the 90 selected frames."""
    seq_dir = out / seq
    tgz = out / f"{seq}.tgz"
    url = f"{_BASE_URL}/{seq}.tgz"
    print(f"[get ] {url}", flush=True)
    _download(url, tgz)
    with tarfile.open(tgz, "r:gz") as tf:
        # 1) the two tiny index files, so we can compute the selection.
        for name in (f"{seq}/rgb.txt", f"{seq}/groundtruth.txt"):
            tf.extract(name, out)
        sel_rgb, _gt = select_tum_frames(seq_dir / "rgb.txt", seq_dir / "groundtruth.txt")
        # 2) only the selected rgb frames.
        members = []
        for rel in sel_rgb:
            try:
                members.append(tf.getmember(f"{seq}/{rel}"))
            except KeyError:
                print(f"[warn] {seq}: archive missing {rel}", file=sys.stderr)
        tf.extractall(out, members=members)
    if not keep_tgz:
        tgz.unlink()
    return len(sel_rgb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    ap.add_argument(
        "--scenes",
        nargs="*",
        default=list(TUM_DYNAMIC_SEQUENCES),
        help="Subset of sequence names (default: all 8 dynamic sequences).",
    )
    ap.add_argument("--keep-tgz", action="store_true", help="Don't delete archives after extract.")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    staged = 0
    for seq in args.scenes:
        seq_dir = out / seq
        if (seq_dir / "rgb.txt").exists() and (seq_dir / "groundtruth.txt").exists():
            n = _prune_to_selected(seq_dir)
            print(f"[prune] {seq}: kept {n} frames (already downloaded)", flush=True)
            staged += 1
            continue
        n = _stage_selective(seq, out, args.keep_tgz)
        print(f"[ok  ] {seq}: {n} frames", flush=True)
        staged += 1

    print(f"\nStaged {staged}/{len(args.scenes)} sequences into {out}", flush=True)
    print(f"Run with:  --data-root {out}   (or export TUM_ROOT={out})")
    return 0 if staged else 1


if __name__ == "__main__":
    raise SystemExit(main())
