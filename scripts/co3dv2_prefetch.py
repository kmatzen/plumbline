#!/usr/bin/env python3
"""Selective CO3Dv2 prefetch for the VGGT / MASt3R pose evaluation.

Co3Dv2 ships at ~4.3 TB raw (276 zips, ~18 GB avg per big chunk), well past
the 200 GB disk budget of the standard vast.ai RTX 3090 box. The
``co3dv2-vggt-pose-eval`` protocol only needs 41 categories × 10 sequences ×
10 frames = **4 100 JPEGs** plus the per-category metadata — about **3 GB
total**. This script does the surgical Range-fetch.

How it works
------------
1. **Metadata** (per category): download ``{category}_000.zip`` (30-90 MB)
   in full and extract it into ``$CO3DV2_ROOT/{category}/``. These small
   archives carry exactly the files
   :class:`plumbline.datasets.co3dv2_vggt_eval.Co3Dv2VGGTPoseEvalLoader`
   reads at construction time: ``frame_annotations.jgz``,
   ``sequence_annotations.jgz``, ``set_lists/set_lists_fewview_dev.json``.

2. **Sampling**: replicate the loader's selection logic verbatim — same
   shared ``random.Random(seed)`` + ``np.random.seed(seed)``, same
   canonical-category iteration order, same ``sorted(seq_names)``, same
   ``np.random.choice(len(clean), num_frames, replace=False)``. That gives
   the exact list of relative filepaths the loader will look for at
   ``self.root / fr["filepath"]``.

3. **Images** (per category, big chunks): for each ``{category}_NNN.zip``
   (NNN >= 001), open the archive over an HTTP-Range-backed seekable
   stream. ``zipfile.ZipFile`` reads the central directory (~20 MB per
   big chunk via Range), then ``zf.open(name).read()`` Range-fetches each
   individual JPEG's local file header + compressed data. The Co3Dv2 CDN
   (CloudFront-fronted S3) advertises ``accept-ranges: bytes``, so this
   works without re-streaming the whole 18 GB chunk. Stop scanning chunks
   once every needed path for that category has been pulled.

4. **Layout**: JPEGs are written to ``$CO3DV2_ROOT/<filepath>``, which is
   the exact path ``Co3Dv2VGGTPoseEvalLoader._load_sample`` does ``open()``
   on. The directory tree under ``$CO3DV2_ROOT/`` is therefore a strict
   subset of the full CO3Dv2 distribution, with only the files the eval
   touches — same layout, same paths.

Run
---
::

    uv run python scripts/co3dv2_prefetch.py --root $CO3DV2_ROOT

    # one category for smoke:
    uv run python scripts/co3dv2_prefetch.py --root $CO3DV2_ROOT \
        --categories apple

    # bigger thread pool on a beefy box:
    uv run python scripts/co3dv2_prefetch.py --root $CO3DV2_ROOT --workers 16

The script is **idempotent**: re-running skips already-extracted metadata
zips (sentinel ``.{category}_000.done``) and already-fetched JPEGs (file
exists at the expected path).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import requests

# Import the canonical category tuple directly from the loader. This
# script's `compute_needed_paths` must iterate the same categories in
# the same order as ``Co3Dv2VGGTPoseEvalLoader._build_records`` for the
# shared py/numpy RNG to land on the same sample set; sharing the
# constant (instead of duplicating it) makes that contract single-source.
# tests/test_co3dv2_prefetch.py asserts path-for-path equivalence so any
# future refactor that desyncs the two is caught at test time.
from plumbline.datasets.co3dv2_vggt_eval import CO3D_VGGT_SEEN_CATEGORIES

CDN = "https://dl.fbaipublicfiles.com/co3dv2_231130"

# Loader defaults — match
# Co3Dv2VGGTPoseEvalLoader.__init__ signature 1:1.
_DEFAULTS = dict(
    num_frames=10,
    sequences_per_category=10,
    min_num_images=50,
    min_quality=0.5,
    max_translation_sum=1e5,
    seed=0,
)


# ---------------------------------------------------------------------------
# HTTP-Range seekable wrapper
# ---------------------------------------------------------------------------


class RangeHTTPFile(io.RawIOBase):
    """Seekable, read-only file-like object backed by HTTP Range requests.

    Suitable for handing to :class:`zipfile.ZipFile`. Issues one Range
    request per ``read()`` call. Uses a shared :class:`requests.Session`
    for keep-alive — passing in a session that's also used by other
    concurrent fetches is fine; ``requests.Session`` is thread-safe for
    independent requests on different threads.
    """

    def __init__(self, url: str, session: requests.Session) -> None:
        super().__init__()
        self.url = url
        self.session = session
        # HEAD to get the total length and confirm Range support.
        r = self.session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            # Be lenient: CloudFront sometimes omits the header but still honours Range.
            pass
        self._pos = 0

    # -- io.RawIOBase API -------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size < 0 or self._pos + size > self._size:
            size = self._size - self._pos
        if size == 0:
            return b""
        end = self._pos + size - 1  # inclusive
        headers = {"Range": f"bytes={self._pos}-{end}"}
        # Retry on transient network errors — CloudFront occasionally
        # 503's during big-zip CD reads.
        for attempt in range(4):
            try:
                r = self.session.get(self.url, headers=headers, timeout=120, stream=False)
                r.raise_for_status()
                data = r.content
                break
            except (requests.RequestException, ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if len(data) != size:
            # CloudFront returned a different length than requested; rare,
            # but tolerated as long as we advance by what we got.
            pass
        self._pos += len(data)
        return data

    @property
    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Metadata download (full *_000.zip)
# ---------------------------------------------------------------------------


def download_metadata_zip(
    root: Path,
    category: str,
    *,
    session: requests.Session,
    force: bool = False,
) -> None:
    """Download ``{category}_000.zip`` and extract its contents into ``root/``.

    The extracted layout under ``root/`` is::

        root/<category>/{frame_annotations.jgz,sequence_annotations.jgz,
                          set_lists/...,eval_batches/...,LICENSE}

    which is exactly what
    :meth:`Co3Dv2VGGTPoseEvalLoader._load_category_annotations` reads.

    Idempotent via the ``.{category}_000.done`` sentinel.
    """
    sentinel = root / f".{category}_000.done"
    if sentinel.exists() and not force:
        return
    url = f"{CDN}/{category}_000.zip"
    # Stream to a temp file under root to avoid loading 30-90 MB into RAM.
    tmp = root / f".{category}_000.zip.tmp"
    print(f"[meta] {category}: GET {url}")
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                got += len(chunk)
        if total and got != total:
            raise RuntimeError(
                f"{category}_000.zip: short read ({got}/{total} bytes)"
            )
    # Extract; the zip's root entry is ``<category>/`` so things land
    # at root/<category>/... directly.
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(root)
    tmp.unlink()
    sentinel.touch()
    print(f"[meta] {category}: done ({got/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Sampling — replicate the loader's selection
# ---------------------------------------------------------------------------


def _load_category_annotations(
    cat_dir: Path,
    *,
    min_quality: float,
) -> dict[str, list[dict[str, Any]]] | None:
    """Mirror Co3Dv2VGGTPoseEvalLoader._load_category_annotations (raw mode)."""
    frame_file = cat_dir / "frame_annotations.jgz"
    seq_file = cat_dir / "sequence_annotations.jgz"
    set_lists_file = cat_dir / "set_lists" / "set_lists_fewview_dev.json"
    if not (frame_file.exists() and seq_file.exists() and set_lists_file.exists()):
        return None

    with gzip.open(frame_file, "rt", encoding="utf-8") as f:
        frame_data: list[dict[str, Any]] = json.load(f)
    with gzip.open(seq_file, "rt", encoding="utf-8") as f:
        seq_data: list[dict[str, Any]] = json.load(f)
    with set_lists_file.open("rt", encoding="utf-8") as f:
        subset_lists: dict[str, list[list[Any]]] = json.load(f)

    good_quality = {
        s["sequence_name"]
        for s in seq_data
        if s["viewpoint_quality_score"] > min_quality
    }

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for fr in frame_data:
        by_key[(fr["sequence_name"], int(fr["frame_number"]))] = fr

    out: dict[str, list[dict[str, Any]]] = {}
    for seq_name, frame_number, filepath in subset_lists.get("test", []):
        if seq_name not in good_quality:
            continue
        fr = by_key.get((seq_name, int(frame_number)))
        if fr is None:
            continue
        out.setdefault(seq_name, []).append(
            {
                "filepath": filepath,
                "T": fr["viewpoint"]["T"],
            }
        )
    return out


def compute_needed_paths(
    root: Path,
    categories: Sequence[str],
    *,
    num_frames: int = _DEFAULTS["num_frames"],
    sequences_per_category: int = _DEFAULTS["sequences_per_category"],
    min_num_images: int = _DEFAULTS["min_num_images"],
    min_quality: float = _DEFAULTS["min_quality"],
    max_translation_sum: float = _DEFAULTS["max_translation_sum"],
    seed: int = _DEFAULTS["seed"],
    fast_eval: bool = True,
) -> list[tuple[str, str, str]]:
    """Return list of (category, sequence, relative_filepath) to fetch.

    Mirrors Co3Dv2VGGTPoseEvalLoader._build_records exactly: same shared
    py-RNG initialized once at startup, same global numpy seed, same
    category iteration order, same per-sequence frame sampling. Any
    deviation here produces a different sample set than the loader picks
    at run time, defeating the prefetch.
    """
    py_rng = random.Random(seed)
    np.random.seed(seed)
    out: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for category in categories:
        cat_dir = root / category
        anno = _load_category_annotations(cat_dir, min_quality=min_quality)
        if anno is None:
            missing.append(category)
            continue

        seq_names = sorted(anno.keys())
        if fast_eval and len(seq_names) >= sequences_per_category:
            seq_names = py_rng.sample(seq_names, sequences_per_category)
            seq_names = sorted(seq_names)

        for seq_name in seq_names:
            frames = anno[seq_name]
            clean = [
                fr for fr in frames
                if (fr["T"][0] + fr["T"][1] + fr["T"][2]) <= max_translation_sum
            ]
            if len(clean) < min_num_images:
                continue
            ids = np.random.choice(len(clean), num_frames, replace=False)
            for i in ids:
                out.append((category, seq_name, clean[int(i)]["filepath"]))

    if missing:
        print(
            f"[sample] WARNING: {len(missing)} categories missing metadata "
            f"(run with --categories that includes them): {missing[:5]}...",
            file=sys.stderr,
        )
    return out


# ---------------------------------------------------------------------------
# Big-zip Range fetch
# ---------------------------------------------------------------------------


def _fetch_one_path(
    zf: zipfile.ZipFile,
    rel_path: str,
    root: Path,
) -> tuple[str, int]:
    """Read one path from the open ZipFile and write to root/<rel_path>."""
    target = root / rel_path
    if target.exists():
        return (rel_path, 0)  # skip
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(rel_path) as src:
        data = src.read()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return (rel_path, len(data))


def fetch_images_for_category(
    root: Path,
    category: str,
    paths: set[str],
    *,
    session: requests.Session,
    max_workers: int = 8,
    max_chunks: int = 32,
) -> tuple[int, int]:
    """For one category, walk big chunks 001+ and fetch every needed path.

    Returns ``(fetched_count, bytes_fetched)``. Stops as soon as
    ``paths`` is empty (every needed JPEG has been written). Errors on
    a missing chunk are tolerated (CO3Dv2 chunk counts vary per
    category) — the loop ends when we 404.
    """
    remaining = set(paths)
    # Skip paths that are already on disk.
    already = {p for p in remaining if (root / p).exists()}
    if already:
        remaining -= already
        print(f"[img]  {category}: {len(already)} JPEGs already on disk")
    if not remaining:
        return (0, 0)

    fetched = 0
    bytes_fetched = 0
    for idx in range(1, max_chunks + 1):
        if not remaining:
            break
        url = f"{CDN}/{category}_{idx:03d}.zip"
        # HEAD first so we can break cleanly on 404 (no more chunks).
        try:
            head = session.head(url, allow_redirects=True, timeout=30)
            if head.status_code == 404:
                break
            head.raise_for_status()
        except requests.HTTPError:
            break
        print(
            f"[img]  {category}: opening {category}_{idx:03d}.zip "
            f"(remaining: {len(remaining)} paths)"
        )
        rfile = RangeHTTPFile(url, session=session)
        with zipfile.ZipFile(rfile) as zf:
            names_in_zip = set(zf.namelist())
            in_this = sorted(remaining & names_in_zip)
            if not in_this:
                continue
            t0 = time.time()
            if max_workers > 1:
                # Each thread needs its own seekable HTTP file (the
                # RangeHTTPFile maintains a cursor that's not thread-safe).
                # Open a fresh ZipFile per thread; the CD-load cost is
                # amortised because the bytes are CloudFront-cached.
                def _worker(p: str) -> tuple[str, int]:
                    rfile_t = RangeHTTPFile(url, session=session)
                    with zipfile.ZipFile(rfile_t) as zf_t:
                        return _fetch_one_path(zf_t, p, root)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_worker, p): p for p in in_this}
                    for fut in as_completed(futs):
                        _p, n = fut.result()
                        fetched += 1
                        bytes_fetched += n
            else:
                for p in in_this:
                    _p, n = _fetch_one_path(zf, p, root)
                    fetched += 1
                    bytes_fetched += n
            dt = time.time() - t0
            print(
                f"[img]  {category}: fetched {len(in_this)} paths from chunk "
                f"{idx:03d} in {dt:.1f}s ({bytes_fetched/1e6:.1f} MB so far)"
            )
            remaining -= set(in_this)

    if remaining:
        print(
            f"[img]  {category}: WARNING — {len(remaining)} paths NOT FOUND "
            f"in any of chunks 001..{idx:03d}. Sample: {sorted(remaining)[:3]}",
            file=sys.stderr,
        )
    return (fetched, bytes_fetched)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("CO3DV2_ROOT"),
        required=("CO3DV2_ROOT" not in os.environ),
        help="Target directory (default: $CO3DV2_ROOT).",
    )
    ap.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of categories (default: all 41 SEEN).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel JPEG fetches per chunk (default: 8).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=_DEFAULTS["seed"],
        help="Match the loader's RNG seed (default: 0).",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=_DEFAULTS["num_frames"],
    )
    ap.add_argument(
        "--sequences-per-category",
        type=int,
        default=_DEFAULTS["sequences_per_category"],
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after sampling; print needed-path count and exit.",
    )
    ap.add_argument(
        "--skip-meta",
        action="store_true",
        help="Don't (re-)download metadata zips; assume root is already populated.",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    categories = tuple(args.categories) if args.categories else CO3D_VGGT_SEEN_CATEGORIES
    unknown = [c for c in categories if c not in CO3D_VGGT_SEEN_CATEGORIES]
    if unknown:
        print(f"Categories not in SEEN: {unknown}", file=sys.stderr)
        return 2

    with requests.Session() as session:
        if not args.skip_meta:
            for cat in categories:
                download_metadata_zip(root, cat, session=session)

        needed = compute_needed_paths(
            root,
            categories,
            num_frames=args.num_frames,
            sequences_per_category=args.sequences_per_category,
            seed=args.seed,
        )
        by_cat: dict[str, set[str]] = {}
        for c, _s, p in needed:
            by_cat.setdefault(c, set()).add(p)
        print(
            f"[plan] {len(needed)} JPEGs across "
            f"{sum(len(set(s for _c, s, _p in needed if _c == cat)) for cat in by_cat)} "
            f"sequences in {len(by_cat)} categories."
        )
        if args.dry_run:
            return 0

        total_fetched = 0
        total_bytes = 0
        for cat in categories:
            if cat not in by_cat:
                continue
            f, b = fetch_images_for_category(
                root,
                cat,
                by_cat[cat],
                session=session,
                max_workers=args.workers,
            )
            total_fetched += f
            total_bytes += b
        print(
            f"[done] fetched {total_fetched} JPEGs, {total_bytes/1e6:.1f} MB"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
