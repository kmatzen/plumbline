"""Tests for the ``scripts/co3dv2_prefetch.py`` selective-fetch contract.

The prefetch script's whole point is "fetch exactly the JPEGs
:class:`plumbline.datasets.co3dv2_vggt_eval.Co3Dv2VGGTPoseEvalLoader`
will look for at iteration time, nothing more, nothing less." That
contract lives in two places at once — ``compute_needed_paths`` in the
script, and ``_build_records`` in the loader — and a silent drift
between them turns the next GPU run into a 60-minute prefetch followed
by FileNotFoundError on the first sample.

These tests pin the contract with a synthetic CO3Dv2 micro-dataset (no
network, no real metadata zips): a few SEEN categories, each with a
handful of sequences + frames whose ``viewpoint_quality_score`` and
``T`` values exercise both the quality filter and the runaway-translation
filter. We then run both implementations against that tree and assert
their outputs are path-for-path equal.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# The prefetch script lives under ``scripts/``, not in the installed
# package, so import it by file path. Same trick the docs/site coverage
# test uses for other scripts.
_PREFETCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "co3dv2_prefetch.py"
_spec = importlib.util.spec_from_file_location("co3dv2_prefetch", _PREFETCH_PATH)
assert _spec is not None and _spec.loader is not None
_prefetch = importlib.util.module_from_spec(_spec)
sys.modules["co3dv2_prefetch"] = _prefetch
_spec.loader.exec_module(_prefetch)

from plumbline.datasets.co3dv2_vggt_eval import (  # noqa: E402
    CO3D_VGGT_SEEN_CATEGORIES,
    Co3Dv2VGGTPoseEvalLoader,
)


# ---------------------------------------------------------------------------
# Synthetic micro-dataset builders
# ---------------------------------------------------------------------------


def _write_category(
    root: Path,
    category: str,
    *,
    sequences: int,
    frames_per_seq: int,
    bad_quality_seqs: int = 0,
    runaway_T_per_seq: int = 0,
) -> None:
    """Write a minimal Co3Dv2-shaped tree for one category under ``root``.

    Produces the three files
    :meth:`Co3Dv2VGGTPoseEvalLoader._load_category_annotations` reads:

    - ``<root>/<cat>/sequence_annotations.jgz`` —
      ``[{sequence_name, viewpoint_quality_score}, ...]``. The first
      ``bad_quality_seqs`` sequences get ``score = 0.3`` (below the
      ``min_quality=0.5`` default) so they're filtered out — exercises
      the quality gate.
    - ``<root>/<cat>/frame_annotations.jgz`` —
      ``[{sequence_name, frame_number, viewpoint: {R, T, focal_length,
      principal_point}}, ...]``. Frames ``> frames_per_seq - 1 -
      runaway_T_per_seq`` get an enormous ``T`` (sum > 1e5) so they're
      dropped by the runaway-translation filter at sample-build time
      — exercises the per-seq frame-clean step.
    - ``<root>/<cat>/set_lists/set_lists_fewview_dev.json`` —
      ``{train, val, test: [[seq, frame_number, filepath], ...]}``.
      Every frame goes into ``test`` (we only consume that key).
    """
    cat_dir = root / category
    (cat_dir / "set_lists").mkdir(parents=True, exist_ok=True)

    seq_data: list[dict[str, Any]] = []
    frame_data: list[dict[str, Any]] = []
    test_subset: list[list[Any]] = []

    for s in range(sequences):
        seq_name = f"{category}_seq_{s:03d}"
        quality = 0.3 if s < bad_quality_seqs else 0.95
        seq_data.append(
            {"sequence_name": seq_name, "viewpoint_quality_score": quality}
        )
        runaway_threshold = frames_per_seq - runaway_T_per_seq
        for fnum in range(frames_per_seq):
            # Distinct, well-behaved T for keepers; sum > 1e5 for runaways.
            if fnum >= runaway_threshold:
                T = [1e6, 0.0, 0.0]
            else:
                T = [0.01 * fnum, 0.02 * fnum, 0.03 * fnum]
            filepath = f"{category}/{seq_name}/images/frame{fnum:06d}.jpg"
            frame_data.append(
                {
                    "sequence_name": seq_name,
                    "frame_number": fnum,
                    "viewpoint": {
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "T": T,
                        "focal_length": [1.0, 1.0],
                        "principal_point": [0.0, 0.0],
                        "intrinsics_format": "ndc_norm_image_bounds",
                    },
                }
            )
            test_subset.append([seq_name, fnum, filepath])

    with gzip.open(cat_dir / "sequence_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(seq_data, f)
    with gzip.open(cat_dir / "frame_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(frame_data, f)
    with (cat_dir / "set_lists" / "set_lists_fewview_dev.json").open(
        "wt", encoding="utf-8"
    ) as f:
        json.dump({"train": [], "val": [], "test": test_subset}, f)


def _write_image_stubs(root: Path, paths: list[str]) -> None:
    """Drop a 4×4 JPEG at each ``root/<rel_path>``.

    The loader will only ``read_rgb_uint8`` these in tests that iterate
    Samples; the path-equivalence test only needs the records, not the
    images. Tests that iterate use this helper to materialise the
    minimum data.
    """
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=(128, 128, 128))
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        img.save(target, format="JPEG", quality=70)


# ---------------------------------------------------------------------------
# The contract: prefetch enumeration matches the loader's records exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "categories",
    [
        ("apple",),                       # leading category, RNG-state-zero
        ("apple", "backpack", "banana"),  # first three canonical
        ("backpack", "banana"),           # skip the canonical leader
    ],
)
def test_prefetch_matches_loader_path_for_path(
    tmp_path: Path, categories: tuple[str, ...]
) -> None:
    """``compute_needed_paths`` and ``_build_records`` enumerate the same set.

    Any RNG-state drift between the two implementations would change the
    sampled ``(seq, frame)`` tuples and produce different filepaths.
    """
    # Sequences-per-cat well above sequences_per_category=10 so the
    # py_rng.sample step actually runs (and advances the RNG).
    for cat in categories:
        _write_category(
            tmp_path, cat, sequences=15, frames_per_seq=60, bad_quality_seqs=2
        )

    # Loader: build records.
    ds = Co3Dv2VGGTPoseEvalLoader(root=tmp_path, categories=categories)
    loader_paths = sorted(
        (rec["category"], rec["sequence"], fr["filepath"])
        for rec in ds._records  # noqa: SLF001 — explicit invariant pin
        for fr in rec["frames"]
    )

    # Script: enumerate needed paths.
    script_paths = sorted(_prefetch.compute_needed_paths(tmp_path, categories))

    assert loader_paths == script_paths, (
        f"prefetch / loader divergence on categories={categories}: "
        f"loader has {len(loader_paths)} paths, script has {len(script_paths)}; "
        f"first diff loader={loader_paths[:2]} script={script_paths[:2]}"
    )
    # Sanity: 10 frames × 10 seqs × len(categories) when fast_eval kicks in.
    assert len(script_paths) == 10 * 10 * len(categories)


def test_prefetch_skips_categories_missing_metadata_without_rng_advance(
    tmp_path: Path,
) -> None:
    """RNG state must not advance for categories whose metadata isn't on disk.

    This is the invariant that makes "prefetch a subset, then run the
    full-default loader on the GPU box" safe: the loader iterates all 41
    canonical categories, but `_load_category_annotations` returns
    ``None`` for those with no files → the for-loop ``continue``s before
    the ``py_rng.sample`` call → RNG state at the next staged category
    is identical to a fresh ``Random(seed)`` reaching that category in a
    subset run.
    """
    # Stage metadata for `backpack` only — apple, banana, ... will be skipped.
    _write_category(
        tmp_path, "backpack", sequences=15, frames_per_seq=60, bad_quality_seqs=2
    )

    # 1) Subset run: loader iterates only ``backpack``.
    ds_subset = Co3Dv2VGGTPoseEvalLoader(root=tmp_path, categories=("backpack",))
    subset_paths = sorted(
        (rec["category"], rec["sequence"], fr["filepath"])
        for rec in ds_subset._records  # noqa: SLF001
        for fr in rec["frames"]
    )

    # 2) Default-all-41 run: loader walks every SEEN category in canonical
    #    order; only ``backpack`` actually loads.
    ds_full = Co3Dv2VGGTPoseEvalLoader(root=tmp_path)
    full_paths = sorted(
        (rec["category"], rec["sequence"], fr["filepath"])
        for rec in ds_full._records  # noqa: SLF001
        for fr in rec["frames"]
    )

    assert subset_paths == full_paths, (
        "loader RNG advanced on missing categories — subset and full-default "
        "produced different sample sets for backpack despite identical metadata "
        "on disk. This breaks the prefetch ↔ loader contract."
    )

    # Script's compute_needed_paths must mirror this exactly.
    script_full = sorted(
        _prefetch.compute_needed_paths(tmp_path, CO3D_VGGT_SEEN_CATEGORIES)
    )
    assert script_full == full_paths


def test_prefetch_respects_runaway_translation_filter(tmp_path: Path) -> None:
    """Frames with ``T[0]+T[1]+T[2] > max_translation_sum`` must be dropped
    BEFORE sequence-length and frame sampling, identically on both sides.

    Build one cat with 15 sequences × 60 frames, where the last 15 frames
    of each sequence are runaway. After filtering, each sequence has 45
    clean frames — still above ``min_num_images=50``? No: 45 < 50, so
    the sequence is skipped entirely. That gives 0 records — exactly what
    the loader produces, exactly what the script computes.
    """
    _write_category(
        tmp_path,
        "apple",
        sequences=15,
        frames_per_seq=60,
        runaway_T_per_seq=15,  # 60 - 15 = 45 clean, below min_num_images=50
    )

    ds = Co3Dv2VGGTPoseEvalLoader(root=tmp_path, categories=("apple",))
    assert len(ds._records) == 0  # noqa: SLF001

    script_paths = _prefetch.compute_needed_paths(tmp_path, ("apple",))
    assert script_paths == []


def test_prefetch_respects_quality_gate(tmp_path: Path) -> None:
    """Sequences with ``viewpoint_quality_score <= min_quality`` are filtered out.

    Stage 12 sequences × 60 frames, 3 of them with ``score=0.3`` (default
    ``min_quality=0.5`` filters them). Remaining 9 < ``sequences_per_category=10``,
    so the ``fast_eval`` sampling path doesn't trigger and we keep all 9.
    Per-seq frame sampling still draws 10 frames each → 90 records.
    """
    _write_category(
        tmp_path, "apple", sequences=12, frames_per_seq=60, bad_quality_seqs=3
    )

    ds = Co3Dv2VGGTPoseEvalLoader(root=tmp_path, categories=("apple",))
    loader_paths = sorted(
        (rec["category"], rec["sequence"], fr["filepath"])
        for rec in ds._records  # noqa: SLF001
        for fr in rec["frames"]
    )
    assert len(loader_paths) == 90  # 9 kept seqs × 10 frames

    script_paths = sorted(_prefetch.compute_needed_paths(tmp_path, ("apple",)))
    assert script_paths == loader_paths


def test_prefetched_layout_feeds_loader_end_to_end(tmp_path: Path) -> None:
    """Round-trip: enumerate paths → write JPEG stubs at those paths →
    instantiate the loader → iterate one Sample.

    Confirms the prefetch's chosen on-disk layout (``root/<filepath>``)
    is exactly what ``Co3Dv2VGGTPoseEvalLoader._load_sample`` opens. If
    the loader ever changes its path convention, this test breaks
    loudly instead of silently FileNotFoundError-ing at GPU time.
    """
    pytest.importorskip("PIL")
    _write_category(
        tmp_path, "apple", sequences=15, frames_per_seq=60, bad_quality_seqs=2
    )

    paths = [fp for _c, _s, fp in _prefetch.compute_needed_paths(tmp_path, ("apple",))]
    _write_image_stubs(tmp_path, paths)

    ds = Co3Dv2VGGTPoseEvalLoader(root=tmp_path, categories=("apple",))
    samples = list(ds)
    assert len(samples) == 10  # 10 sequences sampled (fast_eval picks 10 from 13 kept)
    s = samples[0]
    assert s.images.shape == (10, 4, 4, 3)
    assert s.images.dtype.name == "uint8"
    assert s.intrinsics.shape == (10, 3, 3)
    assert s.extrinsics_gt.shape == (10, 4, 4)
