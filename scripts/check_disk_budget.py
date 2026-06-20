#!/usr/bin/env python3
"""Estimate the disk footprint of a planned GPU session.

Given a list of reproduction YAMLs, sums up the minimum-viable disk
each dataset contributes (from the per-dataset footprint table in
``docs/dataset_footprints.md``), adds rough estimates for model
weights and the prediction cache, and compares the total to a
target budget.

Usage
-----

    scripts/check_disk_budget.py reproductions/*_nyuv2.yaml \\
                                 reproductions/*_kitti.yaml \\
                                 --budget 60GB

    # Or just the YAMLs you plan to run tonight:
    scripts/check_disk_budget.py reproductions/vggt_paper_dtu_mvs.yaml \\
                                 reproductions/da3_eth3d_*.yaml \\
                                 --budget 120GB

Exits non-zero if the total exceeds ``budget - 20 %`` (the 20 %
headroom is for HF-cache surprises, temp unzip peaks, log files).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict

import yaml

# Dataset minimum-viable footprints in gigabytes. Sourced from
# ``docs/dataset_footprints.md``; keep in sync when that table changes.
DATASET_FOOTPRINTS_GB: dict[str, float] = {
    "nyuv2": 3.0,
    "kitti": 6.0,   # 652-frame prune + annotated-depth (~5 GB) + calib
    "diode": 2.8,
    "eth3d": 8.0,   # 3-scene subset covering VGGT Table 3
    "dtu": 7.0,     # SampleSet.zip contents, 22 MVSNet test scans
    "gso": 2.0,
    "co3dv2": 10.0, # pose subset; full set is ~1.5 TB (do not load)
    "sintel": 6.0,  # public bundle only (depth/cam gated)
    "7scenes": 12.0,  # all 7 RGB-D sequences (unzipped)
    "ibims1": 0.2,    # MoGe bundle, 100 scenes (unzipped)
}

MODEL_WEIGHTS_CACHE_GB = 15.0        # VGGT + DA-V2 variants + Metric3D-g + MoGe + DA3 + DepthPro + Marigold
PREDICTION_CACHE_GB = 5.0            # growing; estimate a sustainable number
HEADROOM_FRACTION = 0.20             # 20 % buffer on top of the summed estimate


def parse_size_arg(s: str) -> float:
    """Convert "60GB" / "100G" / "500MB" / raw GB floats to GB."""
    s = s.strip().upper()
    m = re.match(r"^([0-9.]+)\s*(GB|G|MB|M|TB|T)?$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse size: {s!r}")
    num, unit = float(m.group(1)), (m.group(2) or "GB")
    if unit in ("GB", "G"):
        return num
    if unit in ("MB", "M"):
        return num / 1024.0
    if unit in ("TB", "T"):
        return num * 1024.0
    raise argparse.ArgumentTypeError(f"unknown unit: {unit!r}")


def resolve_dataset(cfg: dict) -> str | None:
    """Return the dataset name a reproduction targets, after protocol merge."""
    # Try the direct field first; protocol merge happens lazily at runtime,
    # so we mirror it here without importing plumbline (keeps the script
    # runnable on boxes without the package installed).
    ds = cfg.get("dataset", {})
    if isinstance(ds, dict) and ds.get("name"):
        return ds["name"]
    protocol_name = cfg.get("protocol")
    if not protocol_name:
        return None
    protocol_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "protocols"
        / f"{protocol_name}.yaml"
    )
    if not protocol_path.exists():
        return None
    with protocol_path.open() as f:
        protocol = yaml.safe_load(f)
    return protocol.get("fixed", {}).get("dataset", {}).get("name")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("yamls", nargs="+", type=pathlib.Path,
                        help="Reproduction YAML files to include in the budget.")
    parser.add_argument("--budget", type=parse_size_arg, default=None,
                        help="Target box disk, e.g. '60GB'. If set, script exits non-zero when over budget.")
    args = parser.parse_args()

    per_dataset: dict[str, list[str]] = defaultdict(list)
    unknown_yamls: list[pathlib.Path] = []
    for y in args.yamls:
        try:
            with y.open() as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"!! {y}: cannot parse YAML ({e}); skipping", file=sys.stderr)
            unknown_yamls.append(y)
            continue
        if not isinstance(cfg, dict):
            unknown_yamls.append(y)
            continue
        ds = resolve_dataset(cfg)
        if ds is None:
            unknown_yamls.append(y)
            continue
        per_dataset[ds].append(cfg.get("name", y.stem))

    # Dataset footprints — take MAX (not sum) since reproductions share data.
    dataset_total = 0.0
    print("Dataset footprints (minimum viable):")
    for ds in sorted(per_dataset):
        size = DATASET_FOOTPRINTS_GB.get(ds)
        if size is None:
            print(f"  {ds:15s}  ??  (not in DATASET_FOOTPRINTS_GB — update scripts/check_disk_budget.py)")
            continue
        dataset_total += size
        n = len(per_dataset[ds])
        print(f"  {ds:15s}  {size:6.1f} GB   ({n} reproduction(s))")

    total = dataset_total + MODEL_WEIGHTS_CACHE_GB + PREDICTION_CACHE_GB
    required = total * (1.0 + HEADROOM_FRACTION)

    print()
    print(f"  Datasets subtotal:    {dataset_total:6.1f} GB")
    print(f"  Model weights cache:  {MODEL_WEIGHTS_CACHE_GB:6.1f} GB")
    print(f"  Prediction cache:     {PREDICTION_CACHE_GB:6.1f} GB")
    print(f"  Subtotal:             {total:6.1f} GB")
    print(f"  + 20% headroom:       {required:6.1f} GB")
    print()

    if unknown_yamls:
        print(f"warning: {len(unknown_yamls)} YAML(s) had no resolvable dataset — skipped:", file=sys.stderr)
        for u in unknown_yamls:
            print(f"  {u}", file=sys.stderr)
        print(file=sys.stderr)

    if args.budget is not None:
        fits = required <= args.budget
        status = "OK" if fits else "OVER BUDGET"
        margin_gb = args.budget - required
        pct = 100.0 * margin_gb / args.budget
        print(f"Budget: {args.budget:.1f} GB   Required: {required:.1f} GB   "
              f"Margin: {margin_gb:+.1f} GB ({pct:+.1f}%)   {status}")
        return 0 if fits else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
