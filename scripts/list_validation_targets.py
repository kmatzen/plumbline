#!/usr/bin/env python3
"""List reproduction YAMLs that are eligible for paper-match validation.

Eligible = ``paper_reference.value`` is non-null AND
``paper_reference.source_confidence`` is ``verified_pdf``. Used by the
autonomous-agent GPU runbook (``docs/AGENT_GPU_RUNBOOK.md``) to build
the validation queue.

Usage:
    scripts/list_validation_targets.py                 # JSON to stdout
    scripts/list_validation_targets.py --format md     # markdown table

The output is deliberately stable across runs: sorted by
(estimated-runtime-bucket, name) so the agent's queue is deterministic.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

# Coarse runtime buckets per (model, dataset) on a 3090/4090. Used to
# order the queue cheapest-first so the agent gets early signal. Add to
# this map when new model/dataset combos land. Default if unknown:
# "medium" (treated as 5 min for ordering).
_RUNTIME_BUCKETS: dict[tuple[str, str], str] = {
    # NYU mono-depth (~1-3 min):
    ("depth-anything-v2", "nyuv2"): "fast",
    ("depth-anything-3", "nyuv2"): "fast",
    ("moge", "nyuv2"): "fast",
    # NYU diffusion / metric (~5-10 min):
    ("marigold", "nyuv2"): "medium",
    ("depth-pro", "nyuv2"): "medium",
    ("metric3d-v2", "nyuv2"): "medium",
    # KITTI mono-depth (~2-5 min):
    ("depth-anything-v2", "kitti"): "fast",
    ("moge", "kitti"): "medium",
    ("metric3d-v2", "kitti"): "medium",
    ("marigold", "kitti"): "medium",
    ("depth-pro", "kitti"): "medium",
    # DIODE (~3-5 min, single forward over a few-hundred samples):
    ("moge", "diode"): "medium",
    ("depth-anything-v2", "diode"): "medium",
    # ETH3D multi-view + scene merge (~30-60 min on 3 scenes):
    ("vggt", "eth3d"): "slow",
    ("depth-anything-3", "eth3d"): "slow",
    # DTU multi-view (the v0.1 gate; ~30 min on 22 test scans):
    ("vggt", "dtu"): "slow",
    # Metric3D-Giant variants are larger and slower:
    # (handled by an override below since the variant is in kwargs, not name)
}

_BUCKET_ORDER = {"fast": 0, "medium": 1, "slow": 2}


def _bucket(cfg: dict) -> str:
    model = cfg["model"]["name"]
    dataset = (cfg.get("dataset") or {}).get("name")
    if dataset is None:
        # Resolve via protocol if dataset name isn't inline.
        protocol_name = cfg.get("protocol")
        if protocol_name:
            try:
                pp = pathlib.Path(__file__).resolve().parent.parent / "protocols" / f"{protocol_name}.yaml"
                with pp.open() as f:
                    pcfg = yaml.safe_load(f)
                dataset = (pcfg.get("fixed", {}).get("dataset") or {}).get("name")
            except Exception:
                pass
    bucket = _RUNTIME_BUCKETS.get((model, dataset or "?"), "medium")
    # Variant-specific overrides: Metric3D Giant is meaningfully slower.
    variant = (cfg["model"].get("kwargs") or {}).get("variant", "")
    if model == "metric3d-v2" and "giant" in variant.lower():
        bucket = "slow"
    return bucket


def _resolved_dataset(cfg: dict) -> str:
    """Return the dataset name, following `protocol:` if needed."""
    ds = (cfg.get("dataset") or {}).get("name")
    if ds:
        return ds
    protocol_name = cfg.get("protocol")
    if not protocol_name:
        return "?"
    pp = (
        pathlib.Path(__file__).resolve().parent.parent
        / "protocols"
        / f"{protocol_name}.yaml"
    )
    try:
        with pp.open() as f:
            pcfg = yaml.safe_load(f)
        return (pcfg.get("fixed", {}).get("dataset") or {}).get("name", "?")
    except Exception:
        return "?"


def list_targets(repro_dir: pathlib.Path) -> list[dict]:
    targets = []
    for path in sorted(repro_dir.glob("*.yaml")):
        with path.open() as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            continue
        pr = cfg.get("paper_reference") or {}
        if pr.get("source_confidence") != "verified_pdf":
            continue
        if pr.get("value") is None:
            continue
        targets.append({
            "name": cfg["name"],
            "model": cfg["model"]["name"],
            "model_kwargs": cfg["model"].get("kwargs") or {},
            "dataset": _resolved_dataset(cfg),
            "metric": pr["primary_metric"],
            "value": pr["value"],
            "tolerance_relative": pr.get("tolerance_relative", 0.05),
            "citation": pr.get("citation", ""),
            "runtime_bucket": _bucket(cfg),
        })
    targets.sort(key=lambda t: (_BUCKET_ORDER[t["runtime_bucket"]], t["name"]))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("json", "md"), default="json")
    parser.add_argument("--repro-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent / "reproductions")
    args = parser.parse_args()

    targets = list_targets(args.repro_dir)
    if args.format == "json":
        json.dump(targets, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    # Markdown.
    print(f"# Validation queue ({len(targets)} targets)\n")
    print("| Bucket | Reproduction | Model | Dataset | Metric | Paper value | Tol |")
    print("|---|---|---|---|---|---|---|")
    for t in targets:
        print(
            f"| `{t['runtime_bucket']}` | `{t['name']}` | {t['model']} | "
            f"{t['dataset']} | {t['metric']} | {t['value']} | ±{t['tolerance_relative']:.0%} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
