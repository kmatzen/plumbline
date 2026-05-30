#!/usr/bin/env python3
"""Compare native DIODE GT/mask vs MoGe-bundle on matched val paths (D29, no GPU).

Usage::

    source scripts/pod-localssd-env.sh
    uv run python scripts/probe-diode-d29-native-vs-bundle.py --max-pairs 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from plumbline.datasets.diode import load_diode_depth_m, load_diode_depth_mask, load_moge_depth_png


def _native_key(sample_path: str) -> tuple[str, str, str, str]:
    # val/outdoor/scene/scan/base -> domain, scene, scan, base
    parts = sample_path.split("/")
    return parts[1], parts[2], parts[3], parts[4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diode-root", default=os.environ.get("DIODE_ROOT", ""))
    parser.add_argument("--moge-root", default=os.environ.get("DIODE_MOGE_ROOT", ""))
    parser.add_argument("--max-pairs", type=int, default=30)
    args = parser.parse_args()
    native_root = Path(args.diode_root)
    moge_root = Path(args.moge_root)
    index = moge_root / "DIODE" / ".index.txt"
    if not index.is_file():
        raise SystemExit(f"Missing {index}")

    lines = [ln.strip() for ln in index.read_text().splitlines() if ln.strip()][: args.max_pairs]
    valid_frac_native: list[float] = []
    valid_frac_moge: list[float] = []
    depth_mae: list[float] = []

    for ln in lines:
        domain, scene, scan, base = _native_key(ln)
        scan_dir = native_root / "val" / domain / scene / scan
        rgb = scan_dir / f"{base}.png"
        depth_npy = scan_dir / f"{base}_depth.npy"
        mask_npy = scan_dir / f"{base}_depth_mask.npy"
        bundle = moge_root / "DIODE" / ln
        if not all(p.exists() for p in (rgb, depth_npy, mask_npy, bundle / "image.jpg", bundle / "depth.png")):
            continue
        d_native = load_diode_depth_m(depth_npy)
        m_native = load_diode_depth_mask(mask_npy)
        d_moge, m_moge = load_moge_depth_png(bundle / "depth.png")
        if d_native.shape != d_moge.shape:
            continue
        valid_frac_native.append(float(m_native.mean()))
        valid_frac_moge.append(float(m_moge.mean()))
        both = m_native & m_moge & np.isfinite(d_native) & np.isfinite(d_moge) & (d_native > 0) & (d_moge > 0)
        if both.any():
            depth_mae.append(float(np.mean(np.abs(d_native[both] - d_moge[both]))))

    print(f"pairs={len(depth_mae)}")
    print(f"valid_frac_native mean={np.mean(valid_frac_native):.4f}")
    print(f"valid_frac_moge   mean={np.mean(valid_frac_moge):.4f}")
    print(f"depth MAE (m) on overlap={np.mean(depth_mae):.4f}")


if __name__ == "__main__":
    main()
