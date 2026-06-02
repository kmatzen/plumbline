"""Depth Pro / SUN-RGBD focal probe.

For a balanced sample of 'good' and 'bad' frames (by the saved un-aligned
delta_1), run Depth Pro and record its ESTIMATED focal length, then compare
un-aligned vs per-frame-median-scale-aligned delta_1. Tests the hypothesis
that the bimodal SUN-RGBD result is a focal/metric-scale failure on a frame
subset (structure correct, scale wrong), not a GT-decode/pairing bug.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1])          # dir with rgb/ and depth/
FRAMES = Path(sys.argv[2])        # frame-list txt: "num d1 absrel silog"
OUT = Path(sys.argv[3])

from plumbline.models.depth_pro import DepthProAdapter

def read_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
def read_gt(p):  return np.asarray(Image.open(p), dtype=np.float32) / 10000.0

def metrics(pred, gt, valid):
    p = pred[valid]; g = gt[valid]
    ok = (p > 0) & (g > 0)
    p, g = p[ok], g[ok]
    if p.size == 0: return None
    # un-aligned
    ratio = np.maximum(p / g, g / p)
    d1 = float((ratio < 1.25).mean())
    absrel = float((np.abs(p - g) / g).mean())
    # per-frame median scale alignment (pred *= median(g/p))
    s = float(np.median(g / p))
    pa = p * s
    ratio_a = np.maximum(pa / g, g / pa)
    d1a = float((ratio_a < 1.25).mean())
    return d1, absrel, s, d1a

def main():
    rows = []
    for line in FRAMES.read_text().splitlines():
        num = int(line.split()[0]); saved_d1 = float(line.split()[1])
        rows.append((num, saved_d1))
    model = DepthProAdapter()
    out = []
    for k, (num, saved_d1) in enumerate(rows):
        rgb_p = ROOT / "rgb" / f"img-{num:06d}.jpg"
        gt_p = ROOT / "depth" / f"{num}.png"
        if not rgb_p.exists() or not gt_p.exists():
            print(f"  MISSING {num}", flush=True); continue
        rgb = read_rgb(rgb_p)
        gt = read_gt(gt_p)
        valid = np.isfinite(gt) & (gt > 0) & (gt < 80.0)
        pred = model.predict(rgb[None])
        depth = pred.depth[0]
        fx_est = float(pred.intrinsics[0][0, 0])
        m = metrics(depth, gt, valid)
        if m is None: continue
        d1, absrel, s, d1a = m
        h, w = gt.shape
        out.append(dict(num=num, saved_d1=saved_d1, d1=d1, absrel=absrel,
                        scale=s, d1_aligned=d1a, fx_est=fx_est, w=w, h=h,
                        fx_implied=fx_est * s))
        if k % 10 == 0:
            print(f"[{k+1}/{len(rows)}] frame {num}: d1={d1:.3f} d1_aln={d1a:.3f} "
                  f"fx_est={fx_est:.0f} scale={s:.3f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} rows -> {OUT}")

if __name__ == "__main__":
    main()
