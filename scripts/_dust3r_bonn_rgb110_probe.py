"""Score DUSt3R single-frame Bonn on MonST3R's EXACT eval frames (rgb_110).

MonST3R Table 3's DUSt3R baseline (Bonn AbsRel 0.141) is scored on the prepared
110-frame subsets: sorted(rgb_110)[i] paired with sorted(depth_110)[i], per-frame
median scaling (MonST3R §4.2). This reproduces that exact setup so the dust3r-bonn
cell compares like-for-like.

    python scripts/_dust3r_bonn_rgb110_probe.py --root ~/data/bonn_cut3r/root
"""

import argparse
import glob
import os

import numpy as np

from plumbline.datasets._common import read_rgb_uint8
from plumbline.datasets.bonn import _load_bonn_depth
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel, delta_threshold
from plumbline.models.dust3r import DUSt3RAdapter
from plumbline.runner_metrics import _resize_depth_to_gt

SEQS = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/data/bonn_cut3r/root")
    ap.add_argument("--cache", default="~/dust3r_bonn110_preds")
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    cache = os.path.expanduser(args.cache)
    os.makedirs(cache, exist_ok=True)
    model = DUSt3RAdapter(long_edge=512, device="cuda:0")

    per_seq = {}
    all_ar, all_d1 = [], []
    for seq in SEQS:
        sd = os.path.join(root, f"rgbd_bonn_{seq}")
        rgbs = sorted(glob.glob(os.path.join(sd, "rgb_110", "*.png")))
        deps = sorted(glob.glob(os.path.join(sd, "depth_110", "*.png")))
        assert len(rgbs) == len(deps) == 110, f"{seq}: {len(rgbs)}/{len(deps)}"
        seq_ar = []
        for i, (rp, dp) in enumerate(zip(rgbs, deps)):
            gt, _ = _load_bonn_depth(dp, max_depth=70.0)  # (H,W) meters
            gt = gt.astype(np.float64)
            cf = os.path.join(cache, f"{seq}_{i:04d}.npy")
            if os.path.exists(cf):
                pred = np.load(cf)
            else:
                img = read_rgb_uint8(rp)[None]
                pred = model.predict(img).depth
                np.save(cf, pred)
            pred = _resize_depth_to_gt(pred, gt[None])[0]
            valid = gt > 0
            if valid.sum() == 0:
                continue
            aligned = align_depth(pred, gt, valid, mode="median")
            a = abs_rel(aligned, gt, valid)
            seq_ar.append(a)
            all_ar.append(a)
            all_d1.append(delta_threshold(aligned, gt, valid, threshold=1.25))
        per_seq[seq] = float(np.mean(seq_ar))
        print(f"  {seq:18} {per_seq[seq]:.4f}  ({len(seq_ar)} frames)", flush=True)

    print("\nrgb_110 single-frame, per-frame median (== MonST3R Table 3 setup)")
    print(f"OVERALL AbsRel = {np.mean(all_ar):.4f}   δ₁ = {np.mean(all_d1):.4f}")
    print("  (MonST3R Table 3 DUSt3R baseline: Bonn 0.141 / δ₁ 0.825)")


if __name__ == "__main__":
    main()
