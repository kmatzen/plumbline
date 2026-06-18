"""Resolve the DUSt3R-Bonn off-paper gap (0.1337 vs paper 0.0808, +65%).

DUSt3R's OWN paper reports 0.0808 on Bonn, so the model handles this data — the
+65% is an eval-recipe defect, not a model limitation. Runs DUSt3R F(I,I)
per-frame over the 5 lineage sequences ONCE (caching preds), then sweeps
max_depth cap x estimator x frame-region using plumbline's own scoring fns, with
a per-sequence breakdown to see what drives the gap.

    python scripts/_dust3r_bonn_recipe_probe.py --stride 4
"""

import argparse
import os

import numpy as np

from plumbline.datasets.bonn import BonnDataset
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.models.dust3r import DUSt3RAdapter
from plumbline.runner_metrics import _resize_depth_to_gt

SEQS = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/data/bonn_cut3r/root")
    ap.add_argument("--cache", default="~/dust3r_bonn_preds")
    ap.add_argument("--stride", type=int, default=4, help="evenly subsample every Nth frame/seq")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    cache = os.path.expanduser(args.cache)
    os.makedirs(cache, exist_ok=True)

    # max_depth=70 (loosest) so we keep all GT; caps are re-applied in scoring.
    ds = BonnDataset(root=root, sequences=SEQS, per_frame=True, max_depth=70.0)
    model = DUSt3RAdapter(long_edge=512, device="cuda:0")

    # group samples by sequence, track within-seq order index for region sweep
    by_seq: dict[str, list] = {s: [] for s in SEQS}
    for samp in ds:
        seq = samp.sample_id.split("/")[0]
        by_seq.setdefault(seq, []).append(samp)

    rows = []  # (seq, region_idx, gt(480,640), pred(480,640))
    n_done = 0
    for seq in SEQS:
        samps = by_seq[seq]
        for ridx, samp in enumerate(samps):
            if ridx % args.stride != 0:
                continue
            gt = samp.depth_gt[0].astype(np.float64)  # (480,640) meters
            sid = samp.sample_id.replace("/", "_")
            pf = os.path.join(cache, f"{sid}.npy")
            if os.path.exists(pf):
                pred = np.load(pf)
            else:
                pred = model.predict(samp.images).depth
                np.save(pf, pred)
            pred = _resize_depth_to_gt(pred, gt[None])[0]
            rows.append((seq, ridx, gt, pred))
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done} frames inferred", flush=True)

    def score(subset, cap, mode):
        ar = []
        for seq, ridx, gt, pred in subset:
            valid = gt > 0
            if cap is not None:
                valid &= gt < cap
            if valid.sum() == 0:
                continue
            aligned = align_depth(pred, gt, valid, mode=mode)
            ar.append(abs_rel(aligned, gt, valid))
        return float(np.mean(ar)) if ar else float("nan"), len(ar)

    def score_per_seq_scale(subset, cap, mode):
        """One scale per sequence (video-depth style), AbsRel over all frames."""
        ar = []
        for seq in SEQS:
            ss = [r for r in subset if r[0] == seq]
            if not ss:
                continue
            preds = np.concatenate(
                [r[3][(r[2] > 0) & ((cap is None) | (r[2] < (cap or 1e9)))].ravel() for r in ss]
            )
            gts = np.concatenate(
                [r[2][(r[2] > 0) & ((cap is None) | (r[2] < (cap or 1e9)))].ravel() for r in ss]
            )
            if mode == "median_lineage":
                s = np.median(gts) / max(np.median(preds), 1e-8)
            else:
                s = np.median(gts / np.maximum(preds, 1e-8))
            ar.append(float(np.mean(np.abs(s * preds - gts) / np.maximum(gts, 1e-8))))
        return float(np.mean(ar)) if ar else float("nan")

    print(f"\nBonn (paper DUSt3R: AbsRel 0.0808)  stride={args.stride}\n")
    print(f"{'region':10} {'cap':5} {'estimator':16} {'AbsRel':>8} {'nframes':>8}")
    print("-" * 52)
    for region in ("all", "30:140"):
        sub = rows if region == "all" else [r for r in rows if 30 <= r[1] < 140]
        for cap in (70.0, 10.0, None):
            for mode in ("median", "median_lineage"):
                a, n = score(sub, cap, mode)
                caps = "none" if cap is None else f"{cap:g}"
                print(f"{region:10} {caps:5} {mode:16} {a:8.4f} {n:8d}")

    print("\nPER-SEQUENCE scale (video-depth style, one scale/seq):")
    print(f"{'region':10} {'cap':5} {'estimator':16} {'AbsRel':>8}")
    print("-" * 44)
    for region in ("all", "30:140"):
        sub = rows if region == "all" else [r for r in rows if 30 <= r[1] < 140]
        for cap in (10.0, None):
            for mode in ("median", "median_lineage"):
                a = score_per_seq_scale(sub, cap, mode)
                caps = "none" if cap is None else f"{cap:g}"
                print(f"{region:10} {caps:5} {mode:16} {a:8.4f}")

    # per-sequence breakdown under the current cell recipe (all, cap70, median)
    print("\nper-seq breakdown (all frames, cap=70, median  ==  current cell):")
    for seq in SEQS:
        sub = [r for r in rows if r[0] == seq]
        a, n = score(sub, 70.0, "median")
        a10, _ = score(sub, 10.0, "median")
        print(f"  {seq:18} cap70={a:.4f}  cap10={a10:.4f}  ({n} frames)")


if __name__ == "__main__":
    main()
