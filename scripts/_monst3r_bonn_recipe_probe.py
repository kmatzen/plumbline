"""Test whether MonST3R's notebook recipe reproduces MonST3R-Bonn Table 3 (0.076).

plumbline's monst3r-bonn scores per-frame median (faithful to §4.2 text) → 0.0654
(-13.9%). MonST3R's eval_depth.py actually scores, PER SEQUENCE: stack all rgb_110
frames → ONE scale+shift LAD2 (absolute_value_scaling2) → AbsRel over pooled
pixels → valid-pixel-weighted average across the 5 seqs (align_with_lad2=True).

Runs plumbline's MonST3R F(I,I) per-frame on rgb_110 ONCE, then scores BOTH ways —
the LAD2 path via MonST3R's OWN depth_evaluation so it's exact.

    MONST3R_ROOT=~/deps/monst3r python scripts/_monst3r_bonn_recipe_probe.py
"""

import os
import sys

import numpy as np

from plumbline.datasets.bonn import BonnDataset
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel, delta_threshold
from plumbline.models.monst3r import MonST3RAdapter
from plumbline.runner_metrics import _resize_depth_to_gt

SEQS = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]


def main() -> None:
    root = os.path.expanduser("~/data/bonn_cut3r/root")
    cache = os.path.expanduser("~/monst3r_bonn110_preds")
    os.makedirs(cache, exist_ok=True)

    model = MonST3RAdapter(long_edge=512, device="cuda:0")
    # MonST3R's own scorer (exact LAD2 + metric)
    sys.path.insert(0, os.path.expanduser(os.environ["MONST3R_ROOT"]))
    from dust3r.depth_eval import depth_evaluation

    ds = BonnDataset(root=root, sequences=SEQS, per_frame=True, prepared_110=True, max_depth=70.0)
    by_seq: dict[str, list] = {s: [] for s in SEQS}
    n = 0
    for samp in ds:
        seq = samp.sample_id.split("/")[0]
        gt = samp.depth_gt[0].astype(np.float64)  # (H,W) meters
        cf = os.path.join(cache, f"{samp.sample_id.replace('/', '_')}.npy")
        if os.path.exists(cf):
            pred = np.load(cf)
        else:
            pred = model.predict(samp.images).depth
            np.save(cf, pred)
        pred = _resize_depth_to_gt(pred, gt[None])[0]
        by_seq[seq].append((gt, pred))
        n += 1
        if n % 50 == 0:
            print(f"  {n}/550 inferred", flush=True)

    # ---- (a) plumbline current: per-frame median, equal-weight mean ----
    pf_ar, pf_d1 = [], []
    for seq in SEQS:
        for gt, pred in by_seq[seq]:
            valid = gt > 0
            aligned = align_depth(pred, gt, valid, mode="median")
            pf_ar.append(abs_rel(aligned, gt, valid))
            pf_d1.append(delta_threshold(aligned, gt, valid, threshold=1.25))

    # ---- (b) MonST3R notebook: per-seq LAD2 scale+shift, pixel-weighted mean ----
    seq_ar, seq_w, seq_d1 = [], [], []
    for seq in SEQS:
        gts = np.stack([g for g, _ in by_seq[seq]], axis=0)  # (N,H,W)
        prs = np.stack([p for _, p in by_seq[seq]], axis=0)
        res, *_ = depth_evaluation(prs, gts, max_depth=70, align_with_lad2=True, use_gpu=True)
        seq_ar.append(res["Abs Rel"])
        seq_d1.append(res["δ < 1.25"] if "δ < 1.25" in res else res.get("threshold_1", np.nan))
        seq_w.append(res["valid_pixels"])
        print(f"  {seq:18} lad2 AbsRel={res['Abs Rel']:.4f}  (px={res['valid_pixels']})", flush=True)

    pw = np.average(seq_ar, weights=seq_w)

    # ---- (c) MonST3R eval_depth.py MEDIAN branch: per-seq ratio-of-medians, pixel-weighted ----
    med_ar, med_w = [], []
    for seq in SEQS:
        gts = np.stack([g for g, _ in by_seq[seq]], axis=0)
        prs = np.stack([p for _, p in by_seq[seq]], axis=0)
        res, *_ = depth_evaluation(prs, gts, max_depth=70, align_with_lad2=False, use_gpu=True)
        med_ar.append(res["Abs Rel"])
        med_w.append(res["valid_pixels"])
    med_pw = np.average(med_ar, weights=med_w)

    print("\n=== MonST3R-Bonn (paper Table 3: AbsRel 0.076 / δ₁ 0.939) ===")
    print(f"(a) per-frame median, equal-weight     AbsRel = {np.mean(pf_ar):.4f}  δ₁ = {np.mean(pf_d1):.4f}   (plumbline current)")
    print(f"(b) per-seq LAD2 scale+shift, px-wt    AbsRel = {pw:.4f}                  (Table 2 video recipe)")
    print(f"(c) per-seq median (ratio-of-med), px-wt AbsRel = {med_pw:.4f}                (eval_depth.py median branch)")


if __name__ == "__main__":
    main()
