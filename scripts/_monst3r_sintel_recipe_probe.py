"""Test whether the pixel-weighted per-seq recipe reproduces MonST3R-Sintel (0.345).

plumbline's monst3r-sintel scores per-frame median, equal-weight across frames →
0.3726 (+8%, WORSE than paper). Hypothesis: MonST3R's eval pools each sequence
and takes a valid-pixel-WEIGHTED mean across the 14 scenes, which dilutes the
`temple_*` outlier scenes that dominate plumbline's equal-weight mean.

Runs plumbline's MonST3R F(I,I) per-frame once, then scores per-frame-median
(equal-weight) vs per-seq median/LAD2 (pixel-weighted, max_depth=70,
post_clip_max=70) via MonST3R's own depth_evaluation.

    MONST3R_ROOT=~/deps/monst3r SINTEL_ROOT=~/data/sintel python scripts/_monst3r_sintel_recipe_probe.py
"""

import os
import sys

import numpy as np

from plumbline.datasets.sintel import SintelDataset
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.models.monst3r import MonST3RAdapter
from plumbline.runner_metrics import _resize_depth_to_gt

SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4",
    "market_2", "market_5", "market_6", "shaman_3", "sleeping_1",
    "sleeping_2", "temple_2", "temple_3",
]


def main() -> None:
    cache = os.path.expanduser("~/monst3r_sintel_preds")
    os.makedirs(cache, exist_ok=True)
    model = MonST3RAdapter(long_edge=512, device="cuda:0")
    sys.path.insert(0, os.path.expanduser(os.environ["MONST3R_ROOT"]))
    from dust3r.depth_eval import depth_evaluation

    ds = SintelDataset(
        split="training", pass_name="final", views_per_sample=1,
        max_depth=70, scenes=SCENES,
    )
    by_scene: dict[str, list] = {s: [] for s in SCENES}
    n = 0
    for samp in ds:
        scene = samp.metadata["scene"]
        gt = samp.depth_gt[0].astype(np.float64)
        valid = samp.depth_valid[0] if samp.depth_valid is not None else ((gt > 0) & (gt < 70))
        cf = os.path.join(cache, f"{samp.sample_id.replace('/', '_')}.npy")
        if os.path.exists(cf):
            pred = np.load(cf)
        else:
            pred = model.predict(samp.images).depth
            np.save(cf, pred)
        pred = _resize_depth_to_gt(pred, gt[None])[0]
        by_scene[scene].append((gt, pred, valid))
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames inferred", flush=True)

    # (a) plumbline: per-frame median, post-clip [1e-3,70], equal-weight across frames
    pf = []
    for s in SCENES:
        for gt, pred, valid in by_scene[s]:
            if valid.sum() == 0:
                continue
            aligned = np.clip(align_depth(pred, gt, valid, mode="median"), 1e-3, 70.0)
            pf.append(abs_rel(aligned, gt, valid))

    # (b/c) per-seq pooled, pixel-weighted across scenes (MonST3R eval recipe)
    def per_seq(lad2):
        ar, w, per = [], [], {}
        for s in SCENES:
            gts = np.stack([g for g, _, _ in by_scene[s]], axis=0)
            prs = np.stack([p for _, p, _ in by_scene[s]], axis=0)
            res, *_ = depth_evaluation(
                prs, gts, max_depth=70, post_clip_max=70,
                align_with_lad2=lad2, use_gpu=True,
            )
            ar.append(res["Abs Rel"]); w.append(res["valid_pixels"]); per[s] = res["Abs Rel"]
        return float(np.average(ar, weights=w)), per

    lad2_pw, lad2_per = per_seq(True)
    med_pw, med_per = per_seq(False)

    print("\n=== MonST3R-Sintel (paper Table 3: AbsRel 0.345 / δ₁ 0.565) ===")
    print(f"(a) per-frame median, equal-weight       AbsRel = {np.mean(pf):.4f}   (plumbline current)")
    print(f"(b) per-seq LAD2 scale+shift, pixel-wt    AbsRel = {lad2_pw:.4f}")
    print(f"(c) per-seq median, pixel-weighted        AbsRel = {med_pw:.4f}")
    print("\nper-scene (median path) — note the temple_* outliers:")
    for s in SCENES:
        print(f"  {s:12} {med_per[s]:.4f}")


if __name__ == "__main__":
    main()
