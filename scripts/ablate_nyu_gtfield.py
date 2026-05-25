import numpy as np, glob, os, re
from plumbline.datasets.nyuv2 import NYUv2Dataset, eigen_crop_mask
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel
from plumbline.runner import _resize_depth_to_gt

cache_files = glob.glob(os.path.expanduser("~/.cache/plumbline/predictions/cut3r/*/nyuv2/*.npz"))
cache = {}
for f in cache_files:
    m = re.search(r"(nyuv2_\d+)", f)
    if m: cache[m.group(1)] = f

ds = NYUv2Dataset(split="test", apply_eigen_crop=False, depth_field="raw")
# (crop, clip)
variants = {
  "A crop+clip  [plumbline]": (True, True),
  "B crop only            ": (True, False),
  "C clip only            ": (False, True),
  "D no-crop no-clip [CUT3R]":(False, False),
}
per = {k: [] for k in variants}      # per-sample abs_rel
pix = {k: [] for k in variants}      # valid-pixel counts (for weighted mean)
n = 0
for s in ds:
    sid = s.sample_id
    if sid not in cache: continue
    gt = s.depth_gt[0].astype(np.float64)
    pred = np.load(cache[sid])["depth"][0].astype(np.float64)
    pred_r = _resize_depth_to_gt(pred[None], gt[None])[0]
    base = gt > 0
    cm = eigen_crop_mask(gt.shape).astype(bool)
    for name, (crop, clip) in variants.items():
        valid = (base & cm) if crop else base
        if valid.sum() == 0: continue
        aligned = align_depth(pred_r, gt, valid, mode="median")
        if clip: aligned = np.clip(aligned, 1e-3, 10.0)
        per[name].append(abs_rel(aligned, gt, valid))
        pix[name].append(int(valid.sum()))
    n += 1
print(f"samples scored: {n}  (cache hits {len(cache)})")
print(f"eigen crop keeps {cm.mean()*100:.1f}% of pixels")
for k in variants:
    a = np.array(per[k]); w = np.array(pix[k], float)
    print(f"{k}  mean={a.mean():.4f}  pixwt={np.average(a,weights=w):.4f}  (n={len(a)})")
