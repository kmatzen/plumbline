"""Single-record diff: is plumbline's DUSt3R F(I,I) faithful to dust3r-canonical?

Throwaway diagnostic for D28. Picks one NYU sample and compares, at the model's
native output resolution (no GT resize yet):

  A) REFERENCE  — dust3r's *own* `load_images([png,png], size=512)` →
                  make_pairs(symmetrize) → inference → pred1.pts3d.mean(0)[...,z]
  B) PLUMBLINE  — the cached prediction depth for the same sample_id, which was
                  produced via plumbline's `_images_to_dust3r_dicts` prep.

Also diffs the two input tensors directly (plumbline prep vs dust3r load_images),
since the only place a wrapper bug could hide (given KITTI matches) is image prep
or output extraction. If A≈B, plumbline is faithful → the NYU paper gap is purely
GT-processing/recipe. If A≠B, we've found a prep/extraction discrepancy.

Run on the GPU box:  uv run --no-sync python scripts/_dust3r_nyu_singlediff.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from PIL import Image

DEVICE = "cuda:0"
CKPT = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
CFG_HASH = "16431eabef5f093b"
SAMPLE_IDX = 0  # first Eigen-test sample

# --- dust3r on path ---
root = os.environ.get("DUST3R_ROOT", "/root/deps/mast3r/dust3r")
if root not in sys.path:
    sys.path.insert(0, root)

from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402

from plumbline.datasets.nyuv2 import NYUv2Dataset  # noqa: E402
from plumbline.cache import PredictionCache  # noqa: E402
from plumbline.models.mast3r import _images_to_dust3r_dicts  # noqa: E402
from plumbline.metrics.alignment import align_scale_median  # noqa: E402


def absrel(pred, gt, valid):
    s = align_scale_median(pred, gt, valid)
    p = pred * s
    m = valid & (gt > 0)
    return float(np.mean(np.abs(p[m] - gt[m]) / gt[m]))


def main() -> None:
    # --- one NYU sample (filled GT, no crop = lineage) ---
    ds = NYUv2Dataset(split="test", depth_field="filled", apply_eigen_crop=False)
    sample = next(s for i, s in enumerate(ds) if i == SAMPLE_IDX)
    img_u8 = sample.images[0]  # (H,W,3) uint8
    gt = sample.depth_gt[0].astype(np.float64)  # (H,W)
    valid = sample.depth_valid[0] if sample.depth_valid is not None else (gt > 0)
    sid = sample.sample_id
    print(f"sample_id={sid}  img={img_u8.shape}  gt={gt.shape}")

    png = "/tmp/_nyu_singlediff.png"
    Image.fromarray(img_u8).save(png)

    model = AsymmetricCroCo3DStereo.from_pretrained(CKPT).to(DEVICE).eval()

    # ---- (B) plumbline prep tensor ----
    pl_dicts = _images_to_dust3r_dicts(img_u8[None], long_edge=512)
    pl_img = pl_dicts[0]["img"]  # (1,3,h,w)

    # ---- (A) dust3r-canonical prep tensor ----
    ds_imgs = load_images([png, png], size=512, verbose=False)
    ds_img = ds_imgs[0]["img"]

    print("\n=== INPUT-TENSOR DIFF (plumbline prep vs dust3r load_images) ===")
    print(f"  plumbline img shape={tuple(pl_img.shape)}  dust3r img shape={tuple(ds_img.shape)}")
    if pl_img.shape == ds_img.shape:
        d = (pl_img - ds_img).abs()
        print(f"  max|Δ|={d.max().item():.5f}  mean|Δ|={d.mean().item():.6f}")
    else:
        print("  SHAPE MISMATCH → prep differs structurally")

    # ---- (A) reference F(I,I) via dust3r-canonical prep ----
    pairs = make_pairs(ds_imgs, symmetrize=True, prefilter=None)
    out = inference(pairs, model, DEVICE, batch_size=1, verbose=False)
    pts_ref = out["pred1"]["pts3d"].mean(dim=0).detach().cpu().numpy()
    depth_ref = pts_ref[..., 2]

    # ---- (B) plumbline helper, called directly on the same image ----
    from plumbline.models.dust3r import _dust3r_single_frame_eval
    depth_pl_arr, _, _ = _dust3r_single_frame_eval(
        model, img_u8[None], device=DEVICE, long_edge=512
    )
    depth_pl = np.asarray(depth_pl_arr[0], dtype=np.float64)

    print("\n=== DEPTH-MAP DIFF (reference F(I,I) vs plumbline cached) ===")
    print(f"  ref shape={depth_ref.shape}  plumbline shape={depth_pl.shape}")
    if depth_ref.shape == depth_pl.shape:
        # both unscaled; align each independently then compare relative
        rr = depth_ref / np.median(depth_ref[depth_ref > 0])
        pp = depth_pl / np.median(depth_pl[depth_pl > 0])
        d = np.abs(rr - pp)
        print(f"  after per-map median-norm:  max|Δ|={d.max():.5f}  mean|Δ|={d.mean():.6f}")
        print(f"  correlation={np.corrcoef(depth_ref.ravel(), depth_pl.ravel())[0,1]:.6f}")

    # ---- AbsRel of each vs GT (resize pred → GT, median align, filled, no crop) ----
    import cv2
    def score(dep):
        dr = cv2.resize(dep.astype(np.float32), (gt.shape[1], gt.shape[0]),
                        interpolation=cv2.INTER_CUBIC).astype(np.float64)
        return absrel(dr, gt, valid)

    print("\n=== AbsRel vs GT (single sample, filled+no-crop, median align) ===")
    print(f"  reference F(I,I):  {score(depth_ref):.4f}")
    print(f"  plumbline cached:  {score(depth_pl):.4f}")


if __name__ == "__main__":
    main()
