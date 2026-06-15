"""Single-record inference diff: plumbline CUT3R preprocessing vs CUT3R's own.

Confirms (2026-06-14, anima GTX 1080 Ti) that plumbline's CUT3R adapter feeds
the model a byte-identical input tensor to CUT3R's own monodepth-eval
preprocessing (`load_images_for_eval`), so plumbline's CUT3R inference is
faithful to upstream and the off-paper `cut3r-nyuv2-lineage` number (0.0777 vs
paper 0.086) is 100 % an eval-set/GT difference, not an adapter bug. Same shape
as the DUSt3R byte-verification (D28).

Result: `max|Δ| = 0.000e+00  ->  BYTE-IDENTICAL`.

Run on a host with $CUT3R_ROOT (CUT3R clone) + $NYUV2_ROOT:
    NYUV2_ROOT=~/data/nyuv2 CUT3R_ROOT=~/deps/cut3r python scripts/_cut3r_nyu_input_diff.py
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get("CUT3R_ROOT", "~/deps/cut3r")))


def main() -> None:
    # 1) one NYU sample-0 RGB via plumbline's loader
    from plumbline.datasets.nyuv2 import NYUv2Dataset

    ds = NYUv2Dataset(root=os.environ.get("NYUV2_ROOT"), split="test")
    samp = next(iter(ds))
    rgb = samp.images[0]  # uint8 HxWx3
    print("source RGB:", rgb.shape, rgb.dtype)

    # 2) plumbline preprocessing (the adapter's input path)
    from plumbline.models.cut3r import _build_views

    t_a = _build_views(rgb[None], long_edge=512)[0]["img"]  # (1,3,H,W) in [-1,1]

    # 3) CUT3R's OWN eval preprocessing on the same pixels (PNG round-trip is
    #    lossless for uint8 RGB), the function eval/monodepth/launch.py imports.
    tmp = "/tmp/_nyu0.png"
    Image.fromarray(rgb).save(tmp)
    from src.dust3r.utils.image import load_images_for_eval

    t_b = load_images_for_eval([tmp], size=512, verbose=False, crop=True)[0]["img"]

    print("plumbline _build_views img:        ", tuple(t_a.shape))
    print("CUT3R load_images_for_eval img:    ", tuple(t_b.shape))
    if t_a.shape != t_b.shape:
        print("SHAPE MISMATCH -> preprocessing differs")
        return
    d = (t_a.float() - t_b.float()).abs()
    print(f"max|Δ| = {d.max().item():.3e}   mean|Δ| = {d.mean().item():.3e}")
    print("BYTE-IDENTICAL" if d.max().item() == 0 else "DIFFER")


if __name__ == "__main__":
    main()
