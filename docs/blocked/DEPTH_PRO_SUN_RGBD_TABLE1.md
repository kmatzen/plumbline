# BLOCKED — Depth Pro Table 1 · Sun RGB-D (δ₁)

> **⚠️ Code removed (2026-05-31, pre-release).** The `sun_rgbd` loader, its
> reproduction config, protocol, and fetch script were removed from the
> package. This page documents the attempt — and, as of **2026-06-01**, a
> GPU-verified root-cause characterization that **overturns the original
> "GT-decode / pairing bug" hypothesis**. See `docs/CONFIDENCE_AUDIT.md`.

| Field | Value |
|-------|--------|
| **Status** | 🔒 Blocked (off-paper) — **root cause now characterized (focal/metric-scale), not a parsing bug** |
| **Repro** | `depth-pro-sun-rgbd` (removed) |
| **Protocol** | `sun_rgbd_depth_pro_metric` (removed) |
| **Paper** | δ₁ **0.890** (Table 1; appendix: **5050** samples, 0.001–10 m) |
| **Observed (full 5050)** | δ₁ **0.4505** (2026-05-31) |
| **Direction** | Reads **worse** than paper (−49 %) |

## Summary

Ahanda **test** pack: `rgb/img-{i:06d}.jpg` paired with `depth/{i}.png` (uint16 ÷
10000 → meters), 730×530, 5050 frames. Metric δ₁, **no alignment** (Depth Pro
emits metric meters using its own **estimated focal length** — the model takes no
GT intrinsics; this matches the paper's "metric absolute scale without requiring
camera intrinsics" claim, arXiv:2410.02073).

## Root-cause characterization (GPU-verified, 2026-06-01)

**The gap is a per-frame focal / metric-scale failure in ~45 % of frames — the
depth _structure_ is correct everywhere. It is NOT a GT-decode or RGB↔depth
pairing bug.**

### Evidence A — the full-run δ₁ is bimodal, not uniformly low
Re-analysing the saved 5050-frame run (`runs/.../depth_pro_sun_rgbd_20260531.json`):
δ₁ is sharply **bimodal** — 34 % of frames at δ₁<0.1 and 20 % at δ₁>0.9 (45 %
below 0.3, 31 % above 0.8). A uniform GT-scale/decode bug would shift _every_
frame equally; this is a **mixed population** (consistent with SUN-RGBD pooling
four sensors — Kinect v1/v2, RealSense, Xtion — with different intrinsics).

### Evidence B — GT decode and pairing are correct
- The depth PNGs are stored as **multiples of 8** (low-3-bits always zero); the
  `÷10000 → m` decode is **empirically validated** because the good frames match
  metric, **un-aligned** Depth Pro at δ₁≈0.95. A wrong GT scale cannot produce a
  δ₁ of 0.95 against an un-aligned metric prediction.
- RGB↔depth correspondence visually confirmed for both good and bad frames
  (same scene, aligned depth structure).
- `silog` (scale-invariant) of bad frames (14.8) ≈ good frames (11.3) — both in
  the "good prediction" regime — while their scale-dependent δ₁ differs 20×. The
  structure is right; only the absolute scale is wrong.

### Evidence C — direct focal probe on the GTX 1080Ti
120-frame balanced probe (60 frames with original δ₁>0.9, 60 with δ₁<0.1),
re-running Depth Pro and recording its **estimated focal** plus per-frame
median-scale-aligned δ₁ (`docs/blocked/artifacts/depth_pro_sun_rgbd_focal_probe.py`,
result `..._20260601.json`). The probe reproduces the original per-frame δ₁
**exactly** (corr = 1.000).

| group (by original δ₁) | δ₁ un-aligned | δ₁ **scale-aligned** | est. focal (px) | median scale (gt/pred) |
|---|---|---|---|---|
| GOOD (60) | 0.953 | 0.966 | 540 | 0.91 |
| BAD (60)  | **0.027** | **0.963** | **617** | **0.68** |

- **100 %** of bad frames recover to δ₁>0.8 under a single per-frame scale.
  All-frame scale-aligned δ₁ = **0.964** (> paper 0.890).
- Bad frames get a **systematically higher estimated focal** (617 vs 540), so
  Depth Pro emits metric depth ~1.46× too large (scale 0.68) → δ₁ collapses,
  even though the relative depth is correct.

**Conclusion:** Depth Pro's focal/FoV head mis-estimates on a ~45 % subset of the
**ahanda 730×530 reprocessed** test pack, producing correct-structure /
wrong-metric-scale depth. The paper's 0.890 most likely comes from feeding
Depth Pro **native-resolution** SUN-RGBD images (where its focal head behaves as
in the paper), or from supplying **GT per-frame focal**. Neither is available in
this packaged set (rgb/ + depth/ only; no intrinsics, all resized to 730×530).

## What would unblock (concrete, ranked)

1. **Native-resolution images** — stage original SUN-RGBD (per-sensor native
   resolution + `depth_bfx/`), run Depth Pro with estimated focal. If δ₁→~0.89,
   this is the paper protocol → promotable to ✅. (Largest lift; clearest payoff.)
2. **GT per-frame focal** — stage SUN-RGBD `intrinsics.txt`, pass `f_px` to
   Depth Pro's `infer()`. Promotable **only if** confirmed the paper used GT
   focal (Appendix C.4 is silent; do not assume).

## Do not

- Tune `paper_reference.value` to absorb 0.45.
- Pick a single "canonical" focal that happens to land 0.89 — that is
  tuning-to-the-number unless the focal is independently justified by the
  dataset's documented intrinsics.

## Artifacts

| Artifact | Path |
|----------|------|
| Original 5050-run JSON | `s3://plumbline-bench/runs/tier_depth_pro_table1_20260531/results/depth_pro_sun_rgbd_20260531.json` |
| Focal probe (script + 120-frame result) | `docs/blocked/artifacts/depth_pro_sun_rgbd_focal_probe*.{py,json}` |
| SUN-RGBD test pack | `s3://plumbline-bench/datasets/sun_rgbd/` (rgb/ + depth/, 5050 each) |

## Links

- [`../BLOCKED.md`](../BLOCKED.md)
