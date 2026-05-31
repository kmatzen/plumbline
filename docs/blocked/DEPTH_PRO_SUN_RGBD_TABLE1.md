# BLOCKED — Depth Pro Table 1 · Sun RGB-D (δ₁)

> **⚠️ Code removed (2026-05-31, pre-release).** The `sun_rgbd` loader, its
> reproduction config, protocol, and fetch script were removed from the
> package. No verified result anchored that the loader parsed Sun-RGBD GT
> correctly, and the observed δ₁ **0.451 vs paper 0.890** (2× worse) is the
> shape of a GT-decode / RGB↔depth pairing bug we could not rule out — exactly
> the kind of suspiciously-unverifiable result we do not ship. This page is
> retained to **document the attempt**. See `docs/CONFIDENCE_AUDIT.md`.

| Field | Value |
|-------|--------|
| **Status** | 🔒 Fundamentally blocked (off-paper) |
| **Repro** | `depth-pro-sun-rgbd` |
| **Protocol** | `sun_rgbd_depth_pro_metric` |
| **Paper** | δ₁ **0.890** (Table 1; appendix: **5050** samples, 0.001–10 m) |
| **Observed** | δ₁ **0.4505** (2026-05-31, 5050/5050) |
| **Direction** | Reads **worse** than paper (−49 %) |

## Summary

Ahanda **test** pack: `rgb/img-{i:06d}.jpg` paired with `depth/{i}.png` (uint16 ÷
10000 → meters). Appendix Table 16 clip **0.001–10 m**. Metric δ₁, no alignment.
Unlike Sintel/Middlebury/NuScenes, the model **under-performs** the paper column.

## What we tried

| Item | Detail |
|------|--------|
| Staging | `scripts/download-sun-rgbd.sh` — flat tarball extract (no erroneous `--strip-components`) |
| Pairing | `img-000001.jpg` ↔ `depth/1.png` … 5050 |
| Run | ~8 min H100, all frames scored |

## Why this is blocked

1. **No Sun-RGBD eval** in `ml-depth-pro`.
2. **Reads worse** — suggests **different test split**, depth encoding, invalid-mask rules, or **paper weights** vs our pairing. Booster match on same weights argues against a globally broken adapter.
3. Sun-RGBD has multiple releases (NYU vs Ahanda vs paper subset); paper does not pin which **5050** frames or depth source.

## Hypotheses (unverified)

| Hypothesis | Notes |
|------------|--------|
| Wrong test subset | Paper may not use full Ahanda 5050 |
| Depth scale / invalid pixels | ÷10000 vs other encoding; hole mask |
| Resolution / crop | Table 16 lists 530×730; confirm resize vs paper |
| Public weights | Same README caveat as Sintel |

## What would unblock

- Author frame list + depth file convention for Table 1 Sun-RGBD, **or**
- Confirmed that Ahanda test pack + our pairing matches their eval.

## Do not

- Tune `paper_reference.value` to absorb 0.45

## Artifacts

| Artifact | Path |
|----------|------|
| JSON | `$PLUMBLINE_WORK/runs/depth_pro_sun_rgbd_20260531.json` |
| S3 | `s3://plumbline-bench/runs/tier_depth_pro_table1_20260531/results/` |

## Links

- [`../BLOCKED.md`](../BLOCKED.md)
