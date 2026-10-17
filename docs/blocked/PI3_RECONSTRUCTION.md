# BLOCKED — π³ (pi3) point-cloud reconstruction

> **Historical note:** the adapter was removed on 2026-05-31 after the failed
> reconstruction A/B, then restored and vendored on 2026-06-04 for depth and
> pose work. Its DAGE Table-4 pose path is independently useful; the DTU/ETH3D
> reconstruction result documented here remains blocked and must not be
> presented as verified.

| Field | Value |
|-------|-------|
| **Status** | 🔒 Reconstruction blocked — adapter ships, but this path has no verified anchor |
| **Model** | π³ / Pi3 (Yuan et al. 2025, [arXiv:2507.13347](https://arxiv.org/abs/2507.13347)) |
| **Cells attempted** | DTU MVS chamfer, ETH3D multi-scene chamfer (both informational, `value: null`) |
| **Verified ✅ anywhere?** | No |

## Why reconstruction remains blocked — the A/B inversion

The π³ paper's Table 3 reports point-map reconstruction with **Umeyama + ICP
alignment** (exactly plumbline's chamfer protocol), and it reports VGGT in the
**same** convention:

| | π³ (paper) | VGGT (paper) | paper says |
|---|---|---|---|
| DTU Accuracy | 1.198 | 1.338 | π³ ≈ VGGT (slightly **better**) |
| ETH3D Accuracy | 0.194 | 0.280 | π³ ≈ VGGT (slightly **better**) |

plumbline ran π³ and VGGT through the **same loader, protocol, and ICP
alignment** (`dtu_vggt_table2`, `eth3d_vggt_table3`). Result of the 2026-05-28
GPU run:

| | plumbline π³ | plumbline VGGT | plumbline says |
|---|---|---|---|
| DTU Overall | **17.33 mm** | ~0.76 mm | π³ ~20× **worse** |
| ETH3D Overall | **3.75 m** | ~0.64 m | π³ ~6× **worse** |

Relative ordering is **convention-invariant** — units/normalization cancel when
you compare two models scored the same way. The paper says π³ ≈ VGGT; plumbline
says π³ is 6–20× worse. That inversion cannot be a units mismatch; it is
positive evidence that **plumbline's π³ adapter produces degraded geometry** —
a wiring bug (point-map convention, view handling, or scale) we did not fix.

## Why no paper-match was possible either

Even setting the bug aside, the target was unpinnable: the π³ paper's metric is
in a normalization that matches **neither** plumbline's metric output **nor**
the VGGT paper's own DTU number (the π³ table lists VGGT-DTU as 1.338, whereas
the VGGT paper reports 0.382 mm). Pinning π³ would have required reverse-
engineering an undisclosed normalization — and the built-in VGGT cross-check
already fails to line up.

## To unblock reconstruction

Re-run the π³ vs VGGT A/B on DTU/ETH3D and **debug why π³'s point map is ~20×
worse than VGGT's** under identical alignment. The most likely culprits: the
point-map → camera-frame convention, the per-view scale, or the keyframe
selection ("every 5 images" per the paper). Re-test against the current
vendored adapter. Do **not** promote a DTU/ETH3D reproduction until the A/B
ordering matches the paper. See `docs/CONFIDENCE_AUDIT.md`.
