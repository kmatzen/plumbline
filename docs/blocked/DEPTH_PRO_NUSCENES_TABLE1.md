# BLOCKED — Depth Pro Table 1 · nuScenes (δ₁)

> **⚠️ Code removed (2026-05-31, pre-release).** The `nuscenes` loader, its
> reproduction config, protocol, and fetch script were removed from the
> package. No verified result anchored that the loader parsed nuScenes GT
> correctly; the observed δ₁ **0.594 vs paper 0.491** reads *better* than the
> paper against an unverifiable random-881 subset / recipe. This page is
> retained to **document the attempt**. See `docs/CONFIDENCE_AUDIT.md`.

| Field | Value |
|-------|--------|
| **Status** | 🔒 Fundamentally blocked (off-paper) |
| **Repro** | `depth-pro-nuscenes` |
| **Protocol** | `nuscenes_depth_pro_metric` |
| **Paper** | δ₁ **0.491** (Table 1; appendix: **881** samples, 0.001–80 m) |
| **Observed** | δ₁ **0.5935** (2026-05-31, 881/881) |
| **Direction** | Reads **better** than paper (+20.9 %) |

## Summary

Val-split **CAM_FRONT** keyframes with LiDAR projected to image depth (metric meters),
seed **42** subsample to **881** frames (approximation of paper’s random 881 from val).
GT resolution target **900×1600** per Table 16. Full repro completed on trainval staging.

## What we tried

| Item | Detail |
|------|--------|
| Data | `./scripts/download-nuscenes.sh trainval` → `$NUSCENES_ROOT` |
| Loader | `src/plumbline/datasets/nuscenes.py` |
| Frames | 881/881, no alignment |

## Known approximations (not fixable without paper recipe)

| Topic | Plumbline | Paper (inferred) |
|-------|-----------|------------------|
| Camera | CAM_FRONT only | May mix cameras or use different keyframe policy |
| Subset | `seed=42` on val keyframes | “881 random” — exact draw unknown |
| Depth | LiDAR → image projection | May use different splat / filter |

These may explain **over-performance** but we cannot close the cell without author spec.

## Why this is blocked

1. **No nuScenes eval script** in `ml-depth-pro`.
2. **Reads better** — same pattern as Sintel/Middlebury: likely **weights or undisclosed sampling**, not a broken metric pipeline (Booster matched).
3. Exact **881-frame list** is not published.

## What would unblock

- Published frame list + LiDAR projection recipe used for Table 1, **or**
- Confirmed checkpoint + preprocessing matching the paper run.

## Do not

- Tune `paper_reference.value` or change seed to chase 0.491

## Artifacts

| Artifact | Path |
|----------|------|
| JSON | `$PLUMBLINE_WORK/runs/depth_pro_nuscenes_20260531.json` |
| S3 | `s3://plumbline-bench/runs/tier_depth_pro_nuscenes_20260531/` |

## Links

- [`../BLOCKED.md`](../BLOCKED.md)
