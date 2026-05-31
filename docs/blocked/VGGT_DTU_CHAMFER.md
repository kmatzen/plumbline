# BLOCKED — VGGT · DTU chamfer (Table 2)

| Field | Value |
|-------|--------|
| **Status** | 🔒 Upstream-blocked |
| **Repro** | `vggt-dtu` (and related DTU jobs) |
| **Paper** | Overall **0.382 mm** (Wang 2025 VGGT, Table 2) |
| **Observed** | Overall **~0.75–0.87 mm** (~2× off, stable across levers) |
| **Discrepancy** | D3 in `docs/DISCREPANCIES.md` |

## Summary

Plumbline ports the **CUT3R/MASt3R DTU** recipe: per-view-masked chamfer,
PatchmatchNet geometric-consistency filter, public `facebook/VGGT-1B` weights.
Every adapter and protocol lever tried in-session moves Overall by **&lt;1 %**;
the ~2× gap vs paper is **not** fixable in harness code alone.

## Levers exhausted (D3)

| Lever | Overall (mm) | Effect |
|-------|----------------|--------|
| CUT3R per-view-masked chamfer | ~0.758 | baseline |
| Jensen DTUeval-python toolkit | ~0.868 | worse |
| 49 vs 32 rig views | ~0.849 | negligible |
| PatchmatchNet filter | ~0.756 | negligible |
| PatchmatchNet + fp32 | **~0.750** | negligible |

## Why blocked

1. **Public HF checkpoint** may not match paper weights (same class as Depth Pro README caveat).
2. Paper may include **post-processing** (TSDF, BA, `demo_colmap --use_ba`) not in raw forward pass.
3. Paper cites "Following MASt3R" but **no released DTU eval** in MASt3R repo to diff.

CO3Dv2 pose (Table 1) **did match** on the same codebase — DTU is not a generic VGGT adapter failure.

## What would unblock

- VGGT release with DTU eval script matching Table 2, **or**
- Paper-matched checkpoint + documented fusion/BA pipeline.

## Do not

- Keep burning GPU on filter/dtype/view-count tweaks expecting ~0.38 mm

## Links

- [`../DISCREPANCIES.md`](../DISCREPANCIES.md) § D3
- [`../BLOCKED.md`](../BLOCKED.md)
- `reproductions/vggt_dtu_fp32_probe.yaml` (diagnostic)
