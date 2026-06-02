# Fundamentally blocked reproductions

Some paper cells cannot be closed in plumbline without **upstream changes** (weights,
eval script, undisclosed preprocessing) or **confirmed external protocol** we do not
control. Those are **fundamentally blocked**: we ran the best cited harness recipe,
documented the gap, and **do not tune** `paper_reference` in YAML to force a match.

**Not listed here:** jobs that are only waiting on **data staging** or **new loader
code** (e.g. `depth-pro-eth3d` before official depth is staged) — see handoff docs.

## Depth Pro Table 1 (metric δ₁, no alignment)

| Dataset | Observed / paper | Blocker | Page |
|---------|------------------|---------|------|
| Booster | 0.488 / 0.466 | — | ✅ match (`depth-pro-booster`) |
| Sintel | 0.241 / 0.400 | Protocol aligned; gap upstream / aggregation | [`blocked/DEPTH_PRO_SINTEL_TABLE1.md`](blocked/DEPTH_PRO_SINTEL_TABLE1.md) |
| Middlebury | 0.759 / 0.605 | **Loader REMOVED pre-release** — no verified anchor; reads better, no public eval | [`blocked/DEPTH_PRO_MIDDLEBURY_TABLE1.md`](blocked/DEPTH_PRO_MIDDLEBURY_TABLE1.md) |
| NuScenes | 0.594 / 0.491 | **Loader REMOVED pre-release** — no verified anchor; unknown subset/recipe | [`blocked/DEPTH_PRO_NUSCENES_TABLE1.md`](blocked/DEPTH_PRO_NUSCENES_TABLE1.md) |
| Sun-RGBD | ✅ 0.8682 / 0.890 | **RESOLVED 2026-06-01** (MATCH, 2.4% off) — was a GT-decode (÷10000 vs bit-rotation) + anisotropic-resize + estimated-focal artifact of the removed ahanda pack. New `sun-rgbd-native` loader + `DepthProAdapter(use_gt_focal=True)` reproduces 0.890 | [`blocked/DEPTH_PRO_SUN_RGBD_TABLE1.md`](blocked/DEPTH_PRO_SUN_RGBD_TABLE1.md) |
| ETH3D | — / 0.415 | **Not blocked** — implementation + data | [`ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md`](ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md) |

**Appendix Table 16** (depth clip m, sample count): Booster 0.001–10 / 228 · ETH3D 0.1–200 / 454 ·
Middlebury 0.001–10 / 15 · NuScenes 0.001–80 / 881 · Sintel 0.01–80 / 1064 · Sun-RGBD 0.001–10 / 5050.
Metric δ₁, no alignment, bilinear pred→GT. **iBims sanity** (100 frames, same weights): δ₁ **0.8458** —
adapter OK on indoor GT; Sintel gap is dataset-specific.

> **Removed before public release (2026-05-31).** The Middlebury / NuScenes /
> Sun-RGBD loaders, configs, protocols, and fetch scripts were deleted from the
> package. The Depth Pro *adapter* is verified (Booster ✅) and stays; what was
> removed is the per-dataset **loaders**, which had no verified result proving
> they parsed GT correctly. Sun-RGBD's 2× miss originally looked like a parsing
> bug; a GPU-verified focal probe (2026-06-01) since showed the GT decode and
> pairing are **correct** and the gap is Depth Pro's per-frame focal/metric-scale
> estimate on a frame subset — see its page below. The pages are kept to document
> the attempt. Sintel stays because it runs on the ✅-anchored `sintel` loader.

Queue: `reproductions/gpu_queue.yaml`.

## DA-V2 Table 2 native (affine AbsRel)

| Dataset | Shape | Page |
|---------|-------|------|
| DIODE | Native outdoor blowup; bundle explains gap | [`D29_DIODE_TABLE2_HANDOFF.md`](D29_DIODE_TABLE2_HANDOFF.md) |
| ETH3D | Under paper after loader fix; protocol open | [`ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md) (D31/D33) |
| Sintel | Under paper with sky mask | [`SINTEL_DAV2_TABLE2_HANDOFF.md`](SINTEL_DAV2_TABLE2_HANDOFF.md) (D32) |

## Multi-view / pose (upstream-blocked)

| Job family | Blocker | Page |
|------------|---------|------|
| VGGT DTU chamfer | Public checkpoint ~2× off; filters exhausted | [`blocked/VGGT_DTU_CHAMFER.md`](blocked/VGGT_DTU_CHAMFER.md) |
| GeoWizard / Marigold KITTI | Same class as D17/D18 | [`blocked/UPSTREAM_CHECKPOINT_KITTI.md`](blocked/UPSTREAM_CHECKPOINT_KITTI.md) |
| CUT3R NYU/KITTI/Bonn | Paper uses lineage pipeline, not plumbline strict | [`blocked/CUT3R_DEPTH_LINEAGE.md`](blocked/CUT3R_DEPTH_LINEAGE.md) |
| **π³ (pi3) DTU/ETH3D — ADAPTER REMOVED** | 6–20× worse than VGGT under identical alignment (paper says ≈); suspected adapter bug, no verified anchor | [`blocked/PI3_RECONSTRUCTION.md`](blocked/PI3_RECONSTRUCTION.md) |

## Data-staging blocked (not fundamental)

RealEstate10K pose, ScanNet/ScanNet-1500, etc. — blocked on **fetch**, not on an
exhausted protocol. See `gpu_queue.yaml` `blocked_on` text and `GPU_RUNBOOK.md`.

## Policy

1. **Do not** change `paper_reference.value` or tolerance to absorb off-paper runs.
2. **Do** add a row here + a dedicated `docs/blocked/*.md` when a cell is closed as blocked.
3. **Do** link blocked pages from `reproductions/*.yaml` `notes`, `gpu_queue.yaml`, and `DISCREPANCIES.md`.
