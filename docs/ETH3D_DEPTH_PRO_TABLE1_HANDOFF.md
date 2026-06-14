# ETH3D — Depth Pro Table 1 handoff (2026-05-31)

> **✅ COMPLETED 2026-06-11 — kept for the implementation record only.** This was
> a to-do handoff; the cell has since run (ℹ️ off-paper) and the canonical
> root-cause classification now lives in
> [`CONFIDENCE_AUDIT.md`](CONFIDENCE_AUDIT.md) (L3 recipe — far-range
> metric-scale). The status table here is in
> [`BLOCKED.md`](BLOCKED.md#depth-pro-table-1-metric-δ₁-no-alignment). The notes
> below document *how* it was implemented and *why* it lands off-paper.

Depth Pro Table 16: **454** samples, valid depth **0.1–200 m**, GT **4032×6048** (native
distorted DSLR). Paper δ₁ = **0.415** (41.5 %). This was the last Table 1 column
to be run in plumbline.

## Status: RAN 2026-06-11 — ℹ️ off-paper (explained)

Loader + run landed 2026-06-11 (GTX 1080Ti). Loader `eth3d-native-depth`
(distorted RGB + official float32 depth, GT focal = undistorted pinhole scaled
to distorted size), protocol `eth3d_depth_pro_metric`, reproduction
`depth-pro-eth3d`. Staging: `scripts/stage-eth3d-depth-pro.sh` (curl + py7zr,
no system 7z) for `*_dslr_{depth,jpg}.7z`; calibration via
`*_dslr_undistorted.7z`. All **13 scenes → exactly 454 frames**, a precise
match to the Table 16 manifest below.

| Run (454/454, no align, clip 0.1–200 m) | δ₁ | abs_rel | vs paper 0.415 |
|------------------------------------------|------|---------|----------------|
| GT focal (`use_gt_focal=true`, canonical) | **0.3648** | 0.360 | −12.1% |
| self-focal (Depth Pro default)            | **0.3339** | 0.378 | −19.5% |

**Diagnosis (not tuned).** Bimodal by scene: close indoor scenes match the
paper well (kicker 0.90, office 0.92, pipes 0.93) but 3 far-range outdoor
scenes collapse to δ₁≈0 (meadow 0.00, terrace 0.07, facade 0.00 @ GT focal).
Depth Pro **under-scales far metric depth** — meadow GT median 8.2 m (to 30 m)
vs pred median 2.2 m (~3.7× compression), so the depth ratio is ≫1.25
everywhere on those scenes. Focal is correct (fx≈3323, same scaling gives the
high indoor scores); GT focal lifts mid scenes (terrains 0.43→0.89) but cannot
rescue the far-saturated ones. The residual gap to the paper's already-low
0.415 is therefore concentrated in far-depth handling — likely a paper-private
preprocessing/clip nuance (same shape as the other off-paper Table 1 cells),
not a loader or model-load bug. Result JSONs:
`runlogs/depth_pro_eth3d_{gtfocal,selffocal}_20260611.json`
(config_hash d9a443f62f35e106 / eecb9cb273b5f0e7).

Off-paper cells: [`BLOCKED.md`](BLOCKED.md).

## Why metric δ₁ was not run yet

| Path | GT type | Usable for Table 1 δ₁? |
|------|---------|------------------------|
| `eth3d` + z-buffer PLY | sparse laser render @518 | ❌ wrong metric / resolution |
| `eth3d-moge-eval` | MoGe HF bundle @2048×1365 | ❌ different protocol (DA-V2 Table 2 / MoGe T3) |
| **`ground_truth_depth/dslr_images/*.JPG`** | official float32 depth, distorted grid | ✅ **primary candidate** |

Courtyard probe (`scripts/probe-eth3d-official-depth.py`): **distorted RGB + official
depth** at native res gives AbsRel ~**0.02** (DA-V2); geometry pairing is correct.
Helpers exist: `load_eth3d_official_depth_map`, `official_depth_valid_mask` in
`src/plumbline/datasets/eth3d.py`.

## Staging gap (pod 2026-05-31)

| Item | Status |
|------|--------|
| 13 train scenes under `$ETH3D_ROOT` | ✅ |
| `dslr_scan_eval` PLY (chamfer / DA-V2) | ✅ all scenes |
| `ground_truth_depth/` + `images/dslr_images/` | ⚠️ **partial** (~38 depth files; need per-scene `*_dslr_depth.7z` + `*_dslr_jpg.7z`) |

Top up:

```bash
source scripts/pod-localssd-env.sh
./scripts/stage-eth3d-train-scenes.sh \
  courtyard delivery_area electro facade kicker meadow office pipes \
  playground relief relief_2 terrace terrains
```

~800 MB/scene × missing archives; allow hours. Not on `s3://plumbline-bench/datasets/` yet.

## Implementation plan (when staged)

1. **`Eth3dDepthProDataset`** (or `eth3d` kwargs `gt_source: official_depth`):
   - RGB: `images/dslr_images/DSC_*.JPG` (distorted)
   - Depth: `ground_truth_depth/dslr_images/DSC_*.JPG` (float32 file, `inf` → invalid)
   - Intrinsics: from `dslr_calibration_undistorted` scaled to distorted size, or EXIF if paper used raw
   - **454** frames: same manifest as `eth3d_dav2` train records (13 scenes)
2. **`protocols/eth3d_depth_pro_metric.yaml`**: `depth_clip: [0.1, 200.0]`, `scale_alignment: none`
3. **`reproductions/depth-pro-eth3d.yaml`**: `paper_reference.value: 0.415`
4. **GPU**: ~454 × 4–6 MP native — expect slow; bilinear pred→GT per Table 16

Do **not** tune `paper_reference` if δ₁ mismatches; document vs z-buffer / MoGe paths.

## Quick checks

```bash
source scripts/pod-localssd-env.sh
find "$ETH3D_ROOT" -path '*/ground_truth_depth/dslr_images/*.JPG' | wc -l   # expect ~454
uv run python scripts/probe-eth3d-official-depth.py --scene courtyard
```

## Links

- Table 1 matrix: [`BLOCKED.md`](BLOCKED.md)
- DA-V2 native (parked): [`ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md)
- D31 / official depth: `docs/DISCREPANCIES.md`
