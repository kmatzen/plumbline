# ETH3D — Depth Pro Table 1 handoff (2026-05-31)

Depth Pro Table 16: **454** samples, valid depth **0.1–200 m**, GT **4032×6048** (native
distorted DSLR). Paper δ₁ = **0.415** (41.5 %). Only Table 1 column not yet run in
plumbline.

## Status: pending (not fundamentally blocked)

Unlike the four off-paper Table 1 columns, ETH3D is **not** closed as blocked — we
lack official depth JPGs + a Depth Pro loader, not an exhausted protocol. Off-paper
cells: [`BLOCKED.md`](BLOCKED.md).

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
