# ETH3D DA-V2 Table 2 — handoff (parked 2026-05-30)

Pivoting to other queue work. **Do not tune YAML** to chase paper numbers — OFF-PAPER
is a documented finding. Full narrative: `docs/DISCREPANCIES.md` **D31** (loader fix),
**D33** (protocol investigation).

## What is done

| Item | Status |
|------|--------|
| D31 RGB/GT misalignment (`resize_images_to_pv_render`) | ✅ Shipped (`71ade40`) |
| 13-scene staging (`$ETH3D_ROOT`, 454 frames) | ✅ All scenes + `dslr_scan_eval` |
| Definitive S/B/L harness run (`eth3d_dav2`) | ✅ MISMATCH ~−30–32 % under paper |
| All-scene `dslr_scan_eval` ViT-L re-run | ✅ **0.0782** (still under 0.131) |
| Official depth + distorted JPG probes (courtyard) | ✅ Aligned path still ~0.02 vs 0.131 |
| Code: manifest GT rescan, pv cache by GT tag | ✅ `eth3d.py` |
| Probe script | ✅ `scripts/probe-eth3d-official-depth.py` |
| Staging top-up | ✅ `scripts/stage-eth3d-train-scenes.sh` (`dslr_scan_eval`, `dslr_depth`, `dslr_jpg`) |

## Key numbers (13-scene, 454 frames, `eth3d_dav2`, 2026-05-30)

| Variant | Observed AbsRel | Paper | Notes |
|---------|-----------------|-------|-------|
| ViT-S | 0.1012 | 0.142 | `da_v2_small_eth3d_native_13scene_20260530.json` |
| ViT-B | 0.0936 | 0.137 | `da_v2_base_eth3d_native_13scene_20260530.json` |
| ViT-L (mixed GT) | 0.0888 | 0.131 | `da_v2_large_eth3d_native_13scene_20260530.json` |
| ViT-L (all `dslr_scan_eval`) | **0.0782** | 0.131 | `da_v2_large_eth3d_native_13scene_dslr_eval_20260530.json` |

Variant order **L < B < S** matches Table 2. MoGe-bundle Table 3 cell **0.0473** is a
different dataset/protocol (`eth3d_moge`) — not comparable.

## Ruled out (do not re-litigate without new evidence)

1. **D31 loader bug** — fixed; 3-scene smoke dropped from ~0.33 → ~0.07.
2. **Incomplete staging** — 13 scenes, 454/454 evaluated, 0 skipped.
3. **Frame list vs MoGe bundle** — plumbline manifest maps to the same
   `scene/DSC_*` keys as `$ETH3D_MOGE_ROOT/ETH3D/.index.txt` (454 frames;
   index file may report 453 lines — trailing newline). Not a 454-vs-453
   coverage gap.
3. **Wrong PLY variant** — `dslr_scan_eval` lowers AbsRel further (0.0782), not toward paper.
4. **Sparse official depth mask alone** — courtyard aligned distorted RGB + official
   depth: **0.0204** vs z-buffer **0.0313** (both ≪ 0.131).
5. **Undistorted vs distorted RGB mismatch** — alignment barely moves courtyard AbsRel.

## Artifacts

**Local** (`$PLUMBLINE_WORK/runs/`):

- `da_v2_*_eth3d_native_13scene_20260530.json`
- `da_v2_large_eth3d_native_13scene_dslr_eval_20260530.json`
- `eth3d_official_depth_probe_courtyard.log`
- `eth3d_official_depth_probe_distorted_courtyard.log`

**S3** (`s3://plumbline-bench/runs/`):

- `tier_c_eth3d_13scene_20260530/results/`
- `tier_c_eth3d_dslr_eval_20260530/results/`
- `tier_c_d31_*` (subset / fix runs)

**Data** (`$ETH3D_ROOT`): full 13-scene train; courtyard also has
`ground_truth_depth/` + `images/dslr_images/` (distorted JPG probe only).

## When you return — ranked next steps

1. **MoGe eval harness (primary)** — [`docs/DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md):
   `moge/scripts/eval_baseline.py` + HF bundle @ **2048×1365** + `disparity_affine_invariant`
   with disparity floor at `1/gt.max()`. Matches our **MoGe Table-3** cell (0.047 ✅).
2. **Issue #281** — devkit sparse depth + resize → ~0.5 AbsRel (same failure mode as
   pre-D31 native harness). Not the MoGe-bundle path.
3. **DepthAnythingAC** — possible third-party reference; not verified here.
3. **Optional loader** — `eth3d_official_depth` protocol: distorted JPG + sparse
   `*_depth.7z`, native res, `scale_shift`; only if upstream code confirms that recipe.
   Helpers already in `datasets/eth3d.py` (`load_eth3d_official_depth_map`).
4. **Skip unless needed** — full 13-scene `dslr_jpg` + `dslr_depth` (~800 MB/scene);
   courtyard probe already shows official path won’t explain −32 % gap.

## Quick resume commands

```bash
source /mnt/localssd/plumbline/scripts/pod-localssd-env.sh
export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"

# Re-run harness (cached GT renders exist)
uv run plumbline reproduce da-v2-large-eth3d-native \
  -o "$PLUMBLINE_WORK/runs/da_v2_large_eth3d_native_13scene_20260530.json"

# Official-depth probe (courtyard needs dslr_depth + dslr_jpg once)
uv run python scripts/probe-eth3d-official-depth.py --scene courtyard

# Top up eval GT / official depth / distorted JPG on staged scenes
./scripts/stage-eth3d-train-scenes.sh meadow  # or scene list
```

## Queue / matrix

- `reproductions/gpu_queue.yaml`: `da-v2-{small,base,large}-eth3d-native` → **blocked**
  (D33; `plumbline queue --run` will skip).
- Site matrix: do **not** promote ETH3D Table-2 cells to verified until protocol closes.

## Related

- Protocol: `protocols/eth3d_dav2.yaml`
- Parallel off-paper track: **D32** native Sintel (`depth-anything-v2-sintel`)
