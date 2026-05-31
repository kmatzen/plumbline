# Depth Pro Table 1 — metric δ₁ eval design (2026-05-30)

Depth Pro Table 1 reports **δ₁** on **Sintel** and **ETH3D** with **metric depth in
meters** (no alignment). Plumbline's blocked `depth-pro-sintel` cell used the
**native Sintel** tree + `sintel_dav2`-like depth loading — that reproduces
**AbsRel** under affine alignment, not Depth Pro's metric δ₁ recipe.

## What we ran (blocked)

| Item | Value |
|------|--------|
| Repro | `depth-pro-sintel` |
| Protocol | native `sintel` + metric depth, no alignment |
| Result | δ₁ **0.2418** vs paper **0.400** (−40 %, reads *better*) |
| Upstream | `ml-depth-pro` has **no** public Sintel/ETH3D eval script |

Same “reads better than paper” shape as DA-V2 native Table 2 under wrong metric/protocol.

## What Table 1 likely requires

Before spending GPU on ETH3D or re-pinning Sintel:

1. **Metric depth GT** in meters on the **same RGB frames** Depth Pro ingests
   (512² or model-native resize — confirm from Depth Pro paper §4 / appendix).
2. **δ₁ threshold** definition: `max(d_pred/d_gt, d_gt/d_pred) < 1.25` on valid
   pixels only — confirm mask (finite depth, min depth, max depth cap).
3. **Pass / split**: Sintel `final` vs `clean`; ETH3D which views (dslr vs all).
4. **No** scale-and-shift — adapter `is_metric=True`, `scale_alignment: none`.

## Proposed plumbline protocols (not implemented yet)

### `sintel_depth_pro_metric.yaml`

```yaml
name: sintel_depth_pro_metric
fixed:
  dataset:
    name: sintel
    kwargs:
      pass_name: final   # confirm vs paper
  depth_clip: [0.001, 70.0]   # confirm cap with paper / sky mask
  mask_boundaries: false
  max_views: 1
  tasks: [mono_depth]
metrics:
  primary: delta1
```

### `eth3d_depth_pro_metric.yaml`

**Primary path (2026-05-31):** official `ground_truth_depth/dslr_images/*.JPG`
(float32, distorted DSLR grid) + matching `images/dslr_images/*.JPG`. See
[`ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md`](ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md).

Ruled out for Table 1 δ₁: z-buffer PLY (`eth3d_dav2`), MoGe bundle (`eth3d-moge-eval`),
chamfer clouds. Depth Pro repo has no public ETH3D eval script.

## Appendix C Table 16 (confirmed 2026-05-30)

| Field | Sintel value |
|-------|----------------|
| Valid depth | **0.01 m – 80 m** |
| Samples | **1064** (training) |
| GT resolution | **436 × 1024** |
| Resize | bilinear pred → GT (plumbline default) |

| Run | Protocol | δ₁ (1064) | Notes |
|-----|----------|-----------|--------|
| 2026-05-30 | `sintel_dav2` (0.001–70 m) | **0.2418** | legacy |
| 2026-05-31 | `sintel_depth_pro_metric` (0.01–80 m) | **0.2409** | appendix Table 16 |

80-frame smoke (first scenes): δ₁ **~0.48** — tail frames drive full-set mean down.
Depth clip / max_depth levers ruled out on smoke (identical δ₁ for 70 vs 80 m).

Protocol: `sintel_depth_pro_metric.yaml` · probe: `scripts/probe-depth-pro-sintel-protocol.py`

**Verdict:** protocol aligned with appendix; gap likely **upstream weights** (README:
reference impl re-trained, may not match paper) or undisclosed frame weighting.
Keep `depth-pro-sintel` **blocked**; do not tune `paper_reference.value`.

## Table 16 — all Depth Pro Table 1 datasets (staging on pod)

| Dataset | Depth range (m) | n | Staged locally | Plumbline repro |
|---------|-----------------|---|----------------|-----------------|
| Booster | 0.001–10 | 228 | ✅ | δ₁ **0.4878** vs **0.466** (2026-05-31, **match**) |
| ETH3D | 0.1–200 | 454 | train 13 scenes (partial official depth) | not run — see [`ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md`](ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md) |
| Middlebury | 0.001–10 | 15 | ✅ | δ₁ **0.7589** vs **0.605** (blocked, reads better) |
| NuScenes | 0.001–80 | 881 | ✅ | δ₁ **0.5935** vs **0.491** (blocked, reads better) |
| Sintel | 0.01–80 | 1064 | ✅ `$SINTEL_ROOT` | `depth-pro-sintel` **blocked** (δ₁ 0.2409) |
| Sun-RGBD | 0.001–10 | 5050 | ✅ | δ₁ **0.4505** vs **0.890** (blocked, reads worse) |
| iBims | 0.1–10 | 100 | ✅ `$IBIMS1_ROOT` | ✅ informational δ₁ **0.8458** (2026-05-31) |

**2026-05-31 iBims sanity check:** `depth-pro-ibims1` → δ₁ **0.8458**, AbsRel **0.161** on 100 MoGe-bundle frames (appendix Table 16 clip). Same `depth_pro.pt` as Sintel (δ₁ **0.2409**) — adapter + weights behave on high-quality indoor GT; Sintel miss is **not** a global metric-depth failure.

**2026-05-31 Sintel per-scene breakdown** (`scripts/analyze-depth-pro-sintel-json.py` on
`depth_pro_sintel_table16_20260531.json`):

| Finding | Value |
|---------|--------|
| Loader order: first 80 frames | δ₁ **0.477** |
| Frames 81–1064 | δ₁ **0.221** |
| Equal-scene mean (23 scenes) | δ₁ **0.230** |
| NaN frames | 19 / 1064 |

Six scenes with scene-mean δ₁ **&lt; 0.05** (300 frames): `mountain_1`, `bandage_1`,
`sleeping_2`, `bamboo_1`, `temple_2`, `bamboo_2`. Excluding them → frame-mean δ₁
**~0.31** (still under paper 0.40). Best scenes: `shaman_2` **0.76**, `alley_1` **0.60**.

Hypothesis: paper may use **per-sequence aggregation**, different scene subset, or
official weights — not fixable by `max_depth`/`pass_name` alone (already ruled out).

## Implementation checklist

| Step | Owner | Status |
|------|--------|--------|
| PDF + appendix: exact δ₁ formula, resize, mask | doc | ✅ Table 16 |
| Sintel protocol `sintel_depth_pro_metric` | code | ✅ |
| Full Sintel reproduce | GPU | ✅ 0.2409 vs 0.400 (blocked) |
| Search Depth Pro repo issues / supp for eval hints | doc | 🔎 open |
| ETH3D metric GT staging plan | data | blocked until source defined |
| Booster `depth-pro-booster` | GPU | ✅ δ₁ 0.4878 vs 0.466 (2026-05-31) |
| Middlebury `depth-pro-middlebury` | GPU | ⚠️ δ₁ 0.7589 vs 0.605 (blocked) |
| Sun-RGBD `depth-pro-sun-rgbd` | GPU | ✅ 0.4505 vs 0.890 (blocked) |
| NuScenes `depth-pro-nuscenes` | GPU | ✅ 0.5935 vs 0.491 (blocked) |
| ETH3D official-depth loader + `depth-pro-eth3d` | data/GPU | after `stage-eth3d-train-scenes.sh` top-up |

## Queue stance

Keep `depth-pro-sintel` **blocked** in `gpu_queue.yaml` until this doc's
checklist has a cited GT+mask recipe. No YAML tuning of `paper_reference.value`.

## Links

- `reproductions/depth_pro_sintel.yaml`
- `docs/DISCREPANCIES.md` (D32 Depth Pro notes)
- `docs/GPU_BACKLOG_PLAN.md` §4
