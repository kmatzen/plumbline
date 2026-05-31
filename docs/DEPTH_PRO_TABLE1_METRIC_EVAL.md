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

Requires **metric GT** source — not the z-buffer sparse cloud used for DA-V2
chamfer. Candidates to audit:

- ETH3D laser scan subsampled to image plane (if paper used scan eval)
- Official ETH3D **depth** benchmarks (different from multi-view chamfer)
- Depth Pro authors' unreleased script (none in `ml-depth-pro` today)

**Do not** reuse `eth3d_dav2` / z-buffer projection until a citation ties Table 1
to that geometry.

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
| Booster | 0.001–10 | 228 | ❌ | — |
| ETH3D | 0.1–200 | 454 | native + moge (chamfer) | blocked (no metric GT path) |
| Middlebury | 0.001–10 | 15 | ❌ | — |
| NuScenes | 0.001–80 | 881 | ❌ | — |
| Sintel | 0.01–80 | 1064 | ✅ `$SINTEL_ROOT` | `depth-pro-sintel` **blocked** (δ₁ 0.2409) |
| Sun-RGBD | 0.001–10 | 5050 | ❌ | — |
| iBims | 0.1–10 | 100 | ✅ `$IBIMS1_ROOT` | optional informational |

Next GPU candidate with lowest friction: **iBims** (100 frames, metric indoor) — not a Table 1 column but validates adapter on laser-scanned GT.

## Implementation checklist

| Step | Owner | Status |
|------|--------|--------|
| PDF + appendix: exact δ₁ formula, resize, mask | doc | ✅ Table 16 |
| Sintel protocol `sintel_depth_pro_metric` | code | ✅ |
| Full Sintel reproduce | GPU | ✅ 0.2409 vs 0.400 (blocked) |
| Search Depth Pro repo issues / supp for eval hints | doc | 🔎 open |
| ETH3D metric GT staging plan | data | blocked until source defined |
| Booster / NuScenes / Sun-RGBD staging | data | not started |
| Repro `depth-pro-eth3d` (new) | GPU | after GT |

## Queue stance

Keep `depth-pro-sintel` **blocked** in `gpu_queue.yaml` until this doc's
checklist has a cited GT+mask recipe. No YAML tuning of `paper_reference.value`.

## Links

- `reproductions/depth_pro_sintel.yaml`
- `docs/DISCREPANCIES.md` (D32 Depth Pro notes)
- `docs/GPU_BACKLOG_PLAN.md` §4
