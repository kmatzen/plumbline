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

## Implementation checklist

| Step | Owner | Status |
|------|--------|--------|
| PDF + appendix: exact δ₁ formula, resize, mask | doc | 🔎 open |
| Search Depth Pro repo issues / supp for eval hints | doc | 🔎 open |
| Define `SintelDepthProDataset` or extend `sintel` with `eval_mode: metric_delta1` | code | pending |
| ETH3D metric GT staging plan | data | blocked until source defined |
| Repro `depth-pro-sintel` with new protocol | GPU | after checklist |
| Repro `depth-pro-eth3d` (new) | GPU | after GT |

## Queue stance

Keep `depth-pro-sintel` **blocked** in `gpu_queue.yaml` until this doc's
checklist has a cited GT+mask recipe. No YAML tuning of `paper_reference.value`.

## Links

- `reproductions/depth_pro_sintel.yaml`
- `docs/DISCREPANCIES.md` (D32 Depth Pro notes)
- `docs/GPU_BACKLOG_PLAN.md` §4
