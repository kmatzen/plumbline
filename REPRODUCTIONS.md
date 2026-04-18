# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

## Running

Set the appropriate dataset-root env var first; YAML files deliberately
don't hardcode machine-specific paths:

```bash
export NYUV2_ROOT=~/data/nyuv2      # for any da-v2-*-nyuv2 reproduction
export SCANNET_ROOT=~/data/scannet  # for vggt-paper-scannet-depth
export SINTEL_ROOT=~/data/sintel    # for depth-anything-v2-sintel

plumbline reproduce <name>
```

This loads `reproductions/<name>.yaml`, runs the model on the dataset,
computes metrics, and compares the primary metric against the published
value.

## Status matrix

| Name | Paper | Primary metric | Published | Observed | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `da-v2-small-nyuv2` | DA-V2 ViT-S, NYU Eigen test (Table 2) | `abs_rel` | 0.053 | **0.0623** | ±25% | ⚠️ **+17% vs paper** (matches public code, not paper table). See note below. |
| `da-v2-large-nyuv2` | DA-V2 ViT-L, NYU Eigen test (Table 2) | `abs_rel` | 0.045 | **0.0554** | ±25% | ⚠️ **+23% vs paper** (matches public code, not paper table). See note below. |
| `da-v2-metric-indoor-large-nyuv2` | DA-V2 Metric-Indoor-Large, NYU Eigen | `abs_rel` | _n/a_ | **0.0613** | n/a | Informational. No published AbsRel for Hypersim-finetuned ViT-L on NYU; closest is Table 4a's 0.056 for a distinct NYU-finetuned checkpoint. Median-aligned. |
| `vggt-paper-scannet-depth` | VGGT, ScanNet, 8 views | `abs_rel` | _TBD_ | — | ±5% | VGGT wiring complete (RTX 3090 end-to-end sanity on random images ✓). Blocked on ScanNet ToS signup + `$SCANNET_ROOT` data. |
| `depth-anything-v2-sintel` | DA-V2, Sintel | `abs_rel` | ≈0.075 | — | ±15% | blocked on Sintel depth-archive availability |
| _VGGT / ETH3D courtyard smoke_ | VGGT-1B, 4 views, first sample | `pose_auc@5°` | — | **0.91** | n/a | informational only. Rotation errors <0.3°/view; translation cos <0.6°/view. Chamfer needs similarity alignment (standard ETH3D protocol, not wired in v0.1). |

### Note on the DA-V2 NYUv2 gap

A 2026-04-18 diagnostic ran the author's own public inference code
(`DepthAnything-V2/run.py`) on the 654-sample Eigen test split with the
HuggingFace ViT-S checkpoint, standard Eigen crop, `gt ∈ [1e-3, 10]m`
valid mask, and MiDaS scale+shift in inverse-depth space. The author's
public code produced **AbsRel=0.062** — within 0.3% of plumbline's
0.0623, and ~17% higher than the paper's Table 2 value of 0.053. The
public HF and author-published `.pth` checkpoints agree per-pixel to
within 0.2% on NYU sample 0 (median ratio 1.001, mean 1.002). Taken
together: the plumbline pipeline is faithful to the public-code
baseline; paper's Table 2 numbers are achieved via a protocol detail
(dataset split variant, mask, eval aggregation, or a private checkpoint
refinement) that isn't published. We keep these reproductions so
pipeline regressions are caught early, but tolerance is ±25% until the
gap is resolved upstream.

## Adding a new reproduction

1. Read the target paper's evaluation section carefully. Note:
   - Exact dataset + split + sample list.
   - View count / resolution / crop policy.
   - Scale alignment (metric? median? scale-and-shift?).
   - The metric name and the exact numerical value.
2. Write `reproductions/<short-name>.yaml`:
   - `model.name` + `kwargs` to match the paper's model variant + settings.
   - `dataset.name` + `kwargs` to match the paper's sample selection.
   - `tasks`, `scale_alignment`, `max_views` to match the protocol.
   - `paper_reference.primary_metric`, `.value`, `.tolerance_relative`.
   - `paper_reference.citation` — point a reader at the exact table/line.
3. For sample-level reproducibility, commit a `<short-name>.samples.txt`
   listing sample IDs in evaluation order and reference it from the YAML.
4. On the first successful run, pin the observed value in the YAML's
   `paper_reference.value` (if not already known from the paper) and the
   final `tolerance_relative`.

## Why tolerances

Bitwise reproducibility on CUDA is not possible for most current foundation
models — mixed-precision and cuDNN autotune introduce run-to-run noise. We
therefore express agreement as a **relative** tolerance on the primary
metric (default ±5%). If a run falls outside tolerance, investigate:

- Coordinate-system drift (the `conventions.py` assertions should catch most).
- Resolution / resize interpolation differences.
- Depth vs disparity vs inverse-depth confusion in the adapter.
- Scale alignment mode mismatch.

These failure modes are tracked as known traps in `plan.md § 9`.
