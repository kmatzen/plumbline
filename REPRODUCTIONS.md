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
| `da-v2-small-nyuv2` | DA-V2 ViT-S, NYU Eigen test (Table 2) | `abs_rel` | 0.053 | **0.0510** | ±5% | ✅ **match** (RTX 3090, 2 min). |
| `da-v2-large-nyuv2` | DA-V2 ViT-L, NYU Eigen test (Table 2) | `abs_rel` | 0.045 | **0.0428** | ±10% | ✅ **match** (RTX 3090, 30 s). |
| `da-v2-metric-indoor-large-nyuv2` | DA-V2 Metric-Indoor-Large, NYU Eigen | `abs_rel` | _n/a_ | **0.0496** | n/a | Informational. Median-aligned. No published AbsRel for Hypersim-finetuned ViT-L on NYU. |
| `metric3d-v2-nyuv2` | Metric3Dv2 ViT-L, NYU Eigen (Table I) | `abs_rel` | 0.063 | **0.0660** | ±10% | ✅ **match** (RTX 3090, 4 min). δ₁: paper 0.975, observed 0.974. |
| `da3-nyuv2` | DA3 Large-1.1, NYU Eigen (Table 4) | `delta_1` | 0.974 | **0.9684** | ±2% | ✅ **match** (RTX 3090, 2 min). AbsRel=0.051 (informational; Table 4 only reports δ₁). |
| `vggt-paper-scannet-depth` | VGGT, ScanNet, 8 views | `abs_rel` | _TBD_ | — | ±5% | VGGT wiring complete (RTX 3090 end-to-end sanity on random images ✓). Blocked on ScanNet ToS signup + `$SCANNET_ROOT` data. |
| `depth-anything-v2-sintel` | DA-V2, Sintel | `abs_rel` | ≈0.075 | — | ±15% | blocked on Sintel depth-archive availability |
| _VGGT / ETH3D courtyard smoke_ | VGGT-1B, 4 views, first sample | `pose_auc@5°` | — | **0.91** | n/a | informational only. Rotation errors <0.3°/view; translation cos <0.6°/view. |
| _MASt3R / ETH3D courtyard pairs_ | MASt3R ViT-L, 35 consecutive 2-view samples | `pose_auc@5°` | — | **0.46** | n/a | informational only. Mean rotation error 0.32°/pair; translation cos 3.42°. 2-view setup (PairViewer) — Umeyama needs N≥3 so no chamfer. |

### Note on the NYUv2 Eigen 2014 protocol

Paper matches required three loader/runner details that weren't obvious
from reading the paper itself:

1. **Depth field: `rawDepths`, not `depths`.** NYU's .mat ships both the
   sparse Kinect measurements (`rawDepths`, ~24% holes) and Silberman's
   colorization-filled version (`depths`, dense). Every modern mono-depth
   paper that cites "NYU Eigen" evaluates against `rawDepths`;
   `NYUv2Dataset(depth_field="raw")` is the default.
2. **`depth_clip: [0.001, 10.0]` post-alignment.** Scale+shift alignment
   occasionally produces extreme per-sample predictions (on DA-V2 Large
   sample 88, an aligned value hit 1e8 m). Paper eval clips the aligned
   prediction to the same range as the valid GT mask. Reproduction
   YAMLs set this explicitly.
3. **`gt ∈ [1e-3, 10]m` valid mask.** Standard NYU convention; plumbline
   already applies this via the loader's Eigen crop + positivity mask.

A 2026-04-18 diagnostic confirmed the author's own `run.py` on the
HuggingFace ViT-S checkpoint produces AbsRel=0.0621 against the *filled*
`depths` field — within 0.3% of what plumbline produced before the raw-
default landed. Switching to rawDepths drops that to 0.0510 (vs paper
0.053), and the same switch takes ViT-L from 0.0554 to 0.0428 (vs paper
0.045). Without the clip, ViT-L averaged 77.9 because of sample 88 alone.

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
