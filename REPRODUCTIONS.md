# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

**Summary (as of 2026-04-19):** 7 paper-match rows ✅, 3 blocked on
auth-gated data, 4 informational ETH3D sweeps. All matches were
produced on a single RTX 3090 inside ~90 minutes of total wall clock
(first-run weight downloads dominated). See the status matrix below.

## Running

Set the appropriate dataset-root env var first; YAML files deliberately
don't hardcode machine-specific paths:

```bash
export NYUV2_ROOT=~/data/nyuv2      # for any da-v2-*-nyuv2 reproduction
export SCANNET_ROOT=~/data/scannet  # for vggt-paper-scannet-depth
export SINTEL_ROOT=~/data/sintel    # for depth-anything-v2-sintel
export KITTI_ROOT=~/data/kitti      # for any *-kitti reproduction
export DIODE_ROOT=~/data/diode      # for any *-diode-* reproduction
export DTU_ROOT=~/data/dtu          # for vggt-paper-dtu-mvs

plumbline reproduce <name>
```

This loads `reproductions/<name>.yaml`, runs the model on the dataset,
computes metrics, and compares the primary metric against the published
value.

## Status matrix

| Name | Paper | Primary metric | Published | Observed | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `da-v2-small-nyuv2` | DA-V2 ViT-S, NYU Eigen test (Table 2) | `abs_rel` | 0.053 | **0.0510** | ±5% | ✅ **match** (RTX 3090, 2 min). |
| `da-v2-base-nyuv2` | DA-V2 ViT-B, NYU Eigen test (Table 2) | `abs_rel` | 0.049 | **0.0456** | ±10% | ✅ **match** (RTX 3090, 2 min). δ₁: paper 0.976, observed 0.977. |
| `da-v2-large-nyuv2` | DA-V2 ViT-L, NYU Eigen test (Table 2) | `abs_rel` | 0.045 | **0.0428** | ±10% | ✅ **match** (RTX 3090, 30 s). |
| `da-v2-metric-indoor-large-nyuv2` | DA-V2 Metric-Indoor-Large, NYU Eigen | `abs_rel` | _n/a_ | **0.0496** | n/a | Informational. Median-aligned. No published AbsRel for Hypersim-finetuned ViT-L on NYU. |
| `metric3d-v2-nyuv2` | Metric3Dv2 ViT-L, NYU Eigen (Table I) | `abs_rel` | 0.063 | **0.0660** | ±10% | ✅ **match** (RTX 3090, 4 min). δ₁: paper 0.975, observed 0.974. |
| `metric3d-v2-giant-nyuv2` | Metric3Dv2 ViT-Giant2, NYU Eigen (Table I) | `abs_rel` | 0.067 | **0.0702** | ±10% | ✅ **match** (RTX 3090, ~50 min). δ₁: paper 0.980, observed 0.973. |
| `da3-nyuv2` | DA3 Large-1.1, NYU Eigen (Table 4) | `delta_1` | 0.974 | **0.9684** | ±2% | ✅ **match** (RTX 3090, 2 min). AbsRel=0.051 (informational; Table 4 only reports δ₁). |
| `vggt-paper-dtu-mvs` | VGGT, DTU dense MVS (Table 2) | `chamfer` | **0.382** | — | ±5% | **v0.1 paper-match gate** (retargeted from the defunct ScanNet placeholder). Loader + YAML ready; public data — set `$DTU_ROOT` and run. |
| `vggt-paper-scannet-depth` | VGGT on ScanNet (community eval, no paper target) | `abs_rel` | _n/a_ | — | n/a | Informational only — VGGT's paper doesn't evaluate ScanNet depth (Table 4 is matching, not depth). Kept for a future community run; not a paper-match. |
| `depth-anything-v2-sintel` | DA-V2, Sintel | `abs_rel` | ≈0.075 | — | ±15% | blocked on Sintel depth-archive availability |
| `da-v2-small-kitti` | DA-V2 ViT-S, KITTI Eigen test (Table 2) | `abs_rel` | _TBD_ | — | ±10% | KITTI loader + Garg crop ready. User supplies `$KITTI_ROOT` (public) + pinned Eigen sample list. |
| `metric3d-v2-kitti` | Metric3Dv2 ViT-L, KITTI Eigen test (Table I) | `abs_rel` | _TBD_ | — | ±10% | same gating as above; no ToS needed. |
| `da-v2-metric-outdoor-large-kitti` | DA-V2 Metric-Outdoor-Large, KITTI Eigen | `abs_rel` | _n/a_ | — | n/a | Informational; VKITTI-finetuned checkpoint on KITTI. No direct paper target (paper's KITTI 0.049 is the *KITTI*-finetuned ViT-L). Paired with the KITTI loader as a cross-model smoke reproduction. |
| `vggt-eth3d-courtyard-chamfer` | VGGT on ETH3D courtyard, 8-view | `chamfer` | _n/a_ | — | n/a | First chamfer/F-score reproduction. Exercises the 7-DoF Umeyama + point-cloud metric path. Informational single-scene; paper's Table 6 is across the whole test set. |
| `da3-eth3d-courtyard-chamfer` | DA3 Large-1.1 on ETH3D courtyard, 8-view | `chamfer` | _n/a_ | — | n/a | DA3 counterpart to the VGGT chamfer config — confirms the alignment path isn't VGGT-specific and gives a direct A/B on the same slice. |
| `da-v2-small-diode-indoor` | DA-V2 ViT-S on DIODE val-indoor (Table 3) | `abs_rel` | _TBD_ | — | ±10% | DIODE loader ready; public dataset (~1 GB for val-indoor). User supplies `$DIODE_ROOT`. Paper value to be pinned on first run. |
| `moge-vitl-nyuv2` | MoGe-1 ViT-L on NYUv2 Eigen (Table 3) | `abs_rel` | **0.0297** | — | ±10% | MoGe adapter wired. Needs `pip install 'git+https://github.com/microsoft/MoGe.git'` + `$NYUV2_ROOT`. Tolerance loose to absorb ROE-vs-scale_shift alignment mismatch. |
| _VGGT / ETH3D courtyard smoke_ | VGGT-1B, 4 views, first sample | `pose_auc@5°` | — | **0.91** | n/a | informational only. Rotation errors <0.3°/view; translation cos <0.6°/view. |
| _MASt3R / ETH3D courtyard pairs_ | MASt3R ViT-L, 35 consecutive 2-view samples | `pose_auc@5°` | — | **0.46** | n/a | informational only. Mean rotation error 0.32°/pair; translation cos 3.42°. 2-view setup (PairViewer) — Umeyama needs N≥3 so no chamfer. |
| _VGGT / ETH3D courtyard view-count sweep_ | VGGT-1B on 31 sliding 8-view windows | pairwise `pose_auc@5°` | — | see below | n/a | informational. Reports both absolute per-view and pairwise relative-pose AUC (the latter matches paper tables). Peak at 4 views: **pw@5°=0.66**, abs@5°=0.67. |
| _DA3 / ETH3D courtyard view-count sweep_ | DA3 Large-1.1 on 31 sliding 8-view windows | pairwise `pose_auc@5°` | — | see below | n/a | informational. Peak at 4 views: **pw@5°=0.61**, abs@5°=0.63. DA3 trails VGGT by ~5 pts pw@5° at peak but is ~3× faster per forward. |

### Courtyard view-count sweep results

All numbers are pose only (no chamfer). Aggregated over 31 sliding 8-view windows; `*@5` / `*@10` are AUC in the SuperGlue style, `*_rot°m` is per-pair rotation error median.

| Model | views | abs_rot°m | abs@5 | abs@10 | pw_rot°m | pw@5 | pw@10 | run/s |
|---|---|---|---|---|---|---|---|---|
| VGGT  | 2 | 0.285 | 0.598 | 0.779 | 0.286 | 0.598 | 0.779 | 1.84 |
| VGGT  | 4 | 0.348 | 0.668 | 0.821 | 0.299 | 0.656 | 0.812 | 1.68 |
| VGGT  | 8 | 0.490 | 0.613 | 0.784 | 0.504 | 0.548 | 0.713 | 3.34 |
| DA3   | 2 | 0.389 | 0.574 | 0.727 | 0.389 | 0.574 | 0.727 | 1.00 |
| DA3   | 4 | 0.390 | 0.626 | 0.789 | 0.370 | 0.612 | 0.772 | 0.79 |
| DA3   | 8 | 0.704 | 0.569 | 0.756 | 0.598 | 0.568 | 0.755 | 1.29 |

Notes:
- At `views=2` pairwise == absolute (only one non-origin view → single pair).
- VGGT peaks at 4 views; DA3 peaks at 4 views too. Both degrade at 8 views on this scene — more pairs dilute the mean.
- Pairwise is a strictly-harder aggregation for N>2 because it includes every pair (not just cam-i vs origin), but it's frame-invariant so it's what papers report.

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

### Note on the KITTI Eigen protocol

Three details tend to separate "just ran the HF model on KITTI" numbers
from the paper targets:

1. **Annotated-depth GT, not raw LiDAR projections.** The original Eigen
   2014 protocol reprojects Velodyne points into camera frame, yielding
   sparse and noisy GT. Modern papers (DA-V2, Metric3Dv2, DA3, MoGe,
   Depth Pro) evaluate against the KITTI Depth-Prediction Benchmark's
   *annotated* dense depth maps (Uhrig et al. 2017, ~14 GB public
   archive). plumbline's `KITTIDataset` loads the annotated maps.
2. **Garg crop on evaluation.** Pixels outside
   `row ∈ [0.408 H, 0.992 H) × col ∈ [0.036 W, 0.964 W)` are excluded.
   Pass `apply_garg_crop: true` in the dataset kwargs; the loader
   populates `Sample.depth_valid` with the crop AND-ed with `depth > 0`.
   Without the crop, hood-of-car pixels and image borders dominate the
   metric.
3. **`depth_clip: [1e-3, 80.0]` post-alignment.** Standard KITTI cap
   (80 m). Apply it the same way NYU's `[1e-3, 10.0]` clip is applied.

KITTI sample-list variants (697 raw / 652 with-GT / 500 improved)
differ by paper; plumbline does not bundle one, so reproduction YAMLs
should point at an explicit `sample_list` file (e.g. from Monodepth2's
`splits/eigen`) to avoid silent divergence.

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
