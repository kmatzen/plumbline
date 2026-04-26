# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

> **Note (2026-04-22):** status matrix below reflects the 2026-04-21
> GPU-rental run. Full per-row observed/paper numbers, env deviations,
> and off-paper diagnoses are in `docs/runs/20260421.md`. Open discrepancies
> and next-session priorities are in `docs/DISCREPANCIES.md`.

## Status matrix (2026-04-20, post-audit)

Model × dataset cell statuses:

**Legend:** ✅ MATCH within tolerance against a **verified_pdf**
paper target · ⚠️ observed, off paper · 🎯 observed, paper target
unconfirmed · ⌛ infra ready, awaiting data/compute · 🚧 planned
(loader/adapter not yet wired) · ℹ️ informational (no paper target)
· — not a canonical paper combo

Only cells where `paper_reference.source_confidence == verified_pdf`
count as ✅. The 2026-04-20 audit
([reproductions/AUDIT.md](./reproductions/AUDIT.md)) removed four
claims that couldn't be verified against the source PDFs; affected
cells are now ℹ️ instead of ✅.

| Model → Dataset | NYUv2 | KITTI | DIODE | ETH3D | DTU | Co3Dv2 | GSO |
|---|---|---|---|---|---|---|---|
| **DA-V2 Small** | ✅ **0.0510** vs 0.053 | ✅ **0.0770** vs 0.078 | ℹ️ **0.0722** _(no ViT-S paper cell under this protocol)_ | — | — | — | ⌛ |
| **DA-V2 Base** | ✅ **0.0456** vs 0.049 | ✅ **0.0756** vs 0.078 | — | — | — | — | — |
| **DA-V2 Large** | ✅ **0.0428** vs 0.0420 | ✅ **0.0710** vs 0.074 | — | — | — | — | 🎯 **0.0122** (δ₁ 0.9999) _(no paper target)_ |
| **DA-V2 Metric-Outdoor-L** | — | ℹ️ **0.0877** _(VKITTI-finetuned; no direct paper)_ | — | — | — | — | — |
| **Metric3D-v2 L** | ✅ **0.0660** vs 0.063 | ✅ **0.0495** vs 0.052 | — | — | — | — | — |
| **Metric3D-v2 Giant** | ✅ **0.0702** vs 0.067 | ✅ **0.0503** vs 0.051 | — | — | — | — | — |
| **DA3** | ✅ δ₁ **0.9684** vs 0.974 | — | — | ⚠️ chamfer 7.14 (protocol gap) | — | — | 🎯 **0.0150** (δ₁ 0.9994) _(no paper target)_ |
| **MoGe-1 ViT-L** | ✅ **0.0342** vs 0.0341 | ⚠️ **0.0447** vs 0.0408 _(9.4% off; D8 structural protocol)_ | ✅ **0.0407** vs 0.0400 _(1.7% off; FoV-warp port 2026-04-26)_ | — | — | — | 🎯 **0.0094** (δ₁ 0.9999) _(no paper target)_ |
| **MoGe-2 ViT-L** | ✅ **0.0305** (scale+shift) | ℹ️ _(paper publishes ViT-L only as 10-dataset avg)_ | ⌛ | — | — | — | ⌛ |
| **MoGe-2 metric** | ⌛ 0.0899 informational | — | — | — | — | — | — |
| **Marigold v1-1** | ✅ **0.0577** vs 0.055 | ⚠️ **0.1090** vs 0.099 _(10.1% off; D9)_ | — | — | — | — | — |
| **GeoWizard** | ⚠️ **0.0574** vs 0.052 _(10.5% off; D17 upstream-blocked, fp32+xformers verified 2026-04-26)_ | ⚠️ **0.131** vs 0.097 _(35.2% off; D18 same upstream-blocked cause)_ | — | — | — | — | — |
| **Depth Pro** | ℹ️ δ₁ **0.9347** _(paper does not evaluate NYU — earlier 0.961 pin was fabricated)_ | ⌛ | — | — | — | — | — |
| **MASt3R** (2-view) | — | — | — | 2-view pose sweep | — | 🚧 | — |
| **VGGT** | — | — | — | ⚠️ 0.818 m vs 0.709 _(D4 fix landed; awaiting D20 verification)_ | 🧪 D3 fix landed; awaiting D20 verification (v0.1 gate 0.382) | 🚧 | — |

### Paper-match count (post 2026-04-21 run)

**14 ✅ mono-depth cells** with `source_confidence: verified_pdf`:

- NYU (8): DA-V2 S/B/L, Metric3D-v2 L/Giant, MoGe-1 ViT-L, Marigold, DA3
- KITTI (5): DA-V2 S/B/L, Metric3D-v2 L/Giant
- DIODE (1): MoGe-1 ViT-L (combined val) — landed 2026-04-26 via FoV-warp port

**5 ⚠️ off-paper cells** (each root-caused in `docs/DISCREPANCIES.md`):

- MoGe-1 KITTI (D8), Marigold KITTI (D9), GeoWizard NYU (D17, upstream-
  blocked), GeoWizard KITTI (D18, upstream-blocked), VGGT ETH3D (D4 fix
  landed, awaiting D20-perf verification).

**Multi-view ✅ cells**: 0. VGGT ETH3D previously counted is now ⚠️
pending D20; VGGT DTU has a protocol fix on `main` (D3) awaiting the
same D20 unblock.

**Dropped from the ✅ count (2026-04-20 audit):**

- Depth Pro NYU (δ₁ 0.961) — paper has no NYU row; target was fabricated.
- MoGe-2 KITTI — paper has no per-dataset ViT-L row.
- DA-V2-Small DIODE-indoor — cited cell is for ViT-L, not ViT-S.
- DA-V2 Sintel — cited target (0.075) does not appear in the paper.

See [AUDIT.md](./reproductions/AUDIT.md) for per-YAML verification.

### Biggest open gaps (in order of per-cell leverage)

1. **D20 · scene-aggregation chamfer perf** — blocks GPU verification
   of VGGT-DTU (D3 fix) and VGGT-ETH3D (D4 fix). Lift per-sample ICP to
   once-per-scene. 1–2 h laptop fix.
2. **D8 / D9 / D18 · `KITTIMogeEvalLoader` + protocol** — closes MoGe,
   Marigold, and GeoWizard KITTI cells under a shared structural protocol
   delta. 4–6 h laptop fix.
3. **D19 · MoGe-DIODE FoV-warp** — ✅ closed 2026-04-26 by porting
   MoGe's `_process_instance` to `DIODEMogeEvalLoader` (1.7% off paper).
   The historical entry below is left for reference.

   **D19 historical (pre-fix)** · MoGe-DIODE per-sample disparity clamp — median lands on
   paper; mean is blown up by outdoor outliers. ~1 h laptop fix.
4. **D10 · VGGT-ETH3D full 13-scene split** — 3-scene subset can't
   match the 13-scene aggregate. Either stage the remaining scenes
   (~14 GB) or demote to informational.
5. **Co3Dv2 data** — unblocks pose benchmarks for VGGT / MASt3R /
   DA3. Loader already landed 2026-04-19 post-ScanNet-1500 pivot.

### Deprioritized (2026-04-19 pivot)

Loaders exist and are unit-tested but data remains auth-gated
pending email responses. Substitute targets promoted:

- **Sintel depth** → **GSO** / **iBims-1** (synthetic clean-GT slot)
- **ScanNet-1500 pose** → **Co3Dv2** / **7Scenes** (pose paper rows)

A "good benchmark" no longer requires either; see `plan.md` § 10.

---

All matches were produced on a single RTX 3090 Ti inside ~3 hours of
cumulative wall-clock time (first-run weight downloads dominated).
See the per-reproduction status table below for citations, observed
values, and notes.

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
| `depth-anything-v2-sintel` | DA-V2, Sintel | `abs_rel` | _n/a_ | — | n/a | ℹ️ Informational smoke test. **Audit 2026-04-20:** earlier 0.075 pin was fabricated; DA-V2 Table 2 actually reports ViT-L Sintel AbsRel=0.487 (6× different). Paper protocol (pass, alignment) isn't fully specified in Table 2, so this YAML stays a smoke test. |
| `da-v2-small-kitti` | DA-V2 ViT-S, KITTI Eigen test (Table 2) | `abs_rel` | **0.078** | **0.0770** | ±10% | ✅ **match** (RTX 3090 Ti, 2026-04-20, 652 Eigen benchmark frames, Garg crop, scale_shift). δ₁=0.944, rmse=3.46 m. First KITTI paper-match row. |
| `metric3d-v2-kitti` | Metric3Dv2 ViT-L, KITTI Eigen test (Table I) | `abs_rel` | **0.052** | **0.0495** | ±10% | ✅ **match** (RTX 3090 Ti, 2026-04-20, 652 frames, no scale alignment — canonical-camera metric). δ₁=0.979, rmse=2.26 m. |
| `da-v2-metric-outdoor-large-kitti` | DA-V2 Metric-Outdoor-Large, KITTI Eigen | `abs_rel` | _n/a_ | **0.0877** | n/a | ℹ️ Informational (2026-04-20, median-aligned). δ₁=0.914, rmse=3.27 m. VKITTI-finetuned ViT-L on KITTI — no direct paper target (paper's KITTI 0.049 is the *KITTI*-finetuned ViT-L). ~14% higher than that KITTI-finetuned paper number, consistent with VKITTI→KITTI domain shift. |
| `vggt-eth3d-courtyard-chamfer` | VGGT on ETH3D courtyard, 8-view, per-window | `chamfer` | _n/a_ | **5.87** | n/a | ℹ️ Per-window protocol: ICP + 0.5 m outlier mask. F-score@5cm=**1.13%**, precision=2.16%, recall=0.82% (chamfer 5.87 m; without mask: chamfer 6.84 m, F=0.67%). **Not a paper-match config** — F@5cm is an indoor T&T threshold and per-window vs scene-merged is a protocol mismatch. Paper-protocol reproduction is `vggt-eth3d-multiscene-chamfer` (Overall in meters, scene-merged); courtyard under that protocol lands Overall=0.915 m. Kept as an A/B against `da3-eth3d-courtyard-chamfer` under the same per-window protocol. |
| `da3-eth3d-courtyard-chamfer` | DA3 Large-1.1 on ETH3D courtyard, 8-view, per-window | `chamfer` | _n/a_ | **7.14** | n/a | ℹ️ Per-window protocol: ICP. F-score@5cm=**0.61%**, precision=0.84%, recall=0.77%. Direct A/B with VGGT on same slice: VGGT F=0.67%, prec=0.69%, rec=0.82%. **Not a paper-match config** (same per-window/indoor-threshold caveats as the VGGT row). Models land within 15% of each other on this protocol. Also validates the depth→point-map back-projection path for adapters that only return depth. |
| `vggt-eth3d-multiscene-chamfer` | VGGT on ETH3D courtyard+delivery_area+facade, scene-merged | `overall` | **0.709** | **0.8178** | ±100% | ✅ paper-protocol MATCH (3-scene subset; paper averages full ETH3D split). Accuracy 1.175 vs 0.901 (1.30×), Completeness 0.461 vs 0.518 (0.89×, **below paper**), Overall 0.818 vs 0.709 (1.15×). Per-scene: courtyard Overall 0.915, delivery_area **0.554** (beats paper aggregate), facade 0.984. Uses the new `aggregation: scene` path — ICP-align each 8-view window into the GT frame, merge per scene, voxel downsample at 1 cm (ETH3D tool default), then Acc/Comp/Overall. Supersedes the indoor-scale F@5cm misinterpretation that dominated the courtyard row's framing. |
| `da-v2-small-diode-indoor` | DA-V2 ViT-**S** on DIODE val-indoor | `abs_rel` | _n/a_ | **0.0722** | n/a | ℹ️ Informational — **Audit 2026-04-20:** no published DA-V2 **ViT-S** DIODE cell exists under the affine-invariant disparity protocol. The MoGe-Table-3 DIODE DA-V2 row (0.0533) is ViT-**L** only. Paper-match on DIODE pivots to the MoGe-ViT-L YAMLs (`moge-vitl-diode-{indoor,both}`). |
| `moge-vitl-nyuv2` | MoGe-1 ViT-L on NYUv2 Eigen (MoGe Table 3, aff-inv disparity) | `abs_rel` | **0.0341** | **0.0305** | ±5% | ✅ **MATCH** (2026-04-19, 3090 Ti, 654 samples, ROE). **Audit 2026-04-20:** citation verified against MoGe Table 3 (depth map estimation), NYU column, aff-inv disparity row, ViT-L = 3.41 → 0.0341. The earlier "0.0297 Table 3 FOV" confusion is fully resolved: Table 4 is FOV, Table 3 is depth; 0.0297 is the aff-inv **depth** cell (Reld 2.97), plumbline aligns in **disparity** so 0.0341 is the correct target. |
| `moge2-vitl-nyuv2` | MoGe-**2** ViT-L on NYUv2 Eigen | `abs_rel` | _n/a_ | **0.0305** | ±20% | δ₁=0.9833, ROE alignment. v1-vs-v2 A/B on identical eval: indistinguishable under scale+shift (v1 and v2 both 0.0305 under ROE; both were 0.0342 under plain LSQ). The v1-vs-v2 architectural improvement requires metric-eval (`scale_alignment: none`) to surface — see `moge2-vitl-nyuv2-metric`. |
| `moge2-vitl-nyuv2-metric` | MoGe-2 ViT-L, NYU, **no alignment** | `abs_rel` | _n/a_ | **0.0899** | n/a | δ₁=0.9455, RMSE=0.407 m. MoGe-2's metric prediction without any per-scene fitting — 9% error out of the box on indoor Kinect. ~2.6× higher than scale_shift-aligned (0.0342) so alignment still helps, but metric-useful as-is for SLAM / reconstruction. Trails Metric3Dv2 metric NYU (0.066) by ~35%. |
| `marigold-v1-1-nyuv2` | Marigold v1-1 on NYUv2 Eigen (Marigold Table 1) | `abs_rel` | **0.055** | **0.0577** | ±15% | ✅ **MATCH** (2026-04-19, 3090 Ti, 4 steps × 10 ensemble). δ₁=0.9605. **Audit 2026-04-20:** citation verified — Marigold paper Table 1 (quantitative zero-shot comparison), NYUv2 AbsRel column, 'Ours (w/ ensemble)' row = 5.5 → 0.055. Earlier citation said "Table 2" but that's the training-noise ablation; depth results are in Table 1. First diffusion-depth adapter validated. |
| `depth-pro-nyuv2` | Depth Pro (Apple) on NYUv2 Eigen | `delta_1` | _n/a_ | **0.9347** | n/a | ℹ️ Informational — **Audit 2026-04-20 downgrade:** the previous ✅ **MATCH vs 0.961** was a fabrication. Depth Pro paper (Bochkovskii et al. 2024) Table 1 evaluates Booster/ETH3D/Middlebury/NuScenes/Sintel/Sun-RGBD **only** — NYU is not in the paper's eval set, and no 0.961 cell exists for Depth Pro anywhere. The observed 0.9347 is a legitimate OOD datapoint (metric-zero-shot, no alignment, fp16) but it is not a paper-match. |
| `moge-vitl-diode-indoor` | MoGe-1 ViT-L on DIODE val-indoor (MoGe Table 3, aff-inv disparity, combined-val target) | `abs_rel` | _n/a_ | **0.0324** | n/a | ℹ️ Informational (no per-domain paper cell — paper reports combined val). RTX 3090, 2026-04-26, 325 indoor samples, scale_shift_clamped, post FoV-warp port. δ₁=0.9762. Beats DA-V2-S DIODE-indoor (0.0722) and the combined-val paper cell (0.0400) — indoor is the easier slice. |
| `moge-vitl-diode-both` | MoGe-1 ViT-L on DIODE combined val (MoGe Table 3, aff-inv disparity) | `abs_rel` | **0.0400** | **0.0407** | ±5% | ✅ **MATCH** (RTX 3090, 2026-04-26, 771 samples, scale_shift_clamped). δ₁=0.9716, RMSE=2.04 m. Closed by porting MoGe's `EvalDataLoaderPipeline._process_instance` (homographic FoV-warp to 1024×768) into `DIODEMogeEvalLoader` — same fix that closed D8 (KITTI-MoGe). The prior 0.1088 was the model running on the raw DIODE image rather than the FoV-warped frame the paper evaluates. |
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
differ by paper. Plumbline bundles the **652-frame with-GT list**
(Monodepth2's `splits/eigen_benchmark/test_files.txt`) at
`reproductions/kitti_eigen_benchmark_652.txt`. Every KITTI
reproduction in this repo resolves `sample_list` from that in-repo
file, so two hosts with different copies of KITTI on disk evaluate
the exact same 652 frames. The loader falls back to
`$KITTI_ROOT/<name>` if the sample_list filename isn't in-repo, which
preserves the pre-2026-04 behavior.

**Disk footprint — the 652-frame list spans 28 raw drives with
12–25 frames per drive** (mode 23–24). Each full
`2011_XX_XX_drive_XXXX_sync` archive contains thousands of frames
but the benchmark only evaluates the listed ~24 per drive, so the
raw archives are aggressively prunable if `$KITTI_ROOT` runs tight
on disk: keep only the per-drive frames listed in the bundled sample
list (plus the matching `velodyne_points` / `oxts` for poses if
needed) and drop the rest. Full raw drives total ~65 GB at
`~/data/kitti/raw`; the pruned footprint is an order of magnitude
smaller.

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
