# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

> **Note (2026-05-03):** status matrix below reflects all GPU runs
> through 2026-04-27. Open discrepancies and next-session priorities
> are in `docs/DISCREPANCIES.md`. Per-YAML paper-citation audit (now
> 29 paper-pinned YAMLs, 15 verified after the 2026-05-23 MASt3R
> direct-PDF read) is in
> [`reproductions/AUDIT.md`](./reproductions/AUDIT.md).

## Status matrix (2026-05-03)

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
| **DA-V2 Large** | ✅ **0.0428** vs 0.0420 | ✅ **0.0710** vs 0.074 | ✅ **0.0529** vs 0.0533 _(landed 2026-04-27 via DIODE FoV-warp loader)_ | — | — | — | 🎯 **0.0122** (δ₁ 0.9999) _(no paper target)_ |
| **DA-V2 Metric-Outdoor-L** | — | ℹ️ **0.0877** _(VKITTI-finetuned; no direct paper)_ | — | — | — | — | — |
| **Metric3D-v2 L** | ✅ **0.0660** vs 0.063 | ✅ **0.0495** vs 0.052 | — | — | — | — | — |
| **Metric3D-v2 Giant** | ✅ **0.0702** vs 0.067 | ✅ **0.0503** vs 0.051 | — | — | — | — | — |
| **DA3** | ✅ δ₁ **0.9684** vs 0.974 | — | — | ⚠️ chamfer 7.14 (protocol gap) | — | ⌛ informational only (paper does not evaluate CO3Dv2 pose; A/B vs VGGT/MASt3R) | 🎯 **0.0150** (δ₁ 0.9994) _(no paper target)_ |
| **MoGe-1 ViT-L** | ✅ **0.0342** vs 0.0341 | ⚠️ **0.0447** vs 0.0408 _(9.4% off; D8 structural protocol)_ | ✅ **0.0407** vs 0.0400 _(1.7% off; FoV-warp port 2026-04-26)_ | — | — | — | 🎯 **0.0094** (δ₁ 0.9999) _(no paper target)_ |
| **MoGe-2 ViT-L** | ✅ **0.0305** (scale+shift) | ℹ️ _(paper publishes ViT-L only as 10-dataset avg)_ | ⌛ | — | — | — | ⌛ |
| **MoGe-2 metric** | ⌛ 0.0899 informational | — | — | — | — | — | — |
| **Marigold v1-1** | ✅ **0.0577** vs 0.055 | ⚠️ **0.1090** vs 0.099 _(10.1% off; D9)_ | — | — | — | — | — |
| **GeoWizard** | ⚠️ **0.0574** vs 0.052 _(10.5% off; D17 upstream-blocked, fp32+xformers verified 2026-04-26)_ | ⚠️ **0.131** vs 0.097 _(35.2% off; D18 same upstream-blocked cause)_ | — | — | — | — | — |
| **Depth Pro** | ℹ️ δ₁ **0.9347** _(paper does not evaluate NYU — earlier 0.961 pin was fabricated)_ | ⌛ | — | — | — | — | — |
| **MASt3R** (N-view post-2026-04-27) | — | — | — | 2-view pose sweep | — | ⌛ AUC@30 target **0.818** (Table 3 verified_pdf, awaiting GPU run) | — |
| **VGGT** | — | — | — | ⚠️ 0.642 m vs 0.709 _(D4 per-view-masked landed, 9.4% under paper on 3-scene; D10 needed for full split)_ | ⚠️ 0.756 m vs 0.382 mm _(D3 upstream-blocked: PatchmatchNet filter + fp32 verified no-op, residual ~2× is in public VGGT-1B output)_ | ⌛ AUC@30 target **0.882** (Table 1 verified_pdf, awaiting GPU run) | — |
| **CUT3R** _(video + unordered)_ | ⌛ AbsRel target **0.086** (Table 1 verified_pdf; per-frame median scaling; protocol-diff + GPU pending) | ⌛ AbsRel target **0.092** (Table 1 verified_pdf; same caveat) | — | — | — | ℹ️ recurrent/online — handles ordered video & unordered sets | — |

### Paper-match count

**16 ✅ mono-depth cells** with `source_confidence: verified_pdf`:

- NYU (8): DA-V2 S/B/L, Metric3D-v2 L/Giant, MoGe-1 ViT-L, Marigold, DA3
- KITTI Eigen+Garg (5): DA-V2 S/B/L, Metric3D-v2 L/Giant
- KITTI MoGe-eval (2): MoGe-1 ViT-L (D8 close), DA-V2 ViT-L (2026-04-27)
- DIODE (2): MoGe-1 ViT-L, DA-V2 ViT-L (FoV-warp loader, 2026-04-26/27)

Each cell is verified against the source PDF (table + col + row) per
`reproductions/AUDIT.md`.

**Multi-view ✅ cells**: 0 (paper-match). Two structurally-correct
reproductions on `main`:
- VGGT-DTU (D3) — protocol port complete; ~2 × residual gap declared
  upstream-blocked 2026-04-27 (PatchmatchNet filter + fp32 + Jensen
  toolkit + 49-view all verified ~no-ops).
- VGGT-ETH3D (D4) — per-view-masked path lands Overall 0.642 m on a
  3-scene subset (9.4 % under paper 0.709); apples-to-apples needs
  the full 13-scene split (D10).

**Pose ✅ cells**: 0. Infra landed 2026-04-27; **2 paper-targets
pending GPU run**:

- VGGT CO3Dv2 (Table 1, AUC@30 = 0.882) — `vggt_co3dv2_pose.yaml`,
  `source_confidence: verified_pdf`. Paper cell verified via WebFetch
  2026-05-03.
- MASt3R CO3Dv2 (Table 3, mAA(30) = 0.818, RRA@15 = 0.946, RTA@15 =
  0.919) — `mast3r_co3dv2_pose.yaml`. Paper cell **verified by direct
  PDF read 2026-05-23** (D23 resolved): `arxiv.org/pdf/2406.09756`
  Table 3 row (b) MASt3R CO3Dv2 = 94.6 / 91.9 / 81.8, matching the
  YAML exactly. GPU run is the only thing left before it counts as ✅.

`Co3Dv2VGGTPoseEvalLoader` (41-cat / 10-seq / 10-frame seeded recipe)
+ MASt3R N-view via `PointCloudOptimizer` (N≥3) are tested but have
zero GPU validation as of 2026-05-03. Treat them as untested infra
until the GPU run lands. DA3 has an informational companion
(`da3_co3dv2_pose.yaml`) with no paper target.

**Off-paper / upstream-blocked cells** (each root-caused in
`docs/DISCREPANCIES.md`):

- VGGT-DTU (D3), GeoWizard NYU (D17), GeoWizard KITTI (D18),
  Marigold-KITTI (D9 / D22) — all promoted to **upstream-blocked**.
  The adapter and protocol audits are exhausted; residual gap is in
  the public checkpoint or a paper-private eval config. Cells stay
  as ⚠️ in the matrix; the YAMLs ship on `main` because the protocol
  shape is correct.

**Dropped from the ✅ count (2026-04-20 audit):**

- Depth Pro NYU (δ₁ 0.961) — paper has no NYU row; target was fabricated.
- MoGe-2 KITTI — paper has no per-dataset ViT-L row.
- DA-V2-Small DIODE-indoor — cited cell is for ViT-L, not ViT-S.
- DA-V2 Sintel — cited target (0.075) does not appear in the paper.
- (2026-04-27) "MASt3R Table 5 on 7-Scenes" claim in `seven_scenes.py`
  docstring — MASt3R does not evaluate 7-Scenes for pairwise pose.

### Biggest open gaps (in order of per-cell leverage)

1. **CO3Dv2 GPU run** — converts the only two pending pose
   reproductions (VGGT Table 1, MASt3R Table 3) from ⌛ to ✅ or
   surfaces a real gap. Gates the pose half of the v0.1 release.
2. **D10 · VGGT-ETH3D 13-scene full split** — closes D4's
   3-vs-13-scene caveat. Either stage ~14 GB and run, or formally
   demote to "3-scene informational subset".
3. ~~**D23 · MASt3R Table 3 PDF re-verification**~~ — ✅ done
   2026-05-23. Direct PDF read of `arxiv.org/pdf/2406.09756` Table 3
   confirmed CO3Dv2 row (b) MASt3R = 94.6 / 91.9 / 81.8, matching
   `mast3r_co3dv2_pose.yaml` exactly. Only the GPU run remains.

### Closed-blocked (do not retry without an upstream change)

D3 (VGGT-DTU), D17 / D18 / D9 / D22 (GeoWizard, Marigold KITTI). All
five hit the same wall: adapter + protocol audits exhausted, residual
gap is in the public release. Re-enter the queue if/when upstream
releases an updated checkpoint or eval script.

### Deprioritized (2026-04-19 pivot)

Loaders exist and are unit-tested but data remains auth-gated:

- **Sintel depth** → substituted by **GSO** / **iBims-1** (synthetic clean-GT slot).
- **ScanNet-1500 pose** → substituted by **Co3Dv2** / **7Scenes** (pose paper rows).

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

## Per-YAML detail

Per-YAML observed values, paper citations, and audit status live in
[`reproductions/AUDIT.md`](./reproductions/AUDIT.md). Each YAML's
own `notes:` field carries run-specific detail (RTX-3090 wall-clock,
δ₁/RMSE companions, alignment mode).

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
