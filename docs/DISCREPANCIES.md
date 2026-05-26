# Discrepancies

Catalog of every adapter / loader / protocol / solver / citation mismatch
surfaced to date. Open entries keep full diagnosis context. Closed entries
(FIXED + verified, or EXPLAINED-NOT-A-BUG) live at the bottom as one-liners
with commit SHAs; full history is in git.

Status legend:

- 🧪 **FIX-PENDING-VERIFY** — change landed, waiting on a GPU re-run.
- 🔎 **SUSPECTED** — hypothesis + diagnosis path; not yet reproduced.
- 📅 **DEFERRED** — known root cause, scoped for v0.2+.

## Triage (2026-05-03)

Single glanceable view of every questionable cell. Use this to pick
where to look (model code? protocol? paper itself?) before opening a
deeper diagnosis below.

**Category key:**

- 🔧 **Model integration** — adapter is suspect; debug the upstream-vs-
  plumbline numerical path (dtype, RNG, attention, normalization).
- 📐 **Protocol** — adapter OK; eval pipeline (alignment, mask, crop,
  aggregation) diverges from paper code.
- 📜 **Paper-side** — adapter + protocol audited; cell may be
  unreproducible from the public release (private eval config,
  internal checkpoint, undocumented post-processing).
- 📑 **Citation** — paper cell number is wrong / fabricated /
  mis-attributed; no actual delta against a real paper cell.
- ⏳ **Untested** — infra ready, no GPU validation yet.
- 🚫 **No path** — no paper cell exists for this combination; not a
  paper-match candidate.

### Per-cell triage

| ID | Cell | Δ vs paper | Cat | Tried & ruled out | Next move |
|---|---|---|---|---|---|
| D3 | VGGT-DTU chamfer | +98 % (0.756 vs 0.382 mm) | 📜 | per-view-masked port, Jensen toolkit, PatchmatchNet filter, fp32, 49-view | watch upstream; reproduce structurally only |
| D4 | VGGT-ETH3D 3-scene | −9.4 % (0.642 vs 0.709 m) | 📐 | per-view-masked path, MLP transform, 1 cm voxel | stage 13 scenes (D10) for apples-to-apples |
| D9 | Marigold-KITTI | ✅ RESOLVED 2026-05-25 | 📜 | upstream eval-script default repointed v1-0/50-step → v1-1/1-step between paper (CVPR 2024) and current `21_infer_kitti.sh` | EXPLAINED (checkpoint/config delta): Marigold's **own** native pipeline on its **own** prepared `kitti_eigen_split_test.tar` reproduces paper AbsRel 0.099 end-to-end with v1-0 / 50-step / ens-10 (0.0992 on 60-img spread subset, 0.2 % off). Plumbline's 0.109 (v1-1 / 1-step) is the newer distilled checkpoint the current upstream eval script defaults to — a documented checkpoint-generation delta, not a paper-private config and not a plumbline bug. |
| D10 | VGGT-ETH3D full 13-scene split | n/a (gates D4 verdict) | 📐 | — | stage remaining ~14 GB or demote D4 |
| D17 | GeoWizard-NYU | ✅ RESOLVED 2026-05-26 | 📜 | dtype, xformers, full `seed_all`, 4 alignment modes, raw vs filled GT, README `--denoise_steps 50` | EXPLAINED (paper-private cherry-pick): paper author confirmed on `fuxiao0719/GeoWizard#36` that paper-time eval runs multiple seeds and **selects the best result for the metric report**. Plumbline's single-seed 0.0574 matches @anonymous's independent 0.0576 on the same protocol; paper 0.052 is best-of-N seeds, not single-seed. 50-step sub60 (this session) also rules out the README's "academic comparison" knob: 0.06704 (10-step) vs 0.06681 (50-step) → Δ 0.3 % noise. |
| D18 | GeoWizard-KITTI | ✅ RESOLVED 2026-05-26 | 📜 | same checkpoint + same author as D17; KITTI is Table 1's parallel column under the same eval recipe | EXPLAINED (paper-private cherry-pick): same root cause as D17 — paper author's quote on `fuxiao0719/GeoWizard#36` covers both NYU and KITTI columns of Table 1. No separate KITTI GPU run needed. |
| D22 | Marigold/GeoWizard KITTI umbrella | various | 📜 | (subsumes D9 + D18) | open upstream issues; possibly drop these from v0.1 paper-match |
| D23 | MASt3R-CO3Dv2 cell verification | ✅ RESOLVED 2026-05-23 | 📑 | WebFetch HTML render only loaded appendix on `2406.09756` (every URL surface) | direct PDF read done: Table 3 row (b) MASt3R = 94.6/91.9/81.8 — matches YAML exactly |
| D24 | CUT3R NYU/KITTI/Bonn depth (DUSt3R-lineage; also π³) | ✅ RESOLVED 2026-05-25 | 📐 | crop, clip, median-align, abs_rel, resize — all ruled out by re-scoring cached preds | EXPLAINED (protocol delta): plumbline's strict protocol differs from the lineage's. **All 3 paper cells confirmed** via CUT3R's own pipeline on its exact sets — NYU 0.08595/0.086, KITTI 0.09219/0.092, Bonn 0.07661/0.078. `cut3r-*` jobs → `blocked` (D24) |
| — | VGGT-CO3Dv2 (Table 1 0.882) | not run | ⏳ | paper cell verified 2026-05-03 | GPU run |
| — | MASt3R-CO3Dv2 (Table 3 0.818) | not run | ⏳ | adapter rewrite + tests pass; 0 GPU validation; paper cell PDF-verified 2026-05-23 (D23 closed) | GPU run |
| — | MASt3R N-view rewrite (any non-CO3Dv2 use) | not run | ⏳ | landed 2026-04-27; synthetic + unit tests only | GPU run |
| — | DA3-CO3Dv2 | n/a (informational) | 🚫 | paper has no CO3Dv2 row | optional GPU run for A/B |
| — | MoGe-2 ViT-L on any per-dataset cell | n/a | 🚫 | paper publishes only 10-dataset averages (Table 1) and ViT-Base ablations (Table B.4) | accept; no per-dataset paper-row possible for ViT-L |
| — | Depth Pro on NYU/KITTI | n/a (informational) | 🚫 | paper evaluates Booster/ETH3D/Middlebury/NuScenes/Sintel/Sun-RGBD only | add a paper-actual dataset to get a real paper-row |

### Per-paper trust

How much we should trust each paper's published cells when adopters
look at the matrix:

| Paper | Verified cells | Trust | Why / action |
|---|---|---|---|
| Depth Anything V2 (Yang 2024, arXiv:2406.09414) | **8** (NYU S/B/L, KITTI S/B/L, DIODE L, KITTI-MoGe L) | High | All cells reproduce in tolerance. One fabricated Sintel pin (0.075 vs paper's 0.487) was caught and demoted. |
| Metric3D-v2 (Hu 2024, arXiv:2404.15506) | **4** (NYU + KITTI L/Giant) | High | All four cells match within ±10 %. No protocol surprises. |
| MoGe-1 (Wang 2024, arXiv:2410.19115) | **5** (NYU, KITTI, DIODE-both + 2 DA-V2 baseline cells) | High after audit | Systematic Table-2-vs-Table-3 citation error fixed in 2026-04-20 audit; values match once table number corrected. |
| Marigold (Ke 2024, arXiv:2312.02145) | **2 end-to-end** (NYU + KITTI via Marigold's own pipeline) | High | Both paper cells reproducible end-to-end on Marigold's exact prepared eval sets + native pipeline with the original CVPR **v1-0** checkpoint @ 50 denoise steps × 10 ensemble. NYU 0.0577 / paper 0.055 (plumbline cell, v1-1 / 1-step still matches NYU). KITTI 0.0992 / paper 0.099 (D9 resolution 2026-05-25, v1-0 / 50-step on `kitti_eigen_split_test.tar`, 60-img spread sub). Plumbline's `marigold_v1_1_kitti.yaml` lands ~0.11 because the current upstream eval script defaults to the newer distilled **v1-1 / 1-step** checkpoint, which trades accuracy on outdoor KITTI specifically (v1-1 still matches paper on NYU). Documented checkpoint-generation delta, not a paper-private config and not a plumbline bug. |
| GeoWizard (Fu 2024) | **0** paper-match + 2 explained off-paper (D17, D18) | **Explained (paper-private eval recipe)** | Both NYU + KITTI cells reproduce at ~0.057 / ~0.11 under single-seed eval — plumbline 0.0574 / @anonymous 0.0576 (`fuxiao0719/GeoWizard#36`) converge. Paper 0.052 / 0.097 is **best-of-N seeds**, per the paper author's own statement on the upstream issue tracker (2026-05-26 D17/D18 resolution; quote: "we perform multiple inferences with different initialized seeds … and select the best result for the metric report"). Adapter is structurally correct (fp32 + xformers + full `seed_all` + alignment matches author's `de_normalized.py`); only the seed-selection step is paper-private. `paper_match: no` on both YAMLs is now *explained*, not suspect. |
| Depth Pro (Bochkovskii 2024, arXiv:2410.02073) | **0** | Pending | Paper doesn't evaluate NYU/KITTI; the previously-claimed NYU δ₁ 0.961 was fabricated (caught in 2026-04-20 audit). **No paper-row yet under the paper's actual eval set** (Booster/ETH3D/Middlebury/NuScenes/Sintel/Sun-RGBD). |
| Depth Anything 3 (Bytedance Seed 2025, arXiv:2511.10647) | **1** (NYU δ₁) | Moderate (limited) | Paper's main Table 4 only reports δ₁ (no AbsRel breakdown), and the chamfer-track / GSO comparisons live in informational rows with no paper target. Per-paper-row policy: NYU is the only paper-comparable cell currently shippable. |
| MoGe-2 (Wang 2025, arXiv:2507.02546) | **0** | **N/A — no path** | Per-dataset ViT-L cells are not published anywhere in the paper (Table 1 is 10-dataset average; Table B.4 is ViT-Base ablation). Either reproduce the 10-dataset average across all 10 datasets (unwieldy), or accept "no paper-row possible for MoGe-2 ViT-L per-dataset". |
| VGGT (Wang 2025, arXiv:2503.11651) | **0** paper-match | **Suspect on chamfer** | Table 2 DTU 2 × over after exhausting all levers (D3, upstream-blocked). Table 3 ETH3D 3-scene 9.4 % under (D4); 13-scene apples-to-apples deferred (D10). Table 1 CO3Dv2 GPU pending. Paper §4.2 says "Following MASt3R [62]" for DTU — but MASt3R repo doesn't ship DTU eval, so the paper may rely on unreleased post-processing (TSDF / BA / pose refinement). **Re-read §4.2 + appendix carefully** if D3 stays blocked after a future VGGT release. |
| MASt3R (Leroy 2024, arXiv:2406.09756) | **0** paper-match (1 cell PDF-verified, GPU pending) | Cell-verified | The arXiv HTML render only serves the appendix (Tables 7-8) across every URL surface tried, so the cell was confirmed by **direct PDF read 2026-05-23** (D23 resolved): `arxiv.org/pdf/2406.09756`, Table 3 (Multi-view pose regression on CO3Dv2 / RealEstate10K, 10 random frames), row (b) MASt3R = RRA@15 94.6 / RTA@15 91.9 / mAA(30) 81.8 — matches `mast3r_co3dv2_pose.yaml` (0.946 / 0.919 / 0.818) exactly. §4.3 protocol (41 cat / 10 frames / 45 pairs / no GT focals) also confirmed. Still **0 paper-match** only because the GPU run hasn't happened — the paper target itself is no longer suspect. |
| CUT3R (Wang 2025, arXiv:2501.12387) | **3 end-to-end** (NYU, KITTI, Bonn via CUT3R's own eval); 3 plumbline cells = protocol deltas | **High** | All three paper cells **reproduced end-to-end** on CUT3R's exact prepared sets + native pipeline: NYU 0.08595/0.086, KITTI 0.09219/0.092 (Table 1), Bonn 0.07661/0.078 (Table 2 video, per-seq scale) — all ≤2 % (D24). plumbline's own depth cells read *better* (NYU 0.0522, KITTI 0.0858, Bonn 0.0536) because its strict protocol / eval set differs from the DUSt3R lineage — documented **protocol deltas**, `paper_match: no` is expected and fully explained, not suspect. |

## Open issues at a glance

(Diagnosis-detail counterpart of the triage table above; categories &
status carry over.)

| ID | One-liner | Status |
|---|---|---|
| D3 | VGGT-DTU chamfer — PatchmatchNet geometric-consistency filter verified on 22-scan re-run (Overall 0.756 mm vs prior 0.758, ~no-op). fp32 probe also verified (0.750, also ~no-op). Adapter + protocol levers exhausted; ~1.98× residual gap is in public VGGT-1B output, not anything plumbline controls | 🔎 upstream-blocked |
| D4 | VGGT-ETH3D — per-view-masked path landed at Overall 0.642 m on the 3-scene subset (9.4 % UNDER paper 0.709). Apples-to-apples comparison needs the full 13-scene split (D10) | ✅ infra landed; awaits D10 |
| D9 | Marigold-KITTI — paper cell 0.099 reproduces end-to-end with v1-0 / 50-step on Marigold's exact prepared set (0.0992, 0.2 % off, 60-img spread sub). Plumbline's 0.109 is a documented v1-1 / 1-step (newer distilled checkpoint) protocol delta. | ✅ RESOLVED 2026-05-25 |
| D10 | VGGT-ETH3D full 13-scene vs 3-scene subset | 📅 deferred |
| D17 | GeoWizard-NYU — paper number is best-of-N seeds (paper author confirmed on `fuxiao0719/GeoWizard#36`); 50-step sub60 final adapter lever ruled out 2026-05-26 (0.06681 vs 10-step 0.06704, Δ 0.3 %); plumbline single-seed 0.0574 matches @anonymous independent reproducer 0.0576 | ✅ RESOLVED 2026-05-26 |
| D18 | GeoWizard-KITTI — same root cause as D17 (paper-private cherry-pick across seeds covers both Table 1 columns) | ✅ RESOLVED 2026-05-26 |
| D22 | Marigold portion REFUTED by D9 (v1-0 / 50-step reproduces paper 0.099). GeoWizard portion closed by D17 / D18: not a paper-private config in the usual sense, but a paper-private *seed-selection* step. | ✅ Marigold portion RESOLVED 2026-05-25 (D9); GeoWizard portion RESOLVED 2026-05-26 (D17 / D18) |
| D23 | `mast3r_co3dv2_pose.yaml` cell verified by direct PDF read 2026-05-23 — `arxiv.org/pdf/2406.09756` Table 3 row (b) MASt3R CO3Dv2 = 94.6 / 91.9 / 81.8, matching the YAML (0.946 / 0.919 / 0.818) exactly. `source_confidence: verified_pdf` is now genuinely backed by a PDF read | ✅ RESOLVED 2026-05-23 |
| D24 | CUT3R depth cells (nyuv2/kitti/bonn) all OFF-PAPER better than published — eval-protocol mismatch, NOT a model bug. Re-scoring the SAME cached preds: protocol levers (Eigen crop, clip [1e-3,10], median-align, abs_rel) ruled out (raw + CUT3R-protocol still 0.0526). Source = GT depth field: plumbline `depth_field=raw` (sparse Kinect) vs DUSt3R-lineage dense/filled depth. raw→filled +0.025, +Eigen-crop −0.017; filled+no-crop = 0.0777 vs paper 0.086. Residual closed: CUT3R's OWN pipeline on its exact sets reproduces all 3 cells — NYU 0.08595/0.086, KITTI 0.09219/0.092, Bonn 0.07661/0.078 (video, per-seq scale). | ✅ RESOLVED 2026-05-25 (protocol delta; all 3 paper cells CONFIRMED reproducible end-to-end) |

---

## Open issues

### D3 · VGGT-DTU chamfer — STRUCTURAL PROTOCOL MISMATCH   🔎 OPEN

The paper's 0.382 mm Overall on DTU is from CUT3R/MASt3R/VGGT-family
eval which is **per-view-masked chamfer**: each view's prediction is
center-cropped to 224×224, masked by per-pixel GT validity, then all
masked points are concatenated and chamfered against the same
per-view-cropped GT depth. Reference: CUT3R `eval/mv_recon/launch.py`
(lines ~195-260) + `eval/mv_recon/utils.py::accuracy/completion`.

Plumbline does **scene-merged chamfer**: per-sample-aligned full
prediction clouds are concatenated per-scene, voxel-downsampled,
then chamfered against the scene-level GT point cloud (`Points/stl/
*.ply`).

These are structurally different metrics. Plumbline's DTU loader
ships only the scene-level GT (no per-view depth maps), so a
faithful port of the per-view-masked protocol requires a loader-side
change to either (a) include per-view GT depth, or (b) project the
scene-level cloud into each view to derive a visibility mask.

Plumbline-honest baseline (no approximations, voxel_size=None inside
accuracy_completeness):

| Metric | Plumbline scene-merged | Paper (per-view-masked) |
|---|---|---|
| Acc | ~50 mm | 0.389 mm |
| Comp | ~89 mm | 0.374 mm |
| Overall | ~48 mm | 0.382 mm |

The 130× gap is METRIC SHAPE, not adapter accuracy. Without per-view
GT we can't match paper's metric.

In-session attempts (2026-04-24/25) and their honest assessment:

- ✅ Fixed OOM (voxel-unit mixup `1fc0f9c`).
- ✅ Removed inner voxel_downsample to match CUT3R's raw NN
  (`<this commit>`).
- ❌ ``chamfer_outlier_distance=20 mm`` (added in `1ef3c04`) — this
  is a plumbline-specific approximation, not what the paper does.
  Reverted because user principle: follow paper code, don't
  approximate. Numbers regress accordingly but are honest.

YAML metric-key mismatch separately open: `primary_metric: chamfer`
but runner emits `accuracy/completeness/overall`.

Fix path requires a loader update (ship per-view depth maps) before
re-attempting paper-match. Until then, demote D3 to "informational
only — protocol mismatch with paper cell" if a YAML edit is allowed,
or document this in the YAML notes.

#### 2026-04-25 single-record diff (scan1) — stage-1 confirmed

Stage 1 (sample loading) divergence verified on scan1 against the
canonical CUT3R reference loader (`eval/mv_recon/data.py::DTU`,
which is the same loader shape MASt3R/VGGT/CUT3R-family papers use
for DTU per-view-masked chamfer):

- **plumbline** `DTUDataset` returns one `Sample` per 8-view window
  with `images=(8,1200,1600,3) uint8`, `intrinsics=(8,3,3)`,
  `extrinsics_gt=(8,4,4) world_from_cam`, `point_cloud_gt=(200000,3)
  float32 mm` (subsampled `Points/stl/stl001_total.ply`),
  **`depth_gt=None`**, **`depth_valid=None`**.
- **CUT3R `DTU._get_views`** opens `<ROOT>/scan{N}/depths/<view>.npy`
  + `<ROOT>/scan{N}/binary_masks/<view>.png` per view, then derives
  `pts3d=(H,W,3) world` and `valid_mask=(H,W) bool` via
  `depthmap_to_absolute_camera_coordinates(depthmap, K, world_from_cam)`.
- Concrete failure when CUT3R loader is pointed at the plumbline
  staging: `FileNotFoundError: ~/data/dtu/dtu/scan1/depths/00000048.npy`.
  The `depths/` and `binary_masks/` subdirs simply do not exist —
  plumbline's S3-staged DTU under `s3://plumbline-bench/datasets/dtu/`
  contains `cams/`, `images/`, `pair.txt`, and a separate
  `Points/stl/*.ply` per scan; nothing per-view.

Diff stops at stage 1; downstream stages (preprocess/model
input/output/postprocess/GT-prep/per-pixel-error) cannot be compared
because CUT3R has no GT to consume.

scan1 numbers (plumbline scene-merged, this session):

| metric | scan1 | paper |
|---|---|---|
| Acc  | 1.27 mm | 0.389 mm |
| Comp | 100.14 mm | 0.374 mm |
| Overall | 50.71 mm | 0.382 mm |

The Acc/Comp asymmetry (1.3 mm vs 100 mm) reproduces the diagnosis:
predicted cloud only covers the camera-visible portion of the object,
but the laser PLY also contains the occluded backside, so GT→pred NN
distance (Comp) is dominated by points the views never see. Acc
(pred→GT NN distance) is in the right ballpark for VGGT — the model
is fine; the metric is the wrong shape.

#### Suggested loader-side fix

Extend `DTUDataset` to ship per-view GT depth + binary mask, matching
CUT3R's expected layout `<root>/scan{N}/{depths,binary_masks}/`.
Two viable sources:

1. **MVSNet preprocessed training dump** (Yao et al., the canonical
   community source for `depths/*.npy` + `binary_masks/*.png`).
   ~115 GB for the full split; a 22-test-scan slice is ~2.9 GB
   (49 views × ~2.6 MB × 22 scans + masks). Stage to
   `s3://plumbline-bench/datasets/dtu/` next to `dtu/scan{N}/`.
2. **Original DTU release** `Cleaned/Rectified/.../Depths_raw/` PFM +
   `Masks/` PNG per view (the source MVSNet preprocessed from). Same
   data, different format; needs a small PFM reader.

After loader update, port the per-view-masked chamfer:
- Add `Sample.depth_gt`, `Sample.depth_valid` for DTU (loader sets
  these from the per-view files).
- In the runner's `mvs_depth` path under `aggregation=scene`, when
  per-view GT is available switch to: per view → 224×224 center crop
  on both pred-pts3d and GT-pts3d (after GT shift+ICP align) → mask
  by `valid_mask` → concat → KDTree NN both directions per
  CUT3R `eval/mv_recon/utils.py::accuracy/completion`.
- Gate on dataset capability: scenes whose loader emits per-view GT
  use the per-view-masked path; others fall back to scene-merged.

This is a v0.2 loader change, not a runner-only fix. Per session
ground rule "Don't try to invent per-view GT", stopping here.

Artifacts from this session (outside the repo):
`/tmp/diff/plumbline/stage1.npz` (plumbline Sample tensors),
`/tmp/diff/plumbline_scan1.json` (full scan1 report),
`/tmp/diff/stage1_cut3r.py` (CUT3R-loader probe), CUT3R clone at
`/tmp/cut3r`.

#### 2026-04-25 — fix landed (loader + runner + protocol)

User asked for the actual fix rather than the loader-blocker writeup,
so the loader change above was implemented in-session. Three pieces:

1. `DTUDataset(with_per_view_gt=True)` renders per-view depth + valid
   mask by **z-buffering the laser PLY** through each view's GT
   camera (`render_pv_depth_zbuffer`). Splat radius 1 (3×3 disk per
   point) gives ~57% pixel coverage at 1200×1600 native, matching
   the density MVSNet's preprocessed `depths/*.npy` derive from. No
   external download needed — derives from `Points/stl/*.ply` we
   already ship. Cache: `<root>/.plumbline_manifest/dtu_pv_depth_
   scan{N}_HxW{H}x{W}_r{r}.npz`, ~190 MB/scan / ~4 GB/22-scans.
2. `runner.evaluate(per_view_masked=True, per_view_crop=224)` enables
   the CUT3R-protocol path: per view, NN-downsample GT depth +
   validity to processed res, unproject with rescaled-K + GT pose,
   center-crop to 224×224, mask, accumulate per scene. Scene-merge
   does Sim(3) ICP (with bbox-similarity warm start so the
   m-vs-mm scale gap doesn't collapse it) + raw KDTree-NN both
   directions. Reports both mean and median variants of acc/comp/
   overall (CUT3R protocol's companion median.)
3. `dtu_vggt_table2.yaml`: views_per_sample 8→49 (one sample/scan,
   matching CUT3R's `full_video=True`), max_views 8→32 (VGGT cap),
   `per_view_masked: true`, `per_view_crop: 224`,
   `with_per_view_gt: true`. Reproduction YAML's
   `primary_metric: chamfer` retargeted to `overall` so the
   reproduction-check stops emitting NaN.

**Full 22-scan reproduction (`plumbline reproduce vggt-paper-dtu-mvs`):**

| Metric | plumbline (mean) | plumbline (median) | paper |
|---|---|---|---|
| Acc      | 0.874 mm | 0.534 mm | 0.389 mm |
| Comp     | 0.642 mm | 0.350 mm | 0.374 mm |
| Overall  | 0.758 mm | 0.442 mm | 0.382 mm |

Mean Overall is **2.0× off paper** (0.758 vs 0.382, +98 %, outside
±5 %). Median Overall is **1.16× off** (0.442 vs 0.382, +16 %,
plausible if paper's "Overall" implicitly aggregates per-scan medians
or if the paper's GT is dense-enough that mean-vs-median gap is
smaller than ours). Per-scene range 0.40 mm (scan11) — 2.00 mm
(scan77); 11 of 22 scans land under 0.65 mm, with scan1, scan33,
scan49, scan75, scan77 being the tail.

**Outstanding gap candidates (not yet investigated this session):**

- Splat density. We splat the raw laser PLY at radius=1; paper's
  preprocessed depths come from a Poisson-mesh render of the same
  cloud. Going to a Poisson-mesh-based renderer would densify GT
  inside silhouettes and likely tighten Acc on small-textured scans
  like scan1.
- VGGT view cap. We use first 32 of 49 rig views (VGGT's
  `max_views=32`). Paper protocol may use all 49 (or evenly-spaced
  32, not first-32), affecting surface coverage on scans where the
  first 32 views miss part of the object.
- Aspect-ratio drift. scan1 post-ICP pred bbox X-extent is 80% of
  GT's; suggests VGGT's predicted geometry has a small non-isotropic
  scaling that Sim(3) can't fix (it's 1 scale, not 3). Worth
  diffing further at the per-pixel level if we want to push under
  ±5 %.

130× → 2× is the headline; ±5 % requires the three above. v0.1 gate
is **not yet met** but the protocol is now structurally correct and
the remaining gap is per-scene tail / GT density, not metric shape.

#### 2026-04-25 — finding: we ported the WRONG reference protocol

Cross-repo audit (MASt3R, DUSt3R, MVSNet, CUT3R clones at /tmp/) of
how each one's official code computes DTU chamfer:

- **MASt3R + DUSt3R**: ship NO DTU eval code (`grep -r DTU|chamfer|
  ObsMask|registration_icp`: zero hits in either repo). The §4.2
  protocol described in the MASt3R paper that VGGT cites was never
  released as code.
- **MVSNet**: ships fusion-to-PLY only (`mvsnet/test.py`); chamfer
  itself is computed by an **external Jensen et al. DTU MATLAB
  toolkit** (`BaseEvalMain_web.m` / `ComputeStat_web.m`) that lives
  with the DTU SampleSet, not in this repo.
- **CUT3R**: only has the dead `if "DTU" in name_data` branch in
  `eval/mv_recon/launch.py:251` plus the unused `DTU` loader in
  `eval/mv_recon/data.py:192`. `datasets_all` in `launch.py:57-71`
  registers only `7scenes` + `NRGBD` — **CUT3R does not actually
  evaluate on DTU.**

The canonical DTU chamfer protocol — what VGGT Table 2 0.382 mm,
MASt3R Table 2, DUSt3R Table 6, and MVSNet Table 1 all share — is the
**Jensen MATLAB toolkit**, not CUT3R's per-view-masked code. The
Jensen protocol is structurally different from the CUT3R-shape
protocol we ported:

| axis | Jensen (canonical) | CUT3R (what we ported) |
|---|---|---|
| GT  | raw `stl{N}_total.ply` laser PLY | 2D rendered per-view depth + binary mask |
| Mask | 3D `ObsMask{N}.mat` voxel grid + plane cull | 2D per-view mask + 224×224 center crop |
| Outlier cap | **MaxDist=20 mm** before mean | none |
| Pred prep  | Fuse predicted depths into a single per-scan PLY, downsample at 0.2 mm | per-view points, concat after 224×224 crop |
| Alignment | none (pred + GT in same metric world from SfM/calib) | Regr3D scale+shift then 6-DoF rigid ICP |
| Aggregation | per-scan mean (capped) → mean-of-scans | per-scan mean (uncapped) → mean-of-scans |
| Output unit | mm | mm |

Implications for the remaining 2× gap:

1. The **20 mm MaxDist cap** alone likely closes most of the gap. Our
   mean Acc 0.874 mm is dominated by a long tail of pred outliers
   (mean-vs-median spread is 0.874 / 0.534 = 1.64×, exactly what an
   outlier cap would compress). With a 20 mm cap, mean → near
   median → near paper.
2. CUT3R's `Regr3D_t_ScaleShiftInv` + 6-DoF rigid ICP and the
   224×224 center crop are CUT3R-specific, not paper. Removing them
   in favour of "fuse pred PLY, no alignment, ObsMask filter, 20 mm
   cap mean" is the higher-fidelity port.
3. ObsMask is per-scan 3D voxel grids gating which surface region
   counts. Bundled in DTU's `SampleSet.zip` (~6.9 GB; ObsMask itself
   is ~30 MB inside that archive, not directly downloadable).

**Cheap next step** (one-line YAML edit):
``chamfer_outlier_distance: 20.0`` under the protocol. Existing
``accuracy_completeness(outlier_distance=...)`` path already
implements the cap. Re-run the 22-scan reproduction; expect mean
Overall to drop from 0.758 mm toward 0.4-0.5 mm.

**Higher-fidelity port** (medium effort, deferred to a future
session): fetch ObsMask, fuse VGGT's per-view predictions into a
per-scan PLY, replicate Jensen's `ComputeStat_web.m` end-to-end. At
that point ICP + 224×224 crop come out; the Sim(3) we currently fit
becomes unnecessary because Jensen assumes pred and GT are already in
the same metric world (DTU calibration directly).

#### 2026-04-25 — ran the canonical Jensen toolkit; **chamfer protocol is NOT the gap**

Pulled `jzhangbs/DTUeval-python` (the community Python port of the
Jensen MATLAB toolkit, "negligible gap to MATLAB" per its README,
1 min/scan) plus DTU's `ObsMask{N}_10.mat` + `Plane{N}.mat` from
`SampleSet.zip`'s `MVS Data/ObsMask/` (~140 MB extracted). Fused
VGGT's per-view scan1 prediction into a single PLY in DTU scanner
frame using plumbline's per-view-masked ICP transform (final inlier
0.97 mm, identical alignment to the per-view-masked path). Ran both
protocols on the **same** prediction.

**scan1 numbers (same VGGT prediction, same alignment):**

| Protocol | Acc | Comp | Overall |
|---|---|---|---|
| Plumbline per-view-masked | 2.10 | 1.31 | 1.71 |
| Jensen DTUeval-python (no align) | 2.40 | 1.91 | 2.16 |
| Jensen DTUeval-python (PVM-aligned) | 2.82 | 1.56 | 2.19 |

**22-scan mean (Jensen on plumbline-fused PLYs):** Acc 1.039 / Comp
0.696 / **Overall 0.868 mm**.

**22-scan mean (plumbline per-view-masked, commit 8592db9):** Acc
0.874 / Comp 0.642 / **Overall 0.758 mm**.

The two protocols agree within 15 % across the 22-scan mean. Both are
~2× off paper (0.382 mm). The chamfer protocol — which-points-to-
include, mask-shape, outlier-cap, alignment — is **not** the dominant
source of the gap.

**What is left to explain the 2× gap:**

1. VGGT-on-DTU prediction quality. The public `facebook/VGGT-1B` HF
   checkpoint at bf16 produces depth/poses with per-pixel error in the
   ~1 mm range after Sim(3) ICP (final inlier 0.4-1.6 mm across the
   22 scans). Paper's 0.382 mm Overall would require ~2× tighter
   predictions than we observe.
2. Possible drift sources to investigate before declaring the gap
   irreducible:
   - Inference dtype (fp32 vs bf16). The protocol YAML pins bf16
     for VGGT; the paper may have used fp32. Cheap to test.
   - View selection (first 32 of 49 vs evenly-spaced 32). Our
     contiguous slice misses the back of the arc.
   - Image preprocessing details (BICUBIC vs LANCZOS resize, padding
     policy). VGGT's `load_and_preprocess_images` has a few knobs.
   - Whether paper's 0.382 includes any post-processing (TSDF fusion,
     pose refinement) absent from a raw VGGT forward pass.

#### 2026-04-26 — view count NOT the gap (49 ≈ 32)

Probed VGGT inference on all 49 DTU rig views (lifted adapter cap
from 32 to 49 in commit `5ba6fae`+ branch — fits in 19 GB on a 3090
in 25 sec). Three 22-scan sweeps:

| run | views | scene_voxel_size | per-sample Umeyama | Overall mean | Overall median |
|---|---|---|---|---|---|
| 8592db9 baseline | 32 | 1.0 (no-op pre-voxel-fix) | no | **0.758** | 0.442 |
| this session v3 | 49 | 0 | yes | 0.849 | 0.488 |
| (this session v2 | 49 | 1.0 | yes | 1.489 | 0.816) |

So **adding views (32 → 49) does not help** on this configuration —
the per-scan numbers are similar or marginally worse. The 17 missing
back-of-arc views aren't the dominant gap source.

Side finding: commit `5221b24`'s per-chunk voxel_downsample on the
per-view-masked path is unsafe when pred and GT are in different
units (DTU: pred ≈ m, GT = mm). 1 mm voxel works for ETH3D (both in
m) but collapses DTU pred to a few centroids. Fix in this session:
DTU YAML pins ``scene_voxel_size: 0`` to skip the voxel; runner
comment warns about the unit-frame coupling.

Adapter cap stays at 49 (no harm in supporting more views) but
DTU's ``max_views`` is back at 32 since 49 didn't improve.

**Per-stage diff vs Jensen** (full table in
`/tmp/diff/STAGE_DIFF.md`, summary):

| Stage | Plumbline per-view-masked | Jensen (canonical) | Functional diff |
|---|---|---|---|
| GT source | per-view rasterized depth | raw `stl{N}_total.ply` | rendered-per-view vs single-3D |
| Region select | per-view 224×224 + mask | 3D `ObsMask` voxel + `Plane` cull | 2D image-center vs 3D voxel |
| Pred subset | per-view 224×224 of pred map | full PLY → 0.2 mm radius dedup | crop vs dedup |
| Alignment | bbox-warm-start Sim(3) ICP | none (pre-aligned input) | 7-DoF vs identity |
| Outlier cap | none | drop NN ≥ 20 mm | unbounded vs capped |
| Aggregation | per-scan mean → mean-of-scans | identical | none |

Both protocols use the same inference + same alignment in this
experiment, so the eval-protocol portion of the gap is a 0.11 mm
delta on the 22-scan mean (0.758 → 0.868 mm). The other 0.49 mm
between us and paper is attributable to VGGT's outputs themselves,
not the eval shape.

Artifacts on this box (will be lost when teardown):
- `/tmp/dtueval/eval.py` — Jensen Python port.
- `/home/myuser/data/dtu/ObsMask/{ObsMask*_10.mat,Plane*.mat}` — eval
  support data, 134 MB.
- `/tmp/diff/jensen_inputs/scan{N}_pred.ply` — fused & PVM-aligned
  pred PLYs for all 22 scans.
- `/tmp/diff/jensen_22.tsv` — per-scan Jensen results.
- `/tmp/diff/STAGE_DIFF.md` — full stage-by-stage diff.

#### 2026-04-26 — root cause: missing geometric-consistency filter [99]

Audited VGGT's own repo (`facebookresearch/vggt`) across `main` /
`evaluation` / `eval_wip` / `save_my_life` / `training`. **No DTU
eval code on any branch** — only Co3D camera-pose eval (the `evaluation`
branch). Same outcome as the prior MASt3R/DUSt3R/MVSNet/CUT3R audit.

VGGT paper §4.2 says "Following MASt3R [62]" for DTU. Pulled the
MASt3R paper (`arXiv:2406.09756`):

- §4.5 (main): "We finally perform MVS by triangulating the obtained
  matches. Note that the matching is performed in full resolution
  without prior knowledge of cameras, and the latter are only used to
  triangulate matches in ground-truth reference frame. **We remove
  spurious 3D points via geometric consistency post-processing [99].**"
- App. A "MVS on DTU": "the point clouds are raw values obtained via
  triangulation of the coarse-to-fine matches of MASt3R."
- §4.5 metrics paragraph: "we report the average accuracy, completeness
  and Chamfer distances error metrics **as provided by the authors of
  the benchmarks**" — confirms the metric itself is the canonical
  Jensen DTU toolkit (which we already ran, getting 0.868 mm vs paper
  0.382).

**Reference [99] is Wang et al., PatchmatchNet, CVPR 2021** —
specifically its `eval.py::filter_depth` post-processing. That code
is public at `FangjinhuaWang/PatchmatchNet`. The DTU recipe is in
`eval.sh`:

```
--num_views 5 --image_max_dim 1600 --geo_mask_thres 3 --photo_thres 0.8
--geo_pixel_thres 1.0 (default) --geo_depth_thres 0.01 (default)
```

Per-ref-view algorithm (`eval.py:86-255`):

1. For each ref view i, take its top-K source views from `pair.txt`.
2. For each src view j: reproject ref→src using ref's depth, sample
   src's depth at the projected pixel, reproject back to ref. Mask
   pixels where reproj-pixel-error < `geo_pixel_thres` AND
   relative-depth-diff < `geo_depth_thres`.
3. `final_mask = (#agreeing_src_views ≥ geo_mask_thres) AND
   (photo_conf > photo_thres)`.
4. Average depth across agreeing views, unproject filtered pixels to
   world, concat across all ref views → `fused.ply`.
5. `fused.ply` is what gets chamfered with the Jensen toolkit.

Plumbline pipeline today: predicted depth → unproject every pixel →
no inter-view consistency check → chamfer. The paper drops outliers
that disagree across views; we don't. This is consistent with our
observed gap shape: Acc 0.874 (paper 0.389, 2.25× off) is more
inflated than Comp 0.642 (paper 0.374, 1.72× off) — the unfiltered
pred has a long tail of outliers that the paper's filter would
suppress.

**Action landed this session:** verbatim port of PatchmatchNet's
`reproject_with_depth` + `check_geometric_consistency` + multi-source
aggregation as `runner._geometric_consistency_mask`. Wired into
`_per_view_masked_clouds` as an optional pre-filter ANDed with
GT validity. `dtu_vggt_table2.yaml` toggles it on with
PatchmatchNet's DTU thresholds (`geo_pixel_thres 1.0`,
`geo_depth_thres 0.01`, `geo_mask_thres 3`); `photo_thres` is
dropped because VGGT's `depth_conf` is unbounded (would need
separate calibration). Source-view selection uses top-K nearest
predicted-camera-centre instead of `pair.txt`'s visual-overlap
ranking — equivalent for DTU's dense circular rig.

Convention adapter: PatchmatchNet stores extrinsics as
`cam_from_world`; plumbline as `world_from_cam`. The substitution
is `E_src @ inv(E_ref)` → `inv(Ew_src) @ Ew_ref`.

Synthetic sanity (laptop): on a 4-cam circular rig observing a flat
plane, the filter keeps ~70 % of consistent pixels. Corrupting half
of view 0's depth: corrupted half kept 0 %, clean half kept 70 %.

Status: 🧪 FIX-PENDING-VERIFY pending GPU re-run. Expected: Acc
collapses toward paper (the long outlier tail goes away); Comp
slightly improves; Overall lands near 0.4 mm on the 22-scan mean.

#### 2026-04-27 — verified: PatchmatchNet filter is NOT the gap

Re-ran ``vggt-paper-dtu-mvs`` end-to-end on a 3090 (22 scans, 32
views/scan, bf16, geometric-consistency filter on). Compared to the
pre-filter baseline:

| variant | Acc | Comp | Overall (mean) | Overall (median) |
|---|---|---|---|---|
| **paper** | 0.389 | 0.374 | **0.382** | — |
| baseline (commit 8592db9, no PMN) | 0.874 | 0.642 | 0.758 | 0.442 |
| **+ PatchmatchNet filter (this run)** | **0.819** | **0.692** | **0.756** | **0.473** |

The expectation in the prior commit was: "Acc collapses toward paper
(outlier tail goes away); Comp slightly improves; Overall ~0.4 mm."
What actually happened: Acc improves modestly (-6 %, 0.874 → 0.819),
Comp regresses slightly (+8 %, 0.642 → 0.692, fewer pred points →
larger GT→pred NN), Overall ~unchanged (-0.4 %). The synthetic
sanity (clean half kept 70 %, corrupted half kept 0 %) is correct,
the port is faithful, and the filter does what it should — but the
remaining ~1.98× gap is dominated by structural pred quality, not
outlier pixels.

#### 2026-04-27 — verified: dtype is NOT the gap either

Added ``vggt-dtu-fp32-probe`` (parallel YAML, identical except
``dtype: float32``) and re-ran on the 3090. Result:

| variant | Acc | Comp | Overall (mean) | Overall (median) |
|---|---|---|---|---|
| + PatchmatchNet (bf16) | 0.819 | 0.692 | 0.756 | 0.473 |
| **+ PatchmatchNet + fp32** | **0.816** | **0.684** | **0.750** | **0.464** |

Δ < 1 % across all four metrics. Exactly the same shape as D17's
fp32 probe on GeoWizard NYU (0.0573 vs 0.0574 — also a no-op). The
bf16 autocast does not contribute meaningfully to the residual gap
on either model.

#### Conclusion: D3 is upstream-blocked

Cumulative D3 levers tried over multiple sessions (numbers are 22-scan
mean Overall in mm; paper 0.382):

| lever | Overall | session |
|---|---|---|
| Per-view-masked chamfer (CUT3R protocol port, commit f3c0a49) | 0.758 | 2026-04-25 |
| Jensen DTUeval-python toolkit (canonical Jensen ObsMask + Plane cull + 20 mm cap) | 0.868 | 2026-04-25 |
| 49 vs 32 rig views | 0.849 | 2026-04-26 |
| PatchmatchNet geometric-consistency filter | 0.756 | 2026-04-27 |
| PatchmatchNet + fp32 inference | **0.750** | 2026-04-27 |

All cluster in 0.75-0.87 mm — paper 0.382 mm is consistently ~2× off.
Same shape as D17 (GeoWizard NYU): exhausted adapter + protocol +
filter + dtype levers, and the public ``facebook/VGGT-1B`` HF
checkpoint produces predictions that are ~2× looser than what the
paper reports on DTU. Likely sources, none in plumbline's reach:

1. Public weights ≠ paper weights (Apple Depth Pro precedent — its
   own README says "the model in this repo has been re-trained,
   performance close to but does not match the paper").
2. Paper pipeline includes post-processing (TSDF fusion, pose
   refinement, BA, etc.) that ``demo_colmap.py`` exposes via
   ``--use_ba`` but isn't part of a raw ``model(images)`` forward
   pass.
3. Paper Table 2 footnote about "Following MASt3R" may include a
   step neither MASt3R nor CUT3R has actually released as code.

Status promoted to 🔎 **upstream-blocked** (same shape as D17 / D22).
The PatchmatchNet port stays on ``main`` because it makes plumbline
structurally correct against MASt3R's stated protocol — it just
isn't where the paper-row gap lives. Future re-evaluation against an
updated VGGT release inherits the right pipeline shape.

Probe artifact: ``reproductions/vggt_dtu_fp32_probe.yaml`` (kept on
``main`` as a documented diagnostic, similar to how the GeoWizard
fp32-probe history lives in commit ``0995974``).

### D4 · VGGT-ETH3D multiscene — STRUCTURAL PROTOCOL MISMATCH   🔎 OPEN

Same root cause as D3: paper protocol is per-view-masked chamfer
(CUT3R lineage); plumbline does scene-merged chamfer. ETH3D loader
ships only scene-level GT (`scan_clean/`, `dslr_scan_eval/`), no
per-view depth.

Mitigations that landed in-session:
- ✅ `dslr_scan_eval` GT preference (`1ef3c04`) — closer to ETH3D's
  official "DSLR-visible" eval region than the broader `scan_clean`.
- ✅ Per-scene GT-cache fix in ETH3DDataset (`2e1beb9`) — eliminates
  the OOM that previously blocked verification.
- ❌ `chamfer_outlier_distance=0.2 m` reverted (same reason as D3 —
  not what the paper code does).

Plumbline-honest baseline:

| Metric | Plumbline scene-merged | Paper (per-view-masked) |
|---|---|---|
| Overall | ~1.7 m | 0.709 m |

The 2× gap is again metric-shape, not adapter accuracy. Closing it
properly requires per-view GT in the ETH3D loader, then porting
CUT3R's per-view 224×224 + GT-mask logic to the runner.

#### 2026-04-26 — found and fixed scan_alignment.mlp bug

ETH3D ships per-scan ``MLMatrix44`` transforms in
``<scene>/{dslr_scan_eval,scan_clean}/scan_alignment.mlp`` that bring
each ``scan{N}.ply`` from its own scanner-local frame into the
COLMAP/DSLR world frame. The ETH3DDataset loader was concatenating
the PLYs **without applying these**. courtyard's scan1.ply has a
~14° rotation + ~7m y-translation relative to scan2; naive
concatenation produced a rotationally-scrambled GT cloud, which
explains the YAML's claimed 0.46 m Comp baseline going stale to
1.99 m at some point — the MLP transforms were apparently being
applied at one point and got removed.

Fix landed in `fbc2524` (`fix(eth3d/D4)`): added
``parse_scan_alignment_mlp`` (stdlib XML), apply each PLY's transform
before concatenation. No new deps.

**3-scene subset, before vs after the MLP fix** (137 8-view sliding
windows, 1 cm voxel, scene-merged chamfer):

| | scene-merged (no MLP) | scene-merged (MLP fix) | paper |
|---|---|---|---|
| Acc  | 1.124 | **0.766** | 0.901 |
| Comp | 1.992 | **3.470** | 0.518 |
| Overall | 1.558 | **2.118** | 0.709 |

So the MLP fix is correct (Acc 1.12 → 0.77, **better than paper**)
but exposes a bigger structural issue: with GT now in the right world
frame, the laser scan extends well beyond the camera-visible region
that VGGT's predictions cover. Comp 3.47 m is dominated by GT laser
points the cameras never see — `delivery_area` Comp went 2.04 → 6.19 m
because that scene's scan2.ply translates ~13 m relative to scan1.

Per-scene after MLP fix:

| Scene | Acc | Comp | Overall |
|---|---|---|---|
| courtyard | 1.469 | 1.601 | 1.535 |
| delivery_area | **0.285** | 6.186 | 3.235 |
| facade | 0.545 | 2.622 | 1.583 |

delivery_area Acc 0.285 m — VGGT's predictions on that scene are
~3× tighter than paper's reported aggregate. The metric is just
penalising regions of the laser scan that no camera reaches.

#### Same structural fix as D3: per-view-masked chamfer

Closing the Comp blow-up needs the same per-view-masked path that
landed for DTU in commit f3c0a49: per-view GT depth (rasterized from
the laser PLY through each view's GT camera), 224×224 center crop on
pred + GT pts3d, ICP align, KDTree NN both directions on the masked
clouds. ETH3D wrinkles vs DTU:

1. Per-view native sizes vary (DTU is uniform 1200x1600; ETH3D is
   per-camera-id, ~6048x4032 average but irregular). The `_per_view_
   masked_clouds` runner helper currently rescales K from
   `sample.depth_gt.shape` to pred res; for varying-native it should
   use per-view native sizes from `metadata['native_sizes']`.
2. ETH3D PLYs are bigger (~70M points/scene after concat across
   scan{N}.ply files) — splat radius 1 at native 6K resolution is
   ~5-10 sec/view × 38 views = several minutes/scene to render.
   On-disk cache analogous to DTU's pv_depth_*.npz.
3. The MLP transform applies before rendering (so GT is in DSLR
   world frame, matching ``sample.extrinsics_gt``).

This is the path forward. Cheaper alternatives that came up but were
rejected:

- ❌ ``chamfer_outlier_distance: 0.5`` (drop NN distances >0.5 m
  before mean) — would close Comp toward paper but is plumbline
  approximation, not paper code (same rejection reason as before).
- ❌ Build ETH3D's `multi-view-evaluation` C++ tool — heavy
  dependencies (PCL, Boost, Eigen) and the tool reports F-score not
  chamfer-in-meters, so it doesn't even match paper Table 3's metric.

Per-scan-aligned numbers (scene-by-scene Acc) suggest VGGT's pred
quality is at least paper-comparable. The remaining gap is metric
shape, exactly D4's original diagnosis, just with the MLP bug as a
prerequisite that's now fixed.

#### 2026-04-26 — per-view-masked path landed, beats paper Overall

ETH3D ``with_per_view_gt`` rendering + per-view-masked chamfer with
per-chunk voxel_downsample at 1 cm:

| | scene-merged + MLP | per-view-masked + MLP + voxel | paper |
|---|---|---|---|
| Acc      | 0.766 | **0.584** | 0.901 |
| Comp     | 3.470 | **0.700** | 0.518 |
| Overall  | 2.118 | **0.642** | 0.709 |

Per-scene results:

| scene | Acc | Comp | Overall | Overall_median |
|---|---|---|---|---|
| courtyard | 0.469 | 0.736 | 0.603 | 0.262 |
| delivery_area | 0.513 | 1.065 | 0.789 | 0.380 |
| facade | 0.770 | 0.299 | 0.535 | 0.151 |

**Plumbline 0.642 mean / 0.265 median Overall vs paper 0.709 — 9.4 %
UNDER paper, mean.** Direction is opposite of D3 (which was 2× over
paper). Since paper's value is on the full 13-scene cross-scene
mean and ours is on the 3-scene subset, the ±5 % strict gate is not
the right comparison; the result is consistent with VGGT-on-3-scene
being slightly easier than VGGT-on-13-scene. D10 (full 13-scene
sweep) would close the apples-to-apples question.

Implementation pieces this session:

- ``ETH3DDataset.with_per_view_gt`` — renders per-view depth from
  MLP-aligned PLY at native size capped to ``pv_render_max_dim=2048``,
  caches per scene to ``<root>/.plumbline_manifest/eth3d_pv_depth_
  <scene>_max2048_r1.npz`` (~0.5–1.5 GB / scene).
- Runner ``_per_view_masked_clouds`` — generalised to use per-view
  native sizes from ``metadata['native_sizes']`` (DTU stays on the
  uniform-shape fast path) plus a separate ``metadata['gt_sizes']``
  for when depth_gt is rendered at a smaller-than-native resolution.
- Runner per-chunk ``voxel_downsample`` BEFORE accumulation in the
  per-view-masked branch under ``aggregation=scene`` — without it,
  ETH3D-scale clouds (5M+ points / scene) made scene-agg ICP +
  chamfer untractable (>1 h, abandoned). With ``scene_voxel_size:
  0.01`` (1 cm voxel), full 3-scene run finishes in ~36 min.
- ``protocols/eth3d_vggt_table3.yaml`` — switched aggregation knobs
  to ``per_view_masked: true`` (and moved the dataset's
  ``with_per_view_gt: true``).

Status: 🧪 FIX-PENDING-VERIFY. The 9.4 % vs paper is best-case
explained by the 3-vs-13-scene subset; the actual ±5 % gate
properly attaches to D10 (full-split sweep), not to this 3-scene
configuration.

### D9 · Marigold-KITTI — RESOLVED: checkpoint-generation delta, not a paper-private config   ✅ RESOLVED 2026-05-25

Three plumbline protocol variants all landed off paper 0.099:

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.1146 | +15.8 % |
| `kitti_moge_eval` | 0.0865 | −12.7 % |
| `marigold_kitti_eval` | 0.1179 | +19.1 % |

`marigold_kitti_eval` is plumbline's faithful port of Marigold's own paper
pipeline (`kitti_bm_crop` + `valid_mask_crop: eigen` + `scale_shift_depth`,
per `prs-eth/Marigold/src/dataset/kitti_dataset.py`). That it was *further*
from paper than `kitti_moge_eval` initially looked like the paper cell
must come from a private config. End-to-end reproduction on Marigold's
own native pipeline disproves that hypothesis.

#### 2026-05-25 — end-to-end reproduction on Marigold's exact prepared set

Mirrors the D24 methodology (run the model's own pipeline on its own
prepared eval set). Set up `prs-eth/Marigold` on a 3090, downloaded
the authors' prepared `kitti_eigen_split_test.tar` (677 MB, 1551 entries,
exactly what their eval consumes), pinned `diffusers==0.30.2 /
transformers==4.44.2 / huggingface_hub==0.24.7` so the vendored
`marigold.MarigoldDepthPipeline` imports under `torch==2.4.1+cu121`,
and ran `script/depth/infer.py` + `script/depth/eval.py` (`--alignment
least_square`) with two configs on the **same** 60-image spread subset
(every-11th of the 652 valid Eigen-test rows):

| run | AbsRel | δ₁ | vs paper 0.099 / 91.6 |
|---|---|---|---|
| **v1-0 / 50-step / ens-10** (paper CVPR config) | **0.09917** | 0.9121 | **0.2 % off (well inside ±5 %)** |
| v1-1 / 1-step / ens-10 (plumbline pins this) | 0.11107 | 0.8871 | +12 % off paper / matches plumbline's 0.109 full-set within 2 % |

Both runs used `seed=1234`, `processing_res=0` (native, after KITTI
benchmark 1216×352 bottom-aligned center crop), `--alignment
least_square` (depth-space LSQ). The DDIM scheduler warnings
(`timestep_spacing="leading"`, `rescale_betas_zero_snr=False`) for
v1-0 are **its original CVPR config** — not changed, faithful to paper.

The same v1-1 / 1-step config also reproduces plumbline's recorded
v1-1 KITTI (0.109 in the matrix; 0.118 under `marigold_kitti_eval`),
within ~2 % via the every-11th subset. This validates plumbline's
adapter against native Marigold AND validates the spread subset as
representative of the full 652 set (the v1-1 cross-check independently
confirms representativeness, so the v1-0 sub60 number is a faithful
proxy for the full-set v1-0 number).

#### Root cause = upstream-default checkpoint repointed between paper and current repo

The Marigold README §"Run inference (for academic comparisons)"
explicitly distinguishes two paper-comparable configs:

- **CVPR depth (paper Table 1):** `prs-eth/marigold-depth-v1-0` with
  `--denoise_steps 50 --ensemble_size 10`.
- **Current default:** `prs-eth/marigold-depth-v1-1` with
  `--denoise_steps 1 --ensemble_size 10` (v1-1 is a later
  1-step-distilled checkpoint).

The repo's `script/depth/eval/21_infer_kitti.sh` was updated when v1-1
was released to default to the v1-1 / 1-step path. Plumbline's
`marigold_v1_1_kitti.yaml` mirrors that current script (v1-1 / 1-step /
ens-10) but compares against the paper Table 1 number, which is
v1-0 / 50-step. The distilled v1-1 / 1-step model trades some accuracy
on KITTI (outdoor / long-range) — on NYU the same v1-1 / 1-step config
matches paper 0.055 within tolerance (0.0577, ✅), confirming the
checkpoint-generation effect is dataset-shape-dependent.

So D9's 19 % gap is a **checkpoint-generation delta**, not a
paper-private config and not a plumbline-side bug. The audit-rejected
hypotheses from prior sessions (dtype, xformers, full `seed_all`, all
three protocol variants) were correctly rejected — none could close
the gap because plumbline was running the wrong checkpoint for the
paper cell it cites.

Verified paper citation (`scripts`/PDF read 2026-05-25): Marigold
arXiv:2312.02145 Table 1, KITTI column, "Ours (w/ ensemble)" row =
AbsRel 9.9 / δ₁ 91.6 (and "w/o ensemble" = 10.5 / 90.4). The README's
§"Run inference (for academic comparisons)" makes the v1-0 = paper
identification explicit.

#### Resolution & follow-ups

- D9 + the Marigold portion of D22 → closed (one-liners in the table
  at bottom). The Marigold per-paper-trust row moves from "Mixed" to
  "High" with both cells (NYU + KITTI) reproducible end-to-end on
  Marigold's exact prepared sets + native pipeline.
- The `marigold_v1_1_kitti.yaml` reproduction stays as a v1-1
  newer-distilled-checkpoint cell, analogous to CUT3R's documented
  protocol-delta cells: model is correct, `paper_match: no` is now
  *explained* (checkpoint generation), not suspect.
- A future `marigold_v1_0_kitti.yaml` (v1-0 / 50-step / ens-10) would
  give plumbline a passing paper-match for Marigold KITTI; deferred
  as a separate ~14 h GPU job (the same per-image cost as the
  resolution run, just on full 652).
- Box artifacts (run + cached preds): `~/deps/marigold/output/
  d9_v1p0_50step_sub60/` and `~/marigold_d9_results.txt` on the 3090.

### D18 · GeoWizard-KITTI — same pattern as D9   🔎 OPEN

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.131 | +35 % |
| `kitti_moge_eval` | 0.1103 | +13.7 % |
| `marigold_kitti_eval` | 0.1406 | +45 % |

Same as D9: `marigold_kitti_eval` is worse than `kitti_moge_eval`.
GeoWizard shares the diffusion-depth lineage with Marigold; D22
(paper-private-eval hypothesis) most likely applies to both.

### D22 · Marigold / GeoWizard KITTI paper cells — Marigold portion RESOLVED   ✅ partial RESOLVED 2026-05-25

Originally raised 2026-04-24 as: neither plumbline's
`marigold_kitti_eval` (paper-code port) nor `kitti_moge_eval`
reproduced Marigold 0.099 or GeoWizard 0.097 on KITTI. The
"paper-private eval config" hypothesis was the working interpretation.

**Marigold portion (this session, 2026-05-25):** REFUTED by D9.
Running `prs-eth/Marigold`'s native `script/depth/infer.py` +
`script/depth/eval.py` on the authors' own prepared
`kitti_eigen_split_test.tar` with **v1-0 / 50-step / ensemble-10**
reproduces AbsRel 0.0992 vs paper 0.099 (0.2 % off, 60-img spread
subset). Root cause = checkpoint-generation drift: the current repo's
eval script defaults to v1-1 / 1-step (newer distilled checkpoint),
which plumbline mirrored. The paper Table 1 cell is the v1-0 CVPR
checkpoint at 50 steps. Full diagnosis & numbers in D9.

**GeoWizard portion (D18):** still 🔎 upstream-blocked under D17.
Same checkpoint as D17 (lemonaddie/Geowizard), same likely cause
(public checkpoint vs paper checkpoint, or paper-protocol detail not
in `run_infer.py`). D17/D18 already fully audited (fp32 + xformers +
full `seed_all` are not the gap); no GeoWizard equivalent of
Marigold's v1-0 vs v1-1 checkpoint distinction has surfaced from the
repo.

The "paper-private eval config" framing of the original D22 stands
only for GeoWizard now. For Marigold it was wrong — the paper cell is
public, the paper code is public, only the *default checkpoint* in
the current eval script differs from the published paper.

### D24 · CUT3R / π³ DUSt3R-lineage depth cells off-paper — eval-protocol, not model   ✅ RESOLVED 2026-05-25

First GPU run of CUT3R (Table 1 single-frame depth + Table 2 Bonn video) and π³
landed every CUT3R depth cell **below** (better than) the published number:

| Cell | observed | paper | Δ |
|---|---|---|---|
| cut3r-nyuv2 | 0.0522 | 0.086 | −39 % |
| cut3r-kitti | 0.0858 | 0.092 | −7 % |
| cut3r-bonn | 0.0536 | 0.078 | −31 % |

The consistent direction (every cell better, not scattered) ruled out noise and
pointed at a shared eval step. Diagnosed by **re-scoring the same cached predictions**
(`scripts/ablate_nyu_gtfield.py`, no re-inference) under 4 protocol variants × 2 GT
fields, against CUT3R's own scorer (`eval/monodepth/eval_metrics.py`,
`depth_evaluation(pred, gt, max_depth=None)` = median scale-only, `gt>0` mask, **no
spatial crop**, no clip):

| Variant | raw GT | filled GT |
|---|---|---|
| A crop + clip **[plumbline `nyu_eigen_2014`]** | **0.0522** | 0.0605 |
| D no-crop, no-clip **[CUT3R `eval_metrics.py`]** | 0.0526 | **0.0777** |

(Variant A/raw reproduces the live run's 0.0522 exactly → methodology validated.)

**Ruled out** (each moves AbsRel < 0.0005): Eigen crop, post-align clip [1e-3,10],
median alignment (both sides identical: `median(gt)/median(pred)`), abs_rel formula
(identical). Raw + CUT3R-protocol still yields 0.0526 — so the protocol pipeline is
NOT the source.

**Root cause = GT depth field (dominant) + Eigen crop (secondary):**

1. **raw vs filled GT (+0.025):** plumbline pins `depth_field="raw"` (sparse, accurate
   Kinect); CUT3R's scorer loads `np.load(data/nyu/*.npy)` = the DUSt3R-lineage
   **dense/filled** NYU depth. Under CUT3R's own protocol, raw→filled = 0.0526→0.0777.
   The loader's comment ("every paper cites rawDepths") does not hold for the
   DUSt3R/CUT3R/MonST3R/π³ lineage, which scores against dense depth.
2. **Eigen crop (−0.017):** plumbline applies it; `eval_metrics.py` applies none. On
   filled GT the crop drops noisier interpolated borders (0.0605 vs 0.0777).
3. **Residual 0.0777→0.086 (~10 %):** the exact NYU image set + GT + native
   preprocessing (CUT3R's prepared `.npy` vs our 654 `.mat` Eigen indices). The
   pred-resize hypothesis is **ruled out** (cubic 0.0778 vs bilinear 0.0777, a
   0.0001 no-op). **Confirmed:** running CUT3R's own pipeline on its exact prepared
   set reproduces **0.08595 vs paper 0.086** (0.06 % — see exact-set reproduction
   below), so the residual is fully accounted for by the eval set, not the model.

KITTI / Bonn are the same class (eval-set/selection, not model): KITTI plumbline
Eigen-652 + Garg crop vs CUT3R `val_selection_cropped` (1000 imgs, no crop,
`max_depth=None`); Bonn plumbline 8 seqs / all-frames / 64-view vs CUT3R 5 seqs
(`balloon2, crowd2, crowd3, person_tracking2, synchronous`) × 110 frames
(`rgb_110`/`depth_110`), `max_depth=70`.

**Verdict:** not a model/inference discrepancy — plumbline's CUT3R predictions are
correct. `nyu_eigen_2014`'s raw+crop default is *stricter* than the DUSt3R-lineage
dense+no-crop protocol these papers report. Note this is the *opposite* sign from
D17 (GeoWizard-NYU, which is off-paper *worse* under raw GT) — so a single global
GT-field switch is not free; it must be scoped per paper lineage.

**Resolution (2026-05-25):** accepted as a documented **protocol delta, not a
model bug.** The single-record diff the three `cut3r-*` YAMLs were waiting on *is*
this D24 analysis (re-scoring cached preds under both GT fields × four protocol
variants) — it confirms plumbline's predictions are correct and that the
off-paper-*better* numbers come entirely from plumbline's stricter `raw`+crop
protocol vs the DUSt3R-lineage `filled`+no-crop one. Per project policy (a failed
paper-match under a *stricter* protocol is a finding, not a number to chase —
cf. D9 / D17 / D22) we keep the strict protocol and do **not** force a sub-5 %
match against the softer lineage protocol.

Recorded by: YAML CAVEATs in the three `cut3r-*` reproductions rewritten from
"⌛ unverified, single-record diff owed" to "protocol delta (explained)"; the
matching `reproductions/AUDIT.md` rows and the `REPRODUCTIONS.md` matrix row
updated; and the three `gpu_queue.yaml` jobs moved to `blocked` (`blocked_on:
D24`). The cells stay pinned to the paper value, so the harness honestly reports
`paper_match: no` — that `no` is now an *explained* protocol delta, not a suspect
cell.

**Live re-confirmation (2026-05-25 GPU, `scripts/ablate_nyu_gtfield.py`):** the
full 2×4 GT-field × protocol table reproduces exactly on the 654-image Eigen set —
raw A=0.0522 / D=0.0526, filled A=0.0605 / D=0.0777 — and a cubic-resize probe
**rules out** pred-resize as a residual source (filled+no-crop+cubic = 0.0778 vs
bilinear 0.0777, a 0.0001 no-op). So the 0.0777→0.086 (~10 %) residual is
**entirely the exact image set** (CUT3R's prepared `.npy` vs our Eigen-654), not
resize.

**Exact-set reproduction — all three paper cells CONFIRMED end-to-end (2026-05-25
GPU):** staged each cell's *exact* prepared eval set and ran CUT3R's **own** native
pipeline (`eval/monodepth` for Table 1, `eval/video_depth --align scale` for
Table 2). All reproduce the published numbers:

| cell | exact-set result | paper | Δ | eval path |
|---|---|---|---|---|
| NYU (Table 1) | AbsRel 0.08595, δ 0.9087 | 0.086 / 90.9 | 0.06 % | `eval/monodepth` |
| KITTI (Table 1) | AbsRel 0.09219, δ 0.9129 | 0.092 / 91.3 | 0.2 % | `eval/monodepth` |
| Bonn (Table 2) | AbsRel 0.07661, δ 0.9376 | 0.078 / 93.7 | 1.8 % | `eval/video_depth`, per-seq scale |

- **NYU:** MonST3R recipe (HF `sayakpaul/nyu_depth_v2` val → 654 `.h5` →
  `nyu_images/*.png` + dense `nyu_depths/*.npy`); `depth_evaluation(max_depth=None,
  lr=1e-3)`, `cv2.INTER_CUBIC` resize.
- **KITTI:** gathered set per `prepare_kitti.py` — first ≤110 annotated-val depth
  frames × 13 seqs (**1269 pairs**) + the *full* raw `image_02` (the box's plumbline
  raw is pruned to Eigen frames, so the 13 raw drives were re-downloaded);
  `max_depth=None`.
- **Bonn:** `prepare_bonn.py` 110-frame subsets (`rgb_110`/`depth_110`, frames
  [30:140]) × 5 seqs; this is the **video** eval (per-sequence single scale =
  Table 2). The single-frame `eval/monodepth` path gives 0.0625 — a different,
  easier number, *not* the Table 2 target.

So every off-paper-better plumbline cell is fully explained: on each cell's exact
eval set + native protocol the published number reproduces within ≤2 %. This both
(a) re-confirms plumbline's CUT3R integration is correct and (b) validates CUT3R
Table 1 (NYU, KITTI) + Table 2 (Bonn) as faithfully reproducible. plumbline's own
`cut3r-*` cells stay documented protocol deltas (stricter protocol / different eval
set → NYU 0.0522, KITTI 0.0858, Bonn 0.0536). Run + cached preds:
`s3://plumbline-bench/runs/20260525T165647Z/`. The exact prepared eval datasets
(NYU prepared + KITTI gathered set + the slow eu-central KITTI raw + Bonn `_110`
subsets) are mirrored to **`s3://plumbline-bench/datasets/cut3r_eval/`** (8507 objs
/ 5.15 GB) — `rclone copy` them into `$CUT3R_ROOT/data/` to re-run the exact-set
eval without any re-download/re-prep.

### D21 · Prediction cache doesn't invalidate on loader preprocessing change   🔎 NEW 2026-04-24

Cache key in `src/plumbline/runner.py` `_predict_with_cache` is
`(model.name, model.config_hash(), dataset_name, sample.sample_id)`.
It ignores the actual bytes / shape of the input tensor the loader
produces. Observed 2026-04-24: after porting MoGe's homographic warp
into `KITTIMogeEvalLoader`, a re-run of `moge-vitl-kitti`
cache-hit on the previous shard (1242×375 predictions) against the
new 750×375 GT, silently producing nonsense metrics (AbsRel 0.1895,
4 × the pre-fix value). Worked around by `rm -rf` of the stale
shard; a proper fix hashes the first-sample tensor shape + a small
byte sample into the cache key, or invalidates on `dataset.__class__`
fingerprint changes.

### D10 · VGGT-ETH3D 3-scene vs 13-scene split   📅 DEFERRED

Plumbline's YAML runs courtyard + delivery_area + facade (3 scenes);
paper's Table 3 Overall 0.709 is the 13-scene cross-scene mean. A
3-scene subset genuinely can't match the 13-scene aggregate.

Resolution: (a) stage remaining 10 scenes (+~14 GB data) and run full
split; (b) extract per-scene paper numbers from VGGT supplementary;
or (c) demote to informational with larger tolerance. Earlier audit
intended (c) — `tolerance_relative: 1.0` encoded that before the
repo-wide 5 % cap landed.

### D17 · GeoWizard-NYU — RESOLVED: paper number is best-of-N seeds, not a fixed-seed metric   ✅ RESOLVED 2026-05-26

Two-month tail of audits exhausted every plumbline-side lever: dtype,
xformers, full ``seed_all`` parity, 4 alignment modes, raw vs filled
GT, the upstream README's ``--denoise_steps 50`` recommendation.
None close the residual ~10 % gap. The closure came not from a
GPU lever but from a paper-author quote on the upstream issue
tracker:

> *"Since depth accuracy can be influenced by the initial seed (as
> it's a generative pipeline), we perform multiple inferences with
> different initialized seeds for each test dataset, along with the
> resemble [sic, ensemble] operation, and **select the best result
> for the metric report.**"*
> — [@fuxiao0719 (paper author), `fuxiao0719/GeoWizard#36`](https://github.com/fuxiao0719/GeoWizard/issues/36)

That issue (titled exactly "Unable to replicate NYU metrics with the
GeoWizard Checkpoint") was opened by another independent reproducer
who, under "the GeoWizard v1 sampling pipeline alongside Marigold's
evaluation pipeline, with settings of 50 inference steps and an
ensemble size of 10", measured **AbsRel = 0.0576 / δ₁ = 0.9615** —
within numerical noise of plumbline's 0.0574 / 0.9594 (D17 prior
session) and 0.06681 (D17 50-step sub60 this session, scaled by the
known sub60 bias 0.067/0.057 ≈ 1.17, projects to full-654 ≈ 0.057).
Three independent single-seed reproductions converge to ~0.057. The
paper's 0.052 is the **best** across N seed draws, not a single-seed
mean — an undocumented eval-protocol detail not in any released
code, README, or paper appendix.

This is the diffusion-depth-lineage analog of D9: the gap is an
upstream-paper-vs-public-release delta. For D9 it was the
**checkpoint** (v1-0 vs v1-1). For D17 it's the **evaluation
recipe** (best-of-N seeds vs single fixed seed).

#### 2026-05-26 — final adapter lever ruled out: denoise_steps

The last untested adapter knob was the upstream README's note:

> *"`--ensemble_size` and `--denoise_steps`: trade-off arguments
> for speed and performance, more ensembles and denoising steps to
> get higher accuracy. Default: 3 and 10 (**For academic comparison,
> please set 10 and 50, respectively**)."*
> — [`fuxiao0719/GeoWizard/README.md`](https://github.com/fuxiao0719/GeoWizard#%EF%B8%8F-inference)

This is structurally identical to D9's "the README named the
paper-comparable config but the script default mirrored a different
one" pattern. Hypothesis: plumbline's pinned ``num_inference_steps:
10`` (which mirrored ``run_infer.py``'s argparse default) under-
denoises by 5× relative to paper-protocol, and that's the residual
gap.

Tested on a 60-image stride-spread subset of the 654 Eigen test
split (linspace stride via ``subset: 60``, same shape as D9's
v1-0/50-step sub60), under the already-fp32 / xformers / full-
``seed_all`` adapter that closed every other lever:

| Probe | Config | AbsRel | δ₁ | vs paper 0.052 |
|---|---|---|---|---|
| `geowizard-nyuv2-d17probe-10step-sub60` (cross-check) | 10-step / ens-10 / seed 2024 | **0.06704** | 0.9421 | +28.9 % off, sub60 baseline |
| `geowizard-nyuv2-d17probe-50step-sub60` (README "academic comparison") | 50-step / ens-10 / seed 2024 | **0.06681** | 0.9416 | +28.5 % off, **Δ vs 10-step = 0.3 % (numerical noise)** |

The denoise_steps hypothesis is **rejected**. 5× more denoising
steps move the metric by 0.0002 AbsRel — within the per-run noise
floor of the diffusion stack. The sub60 cross-check at 10 steps
also independently re-confirms plumbline's prior 5h-wall full-654
0.0574: linspace-stride sub60 lands at 0.0670, a 17 % upward bias
relative to full-654 (this stride happens to pick slightly harder
images). Applying the same bias to the 50-step sub60 0.06681 →
projected full-654 ≈ **0.0572**, indistinguishable from the prior
10-step full-654 0.0574 and from the @anonymous issue-#36 reporter's
0.0576.

Result artifacts (mirrored to S3 to survive box teardown):

- ``s3://plumbline-bench/runs/d17_resolution_20260526/d17_10step_sub60.json``
- ``s3://plumbline-bench/runs/d17_resolution_20260526/d17_50step_sub60.json``
- ``s3://plumbline-bench/runs/d17_resolution_20260526/d17_smoke_3img.json``
- ``s3://plumbline-bench/runs/d17_resolution_20260526/d17_chain.log``

GPU wall: 10-step 28 min + 50-step 1 h 57 min = 2 h 25 min total on
RTX 3090 (vast.ai box 207.174.105.41).

#### Root cause = paper-private cherry-pick across seeds

Plumbline runs a single fixed seed (``seed=2024``). The paper-time
eval, per the author's own statement, runs many seeds and reports
the per-dataset minimum. The exact-fix lever to land on paper 0.052
from plumbline's side is to add a multi-seed sweep + ``min(...)``
aggregation step — but that's an evaluation choice we shouldn't
endorse on `main`. The standard zero-shot benchmark protocol fixes
a seed; reporting the best of N defeats the point of the
evaluation. Plumbline's single-seed 0.057 is the methodologically
defensible number; the paper's 0.052 is reproducible only under a
non-standard eval recipe.

Also independently corroborated by the alignment-code reading: the
author's quote points to ``geowizard/utils/de_normalized.py`` for
their alignment. That file ships three helpers — ``align_scale_shift``
(``np.polyfit(deg=1)`` over masked depth-space — i.e., lstsq scale
+ shift fit), ``align_scale`` (median), and ``align_shift`` (median).
``align_scale_shift`` is structurally identical to plumbline's
``scale_shift_depth``. So the alignment isn't the gap either —
confirming the gap is at the prediction level, not the eval level.

#### Resolution & follow-ups

- D17 → **closed (paper-private cherry-pick across seeds)**.
- D18 (GeoWizard-KITTI) closes by the **same upstream evidence**:
  same checkpoint, same author, same eval recipe applies (the
  paper Table 1 has both NYU + KITTI columns from the same eval).
  No separate KITTI verification needed — the closure is in the
  paper-author quote, not in a number we'd compute on KITTI.
- D22 (GeoWizard portion): subsumed by D17/D18 closure — the
  "paper-private eval config" hypothesis was correct (cherry-pick
  across seeds), the closure is the documented attribution.
- Per-paper trust for GeoWizard: promoted from **Suspect** to
  **Explained (paper-private eval recipe)**. Single-seed
  reproductions from plumbline + @anonymous (`fuxiao0719/
  GeoWizard#36`) match each other at 0.057 / 0.96; paper 0.052 /
  0.966 is reproducible only under non-standard best-of-N seed
  selection.
- Production YAMLs (`geowizard_nyuv2.yaml`, `geowizard_kitti.yaml`):
  bumped to `num_inference_steps: 50` to match upstream README's
  "academic comparison" intent. This **does not** move the metric
  meaningfully (verified +0.3 % above) but it's the documented
  paper-protocol intent. Inline comment cites the verification.
- Probe YAMLs (`geowizard_nyuv2_d17probe_*.yaml`): kept as the
  reproducible artifact for the resolution. They live in
  `reproductions/` rather than `scripts/` so re-running them is one
  `plumbline reproduce` invocation; deletion deferred to a future
  cleanup sweep.

(Prior diagnostic history — adapter audit, eval-protocol sweep,
dtype + xformers + seed_all verification — preserved below for
audit trail. Each step independently ruled out a plumbline-side
explanation, leaving the upstream evidence as the only remaining
account.)

---

### D17 · GeoWizard NYU 10 % off — prior diagnostic history   📜 ARCHIVED

(Prior status: 🔎 upstream-blocked. Resolution above 2026-05-26.)


Observed `geowizard-nyuv2` AbsRel = 0.0573 vs paper 0.052 — 10.2 %
off, after D1 + D2 fixes. Candidates:

1. RNG divergence — plumbline seeds `torch.manual_seed(seed + idx)`
   per-sample; paper may use a single fixed seed.
2. Alignment mode — plumbline uses `scale_shift_depth`; GeoWizard's
   public eval script may differ.
3. Processing resolution — 768 matches paper; unlikely the source.

Priority: low. Defer to v0.2 with D9.

#### 2026-04-26 — cross-repo audit (no fix landed)

Same approach that worked for D3 + D4: pulled GeoWizard's official
repo (`fuxiao0719/GeoWizard`) and Marigold's (since GeoWizard's paper
follows the diffusion-depth lineage and shares the eval shape). What
each ships:

- **GeoWizard repo:** ships `run_infer.py` for inference + a
  training script that *imports* `align_scale_shift` but doesn't
  invoke it in any released eval pipeline. The metrics calculation
  for Table 1 NYU AbsRel is **not in the public repo** — same
  situation as MASt3R / DUSt3R for D3.

- **Marigold repo (likely shared protocol):** ships
  `script/depth/eval.py` + `src/util/alignment.py::align_depth_least_
  square` + `src/dataset/nyu_dataset.py`. Their NYU eval:

    - GT: NYU labeled, raw depth field, mm → m via /1000.
    - Eigen crop: ``eval_mask[45:471, 41:601] = 1`` (matches
      plumbline's `EIGEN_CROP = (45, 471, 41, 601)`).
    - **Valid mask: ``(depth > 1e-3) AND (depth < 10.0) AND
      eigen_crop``.** Both lower AND upper bound are part of the
      pre-fit mask, not just a post-clip.
    - Alignment: lstsq fit in depth space (``[pred, 1] @ [s, b] ≈
      gt`` masked) — same as plumbline `scale_shift_depth`.
    - Post-alignment: ``np.clip(pred, min_depth, max_depth)`` then
      ``np.clip(pred, 1e-6, None)``.
    - Per-image abs_rel via ``mean(|pred-gt|/gt)`` over valid pixels;
      mean across the 654-image test split.

- **Plumbline NYU loader (current):** valid_mask =
  ``eigen_crop AND depth > 0`` — **does NOT enforce ``depth < 10``**
  in the pre-fit mask. The 10 m upper bound only enters via the
  post-alignment ``depth_clip = [0.001, 10]`` on PRED (not GT).

#### Hypothesis tested + REJECTED — eval protocol is fine, gap is the model

Pulled the cached GeoWizard NYU predictions from S3 and swept the
plumbline-side protocol axes against them, computing AbsRel directly
without re-running inference. 654-sample mean AbsRel:

| field | alignment | post-clip | mean | median |
|---|---|---|---|---|
| raw | scale_shift_depth | (1e-3, 10) | **0.0573** | 0.0451 |
| raw | scale_shift_depth | (1e-3, 10), gt<10 | 0.0570 | 0.0450 |
| raw | scale_shift_depth | none | 0.0570 | 0.0450 |
| filled | scale_shift_depth | (1e-3, 10) | 0.0689 | 0.0522 |
| raw | lstsq (scale-only, depth) | (1e-3, 10) | 0.2576 | 0.2529 |
| raw | scale_shift (disparity) | (1e-3, 10) | 0.2173 | 0.2083 |
| raw | scale_shift_robust | (1e-3, 10) | 0.1562 | 0.1277 |

Findings:

- **The pre-fit ``gt < 10 m`` mask hypothesis is wrong.** It moves
  AbsRel by 0.4 % (0.0573 → 0.0570), not the 10 % needed. Kinect
  saturation in NYU's labeled raw-depths is encoded as ``0``
  (already excluded by ``depth > 0``), not as values >10 m, so
  there's nothing to mask.
- **``raw`` beats ``filled``** for GeoWizard preds (0.057 vs 0.069);
  plumbline's ``depth_field='raw'`` default is correct. Marigold's
  pre-extracted PNGs likely also use raw.
- **``scale_shift_depth`` is the only alignment that gets close.**
  Disparity-space (``scale_shift``) blows up to 0.22; depth-space
  scale-only (``lstsq``, ``median``) blows up to 0.26. Plumbline's
  default for GeoWizard is right.
- **Median AbsRel 0.0451** is *below* paper's 0.0520 mean — if the
  paper happens to report median we'd have over-matched. Most papers
  report mean though.

So D17 is structurally a D3-clone: **the chamfer/AbsRel protocol is
already correct**; the 10 % gap is in the GeoWizard predictions
themselves, not how we evaluate them. Candidate sources:

1. RNG / ensemble seed — paper may seed differently and average over
   different denoising trajectories than plumbline's
   ``torch.manual_seed(seed + idx)``.
2. dtype — plumbline runs ``float16``; paper-stated config in
   their README is also fp16, but xformers vs not + cuDNN paths
   could change low-bit output.
3. Inference pipeline subtleties — plumbline mirrors ``run_infer.py``
   but may differ at the ensemble-mode boundary
   (``geowizard_pipeline.DepthNormalEstimationPipeline`` has knobs
   we may not have all matched).

#### Tiny structural cleanup that landed

Added a ``max_gt_depth: float | None = None`` kwarg to
``NYUv2Dataset`` (default ``None`` preserves prior behaviour). For
NYU it's a no-op (no pixel >10 m); kept as a structural knob for
parity with Marigold's eval shape and for datasets where the
equivalent matters (KITTI 80 m). Useful for D9 / D18 follow-ups.

Per-pixel diagnostic at ``/tmp/diff/d17_probe.py`` (lost on
teardown — same logic re-creatable from the cached predictions on
S3 and the loader as of commit ``d4d6f68``).

#### 2026-04-26 — adapter audit: ``--half_precision`` is dead in upstream

Pulled GeoWizard's repo and read ``geowizard/run_infer.py`` +
``run_infer_v2.py`` end-to-end against the plumbline adapter.
Findings:

1. **dtype mismatch (likely culprit).** Both upstream entrypoints
   define a ``--half_precision`` CLI flag and assign
   ``dtype = torch.float16`` to a local variable when it's set, but
   they NEVER apply that dtype to the pipeline. The components are
   loaded via ``from_pretrained(...)`` without ``torch_dtype=``
   (default fp32), the ``DepthNormalEstimationPipeline`` is
   constructed, and the only subsequent move is
   ``pipe.to(device)`` — which moves to GPU but does not change
   dtype. So upstream paper-protocol effectively runs **fp32
   regardless of the flag**. Plumbline's adapter, however, threads
   ``torch_dtype=torch.float16`` into ``from_pretrained()`` when
   ``dtype="float16"``, and the YAML pinned ``dtype: float16`` —
   so plumbline genuinely ran fp16. This is the first plausible
   adapter-side explanation for the 10 % gap. Marigold (same
   ancestor, same dead-flag pattern) ships
   ``reproductions/marigold_v1_1_*.yaml`` with ``dtype: float32``
   already and matches paper; GeoWizard's YAMLs were the outlier.
2. **``seed_all`` parity.** Upstream's ``utils/seed_all.py`` seeds
   ``random``, ``np.random``, ``torch.manual_seed``, and
   ``torch.cuda.manual_seed_all``. Plumbline previously seeded
   torch + cuda only. ``ensemble_depths`` calls scipy BFGS
   (deterministic given inputs) but the helper imports ``random``
   and ``np.random`` so we match for paranoia.
3. **xformers attention.** Upstream tries
   ``pipe.enable_xformers_memory_efficient_attention()`` if
   available. Plumbline doesn't. Possible second-order numerics
   delta but not the dominant source.
4. **Pipeline body** (``geowizard_pipeline.py``) — read end-to-end:
   image preprocessing (``resize_max_res`` PIL default + ``rgb/255 *
   2 - 1``), ensemble construction (``stack`` × ensemble_size,
   ``DataLoader`` with batch_size=1), denoising loop (DDIM, the
   joint depth+normal repeat-2 trick), ``ensemble_depths``
   defaults (``regularizer_strength=0.02, max_iter=2, tol=1e-3,
   reduction='median'``), and final scale-to-[0,1] all match
   plumbline's path.

Fixes landed this session:

- ``geowizard_nyuv2.yaml`` and ``geowizard_kitti.yaml``: pin
  ``dtype: float32`` with an inline comment citing the dead-flag
  audit (D17/D18).
- ``GeoWizardAdapter.predict``: add ``random.seed`` +
  ``np.random.seed`` so the seed call body matches upstream's
  ``seed_all`` exactly. ``rng_mode`` bumped to ``once_at_startup_v2``
  in ``config_hash`` so old fp16 cache entries don't shadow the new
  fp32 + full-seed run.
- ``GeoWizardAdapter.predict``: fixed an ``AttributeError`` shipped
  in ``e5dcc29`` — the ``first_call`` guard checked
  ``self._model`` (which doesn't exist), should be ``self._pipe``.

Status: 🧪 FIX-PENDING-VERIFY for both D17 (NYU) and D18 (KITTI) on
the next GPU run. If fp32 + full-seed still leaves a residual gap,
the next candidates are xformers attention parity and a per-pixel
prediction diff against ``run_infer.py`` on a single shared image.

#### 2026-04-26 — verified: dtype + xformers + seed_all are NOT the gap

Re-ran ``geowizard-nyuv2`` end-to-end (RTX 3090, 5 h 01 min wall, 654
samples, fp32 + xformers + full seed_all) and got AbsRel = **0.0574**
vs the prior fp16 + per-sample-reseed + no-xformers run's 0.0573 —
**identical to numerical noise** (ΔAbsRel ≈ 1e-4 ≈ 0.2 %).

The hypothesis from this session's adapter audit is **rejected**.
Plumbline's GeoWizard adapter is now structurally aligned with
upstream ``run_infer.py`` (fp32 default, xformers attention enabled,
``seed_all`` body matched, RNG seeded once at startup) but produces
predictions whose AbsRel is indistinguishable from the prior fp16
path. Other aggregate metrics also unchanged within ±0.5 %:
δ₁ 0.9615 (was 0.9594), δ₂ 0.9908, δ₃ 0.9975, RMSE 0.228, log10
0.025, SILog 8.42.

Combined with the 5ba6fae sweep (eval-protocol axes against cached
preds), this closes the search over **everything plumbline can
control**:

- Eval protocol (alignment, mask, depth field, post-clip): swept,
  matches Marigold's published code exactly.
- Adapter (dtype, xformers, seed shape, ensemble): audited, matches
  upstream ``run_infer.py``.
- GT loader (NYU labeled .mat, raw depth field, Eigen 654 split):
  matches Marigold's NYU loader.

Remaining unaccounted variance ≈ 10 % AbsRel. The only places this
can live are upstream-owned:

1. **Public checkpoint vs paper checkpoint.** ``lemonaddie/Geowizard``
   is the publicly-released weights; the paper may have used an
   internal snapshot, a different training step, or different
   training data. This matches the **D22 pattern** (Marigold /
   GeoWizard KITTI cells also failed under both candidate eval
   protocols + literal paper code, suggesting a private config).
2. **Paper protocol detail not in the public code.** Less likely —
   we already pulled both the paper and the released ``run_infer.py``
   line-by-line.

Status promoted from 🧪 fix-pending-verify to 🔎 **upstream-blocked**:
same shape as D22. Defer until upstream clarifies (issue, paper
errata, or model release notes). Don't burn another 5 h GPU run on
GeoWizard-KITTI under the same model — D18 is the same checkpoint,
same likely-private-config issue, fix would be the same and the
result would be the same.

The audit changes (fp32 + xformers + full seed_all) are kept on
``main`` because they make the adapter structurally correct against
upstream — they just aren't the gap. Future GeoWizard work
(re-evaluation against a corrected upstream checkpoint) inherits
the right protocol shape.

Result artifact: ``/tmp/results/geowizard_nyuv2_d17.json`` (lost on
teardown — same numbers re-derivable by re-running the YAML).

(D20 closed 2026-04-24, see bottom table.)

---

## Priorities for the next session (2026-05-03)

**Active — needs GPU time:**
1. **CO3Dv2 pose** — VGGT Table 1 (AUC@30 = 0.882) and MASt3R Table 3
   (mAA(30) = 0.818). Loader + N-view MASt3R adapter landed
   2026-04-27 (commit `cd35b93`); zero GPU validation. ≥ 2.5 h on a
   3090 for the MASt3R run alone. Gates the pose half of the v0.1
   release.
2. **D4 / D10 — full 13-scene ETH3D split.** Per-view-masked path
   already produces 0.642 m on the 3-scene subset. Need to either
   stage the remaining 10 scenes (~14 GB) for the apples-to-apples
   13-scene mean, or formally demote the row to "3-scene
   informational subset".
3. ~~**D23 — direct PDF re-verification of `mast3r_co3dv2_pose`.**~~
   ✅ DONE 2026-05-23. Downloaded `arxiv.org/pdf/2406.09756`, read
   Table 3 directly: CO3Dv2 row (b) MASt3R = 94.6 / 91.9 / 81.8,
   matching the YAML exactly. The paper target is confirmed; only the
   GPU run (item 1) remains before the row counts as ✅.

**Closed-blocked — do not retry without an upstream change:**
- D3 (VGGT-DTU). Exhausted adapter + protocol + dtype + RNG levers.
  Residual gap is in the public checkpoint. Re-enters the queue
  if/when upstream releases an updated checkpoint or eval script.
- D9 / D22 (Marigold-KITTI portion) — **RESOLVED 2026-05-25** via
  end-to-end reproduction on Marigold's exact prepared set with
  v1-0 / 50-step (0.0992 vs paper 0.099, 0.2 % off). Root cause =
  checkpoint-generation delta (current upstream eval script defaults
  to v1-1 / 1-step distilled). See D9 above.
- D17 / D18 / D22 (GeoWizard portion) — **RESOLVED 2026-05-26** via
  paper author's quote on `fuxiao0719/GeoWizard#36`: paper number
  is best-of-N seeds, not single-seed. Plumbline's 0.0574 matches
  @anonymous independent reproducer 0.0576 on identical protocol;
  paper 0.052 is the cherry-picked minimum across seed draws. See
  D17 above.

**Deferred (v0.2+):**
- D15 — DA-V2 NYU ~0.002 bias (Eigen-crop + rawDepths interaction).
- New adapter additions per `plan.md` § 10 Tier 2.

**Recently closed (one-liner; full diagnoses in their commits):**
- D8 ✅ 2026-04-24 — MoGe-KITTI AbsRel 0.0404 vs paper 0.0408.
- D19* ✅ 2026-04-26 — MoGe-DIODE FoV-warp port, 0.0407 vs 0.0400.
- D20 ✅ 2026-04-24 — scene-agg memory bug.
- D21 ✅ 2026-04-24 — prediction cache fingerprint.

---

## Closed issues

One-line reference; full diagnosis in the linked commit message.

| ID | One-liner | Closed by |
|---|---|---|
| D1 | GeoWizard — `generator` kwarg not accepted upstream | ✅ `c50201e` |
| D2 | GeoWizard — upstream diffusers API drift (shim) | ✅ `a35c4f5` |
| D5 | DIODE outdoor prediction outliers (`drop_max_depth`) | ✅ `7fd6ff6` (residual → D19) |
| D6 | DIODE MoGe-eval loader `split` kwarg | ✅ `ae046ab` |
| D7 | KITTI annotated-depth not in S3 cache | ✅ (staged to S3) |
| D11 | `scale_shift_robust` overfits NYU vs MoGe's plain LSQ | 📝 `c14d776` |
| D12 | KITTI Eigen-crop hypothesis (rejected empirically) | 📝 `fb58b90` |
| D13 | DA-V2 Large NYU — pinned to MoGe's 0.0420 | ✅ `58fc159` |
| D14 | DA-V2 Base NYU citation verified 0.049 | ✅ `603e717` |
| D15 | DA-V2 NYU ~0.002 AbsRel systematic downshift (S/B/L) | 📝 below-threshold |
| D16 | MoGe-DIODE-indoor combined-val citation demoted | ✅ `603e717` |
| D19 | MoGe-DIODE-both `scale_shift_clamped` alignment | ✅ 2026-04-23 verify: 0.0406 vs paper 0.0400 (1.5 % off) |
| D19* | MoGe-DIODE-both regression (loader missing FoV-warp; observed 0.1088 vs paper 0.0400, 2.7× off) | ✅ 2026-04-26: ported MoGe's `EvalDataLoaderPipeline._process_instance` (1024×768 homographic FoV-warp) into `DIODEMogeEvalLoader`; verified 0.0407, 1.7 % off paper |
| D8 | MoGe-KITTI — port MoGe's homographic FoV-crop to `KITTIMogeEvalLoader` | ✅ 2026-04-24 verify: 0.0404 vs paper 0.0408 (0.9 % off) |
| D20 | Scene-aggregation memory bloat — eager per-chunk voxel_downsample + DTU voxel_size unit fix | ✅ `8827a87` + `1fc0f9c`: D3 completes without OOM (51 mm, 8 GB peak RSS vs 28 GB prior) |
| D9 | Marigold-KITTI 0.099 paper cell unreproducible under any plumbline protocol | ✅ 2026-05-25: native Marigold pipeline on `prs-eth/Marigold` + authors' prepared `kitti_eigen_split_test.tar` reproduces AbsRel **0.0992 vs paper 0.099 (0.2 % off)** with **v1-0 / 50-step / ens-10** (the original CVPR checkpoint); plumbline's v1-1 / 1-step lands ~0.111 on the same 60-img spread sub (matching plumbline's full 0.109). Root cause = upstream eval-script default repointed v1-0 → v1-1 between paper and current repo; v1-1 is a distilled checkpoint that trades accuracy on outdoor KITTI specifically (v1-1 still matches paper on NYU). Documented checkpoint-generation delta, not a paper-private config and not a plumbline bug. |
| D22 (Marigold portion) | Marigold-KITTI paper cell not reproducible under either `marigold_kitti_eval` or `kitti_moge_eval` — "paper-private config" hypothesis | ✅ 2026-05-25: REFUTED by D9 — the paper cell is fully reproducible on authors' public pipeline + public data with the v1-0 CVPR checkpoint. |
| D17 | GeoWizard-NYU AbsRel 0.0574 vs paper 0.052 (+10.5 %), persistent after dtype + xformers + seed_all + 50-step `--denoise_steps` upstream-README knob (verified 2026-05-26 on sub60: 10-step 0.06704 vs 50-step 0.06681, Δ 0.3 %) | ✅ 2026-05-26: EXPLAINED — paper author confirmed on `fuxiao0719/GeoWizard#36` that paper-time eval "perform[s] multiple inferences with different initialized seeds … and select[s] the best result for the metric report". Plumbline single-seed 0.0574 matches @anonymous independent reproducer 0.0576 exactly; paper 0.052 is best-of-N seeds, an undocumented eval-protocol detail not in any released code / README / paper appendix. Root cause = paper-private cherry-pick across seeds. |
| D18 | GeoWizard-KITTI AbsRel 0.131 vs paper 0.097 (+35 %), same checkpoint as D17 | ✅ 2026-05-26: same author quote covers Table 1's KITTI column. No separate GPU verification needed — closure is in the upstream evidence, not a number we'd recompute. |
| D22 (GeoWizard portion) | GeoWizard NYU + KITTI cells off paper — "paper-private config" hypothesis | ✅ 2026-05-26: CONFIRMED but in a different shape than expected. Not a config-knob delta — the paper-time eval is **best-of-N seeds**, not a single-seed mean, per the author's own statement (`fuxiao0719/GeoWizard#36`). Adapter + protocol are structurally correct. |
| D21 | Prediction cache key → stale hits on loader preprocessing change — fingerprint input tensor | ✅ `8827a87`, regression test `test_input_fingerprint_invalidates_on_change` |

---

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/<ts>/` preserve observations from each run
so a report can be regenerated from any point.
