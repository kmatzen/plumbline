# Discrepancies

Catalog of every adapter / loader / protocol / solver / citation mismatch
surfaced to date. Open entries keep full diagnosis context. Closed entries
(FIXED + verified, or EXPLAINED-NOT-A-BUG) live at the bottom as one-liners
with commit SHAs; full history is in git.

Status legend:

- 🧪 **FIX-PENDING-VERIFY** — change landed, waiting on a GPU re-run.
- 🔎 **SUSPECTED** — hypothesis + diagnosis path; not yet reproduced.
- 📅 **DEFERRED** — known root cause, scoped for v0.2+.

## Open issues at a glance

| ID | One-liner | Status |
|---|---|---|
| D3 | VGGT-DTU chamfer — 2× off paper; chamfer protocol confirmed not the source (Jensen ref 0.868 vs PVM 0.758); gap is VGGT pred quality | 🔎 correctness |
| D4 | VGGT-ETH3D — scan_alignment.mlp bug fixed (Acc now beats paper at 0.77 vs 0.90); Comp 3.47 m needs per-view-masked path (same fix as D3) | 🔎 correctness |
| D9 | Marigold-KITTI — OFF-PAPER under both candidate protocols (closest 13 % under kitti_moge_eval) | 🔎 secondary-delta |
| D10 | VGGT-ETH3D full 13-scene vs 3-scene subset | 📅 deferred |
| D17 | GeoWizard NYU 10 % off — RNG or secondary protocol detail | 🔎 suspected |
| D18 | GeoWizard-KITTI — OFF-PAPER under both protocols (closest 14 % under kitti_moge_eval, 45 % under marigold_kitti_eval) | 🔎 secondary-delta |
| D22 | Marigold/GeoWizard KITTI paper cells do not reproduce under either Marigold's own eval code or MoGe's bundle — paper likely uses a private eval config | 🔎 new 2026-04-24 |

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

### D9 · Marigold-KITTI — OFF-PAPER under both candidate protocols   🔎 OPEN

Tested under three protocols; paper value 0.099 doesn't reproduce under any:

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.1146 | +15.8 % |
| `kitti_moge_eval` | 0.0865 | −12.7 % |
| `marigold_kitti_eval` | 0.1179 | +19.1 % |

`marigold_kitti_eval` implements Marigold's own paper code (`kitti_bm_crop`
+ `valid_mask_crop: eigen` + `scale_shift_depth`, per
`prs-eth/Marigold/src/dataset/kitti_dataset.py`). That it's *further*
from paper than `kitti_moge_eval` means the paper cell didn't come
from Marigold's public eval pipeline — probably a private config or
different checkpoint.

YAML stays on `marigold_kitti_eval` (the literal paper-code pipeline)
per "never modify YAMLs to fit a number". Closing this requires finding
the paper's actual eval config — upstream issue, not a plumbline bug.

### D18 · GeoWizard-KITTI — same pattern as D9   🔎 OPEN

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.131 | +35 % |
| `kitti_moge_eval` | 0.1103 | +13.7 % |
| `marigold_kitti_eval` | 0.1406 | +45 % |

Same as D9: `marigold_kitti_eval` is worse than `kitti_moge_eval`.
GeoWizard shares the diffusion-depth lineage with Marigold; D22
(paper-private-eval hypothesis) most likely applies to both.

### D22 · Marigold / GeoWizard KITTI paper cells don't reproduce   🔎 NEW 2026-04-24

Neither the literal paper-code pipeline (`marigold_kitti_eval`) nor
the MoGe bundle pipeline (`kitti_moge_eval`) reproduces
Marigold 0.099 or GeoWizard 0.097 on KITTI. Both are consistently
off by 13-45 % in various directions. Under `marigold_kitti_eval`
(the paper-code pipeline), the harness is *further* from paper than
under `kitti_moge_eval` — which rules out "we just need to use the
paper's code". Suggests the paper-reported cells come from a private
eval config (unreleased resolution setting, checkpoint version, or
pre-processing step).

Not paper-match-blocking in the sense that we can't close the gap —
it's a finding about the ground truth being recorded. Document and
move on until Marigold / GeoWizard authors clarify.

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

### D17 · GeoWizard NYU 10 % off   🔎 SUSPECTED

Observed `geowizard-nyuv2` AbsRel = 0.0573 vs paper 0.052 — 10.2 %
off, after D1 + D2 fixes. Candidates:

1. RNG divergence — plumbline seeds `torch.manual_seed(seed + idx)`
   per-sample; paper may use a single fixed seed.
2. Alignment mode — plumbline uses `scale_shift_depth`; GeoWizard's
   public eval script may differ.
3. Processing resolution — 768 matches paper; unlikely the source.

Priority: low. Defer to v0.2 with D9.

(D20 closed 2026-04-24, see bottom table.)

---

## Priorities for the next session

**Completed 2026-04-24:**
- D8 — MATCH (`KITTIMogeEvalLoader` delegates to MoGe's `_process_instance`).
- D20 — scene-agg memory bug fixed: eager per-chunk voxel_downsample (`8827a87`) + unit-mixup fix (`1fc0f9c`, DTU mm vs ETH3D m). D3 now completes without OOM.
- D21 — prediction cache key now mixes input-tensor fingerprint (`8827a87`).
- Marigold-style protocol + YAML repoint landed (`ce4183e`, `6d24c73`). Verify surfaced D22.

**Open correctness investigations:**
1. **D3 — VGGT-DTU chamfer Overall 2× off** (0.758 mm mean / 0.442 mm
   median vs paper 0.382 mm). 2026-04-25 single-record diff against
   CUT3R `eval/mv_recon/` exposed stage-1 loader-side structural
   divergence (plumbline shipped scene-level `Points/stl/*.ply` only;
   CUT3R loader expects per-view `depths/*.npy` + `binary_masks/
   *.png` per scan); fix landed same session — `DTUDataset(
   with_per_view_gt=True)` derives per-view GT by z-buffering the
   laser PLY through each GT pose and the runner now has a
   `per_view_masked` path that ports the CUT3R 224×224-crop +
   GT-mask + KDTree-NN protocol. Numbers went 130× → 2× off paper,
   structurally correct, but ±5 % v0.1 gate not yet met. Remaining
   gap candidates: Poisson-mesh-rendered GT (vs splat), all 49 rig
   views (vs first 32), per-pixel aspect-ratio diagnosis.
   YAML metric-key mismatch (`chamfer` → `overall`) fixed.
2. **D4 — VGGT-ETH3D regression vs prior run** (1.75 m vs prior 0.82 m, 2× worse). A/B `scan_clean` vs `dslr_scan_eval` GT to isolate.
3. **D22 — Marigold + GeoWizard KITTI paper cells** don't reproduce under either candidate protocol. Needs upstream clarification on paper's actual eval config.
4. **D17** — GeoWizard NYU 10 % off. Possibly same family as D22.

**Landed 2026-04-24:**
- D8 — MoGe-KITTI AbsRel 0.0404 vs paper 0.0408 (0.9 % off) ✅
  `KITTIMogeEvalLoader` now delegates to MoGe's own
  `EvalDataLoaderPipeline._process_instance` for the homographic
  FoV-crop. D9 + D18 share the same loader — D9 pending GPU verify,
  D18 pending xformers install + verify.
- D21 — prediction cache stale-hit bug exposed en route to D8;
  workaround is `rm -rf` of the shard before re-run. Proper fix
  deferred.

**Nice-to-have (v0.2):**
- D10 — VGGT-ETH3D 13-scene full split, or demote to informational.
- D17 — GeoWizard NYU 10 % off (RNG or alignment).
- D15 — DA-V2 NYU ~0.002 bias (Eigen-crop + rawDepths).

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
| D8 | MoGe-KITTI — port MoGe's homographic FoV-crop to `KITTIMogeEvalLoader` | ✅ 2026-04-24 verify: 0.0404 vs paper 0.0408 (0.9 % off) |
| D20 | Scene-aggregation memory bloat — eager per-chunk voxel_downsample + DTU voxel_size unit fix | ✅ `8827a87` + `1fc0f9c`: D3 completes without OOM (51 mm, 8 GB peak RSS vs 28 GB prior) |
| D21 | Prediction cache key → stale hits on loader preprocessing change — fingerprint input tensor | ✅ `8827a87`, regression test `test_input_fingerprint_invalidates_on_change` |

---

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/<ts>/` preserve observations from each run
so a report can be regenerated from any point.
