# Discrepancies

Tracker for **outstanding** adapter / loader / protocol / solver / citation work
— open, parked, or investigated-but-unresolved cells only. **Resolved** items are
not kept here: the understanding for them lives in
[`CONFIDENCE_AUDIT.md`](CONFIDENCE_AUDIT.md) (which localizes every off-paper gap
to a layer) and the full investigation history is in git.

> Resolved & closed (see `CONFIDENCE_AUDIT.md` + git): D1, D2, D9, D17, D18, D21,
> D22, D23, D24, D27, D30, D10b. Removed-code attempts: `docs/blocked/`.
>
> D21 (cache vs loader-preprocessing) is fixed: `runner._sample_input_fingerprint`
> mixes the loaded tensor's shape + dtype + intrinsics + first 1 KB of pixel
> bytes into the prediction-cache key (`cache.PredictionCache.path_for`'s
> `input_fingerprint`), so a loader refactor that changes resolution/warp/colour
> stores to a distinct file and can't serve stale predictions against fresh GT.
> Residual (heuristic, accepted): a change that preserves all four of those does
> not invalidate.

**Category key:** 🔧 model integration · 📐 protocol (eval pipeline diverges from
paper code) · 📜 paper-side (unreproducible from the public release) · ⏳ untested.

## Outstanding issues at a glance

| ID | Cell | Status / next move |
|---|---|---|
| D3 | VGGT-DTU chamfer (+98 %, 0.756 vs 0.382 mm) | 📜 upstream-blocked: ~1.98× residual is in the public VGGT-1B output; adapter + protocol levers exhausted (PatchmatchNet filter, fp32, 49-view all no-ops). Watch upstream. |
| D4 | VGGT-ETH3D 3-scene (−9.4 %, 0.642 vs 0.709 m) | 📐 infra landed; apples-to-apples needs the full 13-scene split (D10). |
| D10 | VGGT-ETH3D full 13-scene split (+23.5 %) | 📐 investigated: driven entirely by one scene (`terrains`, Comp 10.18 m); 12/13 scenes beat paper. Open: is `terrains` a model failure or a scene-specific GT artifact. |
| D28 | DUSt3R Table 2 indoor (NYU/Bonn) | ✅ **RESOLVED 2026-06-18**: both were wrong-recipe/wrong-target, not paper-private. NYU → Eigen-2014 crop + ratio-of-medians (`nyu_dust3r_table2`) = 0.0637/0.065 ✅. Bonn → re-target to MonST3R Table 3's DUSt3R baseline (0.141, same rgb_110 seqs) = 0.1384/0.141 ✅. "Static model" claim retracted. |
| D29 | DA-V2 native-DIODE Table 2 (`domain=both`) | 📐 investigated: outdoor preprocessing gap; the MoGe-bundle cells on the same dataset are ✅. |
| D31 | DA-V2 native-ETH3D Table 2 (−32 %) | 📐 RGB/GT misalignment fixed; residual full-13-scene protocol gap still OPEN. |
| D32 | DA-V2 native-Sintel Table 2 (under paper) | 📐 parked: sky-mask / aggregation recipe. |
| D33 | DA-V2 native-ETH3D GT source + recipe | 📜 parked: GT source + eval recipe still diverge from ETH3D official. |

(Full diagnosis per entry below.)

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

### D28 · DUSt3R Table 2 indoor cells off-paper — lineage-recipe ≠ DUSt3R-paper-recipe   ✅ RESOLVED 2026-06-18

> **RESOLVED 2026-06-18 (anima, official `plumbline reproduce` runs).** The
> 2026-05-28 "paper-private indoor recipe" conclusion below was wrong — both
> cells were fixable, and one explanation it gave was self-contradictory.
>
> **NYU (0.0777 → 0.0637 vs 0.065 ✅).** The cell inherited `nyu_dust3r_lineage`
> (filled GT, **no Eigen crop**, median-of-ratios) on the false premise that
> "DUSt3R is the origin of the lineage protocol." But the no-crop convention is a
> *later* MonST3R/CUT3R choice; DUSt3R (2023) reports under the **classical
> Eigen-2014 crop + ratio-of-medians** (its own §4.3: "the medians of the
> predicted depths and the ground-truth ones"). A 654-image recipe sweep over one
> byte-faithful inference pass (`scripts/_dust3r_nyu_recipe_probe.py`):
>
> | recipe (filled GT) | AbsRel | δ₁ |
> |---|---|---|
> | no-crop + median (old cell) | 0.0777 | 0.9101 |
> | Eigen-crop + median | 0.0577 | 0.9495 |
> | **Eigen-crop + median_lineage** | **0.0637** | **0.9456** |
> | (paper) | 0.0650 | 0.9402 |
>
> Independently corroborated: **MonST3R Table 3 re-scores DUSt3R-NYU *no-crop* at
> 0.080** — exactly plumbline's 0.0777 — while DUSt3R's *own* crop gives 0.065 ≈
> plumbline's 0.0637. The Eigen crop *is* the 0.065↔0.080 gap. Fix: protocol →
> `nyu_dust3r_table2`, estimator → `median_lineage`.
>
> **Bonn (0.1337 → 0.1384 vs 0.141 ✅).** The "DUSt3R is a static model, it fails
> on dynamic Bonn" explanation was **self-contradictory** — DUSt3R's own paper
> reports 0.0808 on Bonn, so the model handles the data. A per-sequence probe
> (`scripts/_dust3r_bonn_recipe_probe.py`) ruled out *every* recipe lever (cap
> 10/70/none, estimator, per-seq vs per-frame scale — all near-no-ops); the gap is
> purely the **sequence set**. DUSt3R reproduces the paper on the simpler seqs
> (balloon2 0.085 ≈ 0.0808) and blows up only on the dense crowds. plumbline
> evaluates **MonST3R's 2024 hard-5 selection** (hard-coded in their
> `eval_metadata.py`), which DUSt3R's 2023 Table 2 predates and does not specify
> (no eval code shipped to recover it). The apples-to-apples target is therefore
> **MonST3R Table 3's DUSt3R baseline, 0.141**, scored on exactly that 5-seq
> `rgb_110` set (PDF-verified, ICLR 2025 p.8). plumbline on `rgb_110`
> (`bonn_lineage_110_single`, new loader `prepared_110`) = **0.1384 / δ₁ 0.8331**,
> −1.8 % (the all-frames variant lands 0.1337, −5.2 %). Same pattern as
> `dust3r-co3dv2-pose` targeting MASt3R's DUSt3R baseline.
>
> The full MonST3R Table 3 DUSt3R row cross-checks all three plumbline numbers:
> NYU 0.080 (no-crop) ✓, KITTI 0.112 (plumbline 0.1049, also matches DUSt3R-own
> 0.1074) ✓, Bonn 0.141 ✓.

#### Original 2026-05-28 investigation (superseded by the above)


End-to-end GPU run of the three DUSt3R-own-paper depth pins (PR `dust3r-depth-pin`,
RTX 3090, 2026-05-28), against DUSt3R Table 2 "DUSt3R 512" row (Wang 2024, CVPR,
arXiv:2312.14132, §4.3 *"we simply feed the same input image I to the network as
F(I, I)"*; AbsRel / δ₁ in paper-percent units):

| cell | observed AbsRel | paper | Δ | observed δ₁ | paper δ₁ | n | match |
|---|---|---|---|---|---|---|---|
| dust3r-kitti | **0.1049** | 0.1074 | −2.3 % | 0.8661 | 0.8660 | 1269 | ✅ |
| dust3r-nyuv2 | **0.0777** | 0.0650 | +19.5 % | 0.9101 | 0.9402 | 654 | ❌ |
| dust3r-bonn | **0.1337** | 0.0808 | +65.4 % | 0.8666 | 0.9356 | 3093 | ❌ |

KITTI lands within tolerance on the *lineage-empirical* protocol — the same
`kitti_dust3r_lineage` recipe that reproduces `cut3r-kitti` exactly and
`monst3r-kitti` within ~5.05 % (marginally over the ±5 % gate → ℹ️). So the outdoor recipe matches DUSt3R's own paper
recipe. NYU and Bonn don't.

**KITTI data-root footgun (caught 2026-05-28).** The first KITTI run pointed
`KITTI_ROOT` at an Eigen-652 raw tree and silently evaluated only **82 frames**
(the raw∩GT intersection) with `skipped=0` — it *looked* like a clean run and
coincidentally also matched (0.1086). The lineage KITTI loader enumerates from
raw `image_02/data` and attaches annotated GT, so a root with sparse raw frames
under-counts without raising. Re-run on the proper gathered tree
(`/root/data/kitti_dust3r_lineage`, 1269 frames / 13 drives) gives 0.1049 — the
verdict held, but only the full-set number is trustworthy. Guard landed (both
halves):
1. The runner routes a prediction that yields no metric for the requested task to
   `n_skipped` (with a reason) instead of silently incrementing `n_evaluated`
   (`runner.evaluate`, test `test_metricless_prediction_counts_as_skipped`).
2. A reproduction may declare `min_samples` — a floor on the evaluated-frame count.
   If the run evaluates fewer (the sparse data root above), `run_reproduction`
   forces `paper_match=no` with a COUNT SHORTFALL note instead of letting the
   metric match on the wrong set (`reproduce.py`, tests
   `test_min_samples_shortfall_forces_no_match` / `_met_matches_normally`).
   Adopted on `da-v2-small-diode-indoor` (floor = its 220 pinned ids); other
   reproductions can opt in with their own verified count.

The lineage protocols (`{nyu,kitti,bonn}_dust3r_lineage*`) are named after DUSt3R
because the *file paths and prep conventions* trace from DUSt3R, but the *eval
recipe* in plumbline is calibrated against MonST3R's released
`depth_metric.ipynb` (the only end-to-end runnable artefact in the lineage —
DUSt3R's repo has dataset-prep scripts but no released depth-eval scripts; the
table is implicitly produced by §4.3's `F(I, I)` plus an unspecified scoring
helper). MonST3R's notebook NYU cell uses `depth_evaluation(pred, gt,
max_depth=None, lr=1e-3)` → default branch = per-frame median scale-only (read
2026-05-28 from `/root/deps/monst3r/depth_metric.ipynb` cell 7 + `depth_eval.py:148`).

**Diagnosis (cached-prediction re-score sweep, 2026-05-28, no new GPU).** The
config-hash prediction cache lets us re-score the exact same model outputs under
different scoring recipes in seconds. Two dimensions tested:

*Dimension 1 — scale-alignment (RULED OUT for NYU).* Re-scoring NYU's 654 cached
preds under every alignment mode plumbline supports:

| alignment | NYU AbsRel | Bonn AbsRel |
|---|---|---|
| median (scale-only, what we ship) | **0.0777** | **0.1337** |
| lstsq (scale-only, L2) | 0.0876 | — |
| scale_shift (inv-depth) | 0.1239 | 0.1147 |
| scale_shift_depth (depth, ≈ MonST3R `align_with_lad2`) | 0.0931 | 0.1070 |

For NYU **median is already the best** mode and *nothing* approaches paper 0.0650
— adding a shift term makes it worse. So alignment is **not** the NYU gap. (An
earlier draft of this entry attributed the gap to an alignment-recipe delta on
the strength of δ₁ being closer to paper than AbsRel; that reasoning was wrong —
a shift term changes δ₁ too — and the sweep refutes it. Retracted.)

*Dimension 2 — GT processing (THIS is the NYU gap).* Re-scoring the same NYU
preds, median alignment held fixed, swapping only the GT-side protocol:

| NYU GT processing | AbsRel | δ₁ |
|---|---|---|
| `nyu_eigen_2014` (raw GT, Eigen crop, clip) | **0.0489** | 0.9635 |
| **paper "DUSt3R 512"** | **0.0650** | 0.9402 |
| `nyu_dust3r_lineage` (filled GT, no crop, no clip) | **0.0777** | 0.9101 |

GT processing swings the *same predictions* across a 59 % range (0.0489 → 0.0777)
and the paper number sits **bracketed in between**. This is the exact D24 shape
(CUT3R-NYU swings 0.052 strict-Eigen ↔ 0.086 lineage; MonST3R-NYU 0.0599 ↔
0.0894). The DUSt3R model output is fine — it scores anywhere from 0.049 to 0.078
purely as a function of GT-processing conventions nobody documented. DUSt3R's
2023 paper predates the lineage filled+no-crop convention and used some
intermediate raw/crop/clip recipe (unreleased) that lands at 0.065.

**Bonn (recipe + genuine dynamic-scene weakness).** No alignment mode gets Bonn
near paper 0.0808 (best is scale_shift_depth at 0.1070, still 32 % off). The
per-sequence breakdown shows why:

| Bonn sequence | n | per-frame mean AbsRel | nature |
|---|---|---|---|
| balloon2          | 467 | **0.0785** | mostly static, single object |
| person_tracking2  | 565 | **0.0455** | smooth tracking, low motion |
| crowd2            | 893 | 0.1837 | crowd, dynamic |
| crowd3            | 838 | 0.1516 | crowd, dynamic |
| synchronous       | 330 | 0.1820 | dance, dynamic |

The two low-dynamic sequences land at/below paper; the three dynamic ones (75 %
of frames) drive the aggregate up. DUSt3R is *not* a dynamic-scene model — the
premise of MonST3R existing. The paper 0.0808 is reachable only with a recipe
that suppresses the dynamic-region error (per-sequence scale+shift LAD2 like the
D27 MonST3R notebook, and/or a tighter valid mask) — paper-private, plus a real
model limitation on dynamics that no per-frame scoring choice fixes.

**Inference faithfulness — single-record diff (conclusive, NOT a bug).** To rule
out a plumbline-side inference/prep bug as the NYU gap (rather than just the eval
recipe), a one-sample diff (`scripts/_dust3r_nyu_singlediff.py`, run on the box;
NYU sample 0) compared plumbline's path against a from-scratch reference that uses
dust3r's *own* `load_images` + raw `inference()`:

| comparison | result |
|---|---|
| input tensor: plumbline `_images_to_dust3r_dicts` vs dust3r `load_images` | **max\|Δ\| = 0.00000** (bit-identical) |
| depth map: plumbline `_dust3r_single_frame_eval` vs from-scratch F(I,I) | **max\|Δ\| = 0.00000, corr = 1.000000** |
| single-sample AbsRel (filled, no crop, median) | **0.0412 == 0.0412** |

plumbline's DUSt3R inference is byte-for-byte the canonical `F(I, I)` output —
there is **no prep / extraction / wrapper bug**. The 0.0777 aggregate is DUSt3R's
genuine output quality under the lineage recipe (sample 0 at 0.0412 shows the
high per-sample variance behind that mean). The off-paper gap is therefore *100 %*
eval recipe, mechanically confirmed.

**Authoritative recipe is unrecoverable.** naver/dust3r#180 ("Evaluation on
monocular depth estimation task") asks the exact recipe question — *"affine-
invariant space or scale and shift per image via least squares? … pointers to
existing code?"* — and has **0 maintainer replies**. DUSt3R's repo ships no
monocular-depth eval script (only `datasets_preprocess/` + demo); the paper just
cites refs [6, 117] for "the protocol". So the recipe that yields 0.065 cannot be
recovered, and the OP's own guess (affine-invariant) is *worse* in our sweep.

**Same shape as D9 / D17 / D24 / D27**: the published number comes from an
unreleased / undocumented eval recipe — here specifically the **GT-processing**
dimension for NYU (evidenced by the bracket above + the byte-faithful inference),
not alignment. plumbline ships
the lineage-empirical recipe (reproduces MonST3R / CUT3R cells, and DUSt3R-KITTI
within 2.3 %); it doesn't match DUSt3R's own indoor recipe. KITTI stays ✅ MATCH;
NYU and Bonn are ℹ️ — explained, not a model / adapter bug. No code or protocol
change. Diagnostic variant YAMLs (`dust3r_{nyuv2,bonn}_{lstsq,scale_shift,…}.yaml`,
`dust3r_nyuv2_eigen.yaml`) live only on the GPU box under `reproductions/`, not
committed — the cached preds at `/root/.cache/plumbline/predictions/dust3r/16431eab…`
reproduce the whole sweep.

The DUSt3R-own-paper *pinning* is still valuable: it anchors the lineage at
its source, ships the adapter's N=1 `F(I, I)` branch (v1.1) that the family
depends on, and makes the recipe gap visible / reproducible. Same value
proposition as CUT3R's three D24 cells.

### D29 · DA-V2 native-DIODE Table-2 cells off-paper on `domain=both` — outdoor preprocessing gap   🔬 INVESTIGATED 2026-05-29

End-to-end GPU run (H100, 2026-05-29) of the three `da-v2-{small,base,large}-diode-native`
cells (protocol `diode_dav2`, native `diode` loader, `scale_shift`, depth_clip
`[1e-3, 50]`, `domain=both`), against Depth Anything V2's own Table 2 DIODE
column (Yang et al. 2024, arXiv:2406.09414): ViT-S 0.073 / ViT-B 0.068 / ViT-L 0.066.

| cell | observed AbsRel | paper | Δ | n | match |
|---|---|---|---|---|---|
| da-v2-small-diode-native | **0.2196** | 0.073 | +201 % | 771 | ❌ |
| da-v2-base-diode-native  | **0.2182** | 0.068 | +221 % | 771 | ❌ |
| da-v2-large-diode-native | **0.2142** | 0.066 | +225 % | 771 | ❌ |

**Diagnosis (per-sample domain split of the same run, no extra GPU).** The
combined number is entirely driven by the outdoor split:

| split | n | mean AbsRel (ViT-S) |
|---|---|---|
| indoors | 325 | **0.0720** (≈ paper 0.073, −1.4 %) |
| outdoor | 446 | **0.3271** |

The indoor slice reproduces DA-V2's DIODE number almost exactly — confirming the
model, weights (`source="paper"` + `$DAV2_ROOT`), and indoor recipe are correct
(it's the same config that matched `da-v2-small-diode-indoor` at 0.0722). The
*native* `diode_dav2` outdoor handling is the divergence: DIODE outdoor spans
0–350 m with sky, and `scale_shift`-in-disparity + a hard `[1e-3, 50]` m clip
does not reproduce whatever DA-V2's eval did on outdoor.

**Corroboration that outdoor *can* be done right:** the MoGe-bundle DA-V2 DIODE
cell `da-v2-large-diode` (also `domain=both`, 771 samples, via `diode-moge-eval`
+ MoGe's homographic warp / affine-invariant-disparity) matched at **0.0529** vs
MoGe's reported 0.0533. So MoGe's preprocessing tames outdoor; the native
protocol's does not.

**2026-05-30 follow-up (GPU runbook / D29):**

- ``moge_fov_warp`` on native val is a **no-op** when RGB is already 1024×768
  (FoV warp leaves pixels unchanged; AbsRel identical to native).
- Outdoor 30-frame smoke: native **~0.19** vs ``diode-moge-eval`` **~0.05** —
  divergence is **MoGe HF bundle depth/mask** (log PNG + ``isfinite``), not warp alone.
- Experiment ``diode_dav2_moge_bundle`` (bundle loader + Table-2 ``scale_shift``):
  ViT-S **0.0618** vs 0.073 (−15 %); ViT-L **0.0543** vs 0.066 (−18 %) — much closer
  than native but still **under** paper (MISMATCH), same shape as D31/D32.

Per-domain (bundle + ``scale_shift``): indoor **0.052** / outdoor **0.069** (ViT-S) —
outdoor fixed vs native **0.33**, but combined still ~16 % under paper.

Probes: ``scripts/probe-diode-d29-warp.py``, ``scripts/probe-diode-d29-native-vs-bundle.py``.
Handoff: [`docs/D29_DIODE_TABLE2_HANDOFF.md`](D29_DIODE_TABLE2_HANDOFF.md).
Repros: ``da-v2-*-diode-moge-bundle`` (+ clamped experiment).

**2026-05-30 MoGe upstream harness (ViT-L):** `eval_baseline.py` on HF DIODE
bundle → ``rel`` **0.0529** (matches plumbline ``da-v2-large-diode``); paper Table 2
**0.066** still ~20 % higher (harder / different recipe). See
[`docs/DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md).

**Verdict:** ⚠️ off-paper, protocol gap — NOT a model/adapter bug and NOT tuned.
Two readings, both consistent with the data and neither verifiable without
DA-V2's own (unreleased) zero-shot eval script: (a) DA-V2's Table 2 DIODE number
is effectively indoor-only / outdoor-capped; (b) it is combined but under a
different outdoor depth treatment than `[1e-3, 50]`-clip. The `diode_dav2`
protocol comment already flagged the clip as the "first knob," but widening it
makes outdoor *worse* (more far returns), so the clip is not the lever. Left the
three jobs `pending`; YAMLs/protocol unchanged (per the no-tune rule). Result
JSONs: `s3://plumbline-bench/runs/backlog_20260529/results/`. The indoor cell
(`da-v2-small-diode-indoor`, value:null informational) remains the trustworthy
native-DIODE DA-V2 reference.

### D31 · DA-V2 native-ETH3D Table-2 — RGB/GT misalignment (fixed); full 13-scene still under paper   ✅ FIXED / 🔎 OPEN protocol 2026-05-30

First native ETH3D smoke (`da-v2-large-eth3d-native`, 3 scenes on S3, 158
frames) returned AbsRel **0.330** vs paper **0.131** (+152 %). Root cause was
**not** alignment mode (`scale_shift` vs `scale_shift_clamped` barely moved the
number) but a loader bug: per-view GT was rendered at the DA-V2 inference cap
(`pv_render_max_dim=518`, e.g. 345×518) while RGB stayed at native DSLR
resolution (~4135×6205) padded into the same canvas. Metrics compared each
518×345 GT pixel (full-FOV laser depth at render scale) against the
corresponding native-resolution pred pixel (only the top-left patch of the
FOV) — structurally misaligned.

**Fix (protocol-fidelity, not tuning):** `ETH3DDataset.resize_images_to_pv_render`
+ `protocols/eth3d_dav2.yaml` sets `resize_images_to_pv_render: true` so RGB is
area-resampled to the GT render size before inference. Single-sample check on
`courtyard/000001_v1` dropped from AbsRel **0.224 → 0.024** under `scale_shift`.

**Re-run on the same 3-scene S3 subset (H100, 2026-05-30):**

| cell | observed AbsRel | paper | Δ |
|---|---|---|---|
| da-v2-small-eth3d-native | **0.0758** | 0.142 | −47 % |
| da-v2-base-eth3d-native | **0.0713** | 0.137 | −48 % |
| da-v2-large-eth3d-native | **0.0679** | 0.131 | −48 % |

Variant ordering matches the paper (L < B < S), but all three land ~48 %
**under** tolerance on the 3-scene subset.

**Update (5-scene subset, 2026-05-30):** after staging `meadow` + `electro`
and re-running with the D31 fix, AbsRel moves **toward** the paper as harder
scenes enter the mean (meadow per-scene ~0.30):

| cell | observed (5 scenes, 218 frames) | paper | Δ |
|---|---|---|---|
| da-v2-small-eth3d-native | **0.1078** | 0.142 | −24 % |
| da-v2-base-eth3d-native | **0.0996** | 0.137 | −27 % |
| da-v2-large-eth3d-native | **0.0882** (4 scenes / 173 fr) | 0.131 | −33 % |

Still outside ±5 %, but the direction confirms the earlier miss was mostly
subset bias + the RGB/GT bug, not a wrong model.

**Definitive 13-scene run (454 frames, all train scenes staged, H100 2026-05-30):**

| cell | observed AbsRel | paper | Δ |
|---|---|---|---|
| da-v2-small-eth3d-native | **0.1012** | 0.142 | −29 % |
| da-v2-base-eth3d-native | **0.0936** | 0.137 | −32 % |
| da-v2-large-eth3d-native | **0.0888** | 0.131 | −32 % |

Gap barely moved vs the 8–12 scene interim (~−29–32 %); adding harder scenes did
not close the paper numbers. Variant ordering L < B < S matches Table 2.

**Verdict:** harness + D31 loader fix ✅; paper-match still **OPEN** (protocol
delta, same shape as D32 native-Sintel). JSONs:
`da_v2_*_eth3d_native_13scene_20260530.json` on localssd + S3
`tier_c_eth3d_13scene_20260530/results/` (also `tier_c_d31_20260530` subset runs).
Queue: `pending` / OFF-PAPER — do not tune YAML per GPU_RUNBOOK.

### D33 · DA-V2 native-ETH3D — GT source + eval recipe still diverge from ETH3D official   🔎 PARKED 2026-05-30

**Return checklist:** [`docs/ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md)

After the D31 RGB/GT alignment fix and the definitive 13-scene run (454 frames),
AbsRel stays ~29–32 % **under** paper (ViT-L **0.0888** vs **0.131**). Variant
ordering is correct; the gap is not sample-count or staging incompleteness.

**GT source audit (same run):**

| GT on disk | scenes | frames | ViT-L mean AbsRel |
|---|---|---|---|
| `dslr_scan_eval` (official DSLR-visible) | 3 | 158 | **0.0679** |
| `scan_clean` fallback only | 10 | 296 | **0.1000** |
| all 13 | 454 | **0.0888** |

Initial staging (`stage-eth3d-train-scenes.sh`) fetched `scan_clean` but not
`dslr_scan_eval`; only courtyard / delivery_area / facade had eval GT from an
earlier mirror. **`{scene}_dslr_scan_eval.7z` exists on eth3d.net for the other
train scenes** — staging script updated to fetch it; loader now rescans the
manifest when eval GT appears and tags per-view depth caches by GT source.

**Remaining protocol deltas (likely dominant):**

1. **Official ETH3D mono eval** uses pre-rendered `*_depth.7z` maps on
   **distorted** images (sparse floats, infinity = invalid, occlusion masks).
   Plumbline z-buffers MLP-aligned PLY into **undistorted** views at 518 px
   (denser valid mask) — closer to VGGT Table-3 machinery than to the sparse
   ETH3D depth dumps DA-V2 Table-2 may have used via MiDaS/DA-V1 lineage.
2. Frame set: plumbline evaluates **every** COLMAP image (454). MoGe's
   `.index.txt` lists **453** lines but the same **454** `scene/DSC_*` keys when
   mapped from manifest image names — not a coverage gap (2026-05-30 inventory).

**Re-run with all-`dslr_scan_eval` GT (ViT-L, 454 frames, H100 2026-05-30):**
AbsRel **0.0782** vs paper **0.131** (−40 %, still MISMATCH). Moved **further
under** paper vs the mixed-GT run (**0.0888**) — confirms official frustum-clipped
GT is not the missing lever; the gap is upstream of PLY choice.

JSON: `da_v2_large_eth3d_native_13scene_dslr_eval_20260530.json`. S3:
`tier_c_eth3d_dslr_eval_20260530/results/`.

**Official-depth probe (courtyard, ViT-L, 38 frames, 2026-05-30):**
`scripts/probe-eth3d-official-depth.py` loads `ground_truth_depth/dslr_images`
(float32, distorted 4032×6048) and compares to z-buffer `eth3d_dav2` on the
**same undistorted RGB** preds (geometry intentionally mismatched — interim step).

| GT source | mean AbsRel | valid pixels @518 |
|---|---|---|
| z-buffer (`eth3d_dav2`) | **0.0313** | **83.6 %** |
| official depth (nearest→518) | **0.0222** | **15.0 %** |

Official ETH3D depth is **much sparser** (~6× fewer evaluated pixels). Even
with undistorted/distorted mismatch, AbsRel stays *below* z-buffer on this
scene — so the remaining Table-2 gap is **not** explained by “we forgot the
sparse mask.” The open item is **distorted `dslr_jpg` RGB + pixel-aligned
official depth** (ETH3D doc + DA-V2 issue #281); issue reporter saw AbsRel
~0.5 with a mis-sized pipeline vs paper ~0.13.

Helpers: `load_eth3d_official_depth_map`, `official_depth_valid_mask` in
`datasets/eth3d.py`. Log (undistorted misalign only):
`$PLUMBLINE_WORK/runs/eth3d_official_depth_probe_courtyard.log`.

**Distorted RGB + pixel-aligned official depth (courtyard, ViT-L, 38 frames,
2026-05-30):** staged `courtyard_dslr_jpg.7z` (`images/dslr_images/`, 4032×6048,
same grid as official depth). Extended `scripts/probe-eth3d-official-depth.py`.

| Track | mean AbsRel | valid @518 |
|---|---|---|
| z-buffer (`eth3d_dav2` undistorted) | **0.0313** | **83.6 %** |
| official + undistorted pred (misaligned) | 0.0222 | 15.0 % |
| **official + distorted RGB (aligned)** | **0.0204** | **15.0 %** |
| official + distorted RGB @518 | 0.0208 | 15.0 % |

Pixel alignment does **not** move metrics toward paper **0.131** — courtyard
stays ~6× **under** on the ETH3D-documented sparse-depth recipe. Full 13-scene
harness at **0.0782** (`dslr_scan_eval` + z-buffer @518) is therefore unlikely
to close via official depth dumps alone; remaining deltas are probably **frame
inventory / aggregation** (454 COLMAP views vs MoGe 453, per-scene weighting) or
undocumented DA-V2 eval code (issue #281: mis-sized resize → AbsRel ~0.5, not
~0.13). Log:
`$PLUMBLINE_WORK/runs/eth3d_official_depth_probe_distorted_courtyard.log`.

### D32 · DA-V2 native-Sintel Table-2 under paper with MonST3R-lineage sky mask   🔎 PARKED 2026-05-30

**Return checklist:** [`docs/SINTEL_DAV2_TABLE2_HANDOFF.md`](SINTEL_DAV2_TABLE2_HANDOFF.md)

First native-Sintel run (`depth-anything-v2-sintel`) without `max_depth` included
Sintel sky pixels (~1e5 m) and returned nonsense (AbsRel 64k). Added protocol
`sintel_dav2` (final pass, `max_depth=70`, `scale_shift`, depth clip to 70 m).

Full training split (1064 frames, H100): **AbsRel 0.232** vs DA-V2 Table-2
**0.487** (−52 %, MISMATCH). Same “reads better than paper” shape as the D31
3-scene ETH3D subset — likely a remaining protocol delta (pass, boundary mask,
or DA-V2 §B benchmark detail), not a broken adapter. MoGe-bundle
`da-v2-large-sintel-moge` at **0.214** is Table 3, not comparable. JSON:
`/mnt/localssd/plumbline-work/runs/da_v2_sintel_native_fix_20260530.json`.

Depth Pro Sintel Table 1 (metric, no alignment): **δ₁ 0.2418** (2026-05-30,
`sintel_dav2` 0.001–70 m) and **0.2409** (2026-05-31, appendix Table 16
`sintel_depth_pro_metric` 0.01–80 m) — depth-range levers ruled out. 80-frame
smoke δ₁ ~**0.48**; full 1064-frame mean pulled down by hard tail scenes.
Upstream `ml-depth-pro` has no Sintel eval; README notes re-trained public
weights may not match paper. JSONs: `depth_pro_sintel_*_20260530.json`,
`depth_pro_sintel_table16_20260531.json`. Queue: `depth-pro-sintel` **blocked**.
**Blocked pages:** [`docs/BLOCKED.md`](BLOCKED.md) ·
[`docs/blocked/DEPTH_PRO_SINTEL_TABLE1.md`](blocked/DEPTH_PRO_SINTEL_TABLE1.md) ·
Middlebury / NuScenes / Sun-RGBD in `docs/blocked/`.

**2026-05-31 iBims sanity:** same weights, metric protocol on MoGe iBims bundle → δ₁ **0.8458** (100/100). Weights/adapter OK on indoor laser GT; Sintel gap is dataset-specific (synthetic, sky, scale), not a broken checkpoint.

**Pass probe (2026-05-30, `scripts/probe-sintel-pass.py`, 1064 frames):** `final`
**0.2321** vs `clean` **0.2224** (same `training/depth` GT) — `clean` is slightly
*better*, not worse; pass name does not explain the −52 % gap vs paper.

### D10 · VGGT-ETH3D 3-scene vs 13-scene split   🔬 INVESTIGATED 2026-05-27

**Closed end-to-end: full 13-scene run completed, paper number not
reproduced (+23.5 % over), but the structural reason is now visible
and is one scene, not a protocol gap.**

Plumbline's prior YAML ran only courtyard + delivery_area + facade
(3 of 13). 3-scene subset Overall 0.642 was 9.4 % under paper 0.709
— but that "closer than expected" was misleading: it happened to
exclude the scenes where plumbline diverges.

Full 13-scene run (PID 15445, vast.ai RTX 3090, 363 8-view windows,
~2 h wall):

    scene           Acc      Comp     Overall  Overall_med
    kicker          0.102    0.052    0.077    0.043
    office          0.075    0.101    0.088    0.049
    pipes           0.122    0.078    0.100    0.058
    electro         0.476    0.166    0.321    0.129
    relief_2        0.583    0.403    0.493    0.326
    facade          0.770    0.299    0.535    0.151
    courtyard       0.469    0.736    0.603    0.262
    relief          0.644    0.579    0.612    0.400
    meadow          0.992    0.556    0.774    0.532
    delivery_area   0.513    1.065    0.789    0.380
    terrace         0.795    0.813    0.804    0.479
    playground      0.221    1.759    0.990    0.574
    terrains        0.208   10.185    5.197    5.082   ← outlier
    -----------------------------------------------------
    aggregate       0.459    1.292    0.875    0.651
    paper Table 3   0.901    0.518    0.709    (no median)

**Headline**: Overall 0.875 vs paper 0.709 = **+23.5 % MISMATCH**.

**But it's one scene.** `terrains` Completeness is 10.18 m, 13× the
average of every other scene (median 0.56) and 20× the paper mean.
Excluding `terrains`, the 12-scene mean Overall is **0.515**
(plumbline) vs paper 0.709 — i.e., plumbline is 27 % *tighter* than
paper across the other 12 scenes. The aggregate is dominated by one
pathological case.

**Plumbline-tighter pattern is consistent.** Accuracy across all 13
scenes: plumbline 0.459 vs paper 0.901 (49 % tighter); Completeness
(without terrains): plumbline ~0.55 vs paper 0.518 (6 % looser). The
"plumbline tighter on Acc, slightly looser on Comp" shape that
appeared in the 3-scene subset (D4) replicates across the full
split — modulo the terrains outlier.

**What's terrains-specific.** GT files (scan1.ply, scan2.ply,
scan_alignment.mlp) are present, same structure as working scenes.
Acc is 0.21 (well-behaved); only Comp blows up to 10 m. The
asymmetry is "GT_point → nearest pred_point" diverges, which means
either (a) pred point cloud is missing large regions of the scene
that GT covers, or (b) there's a per-scene scale mismatch that
ICP-per-window aligned poorly. The 5.2 m median (not just mean)
confirms it's not a tail-of-tail outlier within terrains but a
whole-scene shift.

**Resolution path**: filed as ⚠️ off-paper with documented terrains
outlier in REPRODUCTIONS.md. Two follow-ups left for a future
session:
- (a) Per-window ICP-alignment dump for terrains — confirm whether
  ICP is mis-aligning the predicted scene relative to GT scan.
- (b) Compare to paper's own per-scene numbers if VGGT authors
  publish them (currently unavailable per the 2026-05-27
  supplementary search — paper publishes only the 13-scene mean).

The 3-scene subset record stays in `vggt_eth3d_subset_chamfer.yaml`
as an informational regression detector. Result JSON archived at
`docs/runs/plumb_vggt_eth3d_13_20260527T163122Z.json`.

