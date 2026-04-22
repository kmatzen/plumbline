# Discrepancies

Catalog of every adapter / loader / protocol / solver / citation mismatch
surfaced to date, grouped by layer. Triage: is this an **adapter bug**, a
**loader** issue, a **protocol** mismatch, an **alignment-solver** choice,
or a **citation** error?

Status legend:

- ✅ **FIXED** — change landed on `main`.
- 🧪 **FIX-PENDING-VERIFY** — change landed, waiting on a GPU re-run.
- 🔎 **SUSPECTED** — hypothesis + diagnosis path; not yet reproduced.
- 📅 **DEFERRED** — known root cause, scoped for v0.2+.
- 📝 **EXPLAINED-NOT-A-BUG** — discrepancy is real but expected given
  the papers / solvers involved; documented to prevent rediscovery.

Fixed entries are one-liners with commit SHAs — the full diagnosis lives
in the commit message. Open entries keep context because future readers
need it to act.

---

## Adapter bugs

### D1 · GeoWizard — `generator` kwarg not accepted upstream   ✅ FIXED (`c50201e`)

### D2 · GeoWizard — upstream diffusers API drift (`PositionNet`, `dual_transformer_2d`)   ✅ FIXED (`a35c4f5`, shim)

### D3 · VGGT + DTU — per-sample vs scene-aggregated chamfer   ✅ FIXED (`ad924e9`, `2c140b3`)

Protocol now pins `aggregation: scene` + `scene_voxel_size: 0.01`.
2026-04-22 probe (scan1, 15 cached VGGT predictions): ICP 43 mm, Umeyama
81 mm, none 1147 mm — alignment works; the 147× residual vs paper's
0.382 mm was per-sample chamfer instead of scene-merged Acc/Comp. GPU
verification blocked on **D20** (per-sample ICP is too slow to finish a
full-scan sweep).

---

## Loader / dataset issues

### D4 · ETH3D scene_aggregation empty metrics (missing `scan_clean` GT)   ✅ FIXED (`f89cdc4`)

S3 cache only had `dslr_scan_eval`; `scan_clean/scan*.ply` is the point
cloud the loader needs to populate `Sample.point_cloud_gt`. Fetched via
`py7zr`, pushed to S3. GPU verification blocked on **D20**.

### D5 · DIODE outdoor prediction outliers (MoGe `drop_max_depth`)   ✅ FIXED (`7fd6ff6`)

See D19 for the residual mean-blowup after this fix — the `drop_max_depth`
filter is on GT, not predictions.

### D6 · DIODE Moge-eval loader `split` kwarg   ✅ FIXED (`ae046ab`)

### D7 · KITTI annotated-depth not in S3 cache   ✅ FIXED

Retroactive: fetched `data_depth_annotated.zip` (14 GB) from avg-kitti,
pushed to `s3://plumbline-bench/datasets/kitti/depth_annotated/`.
Future `stage_all_data.sh` runs pull it automatically.

---

## Protocol mismatches

### D8 · MoGe KITTI — structural protocol delta   🧪 FIX-PENDING-VERIFY

Symptom: `moge-vitl-kitti` landed 9.4 % off paper (0.0408) under the
Monodepth2 Eigen / Garg protocol; Eigen crop was worse (20.4 % off).

Root cause (subagent read of `microsoft/MoGe` @ HEAD): paper eval uses a
bespoke HF-bundle sample list (`KITTI/.index.txt`, 652 pre-warped frames
at ~750×375), no crop, no `[1e-3, 80]` m clip, depth encoded in MoGe's
log-PNG format, and a disparity-space LSQ with the `1/gt.max()` floor
(D19). Nothing matches Monodepth2-Eigen-Garg.

Fix (this commit): new `KITTIMogeEvalLoader` + `kitti_moge_eval` protocol
mirroring `DIODEMogeEvalLoader` / `diode_moge`. `moge-vitl-kitti` opts
in (alignment: `scale_shift_clamped`); D9 (Marigold) and D18 (GeoWizard)
share the same loader/protocol but keep their own `scale_shift_depth`
fit. GPU verification pending (needs the HF bundle staged to
`$KITTI_MOGE_ROOT`).

### D9 · Marigold KITTI — probable same-protocol delta   🧪 FIX-PENDING-VERIFY

Symptom: 10.1 % off paper under `kitti_eigen_garg` (17.8 % off under
Eigen crop).

`marigold-v1-1-kitti` now points at the shared `kitti_moge_eval` protocol
+ `kitti-moge-eval` loader (D8 fix); keeps its own
`scale_shift_depth` alignment (Marigold's eval.py fits in depth space,
not disparity). GPU verification will confirm whether the shared
loader/protocol closes the 10 % gap, or whether Marigold has a
secondary knob (e.g., `alignment_max_res`) that still differs.

### D10 · VGGT-ETH3D — 3-scene subset vs full-split paper target   📅 DEFERRED

Symptom: plumbline's YAML runs courtyard + delivery_area + facade
(3 scenes); paper's Table 3 Overall 0.709 is cross-scene mean across
the 13 ETH3D train scenes. A 3-scene subset genuinely can't paper-match
the 13-scene aggregate.

Resolution path: one of (a) stage the remaining 10 scenes — +~14 GB
data — and run the full split; (b) extract a per-scene breakdown
from VGGT supplementary if it exists; (c) demote to informational
with a tolerance of 0.20 and cite "3-scene subset for fast sanity
check, not paper-match". Option (c) is what the 2026-04-20 audit
intended; the `tolerance_relative: 1.0` that was there before today's
repo-wide 5 % cap encoded that intent.

---

## Alignment / solver choice

### D11 · Plumbline's `scale_shift_robust` beats MoGe's own LSQ on NYU   📝 EXPLAINED (`c14d776`)

MoGe's own eval uses plain `torch.linalg.lstsq`; plumbline's Huber-IRLS
solver downweights NYU's long-tailed disparity-residual outliers. YAMLs
switched to `scale_shift` (plain LSQ) to match paper protocol. Fix
cascaded through moge-vitl-{kitti,diode-both,diode-indoor}, moge2-vitl-*,
`protocols/gso_moge.yaml`.

### D12 · KITTI Eigen-crop hypothesis   📝 EXPLAINED (rejected empirically, `fb58b90`)

Switching to Eigen crop made both MoGe-KITTI and Marigold-KITTI *worse*.
Real deltas are D8 + D9 (structural protocol). `protocols/kitti_eigen_crop.yaml`
preserved for a paper that genuinely uses it.

---

## Citations

### D13 · DA-V2 Large NYU — two competing paper numbers   ✅ FIXED (`58fc159`)

DA-V2 paper says 0.045, MoGe Table 3 re-eval says 0.0420. Plumbline
reproduces 0.0427 under plain LSQ (1.6 % off MoGe, 5.2 % off DA-V2).
Pinned at MoGe's 0.0420; citation records both.

### D14 · DA-V2 Base NYU citation UNVERIFIED → verified 0.049   ✅ FIXED (`603e717`)

### D15 · DA-V2 NYU ~0.002 AbsRel systematic downshift   📝 EXPLAINED

Plumbline observes 0.002 — 0.003 AbsRel *below* paper across all three
variants (S/B/L). Not Base-specific — Base just looks worst because the
paper denominator is smaller. Per-sample cross-variant Pearson 0.89–0.97
(models agree on hard samples, just score uniformly lower).

Paper-vs-HF checkpoint swap had ~0 effect (<0.0005 delta). Real
suspects: NYU Eigen-crop convention, rawDepths-field filter interaction,
or small protocol detail in DA-V2's own eval we're not replicating.
Below the priority threshold for v0.1 — **document and move on**.

### D16 · MoGe-DIODE-indoor cited combined-val paper value   ✅ FIXED (`603e717`)

Demoted to `source_confidence: approximate`, `value: null`; drops out
of the verified queue.

### D17 · GeoWizard NYU 10 % off paper   🔎 SUSPECTED

Observed `geowizard-nyuv2` AbsRel = 0.0573 vs paper 0.052 — 10.2 % off,
after D1 (generator) + D2 (shim) fixes.

Candidates:

1. **RNG divergence from paper.** Plumbline seeds
   `torch.manual_seed(seed + sample_index)` per-sample (D1 fix). Paper
   may use a single fixed seed for the ensemble latents; per-sample
   re-seeding samples a different latent-chain distribution and can
   shift mean AbsRel ~10 % on diffusion eval.
2. **Paper protocol alignment**: plumbline uses `scale_shift_depth`
   (depth-space LSQ) per YAML. Need to confirm from the public
   GeoWizard eval script that this matches.
3. **Protocol resolution**: paper uses processing_res=768, same as
   plumbline's YAML. Unlikely to be the source.

Priority: low — informational-smoke side. Defer to v0.2 with D9.

### D18 · GeoWizard KITTI 35 % off paper   🧪 FIX-PENDING-VERIFY

AbsRel = 0.131 vs paper 0.097, 35.2 % off under the old Monodepth2-Eigen
+ Garg setup. `geowizard-kitti` now points at the shared
`kitti_moge_eval` protocol + `kitti-moge-eval` loader (D8 fix), with
`scale_shift_depth` alignment (unchanged). GPU verification will show
whether sharing the loader/protocol closes the 35 % gap or GeoWizard has
a secondary protocol detail that still differs.

### D19 · MoGe-DIODE-both mean AbsRel=2481 after `drop_max_depth`   🧪 FIX-PENDING-VERIFY

Median AbsRel was 0.025 (bullseye vs paper 0.040) but mean was 2481 —
outdoor outliers with near-zero post-alignment disparity inverted to
enormous depths. Root cause: MoGe's `drop_max_depth` filters **GT**, not
predictions; they also apply `clamp_min(1 / gt_depth[mask].max())` on
predicted disparity before inverting to depth (`moge/test/metrics.py`
~line 210).

Fix (this commit): new `scale_shift_clamped` alignment mode in
`src/plumbline/metrics/alignment.py` — same LSQ fit as `scale_shift`,
plus a per-sample disparity floor at `1/gt.max()` before inverting.
`moge_vitl_diode_{both,indoor}.yaml` opt in. Unit test in
`tests/test_metrics.py::TestAlignment::test_scale_shift_clamped_caps_at_gt_max`
constructs an outlier-pixel scenario and asserts plain lets it diverge
while clamped caps at `gt.max()`.

GPU verification pending — the next rental re-runs both DIODE YAMLs
(cache-hit on prediction; metrics-only re-score).

---

## Performance

### D20 · Scene-aggregation chamfer slow   🧪 FIX-PENDING-VERIFY

Per-sample `_aligned_point_map` with `pointcloud_alignment=icp` ran a
full ICP refine against the scene's ~200 K-point GT cloud on every
sample (~45 refines/scene on ETH3D, ~42/scan × 22 scans on DTU), which
dominated wall time on the 2026-04-21 rental run.

Fix (this commit): in `aggregation='scene'` mode, per-sample alignment
is downgraded to cheap `camera_centers` Umeyama; a single scene-level
ICP refine runs on the fused + voxel-downsampled prediction cloud in
the merge loop. Regression test in `tests/test_mvs_pipeline.py` counts
`icp_similarity` calls and asserts exactly one per scene.

GPU verification (D3 VGGT-DTU, D4 VGGT-ETH3D) still pending — the fix
only changes wall time; correctness against paper chamfer is the
next-session gate.

---

## Priorities for the next session

Laptop-side prep: **none open.** All previously-queued laptop work (D20,
D19, D8/D9/D18) has landed on `main`. Any further laptop work is
research-scoped (D15 NYU bias, D10 ETH3D full split) rather than
paper-match-blocking.

GPU-side verification (fixes already on `main`):

1. **D3** — VGGT-DTU chamfer under `aggregation: scene` on all 22 scans.
2. **D4** — VGGT-ETH3D multiscene chamfer under the new once-per-scene
   ICP path (D20 fix).
3. **D19** — MoGe-DIODE-{both,indoor} re-score under `scale_shift_clamped`.
   Prediction cache-hits; metrics-only pass.
4. **D8** — `moge-vitl-kitti` under the new `kitti_moge_eval` protocol.
   Needs HF-bundle `KITTI.zip` staged to `$KITTI_MOGE_ROOT`.
5. **D9** — `marigold-v1-1-kitti` under the same protocol.
6. **D18** — `geowizard-kitti` under the same protocol.

Nice-to-have (low priority):

6. **D10** — VGGT-ETH3D full 13-scene split, or demote to informational.
7. **D15** — DA-V2 NYU ~0.002 bias. Inspect Eigen-crop + rawDepths
   interaction.

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/20260421-run1/` preserve the observations
from each run so the report can be regenerated from any point.
