# Changelog

All notable changes to plumbline are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public API may change between 0.x releases.

## [Unreleased]

### Added
- **CUT3R now reproduces on all three depth benchmarks** (NYU, KITTI, Bonn) —
  plumbline-native, closing the off-paper gaps that previously showed only as
  `ℹ️` cells:
  - `cut3r-nyuv2-prepared` — **AbsRel 0.0858 vs 0.086** (Table 1). New
    `nyu-cut3r-eval` loader over CUT3R's exact prepared NYU set (HuggingFace
    `sayakpaul/nyu_depth_v2` val), staged by `scripts/stage_nyu_cut3r_eval.sh`.
  - `cut3r-kitti-lineage` — **AbsRel 0.0953 vs 0.092** (Table 1), 1269/1269, on
    the existing `kitti_dust3r_lineage` protocol. Staged via
    `scripts/stage_kitti_cut3r.sh`.
  - `cut3r-bonn-110` — **AbsRel 0.0768 vs 0.078** (Table 2, video/per-sequence
    scale). Staged via `scripts/stage_bonn_cut3r.sh`.
- **`median_lineage` alignment mode** (`align_scale_ratio_of_medians`) — the
  dust3r-lineage eval code's `s = median(gt)/median(pred)` (ratio-of-medians),
  distinct from `median`'s `median(gt/pred)` (median-of-ratios). This was the
  root cause of CUT3R's per-frame off-paper gap (NYU 0.0777 → 0.0858). The
  estimator is **paper-specific**: MonST3R-NYU stays on `median` (median_lineage
  makes it *worse*), so it is applied only to the CUT3R cells.
- **`scale_weiszfeld` alignment mode** (`align_scale_weiszfeld`) — CUT3R's video
  `eval/video_depth --align scale` (robust scale-only Weiszfeld IRLS), used by
  the Bonn per-sequence cell.
- **Bonn loader `frame_selection` / `frame_start`** — `frame_start=30` selects
  CUT3R's `rgb_110 = sorted(frames)[30:140]` set (per MonST3R's
  `prepare_bonn.py`); the prior `[0:110]` was the wrong set.

### Verified (analysis, no code change)
- **CUT3R inference is byte-verified faithful** to upstream: plumbline's
  `_build_views` preprocessing is byte-identical (`max|Δ|=0`) to CUT3R's
  `load_images_for_eval`, and the adapter calls CUT3R's own `inference()`
  (`scripts/_cut3r_nyu_input_diff.py`). So every CUT3R off-paper number was an
  eval-recipe/data difference, not an adapter bug — each root-caused and fixed
  above rather than tuned.

## [0.2.0] — 2026-06-14

First release that **bundles** the dust3r-lineage + DAGE model code instead of
cloning it. As a result the published wheel now contains NonCommercial vendored
source — **the distribution as a whole is usable for non-commercial purposes
only** (plumbline's own code stays Apache-2.0). See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

### Fixed
- **VDA relative variants now emit depth, not disparity.** The relative
  Video-Depth-Anything variants return the Depth-Anything-V2 DPT head output,
  which is *disparity* (inverse depth); the adapter passed it through as `depth`,
  so the harness's `scale_shift` alignment (which itself fits in inverse-depth
  space) fit `s/D + b` instead of the correct MiDaS `s·D + b`. Relative variants
  now invert disparity→depth, matching the sibling DA-V2 adapter; metric variants
  are unchanged (already meters).
- **Package version no longer drifts from `pyproject`.**
  `src/plumbline/_version.py` reported `0.1.0` while `pyproject.toml` was
  `0.2.0`, so an installed package's `__version__` / `plumbline --version` was
  wrong. Synced to `0.2.0` and guarded by `test_version_sync` so the two can't
  diverge again.
- **Metric models scored under a rescaling alignment now warn.** The runner
  defaults `scale_alignment` to `median` regardless of `is_metric`, so a metric
  cell that forgets `scale_alignment: none` was silently scale-fit to GT (hiding
  its true metric error). The runner now emits a loud warning, and a false
  `depth_pro` docstring claiming `is_metric` auto-selects `none` is corrected.
- **UniK3D now loads for inference.** The vendored inference subset prunes
  `unik3d.ops.losses`, but `UniK3D.build_losses` (run from `__init__` →
  `from_pretrained`) imported it unconditionally, so the model raised
  `ModuleNotFoundError` and could not be instantiated at all. `build_losses`
  now no-ops when the pruned module is absent (the loss dict is read only on the
  training/loss path, never in `infer`).
- **MASt3R pose is now actually MASt3R.** For N≥3 the adapter previously
  recovered pose by running *dust3r's* `PointCloudOptimizer` on MASt3R's point
  maps with the matching head discarded ("MASt3R-via-dust3r-GA") — not MASt3R's
  method. `MASt3RAdapter` now defaults to `pose_backend="sparse_ga"`, calling
  MASt3R's own `sparse_global_alignment` (dense reciprocal matching →
  two-stage global alignment); the legacy path stays available as
  `pose_backend="dust3r_ga"`. On a 50-clip RealEstate10K subset this moved
  MASt3R from 0.674 → **0.850** mAA@30 (vs DUSt3R 0.664), restoring the +18-pt
  MASt3R-over-DUSt3R lead the paper reports (+15.2) and that the old path
  collapsed to +1. A controlled CO3Dv2 A/B (+2.3 pt vs RE10K's +17.6 pt)
  confirms narrow baselines hid the gap; the ✅ `mast3r-co3dv2-pose` cell was
  measured on the old path and survives the fix — full-410 re-run landed
  2026-06-03 on the `sparse_ga` path at **0.8581 vs 0.818** (+4.9 %, MATCH),
  superseding the legacy 0.7960.
- CPU-side correctness and robustness bugs found in a code-review sweep
  (#22): a pointmap nearest-neighbour chunk size that collapsed to one row
  per chunk (operator precedence in `1 << 20 // b`), a NumPy 2.0/2.1
  `np.unique(axis=0, return_inverse=True)` 2-D-inverse regression in
  `voxel_downsample`, a `voxel_downsample` call on `scene_voxel_size <= 0`,
  and silent inflation of the evaluated-sample count when a prediction
  produced no metrics. Adds a `min_samples` reproduction floor that forces
  `paper_match=no` on a sample-count shortfall (the D28 footgun).
- CUT3R checkpoint loading under torch ≥ 2.6, which now defaults
  `weights_only=True` and rejects the checkpoint's embedded
  `omegaconf.DictConfig` (#26).
- Corrected a mislabeled `source_confidence` on `da-v2-small-nyuv2` (#24).

### Added
- **Out-of-tree adapter discovery.** Third-party packages can register models /
  datasets via the `plumbline.adapters` entry-point group (loaded after builtins
  so a plugin can't shadow a builtin name; broken plugins are soft errors), and
  `register_model` / `register_dataset` are now exported at the top level — so
  `pip install plumbline-bench` then adding your own algorithm needs no clone.
- **UniK3D's first reproduction cell** (`unik3d-large-nyuv2`). UniK3D-Large
  (CVPR 2025) on the NYUv2 Eigen test, metric depth with **no alignment**,
  reproduces the paper's Table 18 zero-shot NYUv2 row out of the box: AbsRel
  **0.0749** vs 0.074 (+1.2%, ✅), δ₁ 0.9656 vs 0.965 (≈exact), RMSE 0.2632 vs
  0.259 — 654/654 on a GTX 1080 Ti, UniK3D's default inference bounds (no
  `resolution_level` tuning). Brings the verified-cell count to 39 (32
  mono-depth) and adds UniK3D as a new model family in the matrix.
- **Two more UniK3D zero-shot metric cells** (ℹ️ off-paper, metric/no-align):
  `unik3d-large-eth3d` (UniK3D Table 21) and `unik3d-large-diode` (Table 22),
  each pinned to UniK3D's own dataset-class depth cap ([0.01, 50] m for ETH3D,
  [0.01, 25] m for DIODE Indoor) via new `eth3d_unik3d_metric` /
  `diode_indoor_unik3d_metric` protocols. **DIODE Indoor**: AbsRel 0.1509 vs
  0.161 (6.3% under; δ₁ 0.754 / RMSE 0.718 both better) on the exact 325/325
  official indoor val set — a tight reproduction that narrowly misses the 5%
  band. **ETH3D**: AbsRel 0.1544 vs 0.236, δ₁ 0.814 vs 0.687, RMSE 1.07 vs 2.63
  — off-paper *better* on the 454 native-resolution DSLR frames; the residual
  is a frame-set/resolution protocol difference vs UniK3D's HDF5-packed eval.
  Both stay ℹ️ (no verified-count change).
- **DIODE loader `depth_range` kwarg** (default `None` = no cap, so every
  existing affine-invariant DIODE cell is unchanged) — masks GT outside a
  metric depth range, needed for the UniK3D DIODE Indoor [0.01, 25] m cell.
- **`python -m plumbline`** now works as an alias for the `plumbline` console
  script (added `__main__.py`), so the CLI is reachable even where the script
  isn't on `PATH`.
- **"Did you mean" typo hints** on unknown `--model` / `--dataset` / adapter
  names: a single-character slip like `nyuv` → `nyuv2` gets a direct suggestion
  (`difflib`) ahead of the full `Known:` list.
- **`sq_rel` and `rmse_log` depth metrics**, completing the classic Eigen /
  KITTI-split column set (`AbsRel, SqRel, RMSE, RMSE-log, δ₁/₂/₃`). Every depth
  evaluation now reports them alongside the existing metrics; `log10_error` is
  also now re-exported from `plumbline.metrics`.
- **`scripts/stage_realestate10k.py`** — disk-careful RealEstate10K frame
  scraper (yt-dlp + ffmpeg, low-res, per-clip cleanup, free-space guard,
  resumable), unblocking the dust3r/mast3r/vggt RealEstate10K pose cells.
- **`reproduce-pose` project skill** (`.claude/skills/`) codifying
  stage-pose-dataset → reproduce → compare-mAA on the GPU box (now also covers
  the trajectory-ATE family: dage/cut3r/monst3r on Sintel / TUM-Dynamics).
- **DAGE Table 4 pose — TUM-Dynamics & ScanNet columns.** New `tum-dynamics`
  loader (8 freiburg3 dynamic sequences, MonST3R `prepare_tum.py` prep replicated
  at read time) and `scannet-video-pose` loader (MonST3R `color_90`/`pose_90`
  layout), extending the Sim(3)-aligned trajectory-ATE apparatus beyond Sintel.
  `dage-tum-pose` is a new ✅ cell — **ATE 0.0136 vs 0.014** (−2.9 %, 8/8 clips,
  GTX 1080 Ti). Staged via `scripts/stage_tum_dynamics.py` (public, no ToS;
  member-selective ~366 MB). ScanNet cells are code-ready, data-blocked on
  ToS-gated raw ScanNet. (#44)
- **Vendored model code** under `src/plumbline/_vendor/` for DAGE, CUT3R,
  DUSt3R, MASt3R, and MonST3R (CC BY-NC[-SA]) — no clones needed; `$<m>_ROOT`
  still overrides the vendored path for a dev checkout. The `curope` CUDA RoPE
  extension is vendored as source (required for CUT3R, optional speedup for the
  others). GPL/unlicensed models (GeoWizard) stay clone-only.
- **DAGE adapter** (feed-forward video geometry + pose) plus its Table-4
  baseline reproductions (DAGE / CUT3R Sintel pose).
- **SUN-RGBD native loader** + `DepthProAdapter(use_gt_focal=True)`, closing the
  Depth Pro Table-1 δ₁ cell.

### Changed
- **`install.py` is now the unified Python-dependency view.** Vendored models use
  a new `kind="vendored"` whose only install surface is explicit runtime `pip`
  deps (+ checkpoint/curope build where noted) — no `git clone`, no cloned
  `requirements.txt`. `plumbline doctor` probes a signature dep per model.
- Per-model upstream-license audit with a `vendorable` gate (permissive +
  NonCommercial may be vendored; GPL/unlicensed/bespoke may not).
- The published wheel bundles the NonCommercial `_vendor/*` trees; the package
  metadata (SPDX expression + "Free for non-commercial use" classifier) and the
  bundled `LICENSE` + `THIRD_PARTY_NOTICES.md` reflect this.
- Clearer Depth Anything V2 paper-backend errors that distinguish "repo not
  found" from "repo present but a dependency (e.g. opencv-python) is missing",
  with a matching install note (#23).
- Recorded Metric3D-v2's hard `mmcv` install requirement (the note had said
  "no extra package") (#25).
- Recorded the observed `marigold-v1-1-kitti` result in its YAML note (#27).
- Restructured `docs/DISCREPANCIES.md` into an outstanding-work tracker and
  pruned resolved-issue cruft.

## [0.1.0] — 2026-05-31

First public release.

### Added
- Evaluation harness for 3D geometric foundation models: a model-adapter
  registry, dataset-loader registry, named protocol presets, a results
  runner, and the `plumbline` CLI (`list-models`, `list-datasets`, `run`,
  `reproduce`, `queue`, `install`, `doctor`).
- 12 model adapters: Depth Anything V2 (incl. metric Indoor/Outdoor), Depth
  Anything 3, Metric3Dv2 (S/L/Giant2), MoGe-1, MoGe-2, Marigold v1-1,
  GeoWizard, Depth Pro, MASt3R, DUSt3R, VGGT, CUT3R, and MonST3R.
- 12 dataset loaders: NYUv2, KITTI, DIODE, ETH3D, DTU MVS, CO3Dv2, 7-Scenes,
  GSO, iBims-1, Sintel, ScanNet, and Bonn RGB-D Dynamic.
- Depth, pose (absolute + pairwise relative-pose AUC, RRA/RTA, trajectory
  ATE/RPE), and point-map (7-DoF similarity / per-view-masked chamfer)
  metric families.
- **33 PDF-verified paper-match reproductions** (29 mono-depth + 4 multi-view
  pose/trajectory), each audited table-+-column-+-row against the source
  paper. See [`REPRODUCTIONS.md`](./REPRODUCTIONS.md).

[Unreleased]: https://github.com/kmatzen/plumbline/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kmatzen/plumbline/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kmatzen/plumbline/releases/tag/v0.1.0
