# Changelog

All notable changes to plumbline are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public API may change between 0.x releases.

## [Unreleased]

### Fixed

- Prediction-cache filenames now append a stable hash whenever sanitization
  changes a sample ID, preventing collisions such as `scene/0001` versus
  `scene_0001`; atomic writes use truly unique temp files so concurrent threads
  in one process cannot race. NYU CUT3R sample IDs are now dataset-prefixed.
- The DIODE, KITTI, and ETH3D MoGe-bundle loaders share one audited decode/FoV
  warp path. ETH3D parses and filters its lightweight index before constructing
  the MoGe pipeline, so empty filters fail without loading the model stack.
- **Trajectory ATE/RPE crashed with unbounded evo** — evo 1.38 removed the
  `est_name` keyword from `ape()` / `rpe()`. The integration now uses the
  compatible API, evo is a base dependency pinned below 1.38 because the
  vendored MonST3R reference evaluator also uses the older API, and trajectory
  tests no longer skip in CI. The runner also stopped
  swallowing arbitrary trajectory exceptions as NaNs; only genuine evo
  geometry/metric degeneracy is downgraded to NaN.
- Public coverage documentation now matches the live registries (18 model
  adapters, 23 loaders / 16 dataset families, 47 verified cells). π³ audit
  pages now reflect that the adapter was restored for pose while its
  DTU/ETH3D reconstruction path remains explicitly blocked.
- **`plumbline doctor` produced both false positives and false negatives.** It
  checked only one "signature" dependency, so DAGE read OK with two runtime
  packages missing, while base-complete vendored adapters read MISSING because
  their private package roots are activated lazily. Specs now declare their
  complete lightweight import probes. Metric3Dv2's real `mmengine` +
  `mmcv-lite` install step is also registry-driven instead of being documented
  as a no-op.
- **`plumbline install moge` tried to install incompatible MoGe 3 from moving
  upstream HEAD**, despite the compatible MoGe v1/v2 inference code already
  being vendored. On macOS this failed immediately on MoGe 3's
  `flex-gemm`/Triton dependency; elsewhere it would install the wrong API.
  MoGe is now correctly registered as vendored, its dataset tests exercise the
  bundled pipeline instead of skipping on `import moge`, and stale external
  installation instructions were removed.
- Vendored install plans no longer present optional development-root variables
  as fake installation steps. Base-complete adapters are true no-ops, while
  manual checkpoints and environment overrides are printed as non-executable
  guidance (not shell exports that cannot persist from the installer process).

### Added

- Camera-tracking evaluations can emit versioned trajectory review artifacts
  directly with `plumbline run --tracking-review-dir`, ready for deterministic
  blinded A/B pairing, human or VLM votes, calibration, and task-bootstrap
  Bradley--Terry rankings.

- Camera-tracking review harness inspired by the scorer architecture of
  3DCodeBench/4DCodeBench: versioned pose/metric/manifests, deterministic
  GT-scaled diagnostic PNGs, validity gates, blinded A/B bundles, append-only
  human votes, balanced pair selection, Bradley–Terry ratings, and task-level
  bootstrap confidence intervals. `plumbline tracking-review` exposes the
  render/pair/vote/rank workflow. The design documents human/VLM calibration,
  multi-objective Bayesian optimization, and why RL is reserved for genuinely
  sequential tracking policies.

Codebase review sweep (issues #163–#182). Highest-impact first:

- **Prediction cache ignored `max_views`** (#163) — the `keep = min(n, max_views,
  cap)` trim ran *after* the cache lookup, so a cell run at `max_views=8` and
  later at `max_views=2` shared one entry: the 8-view prediction was served and
  `_trim_sample_to_views` trimmed GT to match, silently reporting the wrong
  protocol. The view count is now part of the cache key. Only *trimmed* samples
  are tagged, so existing entries stay valid.
- **Sintel + DIODE manifest caches were poisoned by scene-filtered scans**
  (#165, #166) — both saved a scene-filtered scan into a scenes-agnostic
  manifest, so a single-scene run silently hid every other scene from later
  runs; DIODE additionally keyed on scene *count*, so two different
  single-scene runs collided and the second evaluated **zero** samples. Both now
  cache the full scan and filter after load (the ETH3D pattern); manifest names
  bumped to `_v2` to invalidate poisoned caches. Regression tests added.
- **Reproduction configs silently dropped `delta_thresholds` /
  `f_score_threshold`** (#164) — validated by `expected_metric_keys` but never
  forwarded to `evaluate()`, so a cell pinning a non-default δ set or F-score
  threshold was scored at the runner's defaults. Also: `expected_metric_keys`
  omitted `sq_rel` / `rmse_log`, rejecting valid KITTI-Eigen cells at load time;
  it now derives from the runner's own key list.
- **Installed wheels could not resolve `protocols/`** (#167) — the YAMLs shipped
  in neither wheel nor sdist and `paths.py` assumed a git checkout, so every
  `protocol:`-bearing cell raised `FileNotFoundError` for pip users. Both trees
  are now packaged, with checkout-preferred fallback resolution and a CI check.
- **`scene_voxel_size: 0` crashed the scene-merge ICP path** (#173) — a
  documented config (DTU's choice), unguarded on one of three call sites, and it
  failed *after* the full inference run, taking the unsaved report with it.
- **NYUv2 `max_gt_depth` was a silent no-op** without `apply_eigen_crop` (#174).
- **Corrupt cache entries were never evicted** (#172) — a failed load fell back
  to re-running inference but left the file, so `has()` stayed true and every
  future run silently redid the work. Cache writes also now use a pid-scoped
  temp name so concurrent runs can't promote a corrupt archive.
- **MonST3R reported `flow_refinement: False` on video-pose runs** that did use
  the flow / motion-mask / temporal pipeline (#182).
- **`load_ply_xyz` silently misparsed non-float32 PLYs** (#182) — double-typed
  or non-`x/y/z` leading properties produced garbage coordinates with no error,
  and `binary_big_endian` died on a confusing decode error. All now raise.
- Empty reproduction runs raised a bare `StopIteration` (recorded by the queue
  as the useless `"StopIteration: "`); they now raise a descriptive error (#176).
- `--scale-alignment` was unvalidated with stale help text listing 4 of 9 modes;
  modes now come from a single `ALIGNMENT_MODES` source and are validated before
  model load (#176).

### Changed
- `evaluate()` no longer takes `device` — it was accepted, documented, and
  never used (the model adapter owns its device). `EvaluateConfig` is removed:
  dead code whose docstring falsely claimed its fields fed the cache key (#163).
- `Prediction.intrinsics` / `.confidence` docstrings now state the real
  contracts (#175): `K` must match the *depth map's* pixel grid (not the input
  image's, as previously documented and contradicted by the runner and every
  multi-view adapter), and `confidence` is model-native and monotone rather than
  normalized to `[0, 1]` — half the adapters emit unbounded values.
- `marigold` gained `per_sample_seed` (default off) — the shared generator makes
  per-sample output depend on evaluation order, so subset/resumed runs disagree
  with full runs. Default is unchanged because the published Marigold cells were
  produced under the streaming-generator semantics (#171).
- ETH3D per-scene GT caches are now single-slot; a 13-scene run no longer
  retains every prior scene's ~450 MB cloud (#182).
- The DUSt3R and MonST3R single-frame eval helpers were byte-identical copies;
  hoisted to one shared `dust3r_single_frame_eval` (#182).
- `torch==2.6.0` pin documented rather than widened — a range would assert
  compatibility that has not been tested against the reproduction suite (#180).

### CI / tooling
- Weights/GPU/data tests no longer auto-run off environment sniffing (#178);
  they are opt-in via `-m weights`. This alone takes the default suite from
  ~880s to ~25s and stops surprise HuggingFace downloads mid-run.
- Hypothesis deadline disabled via a registered profile — the property tests
  assert math, not latency, and flaked under load (#170).
- `wheels.yml` can no longer publish to PyPI from a non-tag `workflow_dispatch`,
  and now verifies the tag matches `[project].version` before building (#168).
- CI installs CPU torch (`UV_TORCH_BACKEND=cpu`) instead of pulling the multi-GB
  CUDA stack into five GPU-less jobs per push (#169).
- Added `concurrency` cancellation and `timeout-minutes` to both workflows (#179).
- mypy now covers the test fakes and conftest; `CITATION.cff` version is pinned
  to `__version__` by a test after drifting two releases (#181).
- New tests: PLY parser guards, packaging/resource resolution, and manifest
  cache-poisoning regressions.

- **DUSt3R indoor depth (NYU + Bonn) now reproduce** — both were off-paper `ℹ️`
  cells under wrong recipes/targets, not "paper-private" as previously recorded
  (D28 RESOLVED). Verified-cell count 42 → **44**.
  - `dust3r-nyuv2` — **AbsRel 0.0637 vs 0.065** (−2.0 %, was 0.0777 / +19.5 %).
    DUSt3R's own Table 2 uses the classical **Eigen-2014 crop + ratio-of-medians**,
    which the *later* lineage protocol (`nyu_dust3r_lineage`) drops. New
    `nyu_dust3r_table2` protocol (filled GT + Eigen crop) + `median_lineage`.
    Independently corroborated by MonST3R Table 3, which re-scores DUSt3R-NYU
    *no-crop* at 0.080 ≈ plumbline's old 0.0777.
  - `dust3r-bonn` — **AbsRel 0.1384 vs 0.141** (−1.8 %, was 0.1337 vs 0.0808 /
    +65.5 %). Re-targeted from DUSt3R's own Table 2 (0.0808, scored on an
    unrecoverable 2023 sequence selection) to **MonST3R Table 3's DUSt3R
    baseline (0.141)**, which is DUSt3R single-frame depth on exactly the
    `rgb_110` 5-sequence set plumbline evaluates. The "DUSt3R is a static model,
    fails on dynamic Bonn" explanation is **retracted** (DUSt3R's own paper
    reports 0.0808 on Bonn). New Bonn loader `prepared_110` mode +
    `bonn_lineage_110_single` protocol.
- Retracted the false claim that DUSt3R is the origin of the `nyu_dust3r_lineage`
  (no-crop) convention — that's a MonST3R/CUT3R-era choice; DUSt3R predates it.

## [0.2.1] — 2026-06-16

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
- **DAGE Table 4 pose — TUM-Dynamics column.** New `tum-dynamics`
  loader (8 freiburg3 dynamic sequences, MonST3R `prepare_tum.py` prep replicated
  at read time), extending the Sim(3)-aligned trajectory-ATE apparatus beyond
  Sintel. `dage-tum-pose` is a new ✅ cell — **ATE 0.0136 vs 0.014** (−2.9 %, 8/8
  clips, GTX 1080 Ti). Staged via `scripts/stage_tum_dynamics.py` (public, no ToS;
  member-selective ~366 MB). (#44)
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
- 11 dataset loaders: NYUv2, KITTI, DIODE, ETH3D, DTU MVS, CO3Dv2, 7-Scenes,
  GSO, iBims-1, Sintel, and Bonn RGB-D Dynamic.
- Depth, pose (absolute + pairwise relative-pose AUC, RRA/RTA, trajectory
  ATE/RPE), and point-map (7-DoF similarity / per-view-masked chamfer)
  metric families.
- **33 PDF-verified paper-match reproductions** (29 mono-depth + 4 multi-view
  pose/trajectory), each audited table-+-column-+-row against the source
  paper. See [`REPRODUCTIONS.md`](./REPRODUCTIONS.md).

[Unreleased]: https://github.com/kmatzen/plumbline/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/kmatzen/plumbline/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kmatzen/plumbline/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kmatzen/plumbline/releases/tag/v0.1.0
