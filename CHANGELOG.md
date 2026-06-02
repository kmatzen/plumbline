# Changelog

All notable changes to plumbline are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public API may change between 0.x releases.

## [Unreleased]

### Added
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

## [0.2.0] — 2026-06-02

First release that **bundles** the dust3r-lineage + DAGE model code instead of
cloning it. As a result the published wheel now contains NonCommercial vendored
source — **the distribution as a whole is usable for non-commercial purposes
only** (plumbline's own code stays Apache-2.0). See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

### Added
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

### Fixed
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

### Changed
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
