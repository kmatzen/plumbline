# Changelog

All notable changes to plumbline are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public API may change between 0.x releases.

## [Unreleased]

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

[Unreleased]: https://github.com/kmatzen/plumbline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kmatzen/plumbline/releases/tag/v0.1.0
