# Changelog

All notable changes to plumbline are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public API may change between 0.x releases.

## [Unreleased]

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
